"""A/B bench: baseline post-hook vs layer-hoist on AMD MI300X.

Uses a raw train loop (forward + backward + optimizer.step) with a fixed
synthetic token batch so we're measuring the scheduler mechanism only, not
dataloader or tokenizer overhead.

Usage:
    python bench_amd.py                        # Qwen2.5-3B, 100 steps
    python bench_amd.py --steps 50 --sticky 25
    python bench_amd.py --skip-baseline        # hoist only
    python bench_amd.py --skip-hoist           # baseline only
"""

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, set_seed

from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler

SEED = 19
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SEQ_LEN = 1024  # tokens per example — long enough to be realistic, short enough to be fast


def reset_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_vram_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def current_vram_mb():
    return torch.cuda.memory_allocated() / 1024**2


def load_model(dtype):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False
    model.generation_config.top_k = None
    return model


def make_batch(model, seq_len, batch_size=2):
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, seq_len),
                              device=next(model.parameters()).device)
    return input_ids


def run_bench(label, model, scheduler, input_ids, steps, warmup=5):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    model.train()

    # Warmup — not measured.
    for i in range(warmup):
        scheduler.step(i)
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        optim.step()

    torch.cuda.synchronize()
    reset_vram()

    per_step_ms = []
    shuffles = 0
    last_shuffle = getattr(scheduler, "last_shuffle_step", -1)

    t_total = time.perf_counter()
    for step in range(warmup, warmup + steps):
        stats = scheduler.step(step)
        if getattr(scheduler, "last_shuffle_step", -1) != last_shuffle:
            shuffles += 1
            last_shuffle = scheduler.last_shuffle_step

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optim.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        loss.backward()
        optim.step()

        torch.cuda.synchronize()
        per_step_ms.append((time.perf_counter() - t0) * 1000)

    total_s = time.perf_counter() - t_total
    peak = peak_vram_mb()
    current = current_vram_mb()

    per_step_ms.sort()
    n = len(per_step_ms)
    median = per_step_ms[n // 2]
    p90 = per_step_ms[int(0.9 * n)]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  total      : {total_s:.2f} s")
    print(f"  median/step: {median:.1f} ms")
    print(f"  p90/step   : {p90:.1f} ms")
    print(f"  peak VRAM  : {peak:.0f} MB")
    print(f"  current    : {current:.0f} MB")
    print(f"  shuffles   : {shuffles}")
    compute = stats.get("compute_pct", float("nan"))
    print(f"  compute_pct: {compute:.1f}%  (last block)")

    return {"total_s": total_s, "median_ms": median, "p90_ms": p90,
            "peak_mb": peak, "shuffles": shuffles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sticky", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-hoist", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    dtype = torch.bfloat16
    print(f"model   : {MODEL_ID}")
    print(f"steps   : {args.steps}  sticky={args.sticky}  seq_len={args.seq_len}")
    print(f"dtype   : {dtype}  device: {torch.cuda.get_device_name(0)}")

    r_baseline = None
    r_hoist = None

    if not args.skip_baseline:
        print("\n--- loading model for BASELINE run ---")
        model = load_model(dtype)
        config = DeepChaosConfig(
            sticky_interval=args.sticky,
            seed=SEED,
            use_layer_hoist=False,
            announce_reshuffles=False,
        )
        scheduler = DeepChaosScheduler(model=model, config=config)
        batch = make_batch(model, args.seq_len)
        r_baseline = run_bench("BASELINE (post-hook mask)", model, scheduler, batch,
                               args.steps)
        del model, scheduler, batch
        reset_vram()

    if not args.skip_hoist:
        print("\n--- loading model for HOIST run ---")
        model = load_model(dtype)
        config = DeepChaosConfig(
            sticky_interval=args.sticky,
            seed=SEED,
            use_layer_hoist=True,
            hoist_stub_kind="bias",
            hoist_stub_init_scale=0.01,
            announce_reshuffles=False,
        )
        scheduler = DeepChaosScheduler(model=model, config=config)
        batch = make_batch(model, args.seq_len)
        r_hoist = run_bench("HOIST (yank dead layers)", model, scheduler, batch,
                            args.steps)
        del model, scheduler, batch
        reset_vram()

    if r_baseline and r_hoist:
        print(f"\n{'='*60}")
        print("  VERDICT")
        print(f"{'='*60}")
        speedup = r_baseline["total_s"] / r_hoist["total_s"]
        vram_delta = r_hoist["peak_mb"] - r_baseline["peak_mb"]
        vram_pct = 100.0 * vram_delta / max(1.0, r_baseline["peak_mb"])
        print(f"  speedup    : {speedup:.2f}x  (hoist vs baseline)")
        print(f"  VRAM delta : {vram_delta:+.0f} MB  ({vram_pct:+.1f}%)")
        print()
        if speedup > 1.1:
            print("  ✓ hoist is faster")
        elif speedup < 0.9:
            print("  ✗ hoist is SLOWER")
        else:
            print("  ~ roughly the same speed")
        if vram_pct < -5:
            print(f"  ✓ hoist uses less VRAM ({vram_pct:.1f}%)")
        elif vram_pct > 5:
            print(f"  ✗ hoist uses MORE VRAM ({vram_pct:.1f}%)")
        else:
            print("  ~ VRAM roughly the same")


if __name__ == "__main__":
    main()
