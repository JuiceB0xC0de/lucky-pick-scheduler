"""Modal version of the instrumentation harness.

Runs train_instrument.py logic on a Modal H100 (or A100-80GB) so we can
keep iterating while the AMD droplet is down.  Same 30 optimizer steps,
same sticky=25, same Qwen2.5-3B target — but on NVIDIA so we can verify
the optimizer is actually firing without waiting on AMD to come back.

Output:
  - prints the summary to stdout (visible in `modal run` output)
  - writes the JSONL + summary to /modal-results volume so we can pull
    them back with `modal volume get`

Usage:
    modal run train_instrument_modal.py
    modal run train_instrument_modal.py::run --steps 30 --sticky 25 --gpu H100
"""

import modal

app = modal.App("deep-chaos-instrument")

SECRETS = [
    modal.Secret.from_name("huggingface"),
]

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("instrument-results", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/results": results_vol,
}

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .uv_pip_install(
        "torch>=2.4",
        "transformers>=4.51.0",
        "trl>=0.12.0",
        "datasets",
        "accelerate",
        "huggingface-hub",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
    # Add the local deep_chaos_scheduler package — we want THIS exact code
    # under test, not whatever is pushed to GitHub.  Edits to the local
    # package are picked up on the next `modal run`.
    .add_local_python_source("deep_chaos_scheduler")
)


def _run_instrument(steps: int, sticky: int, model_id: str, seed: int,
                    use_hoist: bool = True, mode: str = "chaos",
                    dtype: str = "bf16", max_len: int = 8192,
                    attn_impl: str = "sdpa"):
    """mode: 'chaos' (DeepChaos active, hoist on/off via use_hoist),
             'vanilla' (no scheduler at all — control run).
    dtype: 'bf16' | 'fp32'
    max_len: token sequence cap (8192 default; try 2048 to rule out attn overflow)
    attn_impl: 'sdpa' | 'eager'
    """
    """The actual instrumentation logic.  Body matches train_instrument.py
    closely; only the I/O paths differ (writes to /results so we can pull
    back via `modal volume get`)."""
    import json
    import os
    import time

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler

    # Stub class try/imports — older builds may not have them.
    try:
        from deep_chaos_scheduler.deep_chaos import HoistStub
    except ImportError:
        HoistStub = type(None)
    try:
        from deep_chaos_scheduler.deep_chaos import _ZeroAttnStub, _ZeroMLPStub
    except ImportError:
        _ZeroAttnStub = _ZeroMLPStub = type(None)

    if mode == "vanilla":
        suffix = "vanilla"
    else:
        suffix = "hoist" if use_hoist else "baseline"
    diag_tag = ""
    if dtype != "bf16":
        diag_tag += f"_{dtype}"
    if max_len != 8192:
        diag_tag += f"_len{max_len}"
    if attn_impl != "sdpa":
        diag_tag += f"_{attn_impl}"
    suffix += diag_tag
    EVENT_LOG_PATH = f"/results/train_instrument_{suffix}.jsonl"
    SUMMARY_PATH = f"/results/train_instrument_summary_{suffix}.txt"
    os.makedirs("/results", exist_ok=True)

    # ----- event logger ----- #
    class EventLog:
        def __init__(self, path):
            self._fh = open(path, "w")
            self._t0 = time.perf_counter_ns()

        def log(self, event, **fields):
            record = {"t_ns": time.perf_counter_ns() - self._t0, "event": event, **fields}
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()

        def close(self):
            self._fh.close()

    LOG = EventLog(EVENT_LOG_PATH)

    class PhaseTimer:
        def __init__(self, event, **fields):
            self._event = event
            self._fields = fields

        def __enter__(self):
            LOG.log(f"{self._event}.enter", **self._fields)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t0 = time.perf_counter_ns()
            return self

        def __exit__(self, *exc):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dur = time.perf_counter_ns() - self._t0
            LOG.log(f"{self._event}.exit", duration_ns=dur, **self._fields)
            return False

    def vram_snapshot(label, reset_peak=False):
        if not torch.cuda.is_available():
            LOG.log("vram", label=label, allocated_mb=-1, reserved_mb=-1)
            return
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        LOG.log("vram", label=label, allocated_mb=round(alloc, 2),
                reserved_mb=round(reserved, 2), peak_mb=round(peak, 2))
        # Reset peak so the NEXT label captures a fresh window.  Critical
        # for per-step peak measurements — without this every reading
        # reflects the cumulative high-water since process start.
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()

    def per_layer_inventory(model):
        m = model.module if hasattr(model, "module") else model
        try:
            layers = m.model.layers
        except AttributeError:
            LOG.log("layer_inventory.error", reason="no model.model.layers")
            return
        for i, layer in enumerate(layers):
            n_params = sum(p.numel() for p in layer.parameters())
            n_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
            LOG.log("layer_param", idx=i, params=int(n_params),
                    bytes=int(n_bytes), cls=type(layer).__name__)
        total_p = sum(sum(p.numel() for p in l.parameters()) for l in layers)
        total_b = sum(sum(p.numel() * p.element_size() for p in l.parameters()) for l in layers)
        LOG.log("layer_param.total", params=int(total_p), bytes=int(total_b))

    def hook_audit(model):
        fw, fwpre, bw = 0, 0, 0
        for module in model.modules():
            fw += len(module._forward_hooks)
            fwpre += len(module._forward_pre_hooks)
            bw += len(getattr(module, "_backward_hooks", {}) or {}) + \
                  len(getattr(module, "_full_backward_hooks", {}) or {})
        LOG.log("hook_audit", forward=fw, forward_pre=fwpre, backward=bw)

    def identity_check(scheduler, trainer_model, label):
        sched_parent = scheduler._hoist_parent
        sched_attr = scheduler._hoist_attr
        if sched_parent is None or sched_attr is None:
            LOG.log("identity_check", label=label, status="hoist_parent_none")
            return
        sched_layers = getattr(sched_parent, sched_attr, None)
        tm = trainer_model.module if hasattr(trainer_model, "module") else trainer_model
        try:
            trainer_layers = tm.model.layers
        except AttributeError:
            trainer_layers = None
        same_obj = sched_layers is trainer_layers
        swapped = 0
        for layer in scheduler._hoist_originals:
            if isinstance(getattr(layer, "mlp", None), _ZeroMLPStub):
                swapped += 1
            if isinstance(getattr(layer, "self_attn", None), _ZeroAttnStub):
                swapped += 1
        LOG.log(
            "identity_check", label=label,
            sched_layers_id=id(sched_layers) if sched_layers is not None else None,
            trainer_layers_id=id(trainer_layers) if trainer_layers is not None else None,
            same_obj=bool(same_obj),
            sched_layers_len=len(sched_layers) if sched_layers is not None else -1,
            trainer_layers_len=len(trainer_layers) if trainer_layers is not None else -1,
            yanked=len(scheduler._hoist_last_yanked),
            swapped_submodules=swapped,
        )

    # ----- run ----- #
    set_seed(seed)
    LOG.log("config", seed=seed, steps=steps, sticky=sticky, model=model_id, host="modal")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        LOG.log(
            "device", name=torch.cuda.get_device_name(0),
            cuda=str(torch.version.cuda),
            vram_free_gb=round(free / 1024**3, 2),
            vram_total_gb=round(total / 1024**3, 2),
        )
    vram_snapshot("session_start")

    with PhaseTimer("dataset.load"):
        raw = load_dataset("simplescaling/s1K", split="train")

        def format_s1(ex):
            t = ex["thinking_trajectories"]
            if isinstance(t, list):
                t = t[0]
            return {
                "messages": [
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": f"<think>\n{t}\n</think>\n{ex['solution']}"},
                ]
            }

        clean_dataset = raw.map(format_s1, remove_columns=raw.column_names)
    LOG.log("dataset", size=len(clean_dataset))

    with PhaseTimer("tokenizer.load"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    _torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    LOG.log("config.diag", dtype=dtype, max_len=max_len, attn_impl=attn_impl)

    with PhaseTimer("model.load"):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=_torch_dtype,
            device_map="auto", attn_implementation=attn_impl,
        )
    model.config.use_cache = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False
    model.generation_config.top_k = None

    vram_snapshot("after_model_load")
    per_layer_inventory(model)
    hook_audit(model)

    chaos_scheduler = None
    if mode == "chaos":
        with PhaseTimer("scheduler.construct"):
            chaos_config = DeepChaosConfig(
                sticky_interval=sticky, seed=seed,
                use_layer_hoist=use_hoist,
                hoist_stub_kind="bias" if use_hoist else "none",
                hoist_stub_init_scale=0.01,
                announce_reshuffles=False,
            )
            chaos_scheduler = DeepChaosScheduler(model=model, config=chaos_config)
        vram_snapshot("after_scheduler_construct")
        hook_audit(model)
        identity_check(chaos_scheduler, model, "after_scheduler_construct")
    else:
        LOG.log("scheduler.skip", reason="vanilla_mode")

    # Wrap step + surgery for timing.
    if chaos_scheduler is not None:
        _orig_step = chaos_scheduler.step

        def _instrumented_step(global_step):
            LOG.log("scheduler.step.enter", global_step=int(global_step))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            stats = _orig_step(global_step)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            LOG.log(
                "scheduler.step.exit", global_step=int(global_step),
                duration_ns=time.perf_counter_ns() - t0,
                reshuffle=bool(stats.get("reshuffle_event", 0.0) == 1.0),
                kept=int(stats.get("hoist_kept_layers", 0)),
                yanked=int(stats.get("hoist_yanked_layers", 0)),
                active_layers=int(stats.get("active_layers", 0)),
            )
            return stats

        chaos_scheduler.step = _instrumented_step

        _orig_surgery = chaos_scheduler._apply_layer_hoist_surgery

        def _instrumented_surgery():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            result = _orig_surgery()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            LOG.log("surgery", duration_ns=time.perf_counter_ns() - t0,
                    kept=result[0], yanked=result[1])
            return result

        chaos_scheduler._apply_layer_hoist_surgery = _instrumented_surgery

    class InstrumentationCallback(TrainerCallback):
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self._step_t0 = 0

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            LOG.log("trainer.on_train_begin", global_step=int(state.global_step))
            vram_snapshot("on_train_begin")
            if self.scheduler is not None:
                identity_check(self.scheduler, model, "on_train_begin")
            hook_audit(model)

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._step_t0 = time.perf_counter_ns()
            LOG.log("trainer.on_step_begin", global_step=int(state.global_step))
            if self.scheduler is not None:
                self.scheduler.step(state.global_step)
                identity_check(self.scheduler, model, f"on_step_begin_{state.global_step}")
            # Reset peak so this step's peak window starts fresh.
            vram_snapshot(f"on_step_begin_{state.global_step}", reset_peak=True)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            LOG.log("trainer.on_step_end", global_step=int(state.global_step),
                    full_step_ns=time.perf_counter_ns() - self._step_t0)
            # Snapshot WITHOUT reset — captures the just-finished step's peak.
            vram_snapshot(f"on_step_end_{state.global_step}")
            if int(state.global_step) >= steps:
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            LOG.log("trainer.on_log", global_step=int(state.global_step),
                    logs=dict(logs or {}))

    sft_config = SFTConfig(
        output_dir="/tmp/instrument-out",
        max_steps=steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=1,
        save_steps=10_000,
        save_total_limit=1,
        bf16=(dtype == "bf16"),
        fp16=False,
        max_length=max_len,
        report_to=[],
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        include_num_input_tokens_seen=True,
        assistant_only_loss=True,
        dataloader_num_workers=4,
        seed=seed,
    )

    # Diagnose chat_template / assistant_only_loss masking.
    ct = tokenizer.chat_template or ""
    has_generation_tag = "{% generation %}" in ct
    LOG.log("chat_template_check",
            has_generation_tag=has_generation_tag,
            assistant_only_loss=True,
            warning="" if has_generation_tag else "MISSING {% generation %} — loss mask may be wrong")
    if not has_generation_tag:
        print("WARNING: chat_template lacks {% generation %} — verify assistant_only_loss masking")

    with PhaseTimer("trainer.construct"):
        trainer = SFTTrainer(
            model=model, args=sft_config, train_dataset=clean_dataset,
            processing_class=tokenizer,
            callbacks=[InstrumentationCallback(chaos_scheduler)],
        )
    vram_snapshot("after_trainer_construct")
    hook_audit(model)

    LOG.log("trainer.model_identity",
            same_as_original=bool(trainer.model is model),
            trainer_model_type=type(trainer.model).__name__,
            original_model_type=type(model).__name__)
    mw = getattr(trainer, "model_wrapped", None)
    if mw is not None:
        LOG.log("trainer.model_wrapped_identity",
                same_as_original=bool(mw is model),
                same_as_trainer_model=bool(mw is trainer.model),
                type=type(mw).__name__)

    with PhaseTimer("trainer.train"):
        trainer.train()

    vram_snapshot("after_train")
    hook_audit(model)
    if chaos_scheduler is not None:
        identity_check(chaos_scheduler, trainer.model, "after_train")
    per_layer_inventory(trainer.model)

    LOG.log("done")
    LOG.close()

    # Render summary.
    events = []
    with open(EVENT_LOG_PATH) as fh:
        for line in fh:
            events.append(json.loads(line))

    out = []
    out.append("=" * 90)
    out.append("INSTRUMENT SUMMARY (modal)")
    out.append("=" * 90)

    for ev in events:
        if ev["event"] == "config.diag":
            out.append(f"  diag config: dtype={ev['dtype']}  max_len={ev['max_len']}  attn_impl={ev['attn_impl']}")
        if ev["event"] == "chat_template_check":
            tag = "OK" if ev["has_generation_tag"] else "MISSING {% generation %} — loss mask may be wrong"
            out.append(f"  chat_template: {tag}")

    phase_pairs = {}
    for ev in events:
        et = ev["event"]
        if et.endswith(".enter"):
            phase_pairs.setdefault(et[:-6], {})["t0"] = ev["t_ns"]
        elif et.endswith(".exit"):
            slot = phase_pairs.setdefault(et[:-5], {})
            slot["t1"] = ev["t_ns"]
            slot["dur"] = ev.get("duration_ns", ev["t_ns"] - slot.get("t0", ev["t_ns"]))

    out.append("\nPhase timings (ms):")
    for name, p in phase_pairs.items():
        if "dur" in p:
            out.append(f"  {name:<32} {p['dur'] / 1e6:>10.2f} ms")

    out.append("\nPer-step wall-clock (ms):")
    out.append(f"  {'step':>4} {'step_total':>11} {'sched.step':>12} {'reshuffle':>10} {'kept':>5} {'yanked':>7} {'identity_same_obj':>17}")
    step_full, step_sched, step_reshuffle, step_kept, step_yanked, step_same_obj = {}, {}, {}, {}, {}, {}
    for ev in events:
        if ev["event"] == "trainer.on_step_end":
            step_full[ev["global_step"]] = ev["full_step_ns"]
        elif ev["event"] == "scheduler.step.exit":
            step_sched[ev["global_step"]] = ev["duration_ns"]
            step_reshuffle[ev["global_step"]] = ev["reshuffle"]
            step_kept[ev["global_step"]] = ev["kept"]
            step_yanked[ev["global_step"]] = ev["yanked"]
        elif ev["event"] == "identity_check" and ev.get("label", "").startswith("on_step_begin_"):
            try:
                s = int(ev["label"].split("_")[-1])
                step_same_obj[s] = ev.get("same_obj", "?")
            except (ValueError, IndexError):
                pass
    for s in sorted(step_full.keys()):
        full = step_full.get(s, 0) / 1e6
        sched = step_sched.get(s, 0) / 1e6
        rs = "YES" if step_reshuffle.get(s) else ""
        out.append(f"  {s:>4} {full:>11.2f} {sched:>12.2f} {rs:>10} "
                   f"{step_kept.get(s, ''):>5} {step_yanked.get(s, ''):>7} "
                   f"{str(step_same_obj.get(s, '?')):>17}")

    out.append("\nPer-step loss / grad_norm:")
    out.append(f"  {'step':>4} {'loss':>10} {'grad_norm':>12}")
    for ev in events:
        if ev["event"] == "trainer.on_log" and ev.get("logs"):
            lg = ev["logs"]
            loss = lg.get("loss", "")
            gn = lg.get("grad_norm", "")
            s = ev["global_step"]
            out.append(f"  {s:>4} {str(loss):>10} {str(gn):>12}")

    out.append("\nVRAM trajectory (allocated MB):")
    for ev in events:
        if ev["event"] == "vram":
            out.append(f"  {ev['label']:<35} alloc={ev['allocated_mb']:>10.1f}  "
                       f"reserved={ev['reserved_mb']:>10.1f}  "
                       f"peak={ev.get('peak_mb', -1):>10.1f}")

    out.append("\nHook audit history:")
    for ev in events:
        if ev["event"] == "hook_audit":
            out.append(f"  forward={ev['forward']}  forward_pre={ev['forward_pre']}  backward={ev['backward']}")

    out.append("\nIdentity check results:")
    for ev in events:
        if ev["event"] == "identity_check":
            same = ev.get("same_obj", "?")
            marker = "[OK]" if same is True else "[!!]"
            out.append(
                f"  {marker} {ev.get('label', '?'):<32} same_obj={same} "
                f"sched_len={ev.get('sched_layers_len', '?'):<3} "
                f"trainer_len={ev.get('trainer_layers_len', '?'):<3} "
                f"yanked={ev.get('yanked', '?')} swapped={ev.get('swapped_submodules', '?')}"
            )

    out.append("\nTrainer model identity:")
    for ev in events:
        if ev["event"] in ("trainer.model_identity", "trainer.model_wrapped_identity"):
            out.append(f"  {ev['event']}: {ev}")

    text = "\n".join(out) + "\n"
    with open(SUMMARY_PATH, "w") as fh:
        fh.write(text)
    print(text)
    results_vol.commit()


@app.function(
    image=train_image,
    gpu="H100",
    timeout=1800,
    secrets=SECRETS,
    volumes=VOLUMES,
)
def instrument(steps: int = 30, sticky: int = 25,
               model_id: str = "Qwen/Qwen2.5-3B-Instruct", seed: int = 199,
               use_hoist: bool = True, mode: str = "chaos",
               dtype: str = "bf16", max_len: int = 8192, attn_impl: str = "sdpa"):
    _run_instrument(steps, sticky, model_id, seed, use_hoist=use_hoist, mode=mode,
                    dtype=dtype, max_len=max_len, attn_impl=attn_impl)


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=1800,
    secrets=SECRETS,
    volumes=VOLUMES,
)
def instrument_a100(steps: int = 30, sticky: int = 25,
                    model_id: str = "Qwen/Qwen2.5-3B-Instruct", seed: int = 199,
                    use_hoist: bool = True, mode: str = "chaos",
                    dtype: str = "bf16", max_len: int = 8192, attn_impl: str = "sdpa"):
    _run_instrument(steps, sticky, model_id, seed, use_hoist=use_hoist, mode=mode,
                    dtype=dtype, max_len=max_len, attn_impl=attn_impl)


@app.local_entrypoint()
def run(steps: int = 30, sticky: int = 25,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        seed: int = 199, gpu: str = "H100", hoist: bool = True,
        mode: str = "chaos", dtype: str = "bf16", max_len: int = 8192,
        attn_impl: str = "sdpa"):
    """One-off run.

    Examples:
      modal run train_instrument_modal.py::run                          # chaos+hoist
      modal run train_instrument_modal.py::run --no-hoist               # chaos+post-hook
      modal run train_instrument_modal.py::run --mode vanilla           # NO chaos (control)
      modal run train_instrument_modal.py::run --mode vanilla --dtype fp32  # NaN diag: fp32
      modal run train_instrument_modal.py::run --mode vanilla --max-len 2048 # NaN diag: short seq
      modal run train_instrument_modal.py::run --mode vanilla --attn-impl eager # NaN diag: no SDPA
      modal run train_instrument_modal.py::run --gpu A100-80GB
    """
    fn = instrument_a100 if gpu.upper().startswith("A100") else instrument
    fn.remote(steps, sticky, model_id, seed, use_hoist=hoist, mode=mode,
              dtype=dtype, max_len=max_len, attn_impl=attn_impl)
    if mode == "vanilla":
        suffix = "vanilla"
    else:
        suffix = "hoist" if hoist else "baseline"
    diag_tag = ""
    if dtype != "bf16":
        diag_tag += f"_{dtype}"
    if max_len != 8192:
        diag_tag += f"_len{max_len}"
    if attn_impl != "sdpa":
        diag_tag += f"_{attn_impl}"
    suffix += diag_tag
    print(f"\nPull artifacts back with:")
    print(f"  modal volume get instrument-results train_instrument_{suffix}.jsonl ./")
    print(f"  modal volume get instrument-results train_instrument_summary_{suffix}.txt ./")


@app.local_entrypoint()
def nan_diag(steps: int = 10, model_id: str = "Qwen/Qwen2.5-3B-Instruct",
             seed: int = 199, gpu: str = "H100"):
    """Fire 3 vanilla diagnostic runs in parallel to isolate the step-4 NaN.

    Hypotheses being tested:
      A) bf16 precision:          vanilla_fp32   — if no NaN, it's bf16 overflow
      B) sequence-length attn:    vanilla_len2048 — if no NaN, it's long-seq attn overflow
      C) SDPA implementation:     vanilla_eager  — if no NaN, it's H100 SDPA-specific

      modal run train_instrument_modal.py::nan_diag
      modal run train_instrument_modal.py::nan_diag --gpu A100-80GB
    """
    fn = instrument_a100 if gpu.upper().startswith("A100") else instrument
    print("=== NaN diagnostic: 3 runs in parallel ===")
    print("A) fp32 — rules out bf16 overflow")
    fA = fn.spawn(steps, 25, model_id, seed, use_hoist=False, mode="vanilla",
                  dtype="fp32", max_len=8192, attn_impl="sdpa")
    print("B) max_len=2048 — rules out long-sequence attention overflow in bf16")
    fB = fn.spawn(steps, 25, model_id, seed, use_hoist=False, mode="vanilla",
                  dtype="bf16", max_len=2048, attn_impl="sdpa")
    print("C) eager attn — rules out H100 SDPA kernel bug")
    fC = fn.spawn(steps, 25, model_id, seed, use_hoist=False, mode="vanilla",
                  dtype="bf16", max_len=8192, attn_impl="eager")
    fA.get(); fB.get(); fC.get()
    print("\nPull summaries with:")
    print("  modal volume get instrument-results train_instrument_summary_vanilla_fp32.txt ./")
    print("  modal volume get instrument-results train_instrument_summary_vanilla_len2048.txt ./")
    print("  modal volume get instrument-results train_instrument_summary_vanilla_eager.txt ./")


@app.local_entrypoint()
def ab(steps: int = 30, sticky: int = 25,
       model_id: str = "Qwen/Qwen2.5-3B-Instruct",
       seed: int = 199, gpu: str = "H100"):
    """Run baseline (no hoist) AND hoist sequentially.  Two volumes get
    written; their summaries can be diffed for the wall-clock + VRAM
    delta.

      modal run train_instrument_modal.py::ab
    """
    fn = instrument_a100 if gpu.upper().startswith("A100") else instrument
    print("=== run 1: baseline (use_hoist=False) ===")
    fn.remote(steps, sticky, model_id, seed, use_hoist=False)
    print("=== run 2: hoist (use_hoist=True) ===")
    fn.remote(steps, sticky, model_id, seed, use_hoist=True)
    print("\nPull both artifacts back with:")
    print("  modal volume get instrument-results train_instrument_summary_baseline.txt ./")
    print("  modal volume get instrument-results train_instrument_summary_hoist.txt ./")
