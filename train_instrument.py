"""Deep instrumentation harness for a hoist-enabled training run.

Mirrors train_optimizer.py but caps at 30 steps with sticky_interval=25 so
we see two reshuffle boundaries (step 0 and step 25) and a full block in
between.  Logs every event to /tmp/train_instrument.jsonl with
nanosecond timestamps and a final summary table.

What it answers:
  - Is the surgery actually being applied to the same model the trainer
    iterates?  (Same Python object check.)
  - How long does each phase take per step:  callback, scheduler.step,
    surgery, forward, backward, optimizer.step.
  - VRAM peak at every reshuffle and per-step.
  - Per-layer parameter counts + bytes (so we know what hoist is freeing).
  - Are any forward/backward hooks still attached on the model?  (With
    use_layer_hoist=True the published code skips _install_hooks; this
    confirms.)
  - Length of model.layers before/after surgery.
"""

import json
import os
import time

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler

# Hoist + stub classes may not exist in older published versions of the
# package.  Fall back to sentinel types so isinstance() checks always
# return False on those builds — the rest of the harness still runs.
try:
    from deep_chaos_scheduler.deep_chaos import HoistStub
except ImportError:
    HoistStub = type(None)
try:
    from deep_chaos_scheduler.deep_chaos import _ZeroAttnStub, _ZeroMLPStub
    _ZERO_STUBS_AVAILABLE = True
except ImportError:
    _ZeroAttnStub = _ZeroMLPStub = type(None)
    _ZERO_STUBS_AVAILABLE = False


SEED = 199
STEPS_TO_RUN = 30
STICKY = 25
MODEL_ID = "Qwen/Qwen2.5-3b-Instruct"
EVENT_LOG_PATH = "/tmp/train_instrument.jsonl"
SUMMARY_PATH = "/tmp/train_instrument_summary.txt"


# --------------------------------------------------------------------------- #
#  Event logger                                                               #
# --------------------------------------------------------------------------- #


class EventLog:
    def __init__(self, path: str):
        self._path = path
        self._fh = open(path, "w")
        self._t0 = time.perf_counter_ns()

    def log(self, event: str, **fields):
        record = {
            "t_ns": time.perf_counter_ns() - self._t0,
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


LOG = EventLog(EVENT_LOG_PATH)


# --------------------------------------------------------------------------- #
#  Phase timer (context manager)                                              #
# --------------------------------------------------------------------------- #


class PhaseTimer:
    def __init__(self, event: str, **fields):
        self._event = event
        self._fields = fields
        self._t0 = 0

    def __enter__(self):
        LOG.log(f"{self._event}.enter", **self._fields)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dur_ns = time.perf_counter_ns() - self._t0
        LOG.log(f"{self._event}.exit", duration_ns=dur_ns, **self._fields)
        return False


# --------------------------------------------------------------------------- #
#  VRAM / parameter inventory                                                 #
# --------------------------------------------------------------------------- #


def vram_snapshot(label: str):
    if not torch.cuda.is_available():
        LOG.log("vram", label=label, allocated_mb=-1, reserved_mb=-1)
        return
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    LOG.log(
        "vram",
        label=label,
        allocated_mb=round(alloc, 2),
        reserved_mb=round(reserved, 2),
        peak_mb=round(peak, 2),
    )


def per_layer_inventory(model: nn.Module):
    """Walk model.model.layers and report params + bytes per layer."""
    if hasattr(model, "module"):
        m = model.module
    else:
        m = model
    try:
        layers = m.model.layers
    except AttributeError:
        LOG.log("layer_inventory.error", reason="no model.model.layers")
        return
    rows = []
    for i, layer in enumerate(layers):
        n_params = sum(p.numel() for p in layer.parameters())
        n_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
        rows.append((i, n_params, n_bytes, type(layer).__name__))
        LOG.log(
            "layer_param",
            idx=i,
            params=int(n_params),
            bytes=int(n_bytes),
            cls=type(layer).__name__,
        )
    total_params = sum(r[1] for r in rows)
    total_bytes = sum(r[2] for r in rows)
    LOG.log("layer_param.total", params=int(total_params), bytes=int(total_bytes))


def hook_audit(model: nn.Module):
    """Count forward / forward_pre / backward hooks across every module.
    With use_layer_hoist=True the published code skips _install_hooks, so
    we expect this to be near-zero (modulo any HF-internal hooks)."""
    fw, fwpre, bw = 0, 0, 0
    for module in model.modules():
        fw += len(module._forward_hooks)
        fwpre += len(module._forward_pre_hooks)
        bw += len(getattr(module, "_backward_hooks", {}) or {}) + len(
            getattr(module, "_full_backward_hooks", {}) or {}
        )
    LOG.log("hook_audit", forward=fw, forward_pre=fwpre, backward=bw)


def model_structure_tree(model: nn.Module, max_depth: int = 3):
    """Compact structure dump: top-level modules + immediate children."""

    def walk(mod, prefix="", depth=0):
        if depth > max_depth:
            return
        for name, child in mod.named_children():
            n_params = sum(p.numel() for p in child.parameters(recurse=False))
            LOG.log(
                "structure",
                path=f"{prefix}{name}",
                cls=type(child).__name__,
                direct_params=int(n_params),
                depth=depth,
            )
            walk(child, f"{prefix}{name}.", depth + 1)

    walk(model)


# --------------------------------------------------------------------------- #
#  Identity check: scheduler vs trainer model                                 #
# --------------------------------------------------------------------------- #


def identity_check(scheduler: DeepChaosScheduler, trainer_model, label: str):
    sched_parent = scheduler._hoist_parent
    sched_attr = scheduler._hoist_attr
    if sched_parent is None or sched_attr is None:
        LOG.log("identity_check", label=label, status="hoist_parent_none")
        return
    sched_layers = getattr(sched_parent, sched_attr, None)

    tm = trainer_model
    if hasattr(tm, "module"):
        tm = tm.module
    try:
        trainer_layers = tm.model.layers
    except AttributeError:
        trainer_layers = None

    same_obj = sched_layers is trainer_layers
    swapped_count = 0
    for layer in scheduler._hoist_originals:
        if isinstance(getattr(layer, "mlp", None), _ZeroMLPStub):
            swapped_count += 1
        if isinstance(getattr(layer, "self_attn", None), _ZeroAttnStub):
            swapped_count += 1
    yanked_count = len(scheduler._hoist_last_yanked)
    LOG.log(
        "identity_check",
        label=label,
        sched_layers_id=id(sched_layers) if sched_layers is not None else None,
        trainer_layers_id=id(trainer_layers) if trainer_layers is not None else None,
        same_obj=bool(same_obj),
        sched_layers_len=len(sched_layers) if sched_layers is not None else -1,
        trainer_layers_len=len(trainer_layers) if trainer_layers is not None else -1,
        yanked=yanked_count,
        swapped_submodules=swapped_count,
    )


# --------------------------------------------------------------------------- #
#  Setup                                                                      #
# --------------------------------------------------------------------------- #


set_seed(SEED)
LOG.log("config", seed=SEED, steps=STEPS_TO_RUN, sticky=STICKY, model=MODEL_ID)

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    LOG.log(
        "device",
        name=torch.cuda.get_device_name(0),
        rocm=str(torch.version.hip),
        vram_free_gb=round(free / 1024**3, 2),
        vram_total_gb=round(total / 1024**3, 2),
    )

vram_snapshot("session_start")

with PhaseTimer("dataset.load"):
    raw = load_dataset("simplescaling/s1K", split="train")

    def format_s1(example):
        thinking = example["thinking_trajectories"]
        if isinstance(thinking, list):
            thinking = thinking[0]
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {
                    "role": "assistant",
                    "content": f"<think>\n{thinking}\n</think>\n{example['solution']}",
                },
            ]
        }

    clean_dataset = raw.map(format_s1, remove_columns=raw.column_names)

LOG.log("dataset", size=len(clean_dataset))

with PhaseTimer("tokenizer.load"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

with PhaseTimer("model.load"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

model.config.use_cache = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.do_sample = False
model.generation_config.top_k = None

vram_snapshot("after_model_load")
per_layer_inventory(model)
model_structure_tree(model)
hook_audit(model)

# --------------------------------------------------------------------------- #
#  Build the scheduler with hoist + bias stub                                 #
# --------------------------------------------------------------------------- #

with PhaseTimer("scheduler.construct"):
    chaos_config = DeepChaosConfig(
        sticky_interval=STICKY,
        seed=SEED,
        use_layer_hoist=True,
        hoist_stub_kind="bias",
        hoist_stub_init_scale=0.01,
        announce_reshuffles=False,  # keep our log clean
    )
    chaos_scheduler = DeepChaosScheduler(model=model, config=chaos_config)

vram_snapshot("after_scheduler_construct")
hook_audit(model)
identity_check(chaos_scheduler, model, "after_scheduler_construct")


# --------------------------------------------------------------------------- #
#  Wrap scheduler.step + surgery to record entry/exit                         #
# --------------------------------------------------------------------------- #

_orig_scheduler_step = chaos_scheduler.step


def _instrumented_step(global_step):
    LOG.log("scheduler.step.enter", global_step=int(global_step))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    stats = _orig_scheduler_step(global_step)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dur = time.perf_counter_ns() - t0
    LOG.log(
        "scheduler.step.exit",
        global_step=int(global_step),
        duration_ns=dur,
        reshuffle=bool(stats.get("reshuffle_event", 0.0) == 1.0),
        kept=int(stats.get("hoist_kept_layers", 0)),
        yanked=int(stats.get("hoist_yanked_layers", 0)),
        active_layers=int(stats.get("active_layers", 0)),
    )
    return stats


chaos_scheduler.step = _instrumented_step

# Wrap the surgery method too so we time JUST the surgery (not the whole step).
_orig_surgery = chaos_scheduler._apply_layer_hoist_surgery


def _instrumented_surgery():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    result = _orig_surgery()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dur = time.perf_counter_ns() - t0
    LOG.log("surgery", duration_ns=dur, kept=result[0], yanked=result[1])
    return result


chaos_scheduler._apply_layer_hoist_surgery = _instrumented_surgery


# --------------------------------------------------------------------------- #
#  Trainer callback that times every callback firing + verifies state         #
# --------------------------------------------------------------------------- #


class InstrumentationCallback(TrainerCallback):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._step_t0 = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        LOG.log("trainer.on_train_begin", global_step=int(state.global_step))
        vram_snapshot("on_train_begin")
        identity_check(self.scheduler, model, "on_train_begin")
        hook_audit(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_t0 = time.perf_counter_ns()
        LOG.log("trainer.on_step_begin", global_step=int(state.global_step))
        # Drive the scheduler.
        self.scheduler.step(state.global_step)
        # Verify identity + live layer count AFTER surgery may have fired.
        identity_check(self.scheduler, model, f"on_step_begin_{state.global_step}")
        if int(state.global_step) <= 2 or int(state.global_step) % 5 == 0:
            vram_snapshot(f"on_step_begin_{state.global_step}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dur = time.perf_counter_ns() - self._step_t0
        LOG.log(
            "trainer.on_step_end",
            global_step=int(state.global_step),
            full_step_ns=dur,
        )
        if int(state.global_step) <= 2 or int(state.global_step) % 5 == 0:
            vram_snapshot(f"on_step_end_{state.global_step}")

        # Hard stop after our budget so we don't waste droplet time.
        if int(state.global_step) >= STEPS_TO_RUN:
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        LOG.log("trainer.on_log", global_step=int(state.global_step), logs=dict(logs or {}))


# --------------------------------------------------------------------------- #
#  SFT config — keep dataset constants identical to train_optimizer.py        #
# --------------------------------------------------------------------------- #


sft_config = SFTConfig(
    output_dir=f"./instrument-deepchaos-3b-{SEED}",
    max_steps=STEPS_TO_RUN,  # NOTE: max_steps caps total optimizer steps
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    warmup_steps=0,
    logging_steps=1,
    save_steps=10_000,  # don't save during instrumentation
    save_total_limit=1,
    bf16=True,
    max_length=8192,
    report_to=[],  # no wandb during instrumentation; [] is more portable than "none"
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    include_num_input_tokens_seen=True,
    assistant_only_loss=True,
    dataloader_num_workers=4,
    seed=SEED,
)


with PhaseTimer("trainer.construct"):
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=clean_dataset,
        processing_class=tokenizer,
        callbacks=[InstrumentationCallback(chaos_scheduler)],
    )

vram_snapshot("after_trainer_construct")
hook_audit(model)

# Critical: did SFTTrainer / accelerate replace the model object?
# If trainer.model is a different Python object than `model`, the scheduler
# is doing surgery on a phantom.
LOG.log(
    "trainer.model_identity",
    same_as_original=bool(trainer.model is model),
    trainer_model_type=type(trainer.model).__name__,
    original_model_type=type(model).__name__,
)
# Also check trainer.model_wrapped if it exists
mw = getattr(trainer, "model_wrapped", None)
if mw is not None:
    LOG.log(
        "trainer.model_wrapped_identity",
        same_as_original=bool(mw is model),
        same_as_trainer_model=bool(mw is trainer.model),
        type=type(mw).__name__,
    )


# --------------------------------------------------------------------------- #
#  Train                                                                      #
# --------------------------------------------------------------------------- #

with PhaseTimer("trainer.train"):
    trainer.train()

vram_snapshot("after_train")
hook_audit(model)
identity_check(chaos_scheduler, trainer.model, "after_train")
# Re-inventory using trainer.model — this is what the trainer was actually
# iterating.  Compare against the pre-train snapshot to see what hoist left
# in place at the end of the last sticky block.
per_layer_inventory(trainer.model)

LOG.log("done")
LOG.close()


# --------------------------------------------------------------------------- #
#  Render summary table                                                       #
# --------------------------------------------------------------------------- #


def render_summary():
    events = []
    with open(EVENT_LOG_PATH) as fh:
        for line in fh:
            events.append(json.loads(line))

    lines = []
    lines.append("=" * 90)
    lines.append("INSTRUMENT SUMMARY")
    lines.append("=" * 90)

    # Phase timings.
    phase_pairs = {}
    for ev in events:
        et = ev["event"]
        if et.endswith(".enter"):
            phase_pairs.setdefault(et[:-6], {})["t0"] = ev["t_ns"]
        elif et.endswith(".exit"):
            phase_pairs.setdefault(et[:-5], {})["t1"] = ev["t_ns"]
            phase_pairs[et[:-5]]["dur"] = ev.get("duration_ns", ev["t_ns"] - phase_pairs[et[:-5]].get("t0", ev["t_ns"]))

    lines.append("\nPhase timings (ms):")
    for name, p in phase_pairs.items():
        if "dur" in p:
            lines.append(f"  {name:<32} {p['dur'] / 1e6:>10.2f} ms")

    # Per-step breakdown.  "step_total" = wall-clock from on_step_begin entry
    # to on_step_end exit, which includes scheduler.step + surgery + forward
    # + backward + optimizer.step.  NOT just the dense compute.
    lines.append("\nPer-step wall-clock (ms):")
    lines.append(f"  {'step':>4} {'step_total':>11} {'sched.step':>12} {'reshuffle':>10} {'kept':>5} {'yanked':>7} {'identity_same_obj':>17}")
    step_full = {}
    step_sched = {}
    step_reshuffle = {}
    step_kept = {}
    step_yanked = {}
    step_same_obj = {}
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
                step = int(ev["label"].split("_")[-1])
                step_same_obj[step] = ev.get("same_obj", "?")
            except (ValueError, IndexError):
                pass

    for step in sorted(step_full.keys()):
        full = step_full.get(step, 0) / 1e6
        sched = step_sched.get(step, 0) / 1e6
        rs = "YES" if step_reshuffle.get(step) else ""
        kept = step_kept.get(step, "")
        yanked = step_yanked.get(step, "")
        same = str(step_same_obj.get(step, "?"))
        lines.append(f"  {step:>4} {full:>10.2f} {sched:>12.2f} {rs:>10} {kept:>5} {yanked:>7} {same:>17}")

    # VRAM trajectory.
    lines.append("\nVRAM trajectory (allocated MB):")
    for ev in events:
        if ev["event"] == "vram":
            lines.append(f"  {ev['label']:<35} alloc={ev['allocated_mb']:>10.1f}  reserved={ev['reserved_mb']:>10.1f}  peak={ev.get('peak_mb', -1):>10.1f}")

    # Hook audit.
    lines.append("\nHook audit history:")
    for ev in events:
        if ev["event"] == "hook_audit":
            lines.append(f"  forward={ev['forward']}  forward_pre={ev['forward_pre']}  backward={ev['backward']}")

    # Identity warnings.
    lines.append("\nIdentity check results:")
    for ev in events:
        if ev["event"] == "identity_check":
            same = ev.get("same_obj", "?")
            marker = "[OK]" if same is True else "[!!]"
            lines.append(
                f"  {marker} {ev.get('label', '?'):<32} same_obj={same} "
                f"sched_len={ev.get('sched_layers_len', '?'):<3} "
                f"trainer_len={ev.get('trainer_layers_len', '?'):<3} "
                f"yanked={ev.get('yanked', '?')} "
                f"swapped={ev.get('swapped_submodules', '?')}"
            )

    # Trainer model identity (the BIG one).
    lines.append("\nTrainer model identity:")
    for ev in events:
        if ev["event"] in ("trainer.model_identity", "trainer.model_wrapped_identity"):
            lines.append(f"  {ev['event']}: {ev}")

    # Layer params summary.
    lines.append("\nLayer params (top 5 by bytes):")
    layer_params = [ev for ev in events if ev["event"] == "layer_param"]
    layer_params.sort(key=lambda e: e["bytes"], reverse=True)
    for ev in layer_params[:5]:
        lines.append(
            f"  layer {ev['idx']:>2}: {ev['cls']:<30} params={ev['params']:>12,}  bytes={ev['bytes'] / 1024**2:>8.1f} MB"
        )
    total_evs = [ev for ev in events if ev["event"] == "layer_param.total"]
    if total_evs:
        ev = total_evs[-1]
        lines.append(
            f"  TOTAL: params={ev['params']:>14,}  bytes={ev['bytes'] / 1024**2:>10.1f} MB"
        )

    text = "\n".join(lines) + "\n"
    with open(SUMMARY_PATH, "w") as fh:
        fh.write(text)
    print(text)


render_summary()
