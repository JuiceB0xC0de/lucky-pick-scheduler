"""A/B training run: vanilla SFT vs DeepChaos hoist.

Identical hyperparams, same seed, same dataset.
Controls which mode runs via --mode flag.

Usage:
    python train_ab.py --mode vanilla
    python train_ab.py --mode hoist
"""

import argparse
import os
import sys

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["vanilla", "hoist"], required=True)
args = parser.parse_args()

SEED = 199
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
set_seed(SEED)

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")
    print(f"ROCm  : {torch.version.hip}")

# ── AMDSMI telemetry ────────────────────────────────────────────────────────
try:
    import amdsmi
    amdsmi.amdsmi_init()
    _amd_handles = amdsmi.amdsmi_get_processor_handles()
    AMDSMI_OK = len(_amd_handles) > 0
except Exception as e:
    print(f"AMDSMI unavailable: {e}")
    _amd_handles = []
    AMDSMI_OK = False

# ── wandb ────────────────────────────────────────────────────────────────────
wandb.init(
    project="deep-chaos-ab",
    name=f"{args.mode}-3b-s1k-{SEED}",
    tags=[args.mode, "qwen2.5-3b", "s1k"],
    config={"mode": args.mode, "seed": SEED, "model": MODEL_ID},
)

# ── dataset ──────────────────────────────────────────────────────────────────
raw = load_dataset("simplescaling/s1K", split="train")

def format_s1(example):
    thinking = example["thinking_trajectories"]
    if isinstance(thinking, list):
        thinking = thinking[0]
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant",
             "content": f"<think>\n{thinking}\n</think>\n{example['solution']}"},
        ]
    }

clean_dataset = raw.map(format_s1, remove_columns=raw.column_names)
print(f"Dataset: {len(clean_dataset)} examples")

# ── model ────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if "generation" not in (tokenizer.chat_template or ""):
    print("WARNING: chat_template lacks {% generation %} — check assistant_only_loss masking")

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

# ── scheduler (hoist mode only) ──────────────────────────────────────────────
chaos_scheduler = None
if args.mode == "hoist":
    chaos_config = DeepChaosConfig(
        sticky_interval=50,
        seed=SEED,
        use_layer_hoist=True,
        hoist_stub_kind="bias",
        hoist_stub_init_scale=0.01,
        announce_reshuffles=False,
    )
    chaos_scheduler = DeepChaosScheduler(model=model, config=chaos_config)


# ── callbacks ────────────────────────────────────────────────────────────────
class DeepChaosCallback(TrainerCallback):
    def __init__(self, scheduler: DeepChaosScheduler):
        self.scheduler = scheduler
        self._last_stats = {}

    def on_step_begin(self, args, state, control, **kwargs):
        # Only call step() here — cache the stats for on_log to read.
        # Do NOT call step() again in on_log; the scheduler is already
        # advanced and a second call would double-count or return stale data
        # if global_step happens to tick between the two events.
        self._last_stats = self.scheduler.step(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        chaos_logs = {f"chaos/{k}": v for k, v in self._last_stats.items()}
        # commit=False — merges with the trainer's WandbCallback commit on the same step
        # so wandb sees one log entry per logging_steps, not two
        wandb.log(chaos_logs, step=state.global_step, commit=False)


class AMDTelemetryCallback(TrainerCallback):
    def __init__(self, device_index=0):
        self.device_index = device_index

    def on_log(self, args, state, control, logs=None, **kwargs):
        total_mem = torch.cuda.get_device_properties(self.device_index).total_memory
        # torch.cuda VRAM always available — log regardless of AMDSMI
        payload = {
            "gpu/vram_used_mb":     torch.cuda.memory_allocated(self.device_index) / 1024**2,
            "gpu/vram_reserved_mb": torch.cuda.memory_reserved(self.device_index) / 1024**2,
            "gpu/vram_peak_mb":     torch.cuda.max_memory_allocated(self.device_index) / 1024**2,
            "gpu/vram_used_pct":    torch.cuda.memory_allocated(self.device_index) / total_mem * 100,
        }
        # AMDSMI extras: power, temp, clocks — only when available
        if AMDSMI_OK and self.device_index < len(_amd_handles):
            try:
                handle   = _amd_handles[self.device_index]
                metrics  = amdsmi.amdsmi_get_gpu_metrics_info(handle)
                activity = amdsmi.amdsmi_get_gpu_activity(handle)
                clocks   = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
                mem_clk  = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
                payload.update({
                    "gpu/power_w":         metrics.get("average_socket_power"),
                    "gpu/power_limit_w":   metrics.get("power_limit"),
                    "gpu/temp_edge_c":     metrics.get("temperature_edge"),
                    "gpu/temp_hotspot_c":  metrics.get("temperature_hotspot"),
                    "gpu/gfx_util":        activity.get("gfx_activity"),
                    "gpu/mem_util":        activity.get("umc_activity"),
                    "gpu/gfx_clk_mhz":    clocks.get("clk"),
                    "gpu/mem_clk_mhz":    mem_clk.get("clk"),
                    "gpu/throttle_status": metrics.get("throttle_status"),
                })
            except Exception as e:
                print(f"AMDSMI logging error: {e}")
        payload = {k: v for k, v in payload.items()
                   if v is not None and isinstance(v, (int, float))}
        wandb.log(payload, step=state.global_step, commit=False)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if AMDSMI_OK:
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass


# ── training config ──────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=f"./ab-{args.mode}-3b-{SEED}",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=15,
    logging_steps=2,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    max_length=8192,
    report_to="wandb",
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    include_num_input_tokens_seen=True,
    assistant_only_loss=True,
    dataloader_num_workers=4,
    seed=SEED,
)

callbacks = [AMDTelemetryCallback()]
if chaos_scheduler is not None:
    callbacks.append(DeepChaosCallback(chaos_scheduler))

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=clean_dataset,
    processing_class=tokenizer,
    callbacks=callbacks,
)

trainer.train()
wandb.finish()
