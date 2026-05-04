import os

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
import wandb

from transformers import set_seed

SEED = 667
set_seed(SEED)

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"Confirmed: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {free / 1024**3:.2f} GB free of {total / 1024**3:.2f} GB")
    print(f"ROCm: {torch.version.hip}")

wandb.init(project="lucky-pick-benchmark", name=f"run-deepchaos-7b-{SEED}")

model_id = "Qwen/Qwen2.5-7B-Instruct"

from datasets import load_dataset

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
print(f"Dataset size: {len(clean_dataset)}")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

if "generation" not in (tokenizer.chat_template or ""):
    print("WARNING: chat_template lacks {% generation %} — verify assistant_only_loss masking")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

model.config.use_cache = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.do_sample = False
model.generation_config.top_k = None

from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from deep_chaos_scheduler import DeepChaosScheduler, DeepChaosConfig

try:
    import amdsmi
    amdsmi.amdsmi_init()
    _amd_handles = amdsmi.amdsmi_get_processor_handles()
    AMDSMI_OK = len(_amd_handles) > 0
except Exception as e:
    print(f"AMDSMI unavailable: {e}")
    _amd_handles = []
    AMDSMI_OK = False


chaos_config = DeepChaosConfig(
    sticky_interval=50,
    seed=SEED,
    use_layer_hoist=True,
)

chaos_scheduler = DeepChaosScheduler(
    model=model,
    config=chaos_config,
)


class DeepChaosCallback(TrainerCallback):
    def __init__(self, scheduler: DeepChaosScheduler):
        self.scheduler = scheduler

    def on_step_begin(self, args, state, control, **kwargs):
        self.scheduler.step(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        stats = self.scheduler.step(state.global_step)
        chaos_logs = {f"chaos/{k}": v for k, v in stats.items()}
        wandb.log(chaos_logs, step=state.global_step)


class AMDTelemetryCallback(TrainerCallback):
    def __init__(self, device_index=0):
        self.device_index = device_index

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not AMDSMI_OK or self.device_index >= len(_amd_handles):
            return
        try:
            handle   = _amd_handles[self.device_index]
            metrics  = amdsmi.amdsmi_get_gpu_metrics_info(handle)
            activity = amdsmi.amdsmi_get_gpu_activity(handle)
            clocks   = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
            mem_clk  = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
            total_mem = torch.cuda.get_device_properties(0).total_memory

            payload = {
                "gpu/vram_used_mb":     torch.cuda.memory_allocated(0) / 1024**2,
                "gpu/vram_reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
                "gpu/vram_used_pct":    (torch.cuda.memory_allocated(0) / total_mem) * 100,
                "gpu/power_w":          metrics.get("average_socket_power"),
                "gpu/power_limit_w":    metrics.get("power_limit"),
                "gpu/energy_acc_uj":    metrics.get("energy_accumulator"),
                "gpu/temp_edge_c":      metrics.get("temperature_edge"),
                "gpu/temp_hotspot_c":   metrics.get("temperature_hotspot"),
                "gpu/temp_mem_c":       metrics.get("temperature_mem"),
                "gpu/temp_vrgfx_c":     metrics.get("temperature_vrgfx"),
                "gpu/gfx_util":         activity.get("gfx_activity"),
                "gpu/mem_util":         activity.get("umc_activity"),
                "gpu/gfx_clk_mhz":      clocks.get("clk"),
                "gpu/gfx_clk_max_mhz":  clocks.get("max_clk"),
                "gpu/mem_clk_mhz":      mem_clk.get("clk"),
                "gpu/throttle_status":  metrics.get("throttle_status"),
            }
            payload = {k: v for k, v in payload.items()
                       if v is not None and isinstance(v, (int, float))}
            if payload:
                wandb.log(payload, step=state.global_step)
        except Exception as e:
            print(f"AMDSMI logging error: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if AMDSMI_OK:
                amdsmi.amdsmi_shut_down()
        except Exception:
            pass


sft_config = SFTConfig(
    output_dir=f"./benchmark-deepchaos-7b-{SEED}",
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

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=clean_dataset,
    processing_class=tokenizer,
    callbacks=[DeepChaosCallback(chaos_scheduler), AMDTelemetryCallback()],
)

trainer.train()
wandb.finish()
