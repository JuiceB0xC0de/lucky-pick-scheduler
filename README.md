<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/juiceb0xc0de"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD700?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://wandb.ai/ricks-holmberg-juiceb0xc0de/deep-chaos-evals?nw=nwuserricksholmberg"><img src="https://img.shields.io/badge/W%26B-Logging-FFBE00?logo=wandb&logoColor=black" alt="Weights and Biases"></a>
  <a href="https://github.com/huggingface/transformers"><img src="https://img.shields.io/badge/Transformers-4.30+-yellow?logo=python&logoColor=gold" alt="Transformers"></a>
  <a href="https://github.com/huggingface/trl"><img src="https://img.shields.io/badge/TRL-Integration-purple?logo=python&logoColor=white" alt="TRL"></a>
  <a href="https://github.com/huggingface/peft"><img src="https://img.shields.io/badge/PEFT-LoRA-yellowgreen" alt="PEFT"></a>
  <a href="https://github.com/huggingface/accelerate"><img src="https://img.shields.io/badge/Accelerate-Distributed-orange" alt="Accelerate"></a>
  <a href="https://github.com/JuiceB0xC0de/deep-chaos-scheduler/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://rocmdocs.amd.com/en/latest/"><img src="https://img.shields.io/badge/ROCm-7.2-red?logo=amd&logoColor=white" alt="ROCm"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-Compatible-76B900?logo=nvidia&logoColor=white" alt="CUDA"></a>
</p>

# deep-chaos-scheduler

A sticky-topology chaos scheduler for transformer fine-tuning on AMD MI300X (ROCm), paired with a pre/post training neural network diagnostic suite (BoL scans). Developed and benchmarked on AMD hardware — all training runs, evals, and speed numbers below are from MI300X / ROCm 7.2.

## Layer Hoist — Kernel Optimizer

The core performance story. The scheduler physically yanks dead and identity layers out of `model.layers` at every reshuffle boundary instead of running them and zeroing their output. The forward loop never sees them — no saved activations, no backward graph, no FLOPs.

**Measured on Qwen2.5-3B-Instruct, 5-epoch full training run, MI300X / ROCm 7.2:**

| | Baseline (post-hook) | Layer Hoist |
|---|---|---|
| Wall-clock training time | baseline | **2.25× faster** |
| Peak VRAM | baseline | **18% less** |

With ~30% of victim layers drawing `mode=both` (the only mode that runs full compute), roughly 70% of victim-layer compute disappears each sticky block. Layers in `dead`, `identity`, `attn-only`, or `mlp-only` modes cost nothing — they aren't in the graph.

**Hoist is on by default** — you get the speedup unless you opt out:

```python
# default: hoist on, bias stub, sticky=50
DeepChaosConfig(sticky_interval=50, seed=42)

# opt out for unusual architectures or hook-only debugging
DeepChaosConfig(sticky_interval=50, seed=42, use_layer_hoist=False)
```

## Benchmark Results

Evaluated 8 fine-tuned Qwen2.5 checkpoints (4 × 7B, 4 × 3B) — one FFT baseline and three DeepChaos seeds per size — trained on [simplescaling/s1K](https://huggingface.co/datasets/simplescaling/s1K).

**DeepChaos wins on every reasoning benchmark except GSM8K strict-match** (a formatting artifact — the gap inverts on flexible-extract). Key numbers:

| Benchmark | FFT-7B | Best DeepChaos-7B | FFT-3B | Best DeepChaos-3B |
|---|---|---|---|---|
| GSM8K flexible-extract | 68.8% | **70.6%** (+1.8pp) | 54.0% | **58.4%** (+4.4pp) |
| MGSM | 60.0% | **66.4%** (+6.4pp) | 43.2% | **60.4%** (+17.2pp) |
| Minerva Math (math_verify) | 28.0% | **36.6%** (+8.6pp) | 18.8% | **23.6%** (+4.8pp) |
| Minerva Algebra | 48.0% | **58.0%** (+10pp) | 39.0% | **42.0%** (+3pp) |
| Minerva Prealgebra | 45.0% | **58.2%** (+13.2pp) | 33.6% | **41.8%** (+8.2pp) |
| Hendrycks MATH-500 | 1.6% | 0.8% | 2.6% | **4.6%** (+2pp) |

The 3B MGSM result (+17pp over the 3B FFT) is the headline: same model, same data, same compute — the topology lottery alone drives dramatically better generalization.

**→ [Full evaluation breakdown: EVALUATIONS.md](EVALUATIONS.md)**

## Install

### AMD MI300X / ROCm 7.2

```bash
# 1. Purge any NVIDIA / CUDA torch wheels that snuck in
pip uninstall -y torch torchvision torchaudio nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 || true

# 2. Install the ROCm 7.2 PyTorch wheels
pip install --break-system-packages --no-cache-dir \
    --index-url https://download.pytorch.org/whl/rocm7.2 \
    torch==2.11.0 torchvision torchaudio

# 3. Install deep-chaos-scheduler WITHOUT touching torch
pip install --break-system-packages --no-deps --no-cache-dir \
    git+https://github.com/JuiceB0xC0de/deep-chaos-scheduler.git
```

`--no-deps` is critical on AMD — without it pip can helpfully reinstall a CUDA torch
wheel and silently break your environment. To pull updates after setup:

```bash
cd deep-chaos-scheduler && git fetch origin && git reset --hard origin/main && \
    pip install --no-deps .
```

Before launching training, set:

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
```

`TORCH_BLAS_PREFER_HIPBLASLT=1` tells the ROCm matmul dispatcher to prefer
hipBLASLt over rocBLAS where it has a kernel. `HIP_FORCE_DEV_KERNARG=1` forces
kernel arguments to live in device memory instead of being host-managed —
both are small but free throughput gains on MI300X. Set them in the shell
before launching python; setting them via `os.environ[...] = "1"` at the top
of the script also works as long as it's before any torch import.

### Local dev environment (venv, ROCm 6.2)

For a local Linux rig with ROCm 6.2 where you want an isolated venv and a full
Jupyter + W&B dev environment:

```bash
# 1. Create and activate venv
python3 -m venv training-env
source training-env/bin/activate

# 2. Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Install Triton for ROCm (match your Python version, cp312 shown)
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/triton-3.3.1%2Brocm7.2.2.git28a7371e-cp312-cp312-linux_x86_64.whl

# 4. Install ML stack
pip install transformers datasets trl accelerate wandb

# 5. Install missing wandb deps
pip install platformdirs pydantic

# 6. Install Jupyter
pip install notebook ipywidgets

# 7. Clone and install deep_chaos_scheduler
git clone https://github.com/JuiceB0xC0de/deep-chaos-scheduler.git
cd deep-chaos-scheduler
pip install -e .
cd ..

# 8. Login to W&B
wandb login

# 9. Launch Jupyter
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```

Set the ROCm env vars before launching training (same as the DO droplet setup):

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
```

### Other platforms (CUDA / generic)

```bash
pip install git+https://github.com/JuiceB0xC0de/deep-chaos-scheduler.git
```

The scheduler is platform-agnostic — all AMD-specific speed settings and env vars are safe to omit on non-ROCm hardware.

### Dependencies

`torch`, `transformers`. Optional: `wandb` (full integration via `bol_wandb`), `peft` (required for quantized models), `scipy` (used in silhouette scan), `trl` (recommended trainer wrapper).

---

## What Makes It Generic

The scheduler stays model-family agnostic through four design choices:

1. **Layer auto-discovery**: uses structural inference (`resolve_transformer_layers`) instead of hardcoded model maps.
2. **Projection-hook masking**: modifies projection outputs with hooks instead of replacing forward methods.
3. **KV-share awareness**: detects shared K/V regimes and avoids invalid hook placements on shared pathways.
4. **Post-proj norm awareness**: supports architectures where normalization and projection ordering differs from Llama-like defaults.

---

## How the Scheduler Works

`DeepChaosScheduler` runs a **sticky-block topology lottery** across your model's transformer layers on every training step.

On each reshuffle (every `sticky_interval` steps), the scheduler:

1. Walks the model and auto-detects the transformer layer stack — no config files required
2. Marks the first two and last two layers as **sacred** (always active), everything in between is a **victim**
3. Randomly decides how many victim layers are active this block (30–70% by default)
4. Enforces streak limits — a layer can't stay dead more than `max_consecutive_off` blocks or stay on more than `max_consecutive_on` blocks in a row
5. For each active layer, draws a mode: `both` (attn + MLP), `attn only`, `mlp only`, or `identity`
6. Within each active layer, randomly drops groups of Q/K/V heads, MLP gate/up/down channels, and hidden-dim slices using `register_forward_hook` — the base model's forward pass runs completely untouched
7. Holds that topology frozen for the next `sticky_interval` steps, then reshuffles

The result is that the model never trains through the same compute path twice for long. It can't over-rely on specific heads or MLP channels and has to build more distributed representations to stay consistent.

### Hook-based, not forward-replacement

The scheduler installs `register_forward_hook` on individual projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). The base model's own forward — including its LayerNorms, RoPE, causal masking, sliding window, and any remote-code logic — runs exactly as written. The hooks only zero out non-surviving output indices after each projection fires.

This means it works across model families (Llama, Gemma, Mistral, Qwen, Falcon, GPT-NeoX, etc.) without architecture-specific code.

### Auto-detection

On init, `DeepChaosScheduler` walks the model to find the decoder layer stack. It checks a list of known paths (`model.layers`, `transformer.h`, `gpt_neox.layers`, etc.) and falls back to scoring every `ModuleList` by how much it looks like a transformer block. Hidden size, head count, KV head count, intermediate size, and GQA ratio are all inferred from projection weight shapes.

Sacred and victim ranges are computed from `num_hidden_layers`. If the model has a `layer_types` attribute (Gemma-3, some Falcon variants), global-attention layers are automatically added to the sacred set.

---

## Tested Models

The scheduler auto-config path was tested across the following checkpoints:

| Model | Size / Type | Scheduler Auto-Config | Limitation / Caveat |
|---|---|---|---|
| Qwen 2.5 3B Instruct | 3B | ✅ | None observed |
| Falcon-E 3B Instruct | 3B | ✅ | Use `prequantized` revision for training |
| SmolLM2-360M | 360M Tiny | ✅ | None observed |
| Ministral 3 3B Instruct 2512 | 3B | ✅ | `fix_mistral_regex=True` tokenizer compat may be required |
| Doge-320M | 320M Tiny | ✅ | Remote-code revisions can require compat patching |
| Llama 3.2 3B | 3B | ✅ | None observed |
| Llama 3.2 3B Instruct (MI300X / ROCm 7.2) | 3B | ✅ | Use `attn_implementation="sdpa"` for training (41% faster); `eager` only for BoL attention scan. Set `pad_token_id=128001` explicitly. |
| Gemma-4-E4B | ~4B (efficient) | ✅ | Text-only path may require `attn_implementation=\"eager\"` and explicit CUDA move |
| Phi-4-mini-instruct | Mini | ✅ | Prefer native HF load (`trust_remote_code=False`) |
| OLMo-2-0425-1B | 1B | ✅ | None observed |
| Phi-tiny-MoE-instruct | Tiny MoE | ✅ | Prefer fp32 training precision for grouped MoE dtype consistency |

## Compatibility & Limitations

The scheduler was tested on the architectures above, and layer discovery / sacred-victim auto-configuration worked without manual layer mapping, including MoE models.

No permanently unsupported model family is currently known, but highly custom remote-code checkpoints can still require model-specific loader flags.

---

## DeepChaosScheduler Usage

### Minimal

```python
from deep_chaos_scheduler import DeepChaosScheduler, DeepChaosConfig

# model is already loaded (Unsloth, HF, PEFT, anything)
dc = DeepChaosScheduler(
    model,
    DeepChaosConfig(sticky_interval=50, seed=42),
)

# call once per training step
stats = dc.step(global_step)

# when you're done
dc.remove()
```

`stats` is a dict with keys like `layer_density_pct`, `avg_q_surv`, `avg_gate_surv`, `compute_pct`, `reshuffle_event`, and mode counts. Pass it straight to `wandb.log()` if you want.

### Config reference

```python
DeepChaosConfig(
    sticky_interval=50,         # steps between topology reshuffles
    seed=42,

    # layer survival bounds (fraction of victim layers active per block)
    min_layer_survival=0.30,
    max_layer_survival=0.70,

    # attention head survival
    min_head_survival=0.30,
    max_head_survival=0.70,

    # MLP channel group survival
    min_channel_survival=0.30,
    max_channel_survival=0.70,
    channel_group_size=128,

    # MLP gate sub-sampling
    min_mlp_gate_survival=0.35,
    max_mlp_gate_survival=0.80,
    mlp_gate_group_size=128,

    # hidden-dim output survival
    min_hidden_survival=0.60,
    max_hidden_survival=0.95,
    hidden_group_size=64,

    # streak limits
    max_consecutive_on=5,
    max_consecutive_off=10,

    # override auto-detected ranges if needed
    sacred_layers=None,         # e.g. [0, 1, 30, 31] for a 32-layer model
    victim_range=None,          # e.g. (2, 30) — end-exclusive

    # layer hoist (the kernel optimizer — see top of README)
    use_layer_hoist=True,       # physically yank dead/identity layers; default-on
    hoist_stub_kind="bias",     # "bias" | "linear" | "none" — frozen perturbation per yanked run
    hoist_stub_init_scale=0.01,

    announce_reshuffles=True,   # prints a line on each reshuffle
)
```

### Manual sacred/victim override

```python
dc = DeepChaosScheduler(
    model,
    DeepChaosConfig(
        sacred_layers=[0, 1, 31],   # always-on layers
        victim_range=(2, 31),        # candidates for chaos — end-exclusive
    ),
)
```

### freeze_topology

If you need to lock the current topology in place (for a probe or eval mid-run):

```python
dc.freeze_topology(global_step)
```

---

## Using with TRL + plain transformers (recommended on AMD)

This is the cleanest path on MI300X / ROCm 7.2 — no unsloth, no xformers, no
CUDA-specific dependencies.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from deep_chaos_scheduler import (
    DeepChaosConfig, DeepChaosScheduler, patch_clip_grad_norm_disable_foreach,
)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Codified from the gemma-4 PDE-fault incident — also a no-op safety net on AMD.
patch_clip_grad_norm_disable_foreach()

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token_id = 128001                    # explicit Llama-3 EOT, kills PAD/BOS/EOS warning
tok.pad_token = tok.pad_token or tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",              # MI300X / ROCm 7.2: dispatches to Triton fused HIP flash-attn — 41% faster than eager
)
model.config.use_cache = False
model.config.pad_token_id = 128001
model.generation_config.pad_token_id = 128001
model.generation_config.temperature = None   # avoid invalid-flag warnings
model.generation_config.top_p = None
model.generation_config.do_sample = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

dc = DeepChaosScheduler(model, DeepChaosConfig(sticky_interval=50, seed=42))

class DCStep(TrainerCallback):
    def on_step_begin(self, args, state, control, **kw):
        dc.step(state.global_step)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./out",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        bf16=True,
        max_seq_length=8192,
        packing=False,                       # see Known Pitfalls — needs real flash_attn to be safe
        assistant_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",           # small free gain on MI300X
        seed=42,
    ),
    train_dataset=load_dataset("json", data_files="your_data.jsonl", split="train"),
    processing_class=tok,
    callbacks=[DCStep()],
)
trainer.train()
dc.remove()
```

### MI300X / ROCm 7.2 speed stack

| Setting | Effect |
|---|---|
| `attn_implementation="sdpa"` | **Single biggest win — 41% faster epochs (17 min → 10 min on Llama-3.2-3B-Instruct).** Triton 3.6.0 + ROCm's HIP flash-attn headers are pre-installed on the DigitalOcean GPU droplet image, so SDPA dispatches to fused Triton kernels with no extra install. |
| `optim="adamw_torch_fused"` | Small free gain over `adamw_torch`. |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` (env var) | Prefer hipBLASLt for matrix kernels. Set in shell before launching python. |
| `packing=False` | Required without a real `flash_attn` package — see Known Pitfalls. |

The DO ROCm 7.2 droplet has `triton==3.6.0`, `triton-rocm==3.6.0`, and the HIP
flash-attn headers inside torch itself (`torch/include/ATen/native/transformers/hip/flash_attn`).
There's no `flash_attn` pip package installed; getting one would require a
20–40 minute build from source. Until that's done, SDPA via Triton is the fast path.

If you specifically need BoL's `attention_map` scan (which requires
`output_attentions=True`), switch to `attn_implementation="eager"` for that run
only — SDPA on this stack silently returns `None` for attention weights. Use
SDPA for training, eager for diagnostics.

### AMD GPU telemetry (W&B logging via amdsmi)

W&B's built-in system-metrics reader gives you almost nothing useful on AMD —
no per-step gfx activity, no hipBLAS clocks, no throttle status. The fix is to
read straight from `amdsmi` (ships with the ROCm install) inside a
`TrainerCallback` and log the values yourself. The callback below is the one
used to produce the Lucky Pick benchmark dashboard on Qwen2.5-3B-Instruct + s1K
(MI300X, ROCm 7.2):

```python
import torch
import wandb
from transformers import TrainerCallback

try:
    import amdsmi
    amdsmi.amdsmi_init()
    _amd_handles = amdsmi.amdsmi_get_processor_handles()
    AMDSMI_OK = len(_amd_handles) > 0
except Exception as e:
    print(f"AMDSMI unavailable: {e}")
    _amd_handles = []
    AMDSMI_OK = False


class AMDTelemetryCallback(TrainerCallback):
    """Logs ROCm-native GPU telemetry to W&B every Trainer log step."""

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
                # VRAM — torch owns the context, ask torch
                "gpu/vram_used_mb":     torch.cuda.memory_allocated(0) / 1024**2,
                "gpu/vram_reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
                "gpu/vram_used_pct":    (torch.cuda.memory_allocated(0) / total_mem) * 100,

                # Power
                "gpu/power_w":          metrics.get("average_socket_power"),
                "gpu/power_limit_w":    metrics.get("power_limit"),
                "gpu/energy_acc_uj":    metrics.get("energy_accumulator"),

                # Thermals
                "gpu/temp_edge_c":      metrics.get("temperature_edge"),
                "gpu/temp_hotspot_c":   metrics.get("temperature_hotspot"),
                "gpu/temp_mem_c":       metrics.get("temperature_mem"),
                "gpu/temp_vrgfx_c":     metrics.get("temperature_vrgfx"),

                # Utilization
                "gpu/gfx_util":         activity.get("gfx_activity"),
                "gpu/mem_util":         activity.get("umc_activity"),

                # Clocks
                "gpu/gfx_clk_mhz":      clocks.get("clk"),
                "gpu/gfx_clk_max_mhz":  clocks.get("max_clk"),
                "gpu/mem_clk_mhz":      mem_clk.get("clk"),

                # Throttle
                "gpu/throttle_status":  metrics.get("throttle_status"),
            }
            payload = {
                k: v for k, v in payload.items()
                if v is not None and isinstance(v, (int, float))
            }
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


# trainer = SFTTrainer(..., callbacks=[AMDTelemetryCallback(), ChaosStepCallback()])
```

Why each metric matters:

- `gpu/gfx_util` and `gpu/mem_util` distinguish "compute-bound" from "memory-bound" — chaos runs land lower because much of the layer stack is dead each window. If chaos runs match dense gfx_util, the scheduler isn't actually saving compute.
- `gpu/temp_hotspot_c` and `gpu/throttle_status` tell you whether observed slowdowns are algorithmic or thermal. A 70 °C hotspot with throttle_status=0 is a healthy MI300X.
- `gpu/gfx_clk_mhz` vs `gpu/gfx_clk_max_mhz` shows whether the GPU is actually running at boost or being held back by power/thermal envelope.

The scheduler works with Unsloth's `FastLanguageModel` — load your model normally, wrap it in the scheduler, then call `dc.step()` in a custom training loop or via a `TrainerCallback`.

### Option 1 — Custom training loop

```python
from unsloth import FastLanguageModel
from deep_chaos_scheduler import DeepChaosScheduler, DeepChaosConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.2-3b-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

from deep_chaos_scheduler import resolve_scheduler_model
dc = DeepChaosScheduler(
    resolve_scheduler_model(model),   # unwraps PEFT wrapper for hook installation
    DeepChaosConfig(sticky_interval=50),
)

# in your training loop:
for step, batch in enumerate(dataloader):
    stats = dc.step(step)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

dc.remove()
```

`resolve_scheduler_model` unwraps PEFT/Unsloth wrappers so the hooks land on the actual transformer modules.

### Option 2 — SFTTrainer via callback

```python
from unsloth import FastLanguageModel
from deep_chaos_scheduler import DeepChaosScheduler, DeepChaosConfig, resolve_scheduler_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.2-3b-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)

dc = DeepChaosScheduler(
    resolve_scheduler_model(model),
    DeepChaosConfig(sticky_interval=50),
)

class ChaosStepCallback(TrainerCallback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_step_begin(self, args, state, control, **kwargs):
        self.scheduler.step(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.scheduler.remove()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
    ),
    callbacks=[ChaosStepCallback(dc)],
)
trainer.train()
```

### Using with Axolotl or any other trainer

The pattern is the same — load your model, initialize `DeepChaosScheduler`, plug in a callback or hook that calls `dc.step(global_step)` before each forward pass. The scheduler doesn't care how the model was loaded or what trainer wraps it.

---

## Known Pitfalls

- If you use `gradient_checkpointing=True`, keep `sticky_interval >= gradient_accumulation_steps`. If the topology reshuffles in the middle of an accumulation window, backward recomputation can see a different topology than forward, producing incorrect gradients.

- **`CUDA error: an illegal memory access` inside `clip_grad_norm_` on A100.** Traceback lands in `torch._foreach_norm(device_grads, norm_type)`, usually with `XID 31 ... MMU Fault ... FAULT_PDE ACCESS_TYPE_VIRT_READ` in the NVIDIA log. Not a NaN/Inf bug — grads are clean. The fused `_foreach_norm` kernel trips a PDE MMU fault on some A100 + CUDA 12.1 + PyTorch 2.5.1 combos (reproduced on Modal A100-80GB with Gemma-4 E4B). Fix by forcing the single-tensor loop:

  ```python
  from deep_chaos_scheduler import patch_clip_grad_norm_disable_foreach
  patch_clip_grad_norm_disable_foreach()
  # then construct your Trainer / run trainer.train() as usual
  ```

  Idempotent. Also patches the `accelerate.accelerator` reference that HF `Trainer` delegates to.

- **Gemma-4 multimodal + `device_map="auto"` + extracting `.language_model`.** Gemma-4 (`Gemma3ForConditionalGeneration`) wraps `vision_tower`, `multi_modal_projector`, and `language_model`. `device_map="auto"` installs Accelerate `AlignDevicesHook` dispatch metadata on the parent wrapper's params; extracting `.language_model` for training leaves that stale metadata on the vision-side params. `Trainer` still iterates them via `model.parameters()`, and any foreach kernel over the full list PDE-faults. Load with `device_map=None`, drop the vision half, then `.to("cuda")` manually — `model_load_kwargs_for_training(model_name, dtype)` does this for you on Gemma-4 names.

- **Debugging async CUDA faults.** The default async CUDA dispatcher surfaces errors at the next sync op, which may be far from the real faulting kernel. For any `illegal memory access` chase, set `CUDA_LAUNCH_BLOCKING=1` in the container env so the traceback points at the real line.

- **Modal image layer caches a stale scheduler SHA.** `modal.Image.run_commands(...)` hashes the command string for caching. If a pip-install-from-git refuses to update after you push a new SHA, bust the cache with an explicit token in the command string: `"echo 'lps-rev=<date>-<sha>' && pip install ... @<sha>"`. Bump the token whenever you push.

- **AMD: pip silently swaps your ROCm torch for a CUDA one.** Default `pip install git+...` resolves the project's `install_requires`, sees `torch`, and helpfully installs the CUDA build on top of your ROCm wheels. Always use `--no-deps --no-cache-dir` on AMD environments (see Install section). To verify: `python -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. Expect `7.2`, `True`, `AMD Instinct MI300X VF`.

- **MI300X / ROCm 7.2: SDPA silently returns `None` for `output_attentions`.** `attn_implementation="sdpa"` is the fast path (41% epoch-time win on the DO droplet's Triton+HIP stack) but the ROCm SDPA kernel does not surface attention weights, so any code path that reads them — including BoL's `attention_map` scan — gets `None`. Use SDPA for training; switch to `attn_implementation="eager"` only when you want the attention scan to populate.

- **MI300X / ROCm 7.2: `packing=True` is unsafe without a real `flash_attn` package.** The droplet has Triton-fused HIP attention via SDPA but no `flash_attn` wheel (it would need a 20–40 min source build). With packing enabled: `eager` OOMs around 22 GB on the packed-sequence softmax for 8 K context, and `sdpa` raises a warning and risks cross-example contamination because the masked-attention path needed for packing isn't fully implemented in the ROCm SDPA backend. Keep `packing=False` until a flash-attn build is ready. Estimated headroom once it's installed: another 15–25% on top of SDPA's gain.

- **Llama-3 tokenizer warnings on Llama-3.2 / 3.3.** The default tokenizer has BOS/PAD/EOS overlap that triggers a noisy warning during training. Set `tokenizer.pad_token_id = 128001` (the EOT token) explicitly, and mirror it onto `model.config.pad_token_id` and `model.generation_config.pad_token_id`. Also null out `generation_config.temperature` / `top_p` / `do_sample` if you're not generating during training to silence the invalid-generation-flag warning.

---

## Quantized / BitNet models

## Model-family compatibility helpers

For scripts that load many model families, use the built-in helpers:

```python
from deep_chaos_scheduler import (
    model_load_kwargs_for_training,
    resolve_training_precision,
    tokenizer_load_kwargs_for_model,
)

precision = resolve_training_precision(model_name, cuda_available=torch.cuda.is_available())
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    **tokenizer_load_kwargs_for_model(model_name),
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    **model_load_kwargs_for_training(model_name, precision["dtype"]),
)
# Trainer(..., bf16=precision["bf16"], fp16=precision["fp16"])
```

This includes:
- Phi/Phi-MoE defaulting to native HF code path (`trust_remote_code=False`)
- Phi-MoE fp32 precision recommendation to avoid grouped MoE dtype mismatch
- Falcon-E prequantized revision and Mistral regex tokenizer compat

### BitsAndBytes / GPTQ / AWQ (standard quantized)

If your checkpoint is quantized and the trainer rejects full-parameter training, `prepare_model_for_training` auto-attaches LoRA adapters:

```python
from deep_chaos_scheduler import ModelPrepConfig, prepare_model_for_training, resolve_scheduler_model

model, prep_report = prepare_model_for_training(
    model,
    ModelPrepConfig(
        auto_lora_for_quantized=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ),
)
print(prep_report.to_dict())

dc = DeepChaosScheduler(resolve_scheduler_model(model), DeepChaosConfig())
```

For non-quantized models, `prepare_model_for_training` is a no-op.

### Falcon-E and BitNet series (ternary weights)

`tiiuae/Falcon-E-3B-Instruct` (and `Falcon-E-1B-Base`, `Falcon-E-7B-Base`, Microsoft's BitNet series) have three revisions:

| Revision | Use case |
|---|---|
| default | Inference only — packed ternary weights. `transformers` blocks training on this. |
| `prequantized` | Fine-tuning — bfloat16 weights, compatible with `onebitllms`. |
| `bfloat16` | bfloat16 inference without BitNet kernels. |

If you load the default revision and call any trainer, you'll get:

```
ValueError: The model you are trying to fine-tune is quantized with QuantizationMethod.BITNET
but that quantization method do not support training.
```

The correct path:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_chaos_scheduler import apply_bitnet_linear_replacement, DeepChaosScheduler, DeepChaosConfig

model_id = "tiiuae/Falcon-E-3B-Instruct"  # or Falcon-E-1B-Base, Falcon-E-7B-Base, etc.

tokenizer = AutoTokenizer.from_pretrained(model_id, revision="prequantized")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision="prequantized",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Replace standard linears with trainable BitLinear layers (onebitllms QAT).
# This also auto-patches transformers' training validator so the Trainer
# doesn't reject the model — no extra steps needed.
model = apply_bitnet_linear_replacement(model)

dc = DeepChaosScheduler(model, DeepChaosConfig(sticky_interval=50))

# then set up your trainer as normal...
```

After training, convert the checkpoint back to packed ternary for deployment:

```python
from onebitllms import quantize_to_1bit
quantize_to_1bit(output_dir, quantized_output_dir)
```

Requires `pip install onebitllms`. LoRA/PEFT is not currently supported for BitNet models — `apply_bitnet_linear_replacement` does full-parameter QAT using a straight-through estimator.

> **Container note:** `pip install git+https://...` in a container image is cached at build time. If you push changes to this repo and the error persists, force a reinstall in your entrypoint: `pip install --force-reinstall --no-cache-dir git+https://github.com/JuiceB0xC0de/deep-chaos-scheduler.git`

---

## Remote-code model compatibility

Some Hub models (certain Doge/Mistral remote-code revisions) have runtime mismatches across `transformers` versions. Apply the compatibility patch before loading:

```python
from deep_chaos_scheduler import apply_transformers_remote_code_compat

apply_transformers_remote_code_compat(verbose=True)

# then load your model as usual
```

---

## Auto optimizer + LR scheduler

`build_scheduler_stack` introspects the model, groups parameters by role (attention, MLP, embedding, head, norm/bias), and builds an AdamW optimizer + LR scheduler automatically. Three param groups: matrix params (attention/mlp/other_matrix weights) can run at a different LR via `matrix_lr_multiplier`, norm and bias params get `weight_decay=0.0` by default, everything else runs at the base LR.

```python
from deep_chaos_scheduler import build_scheduler_stack, AutoSchedulerConfig

optimizer, lr_scheduler, report = build_scheduler_stack(
    model,
    num_training_steps=1000,
    config=AutoSchedulerConfig(
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        matrix_lr_multiplier=1.0,     # set >1.0 to train matrix params hotter
        no_decay_on_norm_bias=True,   # standard AdamW recipe for transformers
    ),
)
print(report.to_dict())
```

---

## BoL Scans — Neural Network Pre/Post Diagnostics

> If you've read this far and want to see what's actually happening inside the network during training, here's a bonus: BoL (Blocks of Life) runs a full diagnostic suite before and after fine-tuning and prints everything to your terminal. GPU required. Most people skip this section — it's here if you want it.

### The six scans

**Weight Fingerprint**
Per-layer, per-component weight statistics: mean, std, L2 norm, and sparsity for every projection group (q/k/v/o, gate/up/down). Gives you a direct before/after diff of what the optimizer actually changed.

**Layer Sweep**
Progressively zeroes out each layer and measures generation damage. Shows which layers carry the most load and which the model barely uses.

**Component Ablation**
Zeroes out one projection type at a time across the full model and measures the perplexity hit. Shows relative importance of each projection family.

**Silhouette**
Extracts hidden-state representations for related and unrelated word pairs, then computes a silhouette score measuring how well the model separates them. Closer to 1.0 = tighter clustering.

**CKA (Centered Kernel Alignment)**
Computes pairwise representational similarity across all layers. Shows which layers have converged to similar representations and which are doing different things.

**Attention Map**
Per-head attention entropy and cross-token similarity for each layer. Low entropy = focused attention, high = diffuse.

### Usage

```python
from bol_wandb import BoLPrintCallback

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    callbacks=[BoLPrintCallback(model, tokenizer)],
)
trainer.train()
```

Fires `run_all(..., phase="pre")` at train start and `run_all(..., phase="post")` at train end, printing all six scans as readable terminal tables. No W&B required. Pre/post results are also stored on the callback instance for programmatic access.

### Standalone

```python
from bol_scans import run_all

# verbose=True prints the full bol-tools-v2 CLI render (fingerprint table,
# layer sweep with CRITICAL/MODERATE/REMOVABLE tags, ranked importance,
# silhouette ASCII bars, CKA, attention map) to stdout.
pre_results  = run_all(model, tokenizer, phase="pre",  verbose=True)
trainer.train()
post_results = run_all(model, tokenizer, phase="post", verbose=True)
```

If you only want the compact one-liners (good for log scraping), use
`print_summary=True` instead of `verbose=True`. The full render is also
available as a string via `format_results_for_cli(results, phase=phase)`.

All six scans accept custom `eval_texts`, `probes`, `related_pairs`, `unrelated_pairs`, `cluster_words`, `clusters`, and `layer_stride` (skip layers for faster CKA on large models).

---

## File layout

```
deep-chaos-scheduler/
├── deep_chaos_scheduler/
│   ├── __init__.py
│   ├── deep_chaos.py       # DeepChaosScheduler, DeepChaosConfig, LayerBindings, topology logic
│   ├── scheduler.py        # build_scheduler_stack, AutoSchedulerConfig, parameter role classification
│   ├── model_prep.py       # prepare_model_for_training, auto-LoRA for quantized checkpoints
│   └── compat.py           # apply_transformers_remote_code_compat
├── bol_wandb/
│   ├── callback.py         # BoLPrintCallback / BoLWandbCallback (TrainerCallback)
│   ├── scanner.py          # BOLScanner helper
│   └── metrics/            # ablation, attention, cka, fingerprint, silhouette
├── bol_scans.py            # run_all() — the six scans + W&B logging
├── tests/                  # CPU-only smoke tests for the public API (pytest)
├── train_benchmark.py      # canonical SFT training example (Qwen2.5-3B + DeepChaos)
├── train_ab.py             # vanilla SFT vs DeepChaos hoist A/B harness
├── eval_amd.py             # native MI300X / ROCm lm-eval driver (vLLM backend)
├── bench/
│   ├── bench_amd.py              # MI300X wall-clock + VRAM A/B (hoist vs baseline)
│   ├── eval_modal.py             # Modal H100/L4 mirror of eval_amd.py (used during dev)
│   ├── train_instrument.py       # 30-step instrumentation harness (AMD)
│   └── train_instrument_modal.py # Modal H100/A100 mirror of the instrumentation harness
├── EVALUATIONS.md          # full DeepChaos vs FFT eval breakdown across 7B and 3B
├── setup_amd.sh            # one-shot MI300X / ROCm 7.2 environment setup
└── setup.py
```
