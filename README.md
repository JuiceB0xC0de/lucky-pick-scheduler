# lucky-pick-scheduler

A sticky-topology chaos scheduler for transformer fine-tuning, paired with a pre/post training neural network diagnostic suite (BoL scans) that logs structured results to Weights & Biases.

## Install

```bash
pip install git+https://github.com/JuiceB0xC0de/lucky-pick-scheduler.git
```

Dependencies: `torch`, `transformers`, `wandb`. Optional: `peft` (required for quantized models), `scipy` (used in silhouette scan), `muon` (optional optimizer).

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
from lucky_pick_scheduler import DeepChaosScheduler, DeepChaosConfig

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

## Using with Unsloth

The scheduler works with Unsloth's `FastLanguageModel` — load your model normally, wrap it in the scheduler, then call `dc.step()` in a custom training loop or via a `TrainerCallback`.

### Option 1 — Custom training loop

```python
from unsloth import FastLanguageModel
from lucky_pick_scheduler import DeepChaosScheduler, DeepChaosConfig

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

from lucky_pick_scheduler import resolve_scheduler_model
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
from lucky_pick_scheduler import DeepChaosScheduler, DeepChaosConfig, resolve_scheduler_model
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
  from lucky_pick_scheduler import patch_clip_grad_norm_disable_foreach
  patch_clip_grad_norm_disable_foreach()
  # then construct your Trainer / run trainer.train() as usual
  ```

  Idempotent. Also patches the `accelerate.accelerator` reference that HF `Trainer` delegates to.

- **Gemma-4 multimodal + `device_map="auto"` + extracting `.language_model`.** Gemma-4 (`Gemma3ForConditionalGeneration`) wraps `vision_tower`, `multi_modal_projector`, and `language_model`. `device_map="auto"` installs Accelerate `AlignDevicesHook` dispatch metadata on the parent wrapper's params; extracting `.language_model` for training leaves that stale metadata on the vision-side params. `Trainer` still iterates them via `model.parameters()`, and any foreach kernel over the full list PDE-faults. Load with `device_map=None`, drop the vision half, then `.to("cuda")` manually — `model_load_kwargs_for_training(model_name, dtype)` does this for you on Gemma-4 names.

- **Debugging async CUDA faults.** The default async CUDA dispatcher surfaces errors at the next sync op, which may be far from the real faulting kernel. For any `illegal memory access` chase, set `CUDA_LAUNCH_BLOCKING=1` in the container env so the traceback points at the real line.

- **Modal image layer caches a stale scheduler SHA.** `modal.Image.run_commands(...)` hashes the command string for caching. If a pip-install-from-git refuses to update after you push a new SHA, bust the cache with an explicit token in the command string: `"echo 'lps-rev=<date>-<sha>' && pip install ... @<sha>"`. Bump the token whenever you push.

---

## Quantized / BitNet models

## Model-family compatibility helpers

For scripts that load many model families, use the built-in helpers:

```python
from lucky_pick_scheduler import (
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
from lucky_pick_scheduler import ModelPrepConfig, prepare_model_for_training, resolve_scheduler_model

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
from lucky_pick_scheduler import apply_bitnet_linear_replacement, DeepChaosScheduler, DeepChaosConfig

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

> **Container note:** `pip install git+https://...` in a container image is cached at build time. If you push changes to this repo and the error persists, force a reinstall in your entrypoint: `pip install --force-reinstall --no-cache-dir git+https://github.com/JuiceB0xC0de/lucky-pick-scheduler.git`

---

## Remote-code model compatibility

Some Hub models (certain Doge/Mistral remote-code revisions) have runtime mismatches across `transformers` versions. Apply the compatibility patch before loading:

```python
from lucky_pick_scheduler import apply_transformers_remote_code_compat

apply_transformers_remote_code_compat(verbose=True)

# then load your model as usual
```

---

## Auto optimizer + LR scheduler

`build_scheduler_stack` introspects the model, groups parameters by role (attention, MLP, embedding, head, norm/bias), and builds an optimizer + LR scheduler automatically. It prefers Muon for attention and MLP matrices and falls back to AdamW if Muon isn't installed.

```python
from lucky_pick_scheduler import build_scheduler_stack, AutoSchedulerConfig

optimizer, lr_scheduler, report = build_scheduler_stack(
    model,
    num_training_steps=1000,
    config=AutoSchedulerConfig(
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        prefer_muon=True,
    ),
)
print(report.to_dict())
```

---

## BoL Scans

BoL (Blocks of Life) is the diagnostic suite. It runs six scans against a loaded model and tokenizer and logs everything to your active W&B run. The point is to snapshot the model before and after training so you can see concretely what changed.

### Usage

```python
import wandb
from bol_scans import run_all

wandb.init(project="my-project", name="run-name")

# before training
run_all(model, tokenizer, phase="pre")

trainer.train()

# after training
run_all(model, tokenizer, phase="post")
```

All metrics are logged under `pre/*` and `post/*` key prefixes in W&B.

### The six scans

**Weight Fingerprint**
Per-layer, per-component weight statistics: mean, std, L2 norm, sparsity, and kurtosis for every projection group (q/k/v/o, gate/up/down). Gives you a direct before/after diff of what the optimizer actually changed and by how much in each part of the network.

**Layer Sweep**
Progressively zeroes out each layer from the top down and measures how much generation output changes. Shows which layers carry the most load and which the model is barely using.

**Component Ablation**
Zeroes out one projection type at a time (all Q projections, all V projections, all gate projections, etc.) across the full model and measures the perplexity hit. Shows relative importance of each projection family.

**Silhouette**
Extracts hidden-state representations for semantically related and unrelated word pairs, then computes a silhouette score measuring how well the model separates them in representation space. A score closer to 1.0 means tighter clustering of related concepts; a score near 0 means the representations are poorly separated.

**CKA (Centered Kernel Alignment)**
Computes pairwise representational similarity between every layer combination using a fixed set of semantic probe words. Produces a heatmap showing which layers have converged to similar representations and which are doing meaningfully different things.

**Attention Map**
Measures per-head attention entropy and cross-token similarity for each layer. Low entropy heads are attending sharply to specific tokens; high entropy heads are diffuse. Also measures consistency of attention patterns across similar inputs.

### Custom eval texts and probes

All six scans accept custom inputs:

```python
run_all(
    model,
    tokenizer,
    phase="pre",
    eval_texts=["your text here", "another example"],
    probes=["The capital of France is", "def fibonacci(n):"],
    related_pairs=[("dog", "wolf"), ("happy", "joyful")],
    unrelated_pairs=[("dog", "democracy"), ("happy", "concrete")],
    cluster_words=["dog", "wolf", "car", "truck", "happy", "joy"],
    clusters={
        "animals": ["dog", "wolf"],
        "vehicles": ["car", "truck"],
        "emotions": ["happy", "joy"],
    },
    layer_stride=2,   # skip every other layer for faster CKA on large models
)
```

### TrainerCallback (HuggingFace / Unsloth / TRL)

`BoLWandbCallback` in `bol_wandb` handles the pre/post timing automatically:

```python
from bol_wandb import BoLWandbCallback

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    callbacks=[BoLWandbCallback(model, tokenizer)],
)
trainer.train()
```

The callback fires `run_all(..., phase="pre")` on `on_train_begin` and `run_all(..., phase="post")` on `on_train_end`. W&B must be initialized before the trainer starts.

---

## File layout

```
lucky-pick-scheduler/
├── lucky_pick_scheduler/
│   ├── __init__.py
│   ├── deep_chaos.py       # DeepChaosScheduler, DeepChaosConfig, LayerBindings, topology logic
│   ├── scheduler.py        # build_scheduler_stack, AutoSchedulerConfig, parameter role classification
│   ├── model_prep.py       # prepare_model_for_training, auto-LoRA for quantized checkpoints
│   └── compat.py           # apply_transformers_remote_code_compat
├── bol_wandb/
│   ├── callback.py         # BoLWandbCallback (TrainerCallback)
│   └── config.py           # shared eval text and probe defaults
├── bol_scans.py            # run_all() — the six scans + W&B logging
└── setup.py
```
