# Lucky Pick Scheduler + BoL Scans

`lucky-pick-scheduler` now includes:
- A generic, auto-configuring `DeepChaosScheduler` (sticky topology lottery, default sticky interval `50`)
- BoL scan logging utilities (`bol_scans.run_all`)

## Installation

```bash
pip install git+https://github.com/JuiceB0xC0de/lucky-pick-scheduler.git
```

## Deep Chaos Scheduler (Auto-Config)

The scheduler introspects the currently loaded model and auto-detects:
- transformer layer stack
- attention projections (`q/k/v/o`) when present
- MLP projections (`gate/up/down` or `fc1/fc2` style when present)

It then runs sticky-block topology shuffles that activate only subsets of layers/components/hidden dims.

```python
from lucky_pick_scheduler import DeepChaosScheduler, DeepChaosConfig

# model already loaded (HF/Unsloth/etc)
dc = DeepChaosScheduler(
    model,
    DeepChaosConfig(
        sticky_interval=50,  # sticky-50 behavior
        seed=42,
        # sacred_layers / victim_range are optional; auto-inferred by default
    ),
)

# each train step:
stats = dc.step(global_step)

# optional hard freeze at a specific step for probes:
dc.freeze_topology(global_step)

# cleanup at the end
dc.remove()
```

`stats` includes mode mix, layer density, survival percentages, and compute ratio estimates.
It also includes `reshuffle_event` (`1.0` on sticky-boundary reshuffle steps, else `0.0`) and emits
`DeepChaos reshuffle: step=...` console logs when `announce_reshuffles=True` (default).

### Remote-Code Compatibility Patch

For some custom Hub models (for example certain Doge/Mistral remote-code revisions),
you may need compatibility shims for changes across `transformers` versions:

```python
from lucky_pick_scheduler import apply_transformers_remote_code_compat

apply_transformers_remote_code_compat(verbose=True)
```

This patches known runtime mismatches (`OutputRecorder`, `check_model_inputs`,
and tied-weight key expansion list/dict compatibility) before model loading.

### Quantized/BitNet Auto-LoRA Prep

For broad model support (including quantized/BitNet checkpoints), you can let the
package auto-attach LoRA adapters when needed:

```python
from lucky_pick_scheduler import ModelPrepConfig, prepare_model_for_training, resolve_scheduler_model

model, prep = prepare_model_for_training(
    model,
    ModelPrepConfig(
        auto_lora_for_quantized=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ),
)
print(prep.to_dict())

# Use base model for scheduler hooks if model was wrapped (PEFT, etc).
scheduler_model = resolve_scheduler_model(model)
```

This keeps full fine-tuning for non-quantized checkpoints and switches to adapter
training automatically when Trainer rejects pure quantized full-parameter training.

## BoL Scans Usage

Drop this into Unsloth, TRL, HuggingFace Trainer, or any raw training loop:

```python
import wandb
from bol_scans import run_all

# Initialize your model and tokenizer...

# 1. Start W&B
wandb.init(project="my-project", name="bella-v6-eval")

# 2. Run scans before training
run_all(model, tokenizer, phase="pre")

# 3. Train as usual
trainer.train()

# 4. Run scans after training
run_all(model, tokenizer, phase="post")
```

`run_all(...)`:
- Uses the model and tokenizer you already loaded
- Detects architecture from `model.config` (and model structure fallback)
- Runs all 6 scans:
  - weight fingerprint
  - layer sweep
- component ablation
- silhouette
- CKA
- attention map
- Adds component-aware fingerprint panels (`q/k/v/o`, `gate/up/down`) and per-layer dimension profile tables
- Logs metrics/tables/charts to active `wandb.run` using `pre/*` or `post/*` key prefixes
