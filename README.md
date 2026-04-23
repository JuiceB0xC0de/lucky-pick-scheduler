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
