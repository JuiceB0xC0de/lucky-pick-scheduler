# BoL Scans

Trainer-agnostic BoL scan module for running all six model analysis passes against an already-loaded model/tokenizer and logging to W&B.

## Installation

```bash
pip install git+https://github.com/JuiceB0xC0de/lucky-pick-scheduler.git
```

## Usage

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
- Logs metrics/tables/charts to active `wandb.run` using `pre/*` or `post/*` key prefixes
