# BoL W&B Evals

A modular HuggingFace `TrainerCallback` for running the Blocks of Life (BoL) diagnostic suite and logging interactive Custom Charts to Weights & Biases (W&B).

## Installation

```bash
pip install git+https://github.com/juiceb0xc0de/bol-wandb-evals.git
```

## Usage

Drop this into your Unsloth, Axolotl, or standard TRL trainer script to automatically run the BoL suite pre- and post-training:

```python
import wandb
from bol_wandb.callback import BoLWandbCallback
from transformers import SFTTrainer

# Initialize your model and tokenizer...

# 1. Initialize W&B run
wandb.init(project="my-project", name="bella-v6-eval")

# 2. Setup the BoL Callback
bol_evals = BoLWandbCallback(
    model=model,
    tokenizer=tokenizer,
    run_pre_train=True,
    run_post_train=True,
    # Optional overrides:
    # eval_texts=["Custom eval text 1", "Custom eval text 2"]
)

# 3. Add to Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    callbacks=[bol_evals],
    # ... other args
)

# 4. Train!
trainer.train()
```
