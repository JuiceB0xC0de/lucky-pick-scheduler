import torch
import wandb
from transformers import TrainerCallback
from typing import List, Optional

# Import metrics
from .metrics.fingerprint import run_fingerprint
from .metrics.silhouette import run_silhouette
from .metrics.cka import run_cka
from .metrics.attention import run_attention_map
from .metrics.ablation import run_layer_sweep, run_component_ablation

class BoLWandbCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that runs the Blocks of Life (BoL) 
    diagnostic suite and logs custom charts to Weights & Biases.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        tokenizer, 
        run_pre_train: bool = True,
        run_post_train: bool = True,
        eval_texts: Optional[List[str]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.run_pre_train = run_pre_train
        self.run_post_train = run_post_train
        self.eval_texts = eval_texts
        
        # We store the original weights footprint here to compare pre/post
        self.baseline_fingerprint = None
        self.baseline_silhouette = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Fires before training starts. We snapshot the baseline."""
        if not self.run_pre_train:
            return
            
        print("\n[BoL W&B] Running Pre-Train Diagnostics...")
        
        if wandb.run is None:
            print("[BoL W&B] Warning: W&B is not initialized. Metrics will not be logged.")
            return

        # 1. Snapshot base weights
        self.baseline_fingerprint = run_fingerprint(self.model)
        wandb.log({"bol/pre_train_fingerprint": self.baseline_fingerprint})
        
        # 2. Snapshot base semantic separation
        self.baseline_silhouette = run_silhouette(self.model, self.tokenizer)
        wandb.log({"bol/pre_train_silhouette": self.baseline_silhouette})
        
        print("[BoL W&B] Pre-Train Diagnostics Complete.\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Fires after training finishes. Runs the full suite."""
        if not self.run_post_train:
            return
            
        print("\n[BoL W&B] Running Post-Train Diagnostics...")
        
        if wandb.run is None:
            print("[BoL W&B] Warning: W&B is not initialized. Metrics will not be logged.")
            return

        # Run all 6 metrics and log to W&B
        post_fingerprint = run_fingerprint(self.model)
        post_silhouette = run_silhouette(self.model, self.tokenizer)
        cka_heatmap = run_cka(self.model, self.tokenizer)
        attn_entropy = run_attention_map(self.model, self.tokenizer)
        layer_sweep = run_layer_sweep(self.model, self.tokenizer, self.eval_texts)
        comp_ablation = run_component_ablation(self.model, self.tokenizer, self.eval_texts)
        
        wandb.log({
            "bol/post_train_fingerprint": post_fingerprint,
            "bol/post_train_silhouette": post_silhouette,
            "bol/cka_heatmap": cka_heatmap,
            "bol/attention_entropy": attn_entropy,
            "bol/layer_ablation_sweep": layer_sweep,
            "bol/component_ablation": comp_ablation
        })
        
        print("[BoL W&B] Post-Train Diagnostics Complete.\n")
