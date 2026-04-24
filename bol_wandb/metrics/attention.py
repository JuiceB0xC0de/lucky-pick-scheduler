from __future__ import annotations

import math

import numpy as np
import torch
import wandb

from ..config import RELATED_PAIRS, UNRELATED_PAIRS
from .common import resolve_layer_count


def attention_entropy(attn_dist):
    p = attn_dist.clamp(min=1e-10)
    return -(p * p.log()).sum().item()


def max_entropy(seq_len):
    return math.log(seq_len) if seq_len > 0 else 0


def run_attention_map(model: torch.nn.Module, tokenizer) -> wandb.Table:
    device = next(model.parameters()).device
    model.eval()
    num_layers = resolve_layer_count(model)

    def get_attention(word):
        inp = tokenizer(word, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        if not out.attentions:
            return None
        return [layer_attn[0, :, -1, :].cpu().float() for layer_attn in out.attentions]

    all_words = list({w for pair in RELATED_PAIRS + UNRELATED_PAIRS for w in pair})
    test_attn = get_attention(all_words[0])
    if test_attn is None:
        return wandb.Table(columns=["Layer", "Mean_Entropy", "Normalized_Entropy", "Interpretation"], data=[])

    attn_cache = {w: get_attention(w) for w in all_words}
    columns = ["Layer", "Mean_Entropy", "Normalized_Entropy", "Interpretation"]
    data = []

    observed_layers = min(num_layers, len(test_attn))
    for layer in range(observed_layers):
        layer_entropies = []
        seq_len = None
        for word, attn_layers in attn_cache.items():
            attn = attn_layers[layer]
            seq_len = attn.shape[1]
            for head in range(attn.shape[0]):
                layer_entropies.append(attention_entropy(attn[head]))
        mean_h = np.mean(layer_entropies)
        max_possible = max_entropy(seq_len or 0)
        norm_h = mean_h / max_possible if max_possible > 0 else 0
        interp = "focused" if norm_h < 0.5 else "moderate" if norm_h < 0.8 else "diffuse"
        data.append([layer, mean_h, norm_h, interp])

    return wandb.Table(columns=columns, data=data)
