from __future__ import annotations

import numpy as np
import torch
import wandb

from ..config import CLUSTER_WORDS
from .common import resolve_layer_count


def centering_matrix(n):
    return np.eye(n) - np.ones((n, n)) / n


def linear_kernel(x):
    return x @ x.T


def hsic(k, l, h):
    n = k.shape[0]
    return np.trace(h @ k @ h @ l) / ((n - 1) ** 2)


def linear_cka(x, y):
    n = x.shape[0]
    h = centering_matrix(n)
    k = linear_kernel(x)
    l = linear_kernel(y)
    hsic_kl = hsic(k, l, h)
    hsic_kk = hsic(k, k, h)
    hsic_ll = hsic(l, l, h)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return hsic_kl / denom


def run_cka(model: torch.nn.Module, tokenizer) -> wandb.Table:
    device = next(model.parameters()).device
    model.eval()
    num_layers = resolve_layer_count(model)

    def get_hidden(word, layer_idx):
        inputs = tokenizer(word, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]
        return hidden.mean(dim=1).squeeze().cpu().float().numpy()

    sample_layers = list(range(0, num_layers + 1, max(1, num_layers // 10)))
    if num_layers not in sample_layers:
        sample_layers.append(num_layers)
    sample_layers = sorted(set(sample_layers))

    rep_matrices = {}
    for layer in sample_layers:
        vecs = [get_hidden(w, layer) for w in CLUSTER_WORDS]
        rep_matrices[layer] = np.stack(vecs)

    columns = ["Layer_X", "Layer_Y", "CKA_Score"]
    data = []
    for layer_i in sample_layers:
        for layer_j in sample_layers:
            data.append(
                [f"L{layer_i}", f"L{layer_j}", linear_cka(rep_matrices[layer_i], rep_matrices[layer_j])]
            )
    return wandb.Table(columns=columns, data=data)
