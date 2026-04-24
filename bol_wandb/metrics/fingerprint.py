from __future__ import annotations

import torch
import wandb

from .common import resolve_decoder_layers


def run_fingerprint(model: torch.nn.Module) -> wandb.Table:
    columns = ["Layer Name", "Shape", "Mean", "Std", "Abs Mean", "Max", "Min", "Sparsity", "Frobenius"]
    data = []
    for _, layer_name, layer in resolve_decoder_layers(model):
        for local_name, param in layer.named_parameters(recurse=True):
            if param.ndim < 2:
                continue
            full_name = f"{layer_name}.{local_name}"
            p_data = param.data.float()
            data.append(
                [
                    full_name,
                    str(list(param.shape)),
                    p_data.mean().item(),
                    p_data.std().item(),
                    p_data.abs().mean().item(),
                    p_data.max().item(),
                    p_data.min().item(),
                    (p_data.abs() < 1e-6).float().mean().item(),
                    p_data.norm().item(),
                ]
            )
    return wandb.Table(columns=columns, data=data)
