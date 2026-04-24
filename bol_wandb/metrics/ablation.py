from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import wandb

from ..config import DEFAULT_EVAL_TEXTS
from .common import resolve_decoder_layers


def _backup_and_zero_params(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    matcher,
) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    for name, param in named_params:
        if param.ndim < 2:
            continue
        if not matcher(name):
            continue
        backups.append((param, param.data.detach().clone()))
        param.data.zero_()
    return backups


def _restore_backups(backups: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    for param, backup in backups:
        param.data.copy_(backup)


def run_layer_sweep(model: torch.nn.Module, tokenizer, eval_texts=None) -> wandb.Table:
    device = next(model.parameters()).device
    model.eval()
    layers = resolve_decoder_layers(model)

    texts = eval_texts if eval_texts else DEFAULT_EVAL_TEXTS
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    labels = tokenized["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    labels = labels.to(device)

    def get_loss():
        with torch.no_grad():
            outputs = model(**tokenized, labels=labels)
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float("nan")
        return loss.item()

    baseline_loss = get_loss()
    columns = ["Layer", "Loss", "Damage", "Impact"]
    data = []

    for layer_idx, _, layer in layers:
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            backups = _backup_and_zero_params(
                list(layer.named_parameters(recurse=True)),
                lambda _name: True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            loss = get_loss()
            if math.isnan(loss) or math.isinf(loss):
                loss_val = 999.0
                damage = 999.0
                tag = "CRITICAL"
            else:
                loss_val = loss
                damage = loss - baseline_loss
                tag = "CRITICAL" if damage > 1.0 else "MODERATE" if damage > 0.3 else "REMOVABLE"
            data.append([f"L{layer_idx}", loss_val, damage, tag])
        finally:
            _restore_backups(backups)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return wandb.Table(columns=columns, data=data)


def run_component_ablation(model: torch.nn.Module, tokenizer, eval_texts=None) -> wandb.Table:
    device = next(model.parameters()).device
    model.eval()
    layers = resolve_decoder_layers(model)

    components = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    texts = eval_texts if eval_texts else DEFAULT_EVAL_TEXTS
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    labels = tokenized["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    labels = labels.to(device)

    def get_loss():
        with torch.no_grad():
            outputs = model(**tokenized, labels=labels)
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float("nan")
        return loss.item()

    baseline_loss = get_loss()
    columns = ["Component", "Zeroed_Tensors", "Loss", "Damage", "Impact"]
    data = []

    for comp in components:
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for _, _, layer in layers:
                backups.extend(
                    _backup_and_zero_params(
                        list(layer.named_parameters(recurse=True)),
                        lambda name, c=comp: c in name,
                    )
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            loss = get_loss()
            if math.isnan(loss) or math.isinf(loss):
                loss_val = 999.0
                damage = 999.0
                tag = "CRITICAL"
            else:
                loss_val = loss
                damage = loss - baseline_loss
                tag = "CRITICAL" if damage > 1.0 else "MODERATE" if damage > 0.3 else "REMOVABLE"
            data.append([comp, len(backups), loss_val, damage, tag])
        finally:
            _restore_backups(backups)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return wandb.Table(columns=columns, data=data)
