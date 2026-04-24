from __future__ import annotations

from typing import List, Tuple

import torch

from lucky_pick_scheduler import resolve_transformer_layers


def resolve_decoder_layers(model: torch.nn.Module) -> List[Tuple[int, str, torch.nn.Module]]:
    layers = resolve_transformer_layers(model)
    return [(idx, f"layers.{idx}", layer) for idx, layer in enumerate(layers)]


def resolve_layer_count(model: torch.nn.Module) -> int:
    return len(resolve_transformer_layers(model))
