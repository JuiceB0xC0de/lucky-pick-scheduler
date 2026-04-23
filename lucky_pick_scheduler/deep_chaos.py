"""Generic sticky-interval topology scheduler for transformer finetuning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        return model.module
    return model


def resolve_transformer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    base = _unwrap_model(model)

    def _get_dotted_attr(obj: Any, dotted: str):
        current = obj
        for part in dotted.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    explicit_paths = [
        "model.layers",
        "model.model.layers",
        "model.decoder.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "text_model.layers",
        "decoder.layers",
        "transformer.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
    ]
    for path in explicit_paths:
        layers = _get_dotted_attr(base, path)
        if isinstance(layers, (list, tuple, torch.nn.ModuleList)) and len(layers) > 0:
            return list(layers)

    def _block_score(module: torch.nn.Module) -> int:
        score = 0
        if hasattr(module, "self_attn") or hasattr(module, "attention") or hasattr(module, "attn"):
            score += 2
        if hasattr(module, "mlp") or hasattr(module, "feed_forward") or hasattr(module, "ffn"):
            score += 2
        name = module.__class__.__name__.lower()
        if "decoder" in name or "block" in name or "layer" in name:
            score += 1
        return score

    best_layers = None
    best_score = float("-inf")
    for name, module in base.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        samples = list(module[: min(4, len(module))])
        child_score = sum(_block_score(child) for child in samples) / max(1, len(samples))
        path_score = 0.0
        lname = name.lower()
        if "language_model" in lname or "text" in lname or "decoder" in lname:
            path_score += 3.0
        if lname.endswith("layers") or lname.endswith(".layers") or lname.endswith(".h") or lname.endswith("blocks"):
            path_score += 2.0
        if "vision" in lname or "image" in lname or "audio" in lname or "encoder" in lname:
            path_score -= 4.0
        total_score = child_score + path_score + (0.01 * len(module))
        if total_score > best_score:
            best_score = total_score
            best_layers = module

    if best_layers is not None and len(best_layers) > 0:
        return list(best_layers)
    raise AttributeError("Could not infer transformer decoder layers from model structure.")


def _first_attr(module: Any, names: Sequence[str]) -> Any | None:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _as_tensor_output(output):
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def _replace_tensor_output(output, new_tensor):
    if isinstance(output, tuple) and output:
        return (new_tensor,) + output[1:]
    return new_tensor


def _apply_last_dim_mask(tensor: torch.Tensor, alive: torch.Tensor | None) -> torch.Tensor:
    if alive is None:
        return tensor
    if tensor.ndim == 0:
        return tensor
    if tensor.shape[-1] <= 0:
        return tensor
    if alive.numel() == 0:
        return tensor.new_zeros(tensor.shape)
    alive = alive.to(device=tensor.device, dtype=torch.long)
    valid = alive[(alive >= 0) & (alive < int(tensor.shape[-1]))]
    if valid.numel() == 0:
        return tensor.new_zeros(tensor.shape)
    if valid.numel() > 1:
        valid = torch.unique(valid, sorted=True)
    out = tensor.new_zeros(tensor.shape)
    out.index_copy_(-1, valid, tensor.index_select(-1, valid))
    return out


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


@dataclass
class DeepChaosConfig:
    sacred_layers: Sequence[int] | None = None
    victim_range: Tuple[int, int] | None = None  # end-exclusive
    min_layer_survival: float = 0.30
    max_layer_survival: float = 0.70
    min_head_survival: float = 0.30
    max_head_survival: float = 0.70
    channel_group_size: int = 128
    min_channel_survival: float = 0.30
    max_channel_survival: float = 0.70
    mlp_gate_group_size: int = 128
    min_mlp_gate_survival: float = 0.35
    max_mlp_gate_survival: float = 0.80
    hidden_group_size: int = 64
    min_hidden_survival: float = 0.60
    max_hidden_survival: float = 0.95
    max_consecutive_on: int = 5
    max_consecutive_off: int = 10
    sticky_interval: int = 50
    announce_reshuffles: bool = True
    seed: int = 42


@dataclass
class LayerBindings:
    layer_idx: int
    attn_module: torch.nn.Module | None = None
    q_proj: torch.nn.Module | None = None
    k_proj: torch.nn.Module | None = None
    v_proj: torch.nn.Module | None = None
    o_proj: torch.nn.Module | None = None
    mlp_module: torch.nn.Module | None = None
    gate_proj: torch.nn.Module | None = None
    up_proj: torch.nn.Module | None = None
    down_proj: torch.nn.Module | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    supports_attention_masks: bool = False
    supports_mlp_masks: bool = False


@dataclass
class LayerTopology:
    mode: str = "both"  # both | attn | mlp | identity | dead
    alive_q_heads: List[int] | None = None
    alive_kv_heads: List[int] | None = None
    alive_q_out: torch.Tensor | None = None
    alive_k_out: torch.Tensor | None = None
    alive_v_out: torch.Tensor | None = None
    alive_o_out: torch.Tensor | None = None
    alive_gate_out: torch.Tensor | None = None
    alive_up_out: torch.Tensor | None = None
    alive_down_in: torch.Tensor | None = None
    alive_down_out: torch.Tensor | None = None


class DeepChaosScheduler:
    """Generic sticky topology scheduler using projection hooks.

    This preserves the original "sticky block + odds-based subsampling" behavior
    while auto-configuring itself from the loaded model.
    """

    def __init__(self, model: torch.nn.Module, config: DeepChaosConfig):
        self.model = _unwrap_model(model)
        self.config = config
        self.layers = resolve_transformer_layers(self.model)
        self.total_layers = len(self.layers)
        if self.total_layers == 0:
            raise ValueError("No transformer layers found.")

        victim_start, victim_end = self._resolve_victim_range(config.victim_range)
        self.victims = list(range(victim_start, victim_end))
        self.sacred = set(self._resolve_sacred_layers(config.sacred_layers))
        self.on_streak = {layer: 0 for layer in self.victims}
        self.off_streak = {layer: 0 for layer in self.victims}
        self.active_layers: set[int] = set(self.sacred)
        self.topologies: Dict[int, LayerTopology] = {}
        self.last_shuffle_step: int | None = None
        self.cached_stats: Dict[str, float] | None = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.bindings: Dict[int, LayerBindings] = {}
        self.layer_device = next(self.model.parameters()).device

        self._build_layer_bindings()
        self._install_hooks()

        print(
            f"DeepChaosScheduler: victims={self.victims[0] if self.victims else 'none'}-"
            f"{self.victims[-1] if self.victims else 'none'} sacred={sorted(self.sacred)} "
            f"sticky={self.config.sticky_interval}"
        )

    @classmethod
    def from_model(cls, model: torch.nn.Module, **overrides: Any):
        return cls(model, DeepChaosConfig(**overrides))

    def _resolve_victim_range(self, victim_range: Tuple[int, int] | None) -> Tuple[int, int]:
        if victim_range is None:
            if self.total_layers <= 4:
                return (0, self.total_layers)
            return (2, self.total_layers - 2)
        start, end = int(victim_range[0]), int(victim_range[1])
        start = max(0, min(start, self.total_layers - 1))
        end = max(start + 1, min(end, self.total_layers))
        return (start, end)

    def _resolve_sacred_layers(self, sacred_layers: Sequence[int] | None) -> List[int]:
        if sacred_layers is None:
            if self.total_layers <= 2:
                return list(range(self.total_layers))
            if self.total_layers <= 4:
                return [0, self.total_layers - 1]
            return [0, 1, self.total_layers - 2, self.total_layers - 1]
        resolved = []
        for idx in sacred_layers:
            idx_i = max(0, min(int(idx), self.total_layers - 1))
            if idx_i not in resolved:
                resolved.append(idx_i)
        return resolved

    def _build_layer_bindings(self):
        for layer_idx in self.victims:
            layer = self.layers[layer_idx]
            binding = LayerBindings(layer_idx=layer_idx)

            attn = _first_attr(layer, ("self_attn", "attn", "attention"))
            binding.attn_module = attn
            if attn is not None:
                binding.q_proj = _first_attr(attn, ("q_proj", "query", "wq"))
                binding.k_proj = _first_attr(attn, ("k_proj", "key", "wk"))
                binding.v_proj = _first_attr(attn, ("v_proj", "value", "wv"))
                binding.o_proj = _first_attr(attn, ("o_proj", "out_proj", "dense", "wo"))
                binding.num_heads = _safe_int(
                    _first_attr(attn, ("num_heads", "num_attention_heads", "n_heads"))
                )
                binding.num_kv_heads = _safe_int(
                    _first_attr(attn, ("num_key_value_heads", "num_kv_heads", "n_kv_heads")),
                    default=binding.num_heads,
                )
                binding.head_dim = _safe_int(_first_attr(attn, ("head_dim",)))
                binding.supports_attention_masks = (
                    binding.q_proj is not None
                    and binding.k_proj is not None
                    and binding.v_proj is not None
                    and binding.o_proj is not None
                    and isinstance(binding.q_proj, torch.nn.Module)
                )

            mlp = _first_attr(layer, ("mlp", "feed_forward", "ffn", "ff"))
            binding.mlp_module = mlp
            if mlp is not None:
                binding.gate_proj = _first_attr(mlp, ("gate_proj", "w1"))
                binding.up_proj = _first_attr(mlp, ("up_proj", "fc1", "c_fc", "dense_h_to_4h"))
                binding.down_proj = _first_attr(mlp, ("down_proj", "fc2", "c_proj", "dense_4h_to_h"))
                binding.supports_mlp_masks = (
                    binding.up_proj is not None
                    and binding.down_proj is not None
                )

            # infer dimensions from projection weights
            q_weight = getattr(binding.q_proj, "weight", None) if binding.q_proj is not None else None
            up_weight = getattr(binding.up_proj, "weight", None) if binding.up_proj is not None else None
            o_weight = getattr(binding.o_proj, "weight", None) if binding.o_proj is not None else None
            down_weight = getattr(binding.down_proj, "weight", None) if binding.down_proj is not None else None

            if q_weight is not None and q_weight.ndim == 2:
                binding.hidden_size = _safe_int(q_weight.shape[1])
                if binding.num_heads is None:
                    cfg = getattr(self.model, "config", None)
                    binding.num_heads = _safe_int(getattr(cfg, "num_attention_heads", None))
                if binding.num_heads is not None and binding.num_heads > 0:
                    binding.head_dim = binding.head_dim or int(q_weight.shape[0] // binding.num_heads)
                if binding.num_kv_heads is None:
                    binding.num_kv_heads = binding.num_heads

            if up_weight is not None and up_weight.ndim == 2:
                binding.intermediate_size = _safe_int(up_weight.shape[0])
                if binding.hidden_size is None:
                    binding.hidden_size = _safe_int(up_weight.shape[1])

            if binding.hidden_size is None and o_weight is not None and o_weight.ndim == 2:
                binding.hidden_size = _safe_int(o_weight.shape[0])
            if binding.hidden_size is None and down_weight is not None and down_weight.ndim == 2:
                binding.hidden_size = _safe_int(down_weight.shape[0])

            self.bindings[layer_idx] = binding

        if not self.bindings:
            raise ValueError("No victim layers available for DeepChaosScheduler.")

    def _sample_groups(
        self,
        rng: random.Random,
        total_dim: int,
        group_size: int,
        min_rate: float,
        max_rate: float,
        device: torch.device,
    ) -> torch.Tensor:
        if total_dim <= 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        group_size = max(1, int(group_size))
        num_groups = max(1, (total_dim + group_size - 1) // group_size)
        rate = rng.uniform(min_rate, max_rate)
        num_alive = max(1, min(num_groups, int(round(num_groups * rate))))
        alive_groups = sorted(rng.sample(range(num_groups), num_alive))
        indices: List[int] = []
        for group in alive_groups:
            start = group * group_size
            indices.extend(range(start, min(start + group_size, total_dim)))
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _heads_to_indices(self, heads: Sequence[int], head_dim: int, device: torch.device) -> torch.Tensor:
        indices: List[int] = []
        for head in heads:
            start = int(head) * int(head_dim)
            indices.extend(range(start, start + int(head_dim)))
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _subsample_indices(
        self,
        base: torch.Tensor,
        group_size: int,
        min_rate: float,
        max_rate: float,
        rng: random.Random,
    ) -> torch.Tensor:
        if base.numel() <= 1:
            return base
        n = int(base.numel())
        group_size = max(1, int(group_size))
        num_groups = max(1, (n + group_size - 1) // group_size)
        num_alive = max(1, min(num_groups, int(round(num_groups * rng.uniform(min_rate, max_rate)))))
        alive_groups = sorted(rng.sample(range(num_groups), num_alive))
        local_indices: List[int] = []
        for group in alive_groups:
            start = group * group_size
            local_indices.extend(range(start, min(start + group_size, n)))
        return base[torch.tensor(local_indices, dtype=torch.long, device=base.device)]

    def _layer_mode(self, layer_idx: int, rng: random.Random) -> str:
        if layer_idx not in self.active_layers:
            return "dead"
        draw = rng.random()
        if draw < 0.30:
            return "both"
        if draw < 0.55:
            return "attn"
        if draw < 0.80:
            return "mlp"
        return "identity"

    def step(self, global_step: int) -> Dict[str, float]:
        if (
            self.cached_stats is not None
            and self.last_shuffle_step is not None
            and global_step < self.last_shuffle_step + max(1, int(self.config.sticky_interval))
        ):
            cached = dict(self.cached_stats)
            cached["reshuffle_event"] = 0.0
            return cached

        self.last_shuffle_step = int(global_step)
        block_idx = int(global_step) // max(1, int(self.config.sticky_interval))
        rng = random.Random(int(self.config.seed) + int(global_step))
        device = self.layer_device

        num_victims = len(self.victims)
        min_active = max(1, int(round(num_victims * self.config.min_layer_survival)))
        max_active = max(min_active, int(round(num_victims * self.config.max_layer_survival)))
        target_active = rng.randint(min_active, max_active)

        forced_on = [layer for layer in self.victims if self.off_streak[layer] >= int(self.config.max_consecutive_off)]
        forced_off = [layer for layer in self.victims if self.on_streak[layer] >= int(self.config.max_consecutive_on)]
        available = [layer for layer in self.victims if layer not in forced_on and layer not in forced_off]

        active = set(forced_on)
        remaining = target_active - len(active)
        if remaining > 0 and available:
            active.update(rng.sample(available, min(remaining, len(available))))

        for layer in self.victims:
            if layer in active:
                self.on_streak[layer] += 1
                self.off_streak[layer] = 0
            else:
                self.on_streak[layer] = 0
                self.off_streak[layer] += 1

        self.active_layers = active | self.sacred
        self.topologies.clear()
        mode_counts = {"both": 0, "attn": 0, "mlp": 0, "identity": 0, "dead": 0}
        stats = {key: [] for key in ("q", "k", "v", "o", "gate", "up", "down")}

        sliced_elements = 0
        full_elements = 0

        for layer in self.victims:
            binding = self.bindings[layer]
            topo = LayerTopology(mode=self._layer_mode(layer, rng))
            mode_counts[topo.mode] += 1

            if topo.mode in ("dead", "identity"):
                self.topologies[layer] = topo
                continue

            hidden_size = int(binding.hidden_size or 0)
            inter_size = int(binding.intermediate_size or 0)
            num_heads = int(binding.num_heads or 0)
            num_kv_heads = int(binding.num_kv_heads or num_heads or 0)
            head_dim = int(binding.head_dim or 0)

            if topo.mode in ("both", "attn") and binding.supports_attention_masks and num_heads > 0 and head_dim > 0:
                kv_rate = rng.uniform(self.config.min_head_survival, self.config.max_head_survival)
                n_alive_kv = max(1, min(num_kv_heads, int(round(num_kv_heads * kv_rate))))
                alive_kv = sorted(rng.sample(range(num_kv_heads), n_alive_kv))
                gqa_ratio = max(1, num_heads // max(1, num_kv_heads))
                alive_q: List[int] = []
                for kv_head in alive_kv:
                    for q_head in range(kv_head * gqa_ratio, (kv_head + 1) * gqa_ratio):
                        if q_head < num_heads:
                            alive_q.append(q_head)
                topo.alive_kv_heads = alive_kv
                topo.alive_q_heads = alive_q
                topo.alive_q_out = self._heads_to_indices(alive_q, head_dim, device)
                topo.alive_k_out = self._heads_to_indices(alive_kv, head_dim, device)
                topo.alive_v_out = self._heads_to_indices(alive_kv, head_dim, device)
                topo.alive_o_out = self._sample_groups(
                    rng,
                    total_dim=hidden_size,
                    group_size=self.config.hidden_group_size,
                    min_rate=self.config.min_hidden_survival,
                    max_rate=self.config.max_hidden_survival,
                    device=device,
                )
                if hidden_size > 0:
                    stats["o"].append(float(topo.alive_o_out.numel()) / float(hidden_size))
                if topo.alive_q_out is not None and topo.alive_k_out is not None and topo.alive_v_out is not None:
                    qkv_out = topo.alive_q_out.numel() + topo.alive_k_out.numel() + topo.alive_v_out.numel()
                    sliced_elements += int(qkv_out * max(1, hidden_size))
                    full_elements += int(max(1, hidden_size) * max(1, hidden_size) * 4)
                    if num_heads * head_dim > 0:
                        stats["q"].append(float(topo.alive_q_out.numel()) / float(num_heads * head_dim))
                    if num_kv_heads * head_dim > 0:
                        stats["k"].append(float(topo.alive_k_out.numel()) / float(num_kv_heads * head_dim))
                        stats["v"].append(float(topo.alive_v_out.numel()) / float(num_kv_heads * head_dim))

            if topo.mode in ("both", "mlp") and binding.supports_mlp_masks and inter_size > 0 and hidden_size > 0:
                channels = self._sample_groups(
                    rng,
                    total_dim=inter_size,
                    group_size=self.config.channel_group_size,
                    min_rate=self.config.min_channel_survival,
                    max_rate=self.config.max_channel_survival,
                    device=device,
                )
                core = self._subsample_indices(
                    channels,
                    group_size=self.config.mlp_gate_group_size,
                    min_rate=self.config.min_mlp_gate_survival,
                    max_rate=self.config.max_mlp_gate_survival,
                    rng=rng,
                )
                topo.alive_gate_out = core
                topo.alive_up_out = core
                topo.alive_down_in = core
                topo.alive_down_out = self._sample_groups(
                    rng,
                    total_dim=hidden_size,
                    group_size=self.config.hidden_group_size,
                    min_rate=self.config.min_hidden_survival,
                    max_rate=self.config.max_hidden_survival,
                    device=device,
                )
                stats["gate"].append(float(core.numel()) / float(inter_size))
                stats["up"].append(float(core.numel()) / float(inter_size))
                stats["down"].append(float(topo.alive_down_out.numel()) / float(hidden_size))
                sliced_elements += int(core.numel() * hidden_size * 2 + topo.alive_down_out.numel() * core.numel())
                full_elements += int(inter_size * hidden_size * 3)

            self.topologies[layer] = topo

        active_count = sum(1 for layer in self.victims if layer in self.active_layers)
        avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        compute_ratio = float(sliced_elements) / float(max(1, full_elements))

        self.cached_stats = {
            "shuffle_step": float(global_step),
            "shuffle_block": float(block_idx),
            "reshuffle_event": 1.0,
            "active_layers": float(active_count),
            "layer_density_pct": 100.0 * float(active_count) / float(max(1, num_victims)),
            "mode_both": float(mode_counts["both"]),
            "mode_attn_only": float(mode_counts["attn"]),
            "mode_mlp_only": float(mode_counts["mlp"]),
            "mode_identity": float(mode_counts["identity"]),
            "mode_dead": float(mode_counts["dead"]),
            "avg_q_surv": 100.0 * avg(stats["q"]),
            "avg_k_surv": 100.0 * avg(stats["k"]),
            "avg_v_surv": 100.0 * avg(stats["v"]),
            "avg_o_surv": 100.0 * avg(stats["o"]),
            "avg_gate_surv": 100.0 * avg(stats["gate"]),
            "avg_up_surv": 100.0 * avg(stats["up"]),
            "avg_down_surv": 100.0 * avg(stats["down"]),
            "compute_ratio": compute_ratio,
            "compute_pct": 100.0 * compute_ratio,
        }
        if bool(self.config.announce_reshuffles):
            print(
                "DeepChaos reshuffle: "
                f"step={global_step} block={block_idx} "
                f"active={active_count}/{max(1, num_victims)} "
                f"modes(both/attn/mlp/id/dead)="
                f"{mode_counts['both']}/{mode_counts['attn']}/{mode_counts['mlp']}/"
                f"{mode_counts['identity']}/{mode_counts['dead']}"
            )
        return self.cached_stats

    def freeze_topology(self, step: int):
        self.step(step)

    def _install_hook_for_component(self, layer_idx: int, component: str, module: torch.nn.Module):
        def hook_fn(_module, _inputs, output):
            if not self.model.training:
                return output
            topo = self.topologies.get(layer_idx)
            if topo is None:
                return output

            attn_enabled = topo.mode in ("both", "attn")
            mlp_enabled = topo.mode in ("both", "mlp")
            tensor = _as_tensor_output(output)
            if not isinstance(tensor, torch.Tensor):
                return output

            if topo.mode in ("dead", "identity"):
                return _replace_tensor_output(output, tensor.new_zeros(tensor.shape))

            if component in {"q", "k", "v", "o"}:
                if not attn_enabled:
                    return _replace_tensor_output(output, tensor.new_zeros(tensor.shape))
                alive = {
                    "q": topo.alive_q_out,
                    "k": topo.alive_k_out,
                    "v": topo.alive_v_out,
                    "o": topo.alive_o_out,
                }[component]
                return _replace_tensor_output(output, _apply_last_dim_mask(tensor, alive))

            if component in {"gate", "up", "down"}:
                if not mlp_enabled:
                    return _replace_tensor_output(output, tensor.new_zeros(tensor.shape))
                alive = {
                    "gate": topo.alive_gate_out,
                    "up": topo.alive_up_out,
                    "down": topo.alive_down_out,
                }[component]
                return _replace_tensor_output(output, _apply_last_dim_mask(tensor, alive))

            return output

        self.hook_handles.append(module.register_forward_hook(hook_fn))

    def _install_hooks(self):
        for layer_idx in self.victims:
            binding = self.bindings[layer_idx]
            if binding.q_proj is not None:
                self._install_hook_for_component(layer_idx, "q", binding.q_proj)
            if binding.k_proj is not None:
                self._install_hook_for_component(layer_idx, "k", binding.k_proj)
            if binding.v_proj is not None:
                self._install_hook_for_component(layer_idx, "v", binding.v_proj)
            if binding.o_proj is not None:
                self._install_hook_for_component(layer_idx, "o", binding.o_proj)

            if binding.gate_proj is not None:
                self._install_hook_for_component(layer_idx, "gate", binding.gate_proj)
            if binding.up_proj is not None:
                self._install_hook_for_component(layer_idx, "up", binding.up_proj)
            if binding.down_proj is not None:
                self._install_hook_for_component(layer_idx, "down", binding.down_proj)

    def remove(self):
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()
        self.topologies.clear()
        self.cached_stats = None
        self.last_shuffle_step = None
        print("DeepChaosScheduler removed")
