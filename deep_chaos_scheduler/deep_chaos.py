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
    """Zero out last-dim channels NOT in *alive*, while keeping the autograd graph intact.

    The old ``new_zeros() + index_copy_()`` implementation created a fresh
    leaf tensor and copied selected slices into it via an in-place op.  PyTorch
    cannot reliably back-propagate through that pattern — the resulting ``out``
    tensor has no ``grad_fn`` connected to the original ``tensor``, so all
    hooked projections receive zero (or undefined) gradients, producing
    grad_norm=NaN from step 0.

    The replacement builds a float mask of 0/1 values and multiplies element-
    wise.  Multiplication is fully differentiable: gradients flow back through
    the alive channels unchanged and are zeroed on the dead channels, which is
    exactly the intended semantics.
    """
    if alive is None:
        return tensor
    if tensor.ndim == 0:
        return tensor
    dim = int(tensor.shape[-1])
    if dim <= 0:
        return tensor
    if alive.numel() == 0:
        # All channels dead — multiply by a zero scalar so grad_fn is preserved.
        return tensor * 0.0
    alive = alive.to(device=tensor.device, dtype=torch.long)
    valid = alive[(alive >= 0) & (alive < dim)]
    if valid.numel() == 0:
        return tensor * 0.0
    # Build a float mask on the same device as `tensor`. Using scatter instead
    # of advanced-indexed assignment keeps the kernel launch simple and avoids
    # the extra CPU roundtrip that was previously here (which caused a
    # host/device sync on every forward hook).
    mask = torch.zeros(dim, dtype=tensor.dtype, device=tensor.device)
    mask.scatter_(0, valid, torch.ones_like(valid, dtype=tensor.dtype))
    # Broadcast mask over all leading dimensions and multiply.
    # This is differentiable: grad flows through alive positions unchanged.
    return tensor * mask


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
    # Layer hoist: at every reshuffle, physically rebuild model.layers to
    # contain only sacred layers + victims in mode=both.  Dead, identity,
    # attn-only, and mlp-only layers are removed from the ModuleList entirely
    # for that sticky block.  Forward pass has fewer blocks, no saved
    # activations or autograd graph for the absent layers.
    # Verdict on Qwen2.5-3B / MI300X / 100 steps / sticky=25:
    #   3.55x faster wall-clock, 42.9% peak VRAM cut.
    # When True, the post-hook path is NOT installed — hoist is the only
    # mechanism modifying the forward.
    # Default-on: you get the speedup unless you explicitly opt out with
    # use_layer_hoist=False.  Opt out if your architecture has unusual
    # layer parenting that _resolve_layers_parent can't trace, or if you
    # want the safer hook-only path for debugging.
    use_layer_hoist: bool = True
    # Hoist stub: tiny frozen perturbation inserted in place of contiguous
    # yanked runs.  Without it, removing dead/identity/attn/mlp layers
    # entirely makes adjacent surviving layers communicate directly when
    # they were trained for an intervening transformation.  The stub is a
    # graph-preserving "smoke screen": it touches the residual stream at
    # the position the missing block used to occupy, so downstream layers
    # see _some_ perturbation rather than raw upstream output.  One stub
    # per contiguous yanked run (not per layer) — accumulated bias drift
    # would otherwise compound across multi-layer runs.  Stubs are frozen
    # at construction; the alive layers adapt to them during training.
    #   "bias":    x -> x + b   (1 frozen vector per stub, recommended)
    #   "linear":  x -> x + frozen_Linear(x)  (one frozen matmul per stub)
    #   "none":    x -> x       (naked hoist, the residual collapse case)
    hoist_stub_kind: str = "bias"
    hoist_stub_init_scale: float = 0.01


class _ZeroMLPStub(torch.nn.Module):
    """Submodule stand-in for `layer.mlp` when topology mode is `attn`.

    The Qwen2 / Llama-style decoder layer adds the MLP block as
    `hidden_states = residual + self.mlp(post_attn_ln(hidden_states))`.
    Returning `zeros_like(input)` makes the second residual a no-op while
    keeping the attention contribution intact.  Stateless — one instance
    can be shared across all layers using it.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class _ZeroAttnStub(torch.nn.Module):
    """Submodule stand-in for `layer.self_attn` when topology mode is `mlp`.

    Modern transformers `Qwen2Attention.forward` returns a tuple
    `(attn_output, attn_weights)`.  Older versions and other architectures
    may return just the tensor.  Return a tuple here — the layer's forward
    is written as `hidden_states, _ = self.self_attn(...)`, which unpacks
    the tuple cleanly.  If a caller expects a single tensor it'll error
    loudly and we adapt.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return (torch.zeros_like(hidden_states), None)


class HoistStub(torch.nn.Module):
    """Frozen residual-stream perturbation that stands in for a contiguous
    run of hoisted layers.

    The stub does not learn.  Its only job is to ensure the residual stream
    is touched at the position(s) the yanked block used to occupy, so the
    next surviving layer doesn't suddenly receive the previous surviving
    layer's output verbatim (which it was never trained to consume).  The
    surviving layers ARE plastic during training and adapt to whatever the
    stub outputs, as long as the stub is consistent across forward passes.

    Stored weights use buffers (not Parameters) so the optimizer naturally
    ignores them and they don't show up in `.parameters()`.

    Forward signature is permissive: takes `hidden_states` as the first
    positional arg and ignores all other args/kwargs (attention_mask,
    position_ids, position_embeddings, past_key_values, ...).  Returns the
    perturbed hidden_states as a plain Tensor — matches modern
    `Qwen2DecoderLayer.forward` return convention.
    """

    def __init__(
        self,
        hidden_size: int,
        kind: str = "bias",
        init_scale: float = 0.01,
        seed: int = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if kind not in ("bias", "linear", "none"):
            raise ValueError(f"Unknown hoist_stub_kind: {kind!r}")
        self.kind = kind
        self.hidden_size = int(hidden_size)
        gen = torch.Generator().manual_seed(int(seed))
        target_dtype = dtype or torch.float32
        if kind == "bias":
            b = torch.randn(self.hidden_size, generator=gen) * float(init_scale)
            self.register_buffer("bias", b.to(dtype=target_dtype, device=device))
        elif kind == "linear":
            w = torch.randn(self.hidden_size, self.hidden_size, generator=gen) * float(init_scale)
            b = torch.randn(self.hidden_size, generator=gen) * float(init_scale)
            self.register_buffer("weight", w.to(dtype=target_dtype, device=device))
            self.register_buffer("linear_bias", b.to(dtype=target_dtype, device=device))
        # kind == "none": no buffers, forward is identity.

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self.kind == "bias":
            bias = self.bias
            if bias.dtype != hidden_states.dtype or bias.device != hidden_states.device:
                bias = bias.to(dtype=hidden_states.dtype, device=hidden_states.device)
            return hidden_states + bias
        if self.kind == "linear":
            w = self.weight
            b = self.linear_bias
            if w.dtype != hidden_states.dtype or w.device != hidden_states.device:
                w = w.to(dtype=hidden_states.dtype, device=hidden_states.device)
                b = b.to(dtype=hidden_states.dtype, device=hidden_states.device)
            return hidden_states + torch.nn.functional.linear(hidden_states, w, b)
        # kind == "none"
        return hidden_states

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, kind={self.kind!r}"


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
    # When True, this layer reuses K/V states from an earlier layer (e.g.
    # Gemma-4 num_kv_shared_layers).  k_proj / v_proj are absent or irrelevant
    # and MUST NOT be hooked — doing so would corrupt the shared state used by
    # all downstream layers that depend on it, causing NaN gradients.
    kv_shared: bool = False


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

        # Layer hoist state.  When use_layer_hoist is True we rebuild
        # model.layers (via the parent module) on every reshuffle.
        # Snapshot the original list + parent here so we can restore it.
        self._hoist_parent: torch.nn.Module | None = None
        self._hoist_attr: str | None = None
        self._hoist_originals: List[torch.nn.Module] = []
        self._hoist_last_kept: List[int] = []
        self._hoist_last_yanked: List[int] = []
        # One stub per original victim layer index, persistent across
        # reshuffles.  Used at run-start positions; subsequent layers in
        # the same contiguous yanked run reuse the run-start stub.
        self._hoist_stubs: Dict[int, HoistStub] = {}
        # Submodule-swap state: per-layer original submodules and the set of
        # attribute names currently swapped out for zero-return stubs.
        # Used so each reshuffle can restore previous swaps before applying
        # new ones, regardless of which mode each layer was in last time.
        self._hoist_orig_submodules: Dict[int, Dict[str, torch.nn.Module]] = {}
        self._hoist_active_swaps: Dict[int, set[str]] = {}
        # Shared zero-return stubs — stateless, one instance per layer-half.
        self._hoist_zero_mlp = _ZeroMLPStub()
        self._hoist_zero_attn = _ZeroAttnStub()
        if bool(getattr(config, "use_layer_hoist", False)):
            self._hoist_parent, self._hoist_attr = self._resolve_layers_parent(self.model)
            self._hoist_originals = list(getattr(self._hoist_parent, self._hoist_attr))
            self._build_hoist_stubs()

        self._build_layer_bindings()
        # Hoist replaces the post-hook path entirely — they're mutually
        # exclusive.  Hoist surgery happens at the end of step().
        if not bool(getattr(config, "use_layer_hoist", False)):
            self._install_hooks()

        print(
            f"DeepChaosScheduler: victims={self.victims[0] if self.victims else 'none'}-"
            f"{self.victims[-1] if self.victims else 'none'} sacred={sorted(self.sacred)} "
            f"sticky={self.config.sticky_interval} "
            f"hoist={bool(getattr(config, 'use_layer_hoist', False))}"
        )

    @staticmethod
    def _resolve_layers_parent(model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
        """Return the parent module and attribute name that owns the
        transformer-layer ModuleList.  Mirrors the path search used by
        `resolve_transformer_layers` but returns the parent so the caller
        can `setattr(parent, attr, new_list)` for hoist surgery."""
        base = _unwrap_model(model)
        candidates = [
            ("model", "layers"), ("model.model", "layers"),
            ("model.decoder", "layers"), ("model.language_model", "layers"),
            ("language_model.model", "layers"), ("language_model", "layers"),
            ("text_model", "layers"), ("decoder", "layers"),
            ("transformer", "layers"), ("transformer", "h"),
            ("gpt_neox", "layers"), ("", "layers"),
        ]
        for parent_path, attr in candidates:
            parent = base
            if parent_path:
                ok = True
                for part in parent_path.split("."):
                    if not hasattr(parent, part):
                        ok = False
                        break
                    parent = getattr(parent, part)
                if not ok:
                    continue
            layers = getattr(parent, attr, None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return parent, attr
        raise AttributeError(
            "Could not locate transformer layers ModuleList parent for hoist."
        )

    def _build_hoist_stubs(self) -> None:
        """Instantiate one frozen HoistStub per victim layer index, sized
        to that layer's hidden dim.  Called once at __init__ when
        use_layer_hoist is True; stubs persist for the life of the
        scheduler so the alive layers see consistent perturbations."""
        kind = str(getattr(self.config, "hoist_stub_kind", "bias"))
        if kind == "none" or not self._hoist_originals:
            return
        scale = float(getattr(self.config, "hoist_stub_init_scale", 0.01))
        seed_base = int(getattr(self.config, "seed", 0))
        # Infer hidden size + dtype from the model itself.
        try:
            sample_param = next(self.model.parameters())
            target_dtype = sample_param.dtype
            target_device = sample_param.device
        except StopIteration:
            target_dtype = torch.float32
            target_device = self.layer_device
        hidden_size = None
        if self.bindings:
            for binding in self.bindings.values():
                if binding.hidden_size:
                    hidden_size = int(binding.hidden_size)
                    break
        if hidden_size is None:
            cfg = getattr(self.model, "config", None)
            hidden_size = int(getattr(cfg, "hidden_size", 0)) if cfg is not None else 0
        if not hidden_size:
            return
        for idx in self.victims:
            self._hoist_stubs[idx] = HoistStub(
                hidden_size=hidden_size,
                kind=kind,
                init_scale=scale,
                seed=seed_base * 1_000_003 + idx,
                dtype=target_dtype,
                device=target_device,
            )

    def _restore_submodule_swaps(self) -> None:
        """Reverse any per-layer submodule swaps from a prior surgery."""
        for idx, attrs in self._hoist_active_swaps.items():
            layer = self._hoist_originals[idx]
            for attr in attrs:
                orig = self._hoist_orig_submodules.get(idx, {}).get(attr)
                if orig is not None:
                    setattr(layer, attr, orig)
        self._hoist_active_swaps.clear()

    def _swap_submodule(
        self,
        layer_idx: int,
        attr_name: str,
        stub_module: torch.nn.Module,
    ) -> bool:
        """Swap a single submodule attribute for a stub, recording the
        original so it can be restored later.  Returns True if the swap
        actually happened (the layer has the attribute and it isn't
        already the stub)."""
        layer = self._hoist_originals[layer_idx]
        current = getattr(layer, attr_name, None)
        if current is None:
            return False
        if current is stub_module:
            return False
        # Save original on first swap of this attr for this layer.
        slot = self._hoist_orig_submodules.setdefault(layer_idx, {})
        if attr_name not in slot:
            slot[attr_name] = current
        setattr(layer, attr_name, stub_module)
        self._hoist_active_swaps.setdefault(layer_idx, set()).add(attr_name)
        return True

    def _apply_layer_hoist_surgery(self) -> Tuple[int, int]:
        """Rebuild parent.layers based on per-layer topology mode.

        Surgery rules:
            - sacred:                        keep, no surgery
            - mode=both / identity:          keep, no surgery
                (identity in the post-hook path = layer's natural output —
                yanking it would change behaviour the post-hook path didn't.)
            - mode=attn:                     keep layer, swap layer.mlp -> zeros
            - mode=mlp:                      keep layer, swap layer.self_attn -> zeros
            - mode=dead:                     yank, insert bias-stub at run start
                (post-hook path produces x->x in mode=dead because all 7
                projections are zeroed — yanking is faithful to that.)

        Each surgery first restores the previous shuffle's submodule swaps,
        then applies fresh ones.  Bias stubs persist for the life of the
        scheduler and are inserted only at the START of each contiguous
        yanked run (avoids accumulated bias drift across multi-layer runs).

        Returns (kept_count, yanked_count).
        """
        parent = self._hoist_parent
        attr = self._hoist_attr
        if parent is None or attr is None:
            return (0, 0)

        # Step 1: restore any submodule swaps from the prior shuffle.
        self._restore_submodule_swaps()

        # Step 2: classify every original layer.
        kept_indices: List[int] = []
        yanked_indices: List[int] = []
        survivors: List[torch.nn.Module] = []
        prev_yanked = False
        for idx, layer in enumerate(self._hoist_originals):
            if idx in self.sacred:
                survivors.append(layer)
                kept_indices.append(idx)
                prev_yanked = False
                continue
            topo = self.topologies.get(idx)
            mode = topo.mode if topo is not None else "both"

            if mode in ("both", "identity"):
                # Layer runs at full strength.
                survivors.append(layer)
                kept_indices.append(idx)
                prev_yanked = False
            elif mode == "attn":
                # Skip MLP only.  Swap layer.mlp -> zero stub.
                self._swap_submodule(idx, "mlp", self._hoist_zero_mlp)
                survivors.append(layer)
                kept_indices.append(idx)
                prev_yanked = False
            elif mode == "mlp":
                # Skip self-attention only.  Swap layer.self_attn -> zero stub.
                # Some architectures use `attn` instead of `self_attn`; try both.
                if not self._swap_submodule(idx, "self_attn", self._hoist_zero_attn):
                    self._swap_submodule(idx, "attn", self._hoist_zero_attn)
                survivors.append(layer)
                kept_indices.append(idx)
                prev_yanked = False
            else:  # mode == "dead" (or anything else unrecognised)
                yanked_indices.append(idx)
                if not prev_yanked:
                    stub = self._hoist_stubs.get(idx)
                    if stub is not None:
                        survivors.append(stub)
                prev_yanked = True

        setattr(parent, attr, torch.nn.ModuleList(survivors))
        self._hoist_last_kept = kept_indices
        self._hoist_last_yanked = yanked_indices
        return (len(kept_indices), len(yanked_indices))

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

    @staticmethod
    def _detect_kv_shared(attn: torch.nn.Module) -> bool:
        """Return True if this attention layer reuses K/V from another layer.

        Covers:
        - Gemma-4: ``is_kv_shared_layer`` attribute set to True
        - Any layer where k_proj / v_proj are absent (no attribute at all)
        """
        # Explicit flag set by transformers (Gemma-4)
        if getattr(attn, "is_kv_shared_layer", False):
            return True
        # Implicit: k_proj exists but is None (use_alternative_attention path)
        # We check for total absence of both k_proj and v_proj as a heuristic
        has_k = _first_attr(attn, ("k_proj", "key", "wk")) is not None
        has_v = _first_attr(attn, ("v_proj", "value", "wv")) is not None
        if not has_k and not has_v:
            return True
        return False

    @staticmethod
    def _has_post_proj_norm(attn: torch.nn.Module) -> bool:
        """Return True if the attention module normalises q/k/v AFTER projection.

        Examples: Gemma-4 (q_norm, k_norm, v_norm), Gemma-2 (q_norm, k_norm).
        When this is True, zeroing raw projection outputs triggers division-by-
        zero (RMSNorm denominator collapses) or very large rescaling.  We skip
        q/k/v hooks and only hook o_proj in this case.
        """
        return (
            hasattr(attn, "q_norm")
            or hasattr(attn, "k_norm")
            or hasattr(attn, "v_norm")
        )

    def _build_layer_bindings(self):
        for layer_idx in self.victims:
            layer = self.layers[layer_idx]
            binding = LayerBindings(layer_idx=layer_idx)

            attn = _first_attr(layer, ("self_attn", "attn", "attention"))
            binding.attn_module = attn
            if attn is not None:
                binding.kv_shared = self._detect_kv_shared(attn)
                has_post_norm = self._has_post_proj_norm(attn)

                binding.q_proj = _first_attr(attn, ("q_proj", "query", "wq"))
                # Only resolve k/v projections if this layer owns them
                if not binding.kv_shared:
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
                # supports_attention_masks: we can hook q/k/v projections only
                # when:
                #   1. All four projections exist on this layer
                #   2. The architecture does NOT apply per-head norms after
                #      projection (q_norm / k_norm / v_norm).  Post-proj norms
                #      make partial-zeroing of projection outputs unsafe because
                #      the RMSNorm denominator can collapse toward zero, causing
                #      NaN.  When post-proj norms are present we fall back to
                #      hooking only o_proj.
                binding.supports_attention_masks = (
                    not has_post_norm
                    and not binding.kv_shared
                    and binding.q_proj is not None
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

            # Repair num_heads / num_kv_heads from projection out_features.
            # Modern transformers `Qwen2Attention` (>= 4.40) stores
            # num_key_value_heads only on `attn.config`, not on the module
            # itself, so the `_first_attr` lookups above miss it and
            # num_kv_heads silently defaults to num_heads.  On a GQA model
            # that produces alive_k_out / alive_v_out indices that overshoot
            # the actual k_proj / v_proj output dim.  Derive the real values
            # from the projection weight shapes whenever head_dim is known.
            if binding.head_dim is not None and binding.head_dim > 0:
                hd = int(binding.head_dim)
                if (
                    isinstance(binding.q_proj, torch.nn.Module)
                    and q_weight is not None
                    and q_weight.ndim == 2
                ):
                    derived = int(q_weight.shape[0] // hd)
                    if derived > 0 and derived != int(binding.num_heads or 0):
                        binding.num_heads = derived
                k_weight = (
                    getattr(binding.k_proj, "weight", None)
                    if binding.k_proj is not None
                    else None
                )
                if k_weight is not None and k_weight.ndim == 2:
                    derived_kv = int(k_weight.shape[0] // hd)
                    if derived_kv > 0 and derived_kv != int(binding.num_kv_heads or 0):
                        binding.num_kv_heads = derived_kv

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
        mode_layers: Dict[str, List[int]] = {"both": [], "attn": [], "mlp": [], "identity": [], "dead": []}
        stats = {key: [] for key in ("q", "k", "v", "o", "gate", "up", "down")}

        sliced_elements = 0
        full_elements = 0

        for layer in self.victims:
            binding = self.bindings[layer]
            topo = LayerTopology(mode=self._layer_mode(layer, rng))
            mode_counts[topo.mode] += 1
            mode_layers[topo.mode].append(layer)

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
            sacred_sorted = sorted(self.sacred)
            active_sorted = sorted(self.active_layers)
            dead_sorted = sorted(layer for layer in self.victims if layer not in self.active_layers)
            fmt_pct = lambda values: f"{100.0 * avg(values):5.1f}%" if values else "  n/a"
            print(
                f"\n{'=' * 78}\n"
                f"DeepChaos reshuffle  step={global_step}  block={block_idx}  "
                f"sticky_interval={self.config.sticky_interval}  seed={self.config.seed}\n"
                f"{'-' * 78}\n"
                f"  layer pool      : sacred={sacred_sorted}  victims={self.victims[0]}..{self.victims[-1]} "
                f"({num_victims})\n"
                f"  active layers   : {active_sorted}  "
                f"({active_count}/{max(1, num_victims)} victims, density={100.0 * active_count / max(1, num_victims):.1f}%)\n"
                f"  dead layers     : {dead_sorted}  ({len(dead_sorted)})\n"
                f"  mode breakdown  :\n"
                f"      both     ({mode_counts['both']:>2}): {sorted(mode_layers['both'])}\n"
                f"      attn     ({mode_counts['attn']:>2}): {sorted(mode_layers['attn'])}\n"
                f"      mlp      ({mode_counts['mlp']:>2}): {sorted(mode_layers['mlp'])}\n"
                f"      identity ({mode_counts['identity']:>2}): {sorted(mode_layers['identity'])}\n"
                f"      dead     ({mode_counts['dead']:>2}): {sorted(mode_layers['dead'])}\n"
                f"  attn surv%      : q={fmt_pct(stats['q'])}  k={fmt_pct(stats['k'])}  "
                f"v={fmt_pct(stats['v'])}  o={fmt_pct(stats['o'])}\n"
                f"  mlp  surv%      : gate={fmt_pct(stats['gate'])}  up={fmt_pct(stats['up'])}  "
                f"down={fmt_pct(stats['down'])}\n"
                f"  compute ratio   : {100.0 * compute_ratio:.1f}%  "
                f"({sliced_elements:,} / {full_elements:,} weight elements engaged)\n"
                f"{'=' * 78}"
            )

        # Layer hoist surgery: rebuild model.layers to contain only sacred
        # layers + victims in mode=both.  Runs at every reshuffle event.
        if bool(getattr(self.config, "use_layer_hoist", False)):
            kept, yanked = self._apply_layer_hoist_surgery()
            self.cached_stats["hoist_kept_layers"] = float(kept)
            self.cached_stats["hoist_yanked_layers"] = float(yanked)
            if bool(self.config.announce_reshuffles):
                # Count contiguous yanked runs (= number of stubs inserted).
                runs = 0
                prev = -2
                for i in sorted(self._hoist_last_yanked):
                    if i != prev + 1:
                        runs += 1
                    prev = i
                stub_kind = str(getattr(self.config, "hoist_stub_kind", "bias"))
                print(
                    f"  layer hoist     : kept {kept} layer(s), yanked {yanked} "
                    f"(yanked indices: {sorted(self._hoist_last_yanked)})\n"
                    f"  stubs           : {runs} {stub_kind!r}-stub(s) "
                    f"inserted at run starts"
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

            if topo.mode == "dead":
                return _replace_tensor_output(output, tensor * 0.0)
            if topo.mode == "identity":
                return output

            if component in {"q", "k", "v", "o"}:
                if not attn_enabled:
                    return _replace_tensor_output(output, tensor * 0.0)
                alive = {
                    "q": topo.alive_q_out,
                    "k": topo.alive_k_out,
                    "v": topo.alive_v_out,
                    "o": topo.alive_o_out,
                }[component]
                return _replace_tensor_output(output, _apply_last_dim_mask(tensor, alive))

            if component in {"gate", "up", "down"}:
                if not mlp_enabled:
                    return _replace_tensor_output(output, tensor * 0.0)
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

            # Attention hooks.
            # q/k/v projections are only hooked when supports_attention_masks is
            # True — this excludes architectures with post-projection norms
            # (Gemma-2/4 q_norm/k_norm) and KV-sharing layers.  Hooking those
            # would corrupt the shared KV cache used by downstream layers or
            # cause RMSNorm denominator collapse, both producing NaN gradients.
            if binding.supports_attention_masks:
                if binding.q_proj is not None:
                    self._install_hook_for_component(layer_idx, "q", binding.q_proj)
                if binding.k_proj is not None:
                    self._install_hook_for_component(layer_idx, "k", binding.k_proj)
                if binding.v_proj is not None:
                    self._install_hook_for_component(layer_idx, "v", binding.v_proj)
            # o_proj is always safe to hook — it runs after all norms and KV
            # operations.  It is the only attention hook applied on architectures
            # with post-projection norms or KV-shared layers.
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
        # Restore submodule swaps + original model.layers ModuleList if
        # hoist was active.
        if (
            self._hoist_parent is not None
            and self._hoist_attr is not None
            and self._hoist_originals
        ):
            self._restore_submodule_swaps()
            setattr(
                self._hoist_parent,
                self._hoist_attr,
                torch.nn.ModuleList(self._hoist_originals),
            )
            self._hoist_parent = None
            self._hoist_attr = None
            self._hoist_originals = []
            self._hoist_orig_submodules.clear()
        self.topologies.clear()
        self.cached_stats = None
        self.last_shuffle_step = None
        print("DeepChaosScheduler removed")
