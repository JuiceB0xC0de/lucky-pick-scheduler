"""Trainer-agnostic BoL scans with W&B logging.

Usage:
    from bol_scans import run_all

    run_all(model, tokenizer, phase="pre")
    trainer.train()
    run_all(model, tokenizer, phase="post")
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency fallback
    wandb = None  # type: ignore[assignment]

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency fallback
    scipy_stats = None


DEFAULT_EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "In the beginning was the word and the word was with God",
    "def hello_world(): print('hello world')",
    "The population of Tokyo is approximately 14 million people",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure",
]

DEFAULT_PROBES = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time there was",
    "The meaning of life is",
    "def fibonacci(n):",
]

DEFAULT_RELATED_PAIRS = [
    ("dog", "wolf"),
    ("cat", "lion"),
    ("river", "ocean"),
    ("happy", "joyful"),
    ("cold", "frozen"),
    ("king", "queen"),
    ("python", "code"),
    ("car", "vehicle"),
    ("run", "sprint"),
]

DEFAULT_UNRELATED_PAIRS = [
    ("dog", "democracy"),
    ("cat", "algebra"),
    ("river", "justice"),
    ("happy", "concrete"),
    ("cold", "philosophy"),
    ("king", "equation"),
    ("python", "banana"),
    ("car", "silence"),
    ("run", "orange"),
]

DEFAULT_CLUSTER_WORDS = [
    "dog",
    "wolf",
    "puppy",
    "cat",
    "kitten",
    "tiger",
    "car",
    "truck",
    "vehicle",
    "banana",
    "apple",
    "fruit",
    "happy",
    "joy",
    "sadness",
    "code",
    "python",
    "programming",
    "silence",
    "peace",
    "war",
]

DEFAULT_CLUSTERS = {
    "animals": ["dog", "wolf", "puppy"],
    "felines": ["cat", "kitten", "tiger"],
    "vehicles": ["car", "truck", "vehicle"],
    "food": ["banana", "apple", "fruit"],
    "emotions": ["happy", "joy", "sadness"],
    "tech": ["code", "python", "programming"],
    "abstract": ["silence", "peace", "war"],
}

COMPONENT_PRIORITY = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

COMPONENT_DISPLAY_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "attn_other",
    "mlp_other",
    "other",
]


def _component_label(param_name: str) -> str:
    if "self_attn.q_proj" in param_name:
        return "q_proj"
    if "self_attn.k_proj" in param_name:
        return "k_proj"
    if "self_attn.v_proj" in param_name:
        return "v_proj"
    if "self_attn.o_proj" in param_name:
        return "o_proj"
    if "mlp.gate_proj" in param_name:
        return "gate_proj"
    if "mlp.up_proj" in param_name:
        return "up_proj"
    if "mlp.down_proj" in param_name:
        return "down_proj"
    if "self_attn" in param_name:
        return "attn_other"
    if "mlp" in param_name:
        return "mlp_other"
    return "other"


def _component_family(component: str) -> str:
    if component in {"q_proj", "k_proj", "v_proj", "o_proj", "attn_other"}:
        return "attention"
    if component in {"gate_proj", "up_proj", "down_proj", "mlp_other"}:
        return "mlp"
    return "other"


def _component_rank(component: str) -> int:
    try:
        return COMPONENT_DISPLAY_ORDER.index(component)
    except ValueError:
        return len(COMPONENT_DISPLAY_ORDER)


def _fmt_dim(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _layer_label(layer_idx: int, width: int) -> str:
    return f"L{layer_idx:0{width}d}"


def _transformer_block_label(layer_idx: int, width: int, dims: Dict[str, Any]) -> str:
    label = _layer_label(layer_idx, width)
    hidden = _fmt_dim(dims.get("hidden_dim"))
    q_out = _fmt_dim(dims.get("attn_q_out_dim"))
    k_out = _fmt_dim(dims.get("attn_k_out_dim"))
    v_out = _fmt_dim(dims.get("attn_v_out_dim"))
    o_in = _fmt_dim(dims.get("attn_o_in_dim"))
    mlp_mid = _fmt_dim(dims.get("mlp_intermediate_dim"))
    mlp_down = _fmt_dim(dims.get("mlp_down_in_dim"))
    return f"{label} h{hidden} attn({q_out}/{k_out}/{v_out}->{o_in}) mlp({mlp_mid}->{mlp_down})"


def _pad_token_id(tokenizer) -> int | None:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return None


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _phase_prefix(phase: str) -> str:
    normalized = str(phase).strip().lower().rstrip("/")
    if not normalized:
        raise ValueError("phase must be a non-empty string")
    return f"{normalized}/"


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _get_decoder_layers(model: torch.nn.Module) -> List[Tuple[int, str, torch.nn.Module]]:
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        (
            "model.decoder.layers",
            getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        ),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
        ("layers", getattr(model, "layers", None)),
    ]
    for base_name, layer_collection in candidates:
        if isinstance(layer_collection, (list, torch.nn.ModuleList)) and len(layer_collection) > 0:
            return [(idx, f"{base_name}.{idx}", layer) for idx, layer in enumerate(layer_collection)]

    fallback: List[Tuple[int, str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if name.endswith(tuple([f".{i}" for i in range(0, 1000)])):
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] in {"layers", "h"}:
                try:
                    idx = int(parts[-1])
                except ValueError:
                    continue
                fallback.append((idx, name, module))
    fallback.sort(key=lambda x: x[0])
    return fallback


def _infer_architecture(model: torch.nn.Module, layers: Sequence[Tuple[int, str, torch.nn.Module]]) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    hidden_size = None
    num_layers = None
    num_heads = None

    if config is not None:
        for key in ["hidden_size", "n_embd", "d_model", "model_dim"]:
            if hasattr(config, key):
                hidden_size = int(getattr(config, key))
                break
        for key in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if hasattr(config, key):
                num_layers = int(getattr(config, key))
                break
        for key in ["num_attention_heads", "n_head", "num_heads", "n_heads"]:
            if hasattr(config, key):
                num_heads = int(getattr(config, key))
                break

    if num_layers is None:
        num_layers = len(layers)

    return {
        "model_type": getattr(config, "model_type", "unknown") if config is not None else "unknown",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


def _safe_loss(model: torch.nn.Module, tokenized: Dict[str, torch.Tensor], labels: torch.Tensor) -> float:
    with torch.no_grad():
        out = model(**tokenized, labels=labels)
    value = float(out.loss.detach().float().item())
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return value


def _generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = _to_device(inputs, device)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    pad_id = _pad_token_id(tokenizer)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _backup_and_zero_params(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))


def _mannwhitney_greater(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    if scipy_stats is None or len(sample_a) < 3 or len(sample_b) < 3:
        return 1.0
    try:
        _, pval = scipy_stats.mannwhitneyu(sample_a, sample_b, alternative="greater")
        return float(pval)
    except Exception:
        return 1.0


def _extract_hidden_layers(model: torch.nn.Module, tokenizer, word: str, device: torch.device) -> List[np.ndarray]:
    inputs = tokenizer(word, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    return [state[0, -1, :].detach().float().cpu().numpy() for state in outputs.hidden_states]


def _extract_attentions(model: torch.nn.Module, tokenizer, word: str, device: torch.device) -> List[torch.Tensor]:
    inputs = tokenizer(word, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attention tensors")
    layers: List[torch.Tensor] = []
    for layer_attn in outputs.attentions:
        if layer_attn is None or layer_attn.ndim != 4:
            continue
        layers.append(layer_attn[0, :, -1, :].detach().float().cpu())
    if not layers:
        raise RuntimeError("No valid attention tensors were returned")
    return layers


def _compute_weight_fingerprint(
    layers: Sequence[Tuple[int, str, torch.nn.Module]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    layer_name_by_idx: Dict[int, str] = {idx: name for idx, name, _ in layers}
    layer_indices = sorted(layer_name_by_idx.keys())
    layer_label_width = max(2, len(str(layer_indices[-1]))) if layer_indices else 2
    layer_abs_means: Dict[int, List[float]] = {}
    layer_stds: Dict[int, List[float]] = {}
    layer_sparsity: Dict[int, List[float]] = {}
    component_abs_means: Dict[Tuple[int, str], List[float]] = {}
    component_stds: Dict[Tuple[int, str], List[float]] = {}
    component_sparsity: Dict[Tuple[int, str], List[float]] = {}
    component_counts: Dict[Tuple[int, str], int] = {}
    layer_input_dims: Dict[int, List[int]] = {}
    layer_col_energies: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    layer_dims_info: Dict[int, Dict[str, Any]] = {}

    def _layer_info(layer_idx: int) -> Dict[str, Any]:
        if layer_idx not in layer_dims_info:
            layer_dims_info[layer_idx] = {
                "hidden_dim": None,
                "attn_q_out_dim": None,
                "attn_k_out_dim": None,
                "attn_v_out_dim": None,
                "attn_o_in_dim": None,
                "mlp_intermediate_dim": None,
                "mlp_down_in_dim": None,
            }
        return layer_dims_info[layer_idx]

    for layer_idx, layer_name, layer_module in layers:
        for local_name, param in layer_module.named_parameters(recurse=True):
            if param.ndim < 2:
                continue
            full_name = f"{layer_name}.{local_name}"
            component = _component_label(local_name)
            data = param.detach().float()
            stats: Dict[str, Any] = {
                "layer": layer_idx,
                "layer_name": layer_name,
                "layer_label": _layer_label(layer_idx, layer_label_width),
                "name": full_name,
                "component": component,
                "family": _component_family(component),
                "shape": list(param.shape),
                "mean": float(data.mean().item()),
                "std": float(data.std().item()),
                "abs_mean": float(data.abs().mean().item()),
                "sparsity": float((data.abs() < 1e-6).float().mean().item()),
                "frobenius": float(data.norm().item()),
                "spectral_norm": None,
                "condition_number": None,
            }
            if data.ndim == 2 and max(data.shape) <= 2048:
                try:
                    singular = torch.linalg.svdvals(data)
                    stats["spectral_norm"] = float(singular[0].item())
                    stats["condition_number"] = float((singular[0] / (singular[-1] + 1e-9)).item())
                except Exception:
                    pass
            rows.append(stats)
            layer_abs_means.setdefault(layer_idx, []).append(stats["abs_mean"])
            layer_stds.setdefault(layer_idx, []).append(stats["std"])
            layer_sparsity.setdefault(layer_idx, []).append(stats["sparsity"])
            key = (layer_idx, component)
            component_abs_means.setdefault(key, []).append(stats["abs_mean"])
            component_stds.setdefault(key, []).append(stats["std"])
            component_sparsity.setdefault(key, []).append(stats["sparsity"])
            component_counts[key] = component_counts.get(key, 0) + 1

            layer_input_dims.setdefault(layer_idx, []).append(int(param.shape[1]))
            col_energy = data.pow(2).sum(dim=0).detach().cpu().numpy()
            layer_col_energies.setdefault(layer_idx, []).append((int(param.shape[1]), col_energy))

            info = _layer_info(layer_idx)
            if local_name.endswith("self_attn.q_proj.weight"):
                info["attn_q_out_dim"] = int(param.shape[0])
                info["hidden_dim"] = int(param.shape[1])
            elif local_name.endswith("self_attn.k_proj.weight"):
                info["attn_k_out_dim"] = int(param.shape[0])
            elif local_name.endswith("self_attn.v_proj.weight"):
                info["attn_v_out_dim"] = int(param.shape[0])
            elif local_name.endswith("self_attn.o_proj.weight"):
                info["attn_o_in_dim"] = int(param.shape[1])
            elif local_name.endswith("mlp.gate_proj.weight"):
                info["mlp_intermediate_dim"] = int(param.shape[0])
                if info["hidden_dim"] is None:
                    info["hidden_dim"] = int(param.shape[1])
            elif local_name.endswith("mlp.down_proj.weight"):
                info["mlp_down_in_dim"] = int(param.shape[1])

    layer_dimension_summary = []
    layer_dims_by_idx: Dict[int, Dict[str, Any]] = {}
    for layer_idx in sorted(layer_input_dims.keys()):
        info = _layer_info(layer_idx)
        if info["hidden_dim"] is None and layer_input_dims[layer_idx]:
            # Fallback: use most common input width in this layer.
            info["hidden_dim"] = int(Counter(layer_input_dims[layer_idx]).most_common(1)[0][0])
        hidden_dim = info["hidden_dim"]
        effective_nonzero = None
        effective_1pct = None
        if hidden_dim is not None and hidden_dim > 0:
            energies = [
                col_energy
                for dim, col_energy in layer_col_energies.get(layer_idx, [])
                if dim == hidden_dim and col_energy.shape[0] == hidden_dim
            ]
            if energies:
                combined = np.sum(np.stack(energies), axis=0)
                max_energy = float(np.max(combined)) if combined.size else 0.0
                effective_nonzero = int(np.sum(combined > 0.0))
                effective_1pct = int(np.sum(combined > (0.01 * max_energy))) if max_energy > 0 else 0

        dims_row = {
            "layer": layer_idx,
            "layer_name": layer_name_by_idx.get(layer_idx, f"layer.{layer_idx}"),
            "layer_label": _layer_label(layer_idx, layer_label_width),
            "hidden_dim": hidden_dim,
            "effective_hidden_dims_nonzero": effective_nonzero,
            "effective_hidden_dims_1pct": effective_1pct,
            "attn_q_out_dim": info["attn_q_out_dim"],
            "attn_k_out_dim": info["attn_k_out_dim"],
            "attn_v_out_dim": info["attn_v_out_dim"],
            "attn_o_in_dim": info["attn_o_in_dim"],
            "mlp_intermediate_dim": info["mlp_intermediate_dim"],
            "mlp_down_in_dim": info["mlp_down_in_dim"],
        }
        dims_row["transformer_block"] = _transformer_block_label(layer_idx, layer_label_width, dims_row)
        layer_dimension_summary.append(dims_row)
        layer_dims_by_idx[layer_idx] = dims_row

    layer_summary = []
    for layer_idx in sorted(layer_abs_means):
        dims_row = layer_dims_by_idx.get(
            layer_idx,
            {
                "hidden_dim": None,
                "attn_q_out_dim": None,
                "attn_k_out_dim": None,
                "attn_v_out_dim": None,
                "attn_o_in_dim": None,
                "mlp_intermediate_dim": None,
                "mlp_down_in_dim": None,
            },
        )
        layer_summary.append(
            {
                "layer": layer_idx,
                "layer_name": layer_name_by_idx.get(layer_idx, f"layer.{layer_idx}"),
                "layer_label": _layer_label(layer_idx, layer_label_width),
                "transformer_block": _transformer_block_label(layer_idx, layer_label_width, dims_row),
                "abs_mean": float(np.mean(layer_abs_means[layer_idx])),
                "std": float(np.mean(layer_stds[layer_idx])),
                "sparsity": float(np.mean(layer_sparsity[layer_idx])),
            }
        )

    layer_component_summary = []
    for (layer_idx, component), values in sorted(
        component_abs_means.items(), key=lambda x: (x[0][0], _component_rank(x[0][1]), x[0][1])
    ):
        layer_label = _layer_label(layer_idx, layer_label_width)
        component_rank = _component_rank(component)
        layer_component_summary.append(
            {
                "layer": layer_idx,
                "layer_name": layer_name_by_idx.get(layer_idx, f"layer.{layer_idx}"),
                "layer_label": layer_label,
                "layer_component": f"{layer_label}.{component}",
                "transformer_block": _transformer_block_label(
                    layer_idx, layer_label_width, layer_dims_by_idx.get(layer_idx, {})
                ),
                "component": component,
                "component_rank": component_rank,
                "component_ordered": f"{component_rank:02d}_{component}",
                "component_display": f"{component_rank + 1}. {component}",
                "family": _component_family(component),
                "abs_mean": float(np.mean(values)),
                "std": float(np.mean(component_stds[(layer_idx, component)])),
                "sparsity": float(np.mean(component_sparsity[(layer_idx, component)])),
                "tensor_count": int(component_counts[(layer_idx, component)]),
            }
        )

    effective_counts = [r["effective_hidden_dims_1pct"] for r in layer_dimension_summary if r["effective_hidden_dims_1pct"] is not None]

    return {
        "rows": rows,
        "layer_summary": layer_summary,
        "layer_component_summary": layer_component_summary,
        "layer_dimension_summary": layer_dimension_summary,
        "summary": {
            "tensor_count": len(rows),
            "avg_abs_mean": float(np.mean([r["abs_mean"] for r in rows])) if rows else 0.0,
            "avg_std": float(np.mean([r["std"] for r in rows])) if rows else 0.0,
            "avg_sparsity": float(np.mean([r["sparsity"] for r in rows])) if rows else 0.0,
            "avg_effective_hidden_dims_1pct": float(np.mean(effective_counts)) if effective_counts else 0.0,
        },
    }


def _compute_layer_sweep(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    layers: Sequence[Tuple[int, str, torch.nn.Module]],
    eval_texts: Sequence[str],
    max_new_tokens: int,
) -> Dict[str, Any]:
    tokenized = tokenizer(list(eval_texts), return_tensors="pt", padding=True, truncation=True)
    pad_id = _pad_token_id(tokenizer)
    labels = tokenized["input_ids"].clone()
    if pad_id is not None:
        labels[labels == pad_id] = -100
    tokenized = _to_device(tokenized, device)
    labels = labels.to(device)

    baseline_loss = _safe_loss(model, tokenized, labels)
    baseline_prompt = DEFAULT_PROBES[0]
    baseline_output = _generate_text(model, tokenizer, baseline_prompt, device, max_new_tokens=max_new_tokens)

    rows = []
    for layer_idx, layer_name, layer_module in layers:
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            backups = _backup_and_zero_params(layer_module.named_parameters(recurse=True), lambda _: True)
            loss = _safe_loss(model, tokenized, labels)
            sample = _generate_text(model, tokenizer, baseline_prompt, device, max_new_tokens=max_new_tokens)
            if math.isnan(loss):
                loss = 999.9999
            damage = loss - baseline_loss if not math.isnan(baseline_loss) else float("nan")
            rows.append(
                {
                    "layer": layer_idx,
                    "layer_name": layer_name,
                    "loss": float(loss),
                    "damage": float(damage),
                    "changed": bool(sample != baseline_output),
                    "sample": sample[:200],
                }
            )
        finally:
            _restore_backups(backups)

    ranked = sorted(rows, key=lambda x: x["damage"], reverse=True)
    return {
        "baseline_loss": baseline_loss,
        "rows": rows,
        "ranked": ranked,
        "summary": {
            "max_damage": float(max([r["damage"] for r in rows])) if rows else 0.0,
            "changed_layers": int(sum(1 for r in rows if r["changed"])),
        },
    }


def _discover_components(layers: Sequence[Tuple[int, str, torch.nn.Module]], max_components: int = 8) -> List[str]:
    counts: Counter[str] = Counter()
    for _, _, layer in layers:
        for name, param in layer.named_parameters(recurse=True):
            if param.ndim < 2 or not name.endswith(".weight"):
                continue
            counts[name[: -len(".weight")]] += 1

    selected = [name for name in COMPONENT_PRIORITY if name in counts]
    for name, _ in counts.most_common():
        if name not in selected:
            selected.append(name)
        if len(selected) >= max_components:
            break
    return selected[:max_components]


def _compute_component_ablation(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    layers: Sequence[Tuple[int, str, torch.nn.Module]],
    eval_texts: Sequence[str],
    probes: Sequence[str],
    max_new_tokens: int,
) -> Dict[str, Any]:
    components = _discover_components(layers)
    tokenized = tokenizer(list(eval_texts), return_tensors="pt", padding=True, truncation=True)
    pad_id = _pad_token_id(tokenizer)
    labels = tokenized["input_ids"].clone()
    if pad_id is not None:
        labels[labels == pad_id] = -100
    tokenized = _to_device(tokenized, device)
    labels = labels.to(device)

    baseline_loss = _safe_loss(model, tokenized, labels)
    baseline_outputs = {
        prompt: _generate_text(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        for prompt in probes
    }

    rows = []
    for component in components:
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for _, _, layer in layers:
                backups.extend(
                    _backup_and_zero_params(
                        layer.named_parameters(recurse=True),
                        lambda n, c=component: n.startswith(c),
                    )
                )

            loss = _safe_loss(model, tokenized, labels)
            if math.isnan(loss):
                loss = 999.9999
            damage = loss - baseline_loss if not math.isnan(baseline_loss) else float("nan")

            changed = 0
            outputs = {}
            for prompt in probes:
                out = _generate_text(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
                is_changed = out != baseline_outputs[prompt]
                outputs[prompt] = {"changed": bool(is_changed), "text": out[:200]}
                changed += int(is_changed)

            rows.append(
                {
                    "component": component,
                    "zeroed_tensors": len(backups),
                    "loss": float(loss),
                    "damage": float(damage),
                    "changed_prompts": changed,
                    "outputs": outputs,
                }
            )
        finally:
            _restore_backups(backups)

    rows.sort(key=lambda x: x["damage"], reverse=True)
    return {
        "baseline_loss": baseline_loss,
        "rows": rows,
        "summary": {
            "component_count": len(rows),
            "max_damage": float(max([r["damage"] for r in rows])) if rows else 0.0,
        },
    }


def _compute_silhouette(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    related_pairs: Sequence[Tuple[str, str]],
    unrelated_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    words = sorted({word for pair in list(related_pairs) + list(unrelated_pairs) for word in pair})
    cache = {word: _extract_hidden_layers(model, tokenizer, word, device) for word in words}
    num_layers = min(len(hidden) for hidden in cache.values())

    rows = []
    best_layer = 0
    best_sep = -1e9
    for layer in range(num_layers):
        rel_sims = [_cosine(cache[a][layer], cache[b][layer]) for a, b in related_pairs]
        unrel_sims = [_cosine(cache[a][layer], cache[b][layer]) for a, b in unrelated_pairs]
        rel_avg = float(np.mean(rel_sims))
        unrel_avg = float(np.mean(unrel_sims))
        separation = rel_avg - unrel_avg
        pval = _mannwhitney_greater(rel_sims, unrel_sims)

        if separation > best_sep:
            best_sep = separation
            best_layer = layer
        rows.append(
            {
                "layer": layer,
                "related_avg": rel_avg,
                "unrelated_avg": unrel_avg,
                "separation": separation,
                "p_value": pval,
                "significant": bool(pval < 0.05),
            }
        )

    ranked = sorted(rows, key=lambda x: x["separation"], reverse=True)
    return {
        "rows": rows,
        "ranked": ranked,
        "best_layer": best_layer,
        "best_separation": float(best_sep),
    }


def _centering_matrix(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    h = _centering_matrix(x.shape[0])
    k = x @ x.T
    l = y @ y.T
    hsic_kl = np.trace(h @ k @ h @ l) / ((x.shape[0] - 1) ** 2)
    hsic_kk = np.trace(h @ k @ h @ k) / ((x.shape[0] - 1) ** 2)
    hsic_ll = np.trace(h @ l @ h @ l) / ((x.shape[0] - 1) ** 2)
    denom = math.sqrt(max(hsic_kk * hsic_ll, 0.0))
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


def _compute_cka(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    cluster_words: Sequence[str],
    clusters: Dict[str, Sequence[str]],
    layer_stride: int | None,
) -> Dict[str, Any]:
    cache = {word: _extract_hidden_layers(model, tokenizer, word, device) for word in cluster_words}
    num_layers = min(len(hidden) for hidden in cache.values()) - 1
    stride = layer_stride if layer_stride and layer_stride > 0 else max(1, num_layers // 10)
    sampled_layers = sorted(set(list(range(0, num_layers + 1, stride)) + [num_layers]))

    rep_matrices = {}
    for layer in sampled_layers:
        rep_matrices[layer] = np.stack([cache[word][layer] for word in cluster_words])

    matrix_rows = []
    matrix = {}
    for layer_i in sampled_layers:
        matrix[layer_i] = {}
        for layer_j in sampled_layers:
            score = _linear_cka(rep_matrices[layer_i], rep_matrices[layer_j])
            matrix[layer_i][layer_j] = score
            matrix_rows.append({"layer_x": layer_i, "layer_y": layer_j, "cka": score})

    cluster_rows = []
    for layer in sampled_layers:
        reps = rep_matrices[layer]
        cluster_reps = {}
        for cluster_name, cluster_words_in_group in clusters.items():
            indices = [cluster_words.index(word) for word in cluster_words_in_group if word in cluster_words]
            if len(indices) >= 2:
                cluster_reps[cluster_name] = reps[indices]

        within = []
        for rep in cluster_reps.values():
            for i in range(rep.shape[0]):
                for j in range(i + 1, rep.shape[0]):
                    within.append(_cosine(rep[i], rep[j]))
        cross = []
        items = list(cluster_reps.values())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                for vec_i in items[i]:
                    for vec_j in items[j]:
                        cross.append(_cosine(vec_i, vec_j))

        within_avg = float(np.mean(within)) if within else 0.0
        cross_avg = float(np.mean(cross)) if cross else 0.0
        sep = within_avg - cross_avg
        cluster_rows.append(
            {
                "layer": layer,
                "within_cka": within_avg,
                "cross_cka": cross_avg,
                "separation": sep,
            }
        )

    evolution_rows = []
    transform_point = None
    for layer in sampled_layers:
        score = _linear_cka(rep_matrices[0], rep_matrices[layer])
        evolution_rows.append({"layer": layer, "cka_vs_input": score})
        if transform_point is None and score < 0.5:
            transform_point = layer

    return {
        "sampled_layers": sampled_layers,
        "matrix_rows": matrix_rows,
        "matrix": matrix,
        "cluster_rows": cluster_rows,
        "evolution_rows": evolution_rows,
        "transformation_point": transform_point,
    }


def _attention_entropy(attn_distribution: torch.Tensor) -> float:
    p = attn_distribution.clamp(min=1e-10)
    return float(-(p * p.log()).sum().item())


def _attention_similarity(attn_a: List[torch.Tensor], attn_b: List[torch.Tensor]) -> List[float]:
    layer_sims = []
    for layer_a, layer_b in zip(attn_a, attn_b):
        min_heads = min(layer_a.shape[0], layer_b.shape[0])
        min_len = min(layer_a.shape[1], layer_b.shape[1])
        a = layer_a[:min_heads, :min_len]
        b = layer_b[:min_heads, :min_len]
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        b = b / b.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        m = 0.5 * (a + b)
        kl_am = (a * (a / m.clamp(min=1e-10)).log()).sum(dim=-1)
        kl_bm = (b * (b / m.clamp(min=1e-10)).log()).sum(dim=-1)
        jsd = 0.5 * (kl_am + kl_bm)
        layer_sims.append(1.0 - float(jsd.mean().item()))
    return layer_sims


def _compute_attention_map(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    related_pairs: Sequence[Tuple[str, str]],
    unrelated_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    words = sorted({word for pair in list(related_pairs) + list(unrelated_pairs) for word in pair})
    cache = {word: _extract_attentions(model, tokenizer, word, device) for word in words}
    num_layers = min(len(v) for v in cache.values())

    entropy_rows = []
    for layer in range(num_layers):
        per_head_entropy = []
        per_head_norm_entropy = []
        for word in words:
            attn = cache[word][layer]
            seq_len = attn.shape[1]
            max_entropy = math.log(seq_len) if seq_len > 1 else 1.0
            for head in range(attn.shape[0]):
                entropy = _attention_entropy(attn[head])
                per_head_entropy.append(entropy)
                per_head_norm_entropy.append(entropy / max_entropy if max_entropy > 0 else 0.0)

        norm_mean = float(np.mean(per_head_norm_entropy)) if per_head_norm_entropy else 0.0
        entropy_rows.append(
            {
                "layer": layer,
                "mean_entropy": float(np.mean(per_head_entropy)) if per_head_entropy else 0.0,
                "normalized_entropy": norm_mean,
                "interpretation": (
                    "focused" if norm_mean < 0.5 else "moderate" if norm_mean < 0.8 else "diffuse"
                ),
            }
        )

    separation_rows = []
    for layer in range(num_layers):
        rel_scores = []
        for a, b in related_pairs:
            rel_scores.append(_attention_similarity(cache[a], cache[b])[layer])
        unrel_scores = []
        for a, b in unrelated_pairs:
            unrel_scores.append(_attention_similarity(cache[a], cache[b])[layer])

        rel_avg = float(np.mean(rel_scores))
        unrel_avg = float(np.mean(unrel_scores))
        sep = rel_avg - unrel_avg
        pval = _mannwhitney_greater(rel_scores, unrel_scores)
        verdict = (
            "structured"
            if sep > 0.02 and pval < 0.05
            else "weak"
            if sep > 0
            else "no_structure"
        )
        separation_rows.append(
            {
                "layer": layer,
                "related_sim": rel_avg,
                "unrelated_sim": unrel_avg,
                "separation": sep,
                "p_value": pval,
                "verdict": verdict,
            }
        )

    head_rows = []
    for layer in range(num_layers):
        head_patterns: List[List[np.ndarray]] = []
        max_heads = max(cache[word][layer].shape[0] for word in words)
        head_patterns = [[] for _ in range(max_heads)]
        for word in words:
            attn = cache[word][layer]
            for head in range(attn.shape[0]):
                head_patterns[head].append(attn[head].numpy())

        averaged = []
        for patterns in head_patterns:
            if not patterns:
                continue
            max_len = max(arr.shape[0] for arr in patterns)
            padded = [np.pad(arr, (0, max_len - arr.shape[0])) for arr in patterns]
            averaged.append(np.mean(padded, axis=0))

        distances = []
        for i in range(len(averaged)):
            for j in range(i + 1, len(averaged)):
                distances.append(1.0 - _cosine(averaged[i], averaged[j]))
        diversity = float(np.mean(distances)) if distances else 0.0
        head_rows.append(
            {
                "layer": layer,
                "head_diversity": diversity,
                "interpretation": (
                    "specialized" if diversity > 0.3 else "moderate" if diversity > 0.1 else "redundant"
                ),
            }
        )

    return {
        "entropy_rows": entropy_rows,
        "separation_rows": separation_rows,
        "head_rows": head_rows,
        "summary": {
            "structured_layers": int(sum(1 for row in separation_rows if row["verdict"] == "structured")),
            "specialized_layers": int(sum(1 for row in head_rows if row["interpretation"] == "specialized")),
            "focused_layers": int(sum(1 for row in entropy_rows if row["interpretation"] == "focused")),
        },
    }


def _try_plot_line(table: wandb.Table, x: str, y: str, title: str):
    if wandb is None:
        return None
    try:
        return wandb.plot.line(table, x, y, title=title)
    except Exception:
        return None


def _try_plot_bar(table: wandb.Table, x: str, y: str, title: str):
    if wandb is None:
        return None
    try:
        return wandb.plot.bar(table, x, y, title=title)
    except Exception:
        return None


def _try_plot_heatmap(table: wandb.Table, x: str, y: str, value: str, title: str):
    if wandb is None:
        return None
    try:
        return wandb.plot.heatmap(table, x, y, value, title=title)
    except Exception:
        return None


def run_all(
    model: torch.nn.Module,
    tokenizer,
    phase: str,
    *,
    eval_texts: Sequence[str] = DEFAULT_EVAL_TEXTS,
    probes: Sequence[str] = DEFAULT_PROBES,
    related_pairs: Sequence[Tuple[str, str]] = DEFAULT_RELATED_PAIRS,
    unrelated_pairs: Sequence[Tuple[str, str]] = DEFAULT_UNRELATED_PAIRS,
    cluster_words: Sequence[str] = DEFAULT_CLUSTER_WORDS,
    clusters: Dict[str, Sequence[str]] = DEFAULT_CLUSTERS,
    max_new_tokens: int = 32,
    layer_stride: int | None = None,
) -> Dict[str, Any]:
    """Run all six BoL scans and log to an active W&B run.

    Requirements:
    - `model` and `tokenizer` are already loaded.
    - `wandb.run` is active if logging is desired.
    """

    prefix = _phase_prefix(phase)
    device = _model_device(model)
    layers = _get_decoder_layers(model)
    architecture = _infer_architecture(model, layers)

    was_training = model.training
    model.eval()

    results: Dict[str, Any] = {
        "phase": phase,
        "prefix": prefix,
        "architecture": architecture,
        "errors": {},
    }

    try:
        results["weight_fingerprint"] = _compute_weight_fingerprint(layers)
    except Exception as exc:  # pragma: no cover - runtime safety
        results["errors"]["weight_fingerprint"] = str(exc)

    try:
        results["layer_sweep"] = _compute_layer_sweep(
            model=model,
            tokenizer=tokenizer,
            device=device,
            layers=layers,
            eval_texts=eval_texts,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        results["errors"]["layer_sweep"] = str(exc)

    try:
        results["component_ablation"] = _compute_component_ablation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            layers=layers,
            eval_texts=eval_texts,
            probes=probes,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        results["errors"]["component_ablation"] = str(exc)

    try:
        results["silhouette"] = _compute_silhouette(
            model=model,
            tokenizer=tokenizer,
            device=device,
            related_pairs=related_pairs,
            unrelated_pairs=unrelated_pairs,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        results["errors"]["silhouette"] = str(exc)

    try:
        results["cka"] = _compute_cka(
            model=model,
            tokenizer=tokenizer,
            device=device,
            cluster_words=cluster_words,
            clusters=clusters,
            layer_stride=layer_stride,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        results["errors"]["cka"] = str(exc)

    try:
        results["attention_map"] = _compute_attention_map(
            model=model,
            tokenizer=tokenizer,
            device=device,
            related_pairs=related_pairs,
            unrelated_pairs=unrelated_pairs,
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        results["errors"]["attention_map"] = str(exc)

    if was_training:
        model.train()

    if wandb is None:
        results["errors"]["wandb"] = "wandb is not installed; logging skipped."
        return results

    if wandb.run is None:
        return results

    log_payload: Dict[str, Any] = {
        f"{prefix}architecture/num_layers": architecture["num_layers"],
    }
    if architecture["hidden_size"] is not None:
        log_payload[f"{prefix}architecture/hidden_size"] = architecture["hidden_size"]
    if architecture["num_heads"] is not None:
        log_payload[f"{prefix}architecture/num_heads"] = architecture["num_heads"]

    for scan_name, error in results["errors"].items():
        log_payload[f"{prefix}errors/{scan_name}"] = error

    if "weight_fingerprint" in results:
        fp = results["weight_fingerprint"]
        fp_table = wandb.Table(
            columns=[
                "layer",
                "layer_label",
                "layer_name",
                "name",
                "component",
                "family",
                "mean",
                "std",
                "abs_mean",
                "sparsity",
                "frobenius",
                "spectral_norm",
            ],
            data=[
                [
                    row["layer"],
                    row["layer_label"],
                    row["layer_name"],
                    row["name"],
                    row["component"],
                    row["family"],
                    row["mean"],
                    row["std"],
                    row["abs_mean"],
                    row["sparsity"],
                    row["frobenius"],
                    row["spectral_norm"],
                ]
                for row in fp["rows"]
            ],
        )
        fp_layer_table = wandb.Table(
            columns=["layer", "layer_label", "layer_name", "transformer_block", "abs_mean", "std", "sparsity"],
            data=[
                [
                    row["layer"],
                    row["layer_label"],
                    row["layer_name"],
                    row["transformer_block"],
                    row["abs_mean"],
                    row["std"],
                    row["sparsity"],
                ]
                for row in fp["layer_summary"]
            ],
        )
        fp_component_table = wandb.Table(
            columns=[
                "layer",
                "layer_label",
                "layer_name",
                "layer_component",
                "transformer_block",
                "component",
                "component_rank",
                "component_ordered",
                "component_display",
                "family",
                "abs_mean",
                "std",
                "sparsity",
                "tensor_count",
            ],
            data=[
                [
                    row["layer"],
                    row["layer_label"],
                    row["layer_name"],
                    row["layer_component"],
                    row["transformer_block"],
                    row["component"],
                    row["component_rank"],
                    row["component_ordered"],
                    row["component_display"],
                    row["family"],
                    row["abs_mean"],
                    row["std"],
                    row["sparsity"],
                    row["tensor_count"],
                ]
                for row in fp["layer_component_summary"]
            ],
        )
        fp_dim_table = wandb.Table(
            columns=[
                "layer",
                "layer_label",
                "layer_name",
                "transformer_block",
                "hidden_dim",
                "effective_hidden_dims_nonzero",
                "effective_hidden_dims_1pct",
                "attn_q_out_dim",
                "attn_k_out_dim",
                "attn_v_out_dim",
                "attn_o_in_dim",
                "mlp_intermediate_dim",
                "mlp_down_in_dim",
            ],
            data=[
                [
                    row["layer"],
                    row["layer_label"],
                    row["layer_name"],
                    row["transformer_block"],
                    row["hidden_dim"],
                    row["effective_hidden_dims_nonzero"],
                    row["effective_hidden_dims_1pct"],
                    row["attn_q_out_dim"],
                    row["attn_k_out_dim"],
                    row["attn_v_out_dim"],
                    row["attn_o_in_dim"],
                    row["mlp_intermediate_dim"],
                    row["mlp_down_in_dim"],
                ]
                for row in fp["layer_dimension_summary"]
            ],
        )
        log_payload[f"{prefix}fingerprint/table"] = fp_table
        log_payload[f"{prefix}fingerprint/layer_table"] = fp_layer_table
        log_payload[f"{prefix}fingerprint/component_layer_table"] = fp_component_table
        log_payload[f"{prefix}fingerprint/layer_dimension_table"] = fp_dim_table
        log_payload[f"{prefix}fingerprint/tensor_count"] = fp["summary"]["tensor_count"]
        log_payload[f"{prefix}fingerprint/avg_abs_mean"] = fp["summary"]["avg_abs_mean"]
        log_payload[f"{prefix}fingerprint/avg_std"] = fp["summary"]["avg_std"]
        log_payload[f"{prefix}fingerprint/avg_sparsity"] = fp["summary"]["avg_sparsity"]
        log_payload[f"{prefix}fingerprint/avg_effective_hidden_dims_1pct"] = fp["summary"]["avg_effective_hidden_dims_1pct"]
        line_plot = _try_plot_line(fp_layer_table, "layer_label", "std", "Fingerprint Std by Layer")
        if line_plot is not None:
            log_payload[f"{prefix}fingerprint/std_curve"] = line_plot
        numeric_line_plot = _try_plot_line(fp_layer_table, "layer", "std", "Fingerprint Std by Layer (Index)")
        if numeric_line_plot is not None:
            log_payload[f"{prefix}fingerprint/std_curve_numeric"] = numeric_line_plot
        transformer_std_plot = _try_plot_bar(
            fp_layer_table,
            "transformer_block",
            "std",
            "Fingerprint Std by Transformer Block",
        )
        if transformer_std_plot is not None:
            log_payload[f"{prefix}fingerprint/std_by_transformer_block"] = transformer_std_plot
        component_std_heatmap = _try_plot_heatmap(
            fp_component_table,
            "layer",
            "component_display",
            "std",
            "Fingerprint Std by Layer + Component",
        )
        if component_std_heatmap is not None:
            log_payload[f"{prefix}fingerprint/component_std_heatmap"] = component_std_heatmap
        component_abs_heatmap = _try_plot_heatmap(
            fp_component_table,
            "layer",
            "component_display",
            "abs_mean",
            "Fingerprint Abs Mean by Layer + Component",
        )
        if component_abs_heatmap is not None:
            log_payload[f"{prefix}fingerprint/component_abs_mean_heatmap"] = component_abs_heatmap
        component_std_bar = _try_plot_bar(
            fp_component_table,
            "layer_component",
            "std",
            "Fingerprint Std by Transformer Component",
        )
        if component_std_bar is not None:
            log_payload[f"{prefix}fingerprint/component_std_by_layer_component"] = component_std_bar
        effective_dim_table = wandb.Table(
            columns=["layer", "layer_label", "effective_hidden_dims_1pct"],
            data=[
                [row["layer"], row["layer_label"], row["effective_hidden_dims_1pct"]]
                for row in fp["layer_dimension_summary"]
                if row["effective_hidden_dims_1pct"] is not None
            ],
        )
        if len(effective_dim_table.data) > 0:
            effective_line = _try_plot_line(
                effective_dim_table,
                "layer",
                "effective_hidden_dims_1pct",
                "Effective Hidden Dims (>1% Column Energy)",
            )
            if effective_line is not None:
                log_payload[f"{prefix}fingerprint/effective_hidden_dims_curve"] = effective_line
            effective_labeled_line = _try_plot_line(
                effective_dim_table,
                "layer_label",
                "effective_hidden_dims_1pct",
                "Effective Hidden Dims by Layer (Labeled)",
            )
            if effective_labeled_line is not None:
                log_payload[f"{prefix}fingerprint/effective_hidden_dims_curve_labeled"] = effective_labeled_line

    if "layer_sweep" in results:
        sweep = results["layer_sweep"]
        sweep_table = wandb.Table(
            columns=["layer", "loss", "damage", "changed"],
            data=[[row["layer"], row["loss"], row["damage"], int(row["changed"])] for row in sweep["rows"]],
        )
        log_payload[f"{prefix}layer_sweep/table"] = sweep_table
        log_payload[f"{prefix}layer_sweep/baseline_loss"] = sweep["baseline_loss"]
        log_payload[f"{prefix}layer_sweep/max_damage"] = sweep["summary"]["max_damage"]
        log_payload[f"{prefix}layer_sweep/changed_layers"] = sweep["summary"]["changed_layers"]
        line_plot = _try_plot_line(sweep_table, "layer", "damage", "Layer Sweep Damage")
        if line_plot is not None:
            log_payload[f"{prefix}layer_sweep/damage_curve"] = line_plot

    if "component_ablation" in results:
        comp = results["component_ablation"]
        comp_table = wandb.Table(
            columns=["component", "zeroed_tensors", "loss", "damage", "changed_prompts"],
            data=[
                [
                    row["component"],
                    row["zeroed_tensors"],
                    row["loss"],
                    row["damage"],
                    row["changed_prompts"],
                ]
                for row in comp["rows"]
            ],
        )
        log_payload[f"{prefix}component_ablation/table"] = comp_table
        log_payload[f"{prefix}component_ablation/baseline_loss"] = comp["baseline_loss"]
        log_payload[f"{prefix}component_ablation/max_damage"] = comp["summary"]["max_damage"]
        bar_plot = _try_plot_bar(comp_table, "component", "damage", "Component Ablation Damage")
        if bar_plot is not None:
            log_payload[f"{prefix}component_ablation/damage_bar"] = bar_plot

    if "silhouette" in results:
        sil = results["silhouette"]
        sil_table = wandb.Table(
            columns=["layer", "related_avg", "unrelated_avg", "separation", "p_value", "significant"],
            data=[
                [
                    row["layer"],
                    row["related_avg"],
                    row["unrelated_avg"],
                    row["separation"],
                    row["p_value"],
                    int(row["significant"]),
                ]
                for row in sil["rows"]
            ],
        )
        log_payload[f"{prefix}silhouette/table"] = sil_table
        log_payload[f"{prefix}silhouette/best_layer"] = sil["best_layer"]
        log_payload[f"{prefix}silhouette/best_separation"] = sil["best_separation"]
        line_plot = _try_plot_line(sil_table, "layer", "separation", "Silhouette Separation by Layer")
        if line_plot is not None:
            log_payload[f"{prefix}silhouette/separation_curve"] = line_plot

    if "cka" in results:
        cka = results["cka"]
        cka_table = wandb.Table(
            columns=["layer_x", "layer_y", "cka"],
            data=[[row["layer_x"], row["layer_y"], row["cka"]] for row in cka["matrix_rows"]],
        )
        cka_cluster_table = wandb.Table(
            columns=["layer", "within_cka", "cross_cka", "separation"],
            data=[
                [row["layer"], row["within_cka"], row["cross_cka"], row["separation"]]
                for row in cka["cluster_rows"]
            ],
        )
        evolution_table = wandb.Table(
            columns=["layer", "cka_vs_input"],
            data=[[row["layer"], row["cka_vs_input"]] for row in cka["evolution_rows"]],
        )
        log_payload[f"{prefix}cka/matrix_table"] = cka_table
        log_payload[f"{prefix}cka/cluster_table"] = cka_cluster_table
        log_payload[f"{prefix}cka/evolution_table"] = evolution_table
        if cka["transformation_point"] is not None:
            log_payload[f"{prefix}cka/transformation_point"] = cka["transformation_point"]
        heatmap = _try_plot_heatmap(cka_table, "layer_x", "layer_y", "cka", "CKA Layer Heatmap")
        if heatmap is not None:
            log_payload[f"{prefix}cka/heatmap"] = heatmap
        line_plot = _try_plot_line(evolution_table, "layer", "cka_vs_input", "CKA vs Input Layer")
        if line_plot is not None:
            log_payload[f"{prefix}cka/evolution_curve"] = line_plot

    if "attention_map" in results:
        attn = results["attention_map"]
        entropy_table = wandb.Table(
            columns=["layer", "mean_entropy", "normalized_entropy", "interpretation"],
            data=[
                [row["layer"], row["mean_entropy"], row["normalized_entropy"], row["interpretation"]]
                for row in attn["entropy_rows"]
            ],
        )
        separation_table = wandb.Table(
            columns=["layer", "related_sim", "unrelated_sim", "separation", "p_value", "verdict"],
            data=[
                [
                    row["layer"],
                    row["related_sim"],
                    row["unrelated_sim"],
                    row["separation"],
                    row["p_value"],
                    row["verdict"],
                ]
                for row in attn["separation_rows"]
            ],
        )
        head_table = wandb.Table(
            columns=["layer", "head_diversity", "interpretation"],
            data=[[row["layer"], row["head_diversity"], row["interpretation"]] for row in attn["head_rows"]],
        )
        log_payload[f"{prefix}attention_map/entropy_table"] = entropy_table
        log_payload[f"{prefix}attention_map/separation_table"] = separation_table
        log_payload[f"{prefix}attention_map/head_table"] = head_table
        log_payload[f"{prefix}attention_map/structured_layers"] = attn["summary"]["structured_layers"]
        log_payload[f"{prefix}attention_map/specialized_layers"] = attn["summary"]["specialized_layers"]
        log_payload[f"{prefix}attention_map/focused_layers"] = attn["summary"]["focused_layers"]
        entropy_curve = _try_plot_line(
            entropy_table,
            "layer",
            "normalized_entropy",
            "Attention Normalized Entropy by Layer",
        )
        if entropy_curve is not None:
            log_payload[f"{prefix}attention_map/entropy_curve"] = entropy_curve
        separation_curve = _try_plot_line(
            separation_table,
            "layer",
            "separation",
            "Attention Semantic Separation by Layer",
        )
        if separation_curve is not None:
            log_payload[f"{prefix}attention_map/separation_curve"] = separation_curve
        head_curve = _try_plot_line(head_table, "layer", "head_diversity", "Attention Head Diversity by Layer")
        if head_curve is not None:
            log_payload[f"{prefix}attention_map/head_diversity_curve"] = head_curve

    wandb.log(log_payload)
    return results


__all__ = ["run_all"]
