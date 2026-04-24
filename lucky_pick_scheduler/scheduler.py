"""Auto-configured optimizer + LR scheduler builder for transformer finetuning.

AdamW only. Introspects the model, groups parameters by role (attention, MLP,
embedding, head, norm/bias), and builds a single AdamW optimizer + LR scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import torch
from transformers import get_scheduler


ROLE_ORDER = (
    "attention",
    "mlp",
    "embedding",
    "head",
    "norm_bias",
    "other_matrix",
    "other_scalar",
)

ATTN_TOKENS = ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo")
MLP_TOKENS = ("mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj", "c_fc", "c_proj")
EMBED_TOKENS = ("embed", "embeddings", "wte", "word_embeddings", "token_embeddings", "position_embeddings")
HEAD_TOKENS = ("lm_head", "classifier", "score", "output_projection", "final_logits_bias")
NORM_TOKENS = ("norm", "ln_", "layernorm", "rmsnorm", "bias")


@dataclass
class ModelProfile:
    model_type: str | None
    num_layers: int | None
    hidden_size: int | None
    intermediate_size: int | None
    num_attention_heads: int | None


@dataclass
class AutoSchedulerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int | None = None
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-10
    # Per-role LR multipliers. The matrix group (attention/mlp/other_matrix
    # parameters with ndim >= matrix_ndim) uses `matrix_lr_multiplier * learning_rate`,
    # everything else uses the base learning_rate. Defaults keep both groups at the
    # same LR — set matrix_lr_multiplier > 1.0 if you want matrix params trained hotter.
    matrix_lr_multiplier: float = 1.0
    matrix_ndim: int = 2
    # When True, norm/bias params (ndim < 2 or matched by NORM_TOKENS) get
    # weight_decay=0.0 instead of cfg.weight_decay. This is the standard AdamW
    # recipe for transformer training.
    no_decay_on_norm_bias: bool = True


@dataclass
class SchedulerBuildReport:
    optimizer_name: str
    scheduler_name: str
    num_training_steps: int
    num_warmup_steps: int
    model_profile: ModelProfile
    role_param_counts: Dict[str, int]
    role_numel_counts: Dict[str, int]
    matrix_param_count: int
    scalar_param_count: int
    no_decay_param_count: int
    matrix_numel: int
    scalar_numel: int
    no_decay_numel: int
    skipped_param_count: int

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["model_profile"] = asdict(self.model_profile)
        return payload


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        return model.module
    return model


def _config_value(config: Any, *keys: str) -> int | None:
    if config is None:
        return None
    for key in keys:
        if hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    return None
    return None


def _infer_num_layers_from_modules(model: torch.nn.Module) -> int | None:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
        getattr(model, "layers", None),
    ]
    for layer_collection in candidates:
        if isinstance(layer_collection, (list, torch.nn.ModuleList)) and len(layer_collection) > 0:
            return int(len(layer_collection))
    return None


def infer_model_profile(model: torch.nn.Module) -> ModelProfile:
    base_model = _unwrap_model(model)
    config = getattr(base_model, "config", None)
    num_layers = _config_value(config, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    if num_layers is None:
        num_layers = _infer_num_layers_from_modules(base_model)
    return ModelProfile(
        model_type=getattr(config, "model_type", None) if config is not None else None,
        num_layers=num_layers,
        hidden_size=_config_value(config, "hidden_size", "n_embd", "d_model", "model_dim"),
        intermediate_size=_config_value(config, "intermediate_size", "ffn_dim", "n_inner"),
        num_attention_heads=_config_value(config, "num_attention_heads", "n_head", "num_heads"),
    )


def _role_for_param(name: str, param: torch.nn.Parameter) -> str:
    name_l = name.lower()
    if any(token in name_l for token in EMBED_TOKENS):
        return "embedding"
    if any(token in name_l for token in HEAD_TOKENS):
        return "head"
    if param.ndim < 2 or any(token in name_l for token in NORM_TOKENS):
        return "norm_bias" if any(token in name_l for token in NORM_TOKENS) else "other_scalar"
    if any(token in name_l for token in ATTN_TOKENS):
        return "attention"
    if any(token in name_l for token in MLP_TOKENS):
        return "mlp"
    return "other_matrix"


def classify_model_parameters(model: torch.nn.Module) -> List[Dict[str, Any]]:
    base_model = _unwrap_model(model)
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for name, param in base_model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        role = _role_for_param(name, param)
        rows.append(
            {
                "name": name,
                "role": role,
                "shape": list(param.shape),
                "numel": int(param.numel()),
                "ndim": int(param.ndim),
                "requires_grad": bool(param.requires_grad),
                "param_ref": param,
            }
        )
    rows.sort(key=lambda row: (ROLE_ORDER.index(row["role"]) if row["role"] in ROLE_ORDER else 99, row["name"]))
    return rows


def _warmup_steps(cfg: AutoSchedulerConfig, total_steps: int) -> int:
    if total_steps <= 0:
        raise ValueError("num_training_steps must be > 0")
    if cfg.warmup_steps is not None:
        return max(0, min(int(cfg.warmup_steps), int(total_steps)))
    return max(0, min(int(round(total_steps * cfg.warmup_ratio)), int(total_steps)))


def build_scheduler_stack(
    model: torch.nn.Module,
    num_training_steps: int,
    config: AutoSchedulerConfig | None = None,
):
    """Build an AdamW optimizer + LR scheduler using role-based parameter groups.

    Parameters are split into three groups:
      - matrix params (attention/mlp/other_matrix with ndim >= cfg.matrix_ndim):
        lr = cfg.learning_rate * cfg.matrix_lr_multiplier, weight_decay = cfg.weight_decay
      - scalar/misc params (embedding, head, other_scalar, or ndim < matrix_ndim
        not matching norm/bias): lr = cfg.learning_rate, weight_decay = cfg.weight_decay
      - no-decay params (norm_bias role, when cfg.no_decay_on_norm_bias is True):
        lr = cfg.learning_rate, weight_decay = 0.0
    """
    cfg = config or AutoSchedulerConfig()
    rows = classify_model_parameters(model)
    profile = infer_model_profile(model)

    matrix_params: List[torch.nn.Parameter] = []
    scalar_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    role_param_counts: Dict[str, int] = {role: 0 for role in ROLE_ORDER}
    role_numel_counts: Dict[str, int] = {role: 0 for role in ROLE_ORDER}
    skipped_param_count = 0

    for row in rows:
        role = row["role"]
        param = row["param_ref"]
        if role not in role_param_counts:
            role_param_counts[role] = 0
            role_numel_counts[role] = 0
        role_param_counts[role] += 1
        role_numel_counts[role] += row["numel"]
        if not row["requires_grad"]:
            skipped_param_count += 1
            continue

        is_no_decay = bool(cfg.no_decay_on_norm_bias) and role == "norm_bias"
        is_matrix = (
            row["ndim"] >= cfg.matrix_ndim
            and role in {"attention", "mlp", "other_matrix"}
        )

        if is_no_decay:
            no_decay_params.append(param)
        elif is_matrix:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    matrix_lr = cfg.learning_rate * cfg.matrix_lr_multiplier

    param_groups = []
    if matrix_params:
        param_groups.append({"params": matrix_params, "lr": matrix_lr, "weight_decay": cfg.weight_decay})
    if scalar_params:
        param_groups.append({"params": scalar_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": cfg.learning_rate, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("No trainable parameters found for AdamW setup.")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    warmup_steps = _warmup_steps(cfg, num_training_steps)
    scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    matrix_numel = sum(int(p.numel()) for p in matrix_params)
    scalar_numel = sum(int(p.numel()) for p in scalar_params)
    no_decay_numel = sum(int(p.numel()) for p in no_decay_params)
    report = SchedulerBuildReport(
        optimizer_name="AdamW",
        scheduler_name=cfg.lr_scheduler_type,
        num_training_steps=int(num_training_steps),
        num_warmup_steps=warmup_steps,
        model_profile=profile,
        role_param_counts=role_param_counts,
        role_numel_counts=role_numel_counts,
        matrix_param_count=len(matrix_params),
        scalar_param_count=len(scalar_params),
        no_decay_param_count=len(no_decay_params),
        matrix_numel=matrix_numel,
        scalar_numel=scalar_numel,
        no_decay_numel=no_decay_numel,
        skipped_param_count=skipped_param_count,
    )
    return optimizer, scheduler, report