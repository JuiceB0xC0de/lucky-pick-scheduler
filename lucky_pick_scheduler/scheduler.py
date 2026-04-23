"""Auto-configured optimizer + LR scheduler builder for transformer finetuning."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import get_scheduler

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - optional distributed runtime
    dist = None  # type: ignore[assignment]


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
    prefer_muon: bool = True
    muon_lr_multiplier: float = 1.0
    muon_momentum: float = 0.95
    muon_weight_decay: float | None = None
    min_muon_ndim: int = 2


@dataclass
class SchedulerBuildReport:
    optimizer_name: str
    scheduler_name: str
    used_muon: bool
    num_training_steps: int
    num_warmup_steps: int
    model_profile: ModelProfile
    role_param_counts: Dict[str, int]
    role_numel_counts: Dict[str, int]
    muon_param_count: int
    adam_param_count: int
    muon_numel: int
    adam_numel: int
    skipped_param_count: int
    fallback_reason: str | None = None

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


def _is_distributed_ready() -> bool:
    if dist is None:
        return False
    try:
        return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
    except Exception:
        return False


def _build_muon_optimizer(
    muon_params: List[torch.nn.Parameter],
    adam_params: List[torch.nn.Parameter],
    cfg: AutoSchedulerConfig,
):
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam  # type: ignore[import-not-found]

    muon_lr = cfg.learning_rate * cfg.muon_lr_multiplier
    muon_wd = cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay
    groups = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr,
                "momentum": cfg.muon_momentum,
                "weight_decay": muon_wd,
            }
        )
    if adam_params:
        groups.append(
            {
                "params": adam_params,
                "use_muon": False,
                "lr": cfg.learning_rate,
                "betas": cfg.adam_betas,
                "eps": cfg.adam_eps,
                "weight_decay": cfg.weight_decay,
            }
        )
    if not groups:
        raise ValueError("No trainable parameters found for Muon/Adam setup.")

    if _is_distributed_ready():
        return MuonWithAuxAdam(groups), "MuonWithAuxAdam"
    return SingleDeviceMuonWithAuxAdam(groups), "SingleDeviceMuonWithAuxAdam"


def _build_adamw_optimizer(
    muon_like_params: List[torch.nn.Parameter],
    adam_params: List[torch.nn.Parameter],
    cfg: AutoSchedulerConfig,
):
    muon_lr = cfg.learning_rate * cfg.muon_lr_multiplier
    param_groups = []
    if muon_like_params:
        param_groups.append({"params": muon_like_params, "lr": muon_lr, "weight_decay": cfg.weight_decay})
    if adam_params:
        param_groups.append({"params": adam_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
    if not param_groups:
        raise ValueError("No trainable parameters found for AdamW setup.")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    return optimizer, "AdamW"


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
    """Build optimizer + lr scheduler from a loaded model using role-based auto grouping."""
    cfg = config or AutoSchedulerConfig()
    rows = classify_model_parameters(model)
    profile = infer_model_profile(model)

    muon_candidates: List[torch.nn.Parameter] = []
    adam_candidates: List[torch.nn.Parameter] = []
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

        muon_eligible = (
            cfg.prefer_muon
            and row["ndim"] >= cfg.min_muon_ndim
            and role in {"attention", "mlp", "other_matrix"}
        )
        if muon_eligible:
            muon_candidates.append(param)
        else:
            adam_candidates.append(param)

    used_muon = False
    fallback_reason = None
    try:
        if cfg.prefer_muon and muon_candidates:
            optimizer, optimizer_name = _build_muon_optimizer(muon_candidates, adam_candidates, cfg)
            used_muon = True
        else:
            optimizer, optimizer_name = _build_adamw_optimizer(muon_candidates, adam_candidates, cfg)
    except Exception as exc:
        fallback_reason = str(exc)
        optimizer, optimizer_name = _build_adamw_optimizer(muon_candidates, adam_candidates, cfg)

    warmup_steps = _warmup_steps(cfg, num_training_steps)
    scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    muon_numel = sum(int(p.numel()) for p in muon_candidates)
    adam_numel = sum(int(p.numel()) for p in adam_candidates)
    report = SchedulerBuildReport(
        optimizer_name=optimizer_name,
        scheduler_name=cfg.lr_scheduler_type,
        used_muon=used_muon,
        num_training_steps=int(num_training_steps),
        num_warmup_steps=warmup_steps,
        model_profile=profile,
        role_param_counts=role_param_counts,
        role_numel_counts=role_numel_counts,
        muon_param_count=len(muon_candidates),
        adam_param_count=len(adam_candidates),
        muon_numel=muon_numel,
        adam_numel=adam_numel,
        skipped_param_count=skipped_param_count,
        fallback_reason=fallback_reason,
    )
    return optimizer, scheduler, report

