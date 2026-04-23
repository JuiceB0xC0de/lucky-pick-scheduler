"""Model preparation helpers for broad checkpoint compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass
class ModelPrepConfig:
    auto_lora_for_quantized: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: Sequence[str] | None = None
    verbose: bool = True


@dataclass
class ModelPrepReport:
    quantized_detected: bool
    auto_lora_applied: bool
    lora_target_modules: List[str]

    def to_dict(self):
        return {
            "quantized_detected": bool(self.quantized_detected),
            "auto_lora_applied": bool(self.auto_lora_applied),
            "lora_target_modules": list(self.lora_target_modules),
        }


def is_quantized_model(model: torch.nn.Module) -> bool:
    return bool(
        getattr(model, "is_quantized", False)
        or getattr(model, "hf_quantizer", None) is not None
        or getattr(model, "quantization_method", None) is not None
    )


def infer_lora_target_modules(model: torch.nn.Module, preferred: Sequence[str] | None = None) -> List[str]:
    preferred = list(
        preferred
        or (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        )
    )

    linear_like_leaf_names: set[str] = set()
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        if isinstance(module, torch.nn.Linear):
            linear_like_leaf_names.add(leaf)
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            class_name_l = module.__class__.__name__.lower()
            if "norm" in class_name_l or "embed" in class_name_l:
                continue
            linear_like_leaf_names.add(leaf)

    selected = [name for name in preferred if name in linear_like_leaf_names]
    if selected:
        return selected
    return sorted(linear_like_leaf_names)


def resolve_scheduler_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            if isinstance(base, torch.nn.Module):
                return base
        except Exception:
            pass
    return model


def prepare_model_for_training(
    model: torch.nn.Module,
    config: ModelPrepConfig | None = None,
) -> tuple[torch.nn.Module, ModelPrepReport]:
    cfg = config or ModelPrepConfig()
    quantized = is_quantized_model(model)
    auto_lora_applied = False
    lora_target_modules: List[str] = []

    if quantized and bool(cfg.auto_lora_for_quantized):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError(
                "Quantized model detected but PEFT is unavailable; install `peft`."
            ) from exc

        lora_target_modules = list(cfg.lora_target_modules or infer_lora_target_modules(model))
        if not lora_target_modules:
            raise RuntimeError(
                "Quantized model detected but no linear target modules were found for LoRA adapters."
            )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg.lora_r),
            lora_alpha=int(cfg.lora_alpha),
            lora_dropout=float(cfg.lora_dropout),
            bias=str(cfg.lora_bias),
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        auto_lora_applied = True
        if cfg.verbose:
            print(
                "Auto-LoRA applied for quantized model. "
                f"targets={lora_target_modules}"
            )
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

    report = ModelPrepReport(
        quantized_detected=bool(quantized),
        auto_lora_applied=bool(auto_lora_applied),
        lora_target_modules=lora_target_modules,
    )
    return model, report

