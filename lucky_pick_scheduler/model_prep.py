"""Model preparation helpers for broad checkpoint compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from .compat import allow_quantized_training_in_trainer


@dataclass
class ModelPrepConfig:
    auto_lora_for_quantized: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: Sequence[str] | None = None
    try_dequantize_if_available: bool = True
    allow_trainer_quantization_bypass: bool = True
    verbose: bool = True


@dataclass
class ModelPrepReport:
    quantized_detected: bool
    auto_lora_applied: bool
    dequantized: bool
    trainer_quantization_bypass_applied: bool
    lora_target_modules: List[str]
    notes: List[str]

    def to_dict(self):
        return {
            "quantized_detected": bool(self.quantized_detected),
            "auto_lora_applied": bool(self.auto_lora_applied),
            "dequantized": bool(self.dequantized),
            "trainer_quantization_bypass_applied": bool(self.trainer_quantization_bypass_applied),
            "lora_target_modules": list(self.lora_target_modules),
            "notes": list(self.notes),
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

    supported_leaf_names: set[str] = set()

    supported_types = [torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]
    try:
        from transformers.pytorch_utils import Conv1D as HFConv1D  # type: ignore

        supported_types.append(HFConv1D)
    except Exception:
        pass
    supported_tuple = tuple(supported_types)

    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]
        if isinstance(module, supported_tuple):
            supported_leaf_names.add(leaf)

    selected = [name for name in preferred if name in supported_leaf_names]
    if selected:
        return selected
    return sorted(supported_leaf_names)


def resolve_scheduler_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            if isinstance(base, torch.nn.Module):
                return base
        except Exception:
            pass
    return model


def _set_attr_best_effort(obj: object, name: str, value, notes: List[str]):
    try:
        setattr(obj, name, value)
    except Exception as exc:
        notes.append(f"setattr_failed:{type(obj).__name__}.{name}:{exc}")


def _clear_quantization_training_flags(model: torch.nn.Module, notes: List[str]) -> bool:
    touched = False
    for obj in (model, resolve_scheduler_model(model)):
        for name, value in (
            ("is_quantized", False),
            ("quantization_method", None),
            ("hf_quantizer", None),
        ):
            before = getattr(obj, name, None)
            _set_attr_best_effort(obj, name, value, notes)
            after = getattr(obj, name, None)
            if before is not after:
                touched = True
    return touched


def prepare_model_for_training(
    model: torch.nn.Module,
    config: ModelPrepConfig | None = None,
) -> tuple[torch.nn.Module, ModelPrepReport]:
    cfg = config or ModelPrepConfig()
    quantized = is_quantized_model(model)
    auto_lora_applied = False
    dequantized = False
    trainer_quantization_bypass_applied = False
    lora_target_modules: List[str] = []
    notes: List[str] = []

    if quantized and bool(cfg.auto_lora_for_quantized):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise RuntimeError(
                "Quantized model detected but PEFT is unavailable; install `peft`."
            ) from exc

        lora_target_modules = list(cfg.lora_target_modules or infer_lora_target_modules(model))
        if lora_target_modules:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(cfg.lora_r),
                lora_alpha=int(cfg.lora_alpha),
                lora_dropout=float(cfg.lora_dropout),
                bias=str(cfg.lora_bias),
                target_modules=lora_target_modules,
            )
            try:
                model = get_peft_model(model, lora_cfg)
                auto_lora_applied = True
                if cfg.verbose:
                    print(
                        "Auto-LoRA applied for quantized model. "
                        f"targets={lora_target_modules}"
                    )
                    if hasattr(model, "print_trainable_parameters"):
                        model.print_trainable_parameters()
            except Exception as exc:
                notes.append(f"auto_lora_failed:{exc}")
                if cfg.verbose:
                    print(f"Auto-LoRA injection failed for quantized model: {exc}")
        else:
            notes.append("no_supported_lora_target_modules")
            if cfg.verbose:
                print("No PEFT-supported target modules found for quantized model; skipping auto-LoRA.")

    if quantized and not auto_lora_applied and bool(cfg.try_dequantize_if_available) and hasattr(model, "dequantize"):
        try:
            maybe_model = model.dequantize()  # some APIs mutate in place and return None
            if isinstance(maybe_model, torch.nn.Module):
                model = maybe_model
            quantized = is_quantized_model(model)
            dequantized = not quantized
            if dequantized:
                notes.append("dequantized_model")
                if cfg.verbose:
                    print("Dequantized model for training compatibility.")
        except Exception as exc:
            notes.append(f"dequantize_failed:{exc}")
            if cfg.verbose:
                print(f"Dequantize attempt failed: {exc}")

    if quantized and not auto_lora_applied and bool(cfg.allow_trainer_quantization_bypass):
        patched = allow_quantized_training_in_trainer(verbose=cfg.verbose)
        trainer_quantization_bypass_applied = len(patched) > 0
        if trainer_quantization_bypass_applied:
            notes.append("trainer_quantization_validation_bypassed")
        if _clear_quantization_training_flags(model, notes):
            notes.append("quantization_flags_cleared_for_trainer")
        quantized = is_quantized_model(model)

    report = ModelPrepReport(
        quantized_detected=bool(quantized),
        auto_lora_applied=bool(auto_lora_applied),
        dequantized=bool(dequantized),
        trainer_quantization_bypass_applied=bool(trainer_quantization_bypass_applied),
        lora_target_modules=lora_target_modules,
        notes=notes,
    )
    return model, report
