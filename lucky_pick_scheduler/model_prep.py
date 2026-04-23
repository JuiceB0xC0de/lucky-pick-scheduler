"""Model preparation helpers for broad checkpoint compatibility."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# BitNet detection helpers
# ---------------------------------------------------------------------------

def _get_quantization_method_name(model: torch.nn.Module) -> str | None:
    """Return the string name of the quantization method if present."""
    qm = getattr(model, "quantization_method", None)
    if qm is None:
        return None
    # Can be a string or a QuantizationMethod enum
    name = str(qm).upper()
    # Normalise: transformers uses QuantizationMethod.BITNET which str()'s to 'bitnet'
    return name.split(".")[-1]  # e.g. 'BITNET', 'BITS_AND_BYTES', etc.


def is_bitnet_model(model: torch.nn.Module) -> bool:
    """Return True if the model is a native BitNet quantized checkpoint.

    This is the case when transformers sets quantization_method = BITNET
    (i.e. the model was loaded from the packed ternary checkpoint, not the
    pre-quantized bfloat16 revision).  Training this directly is blocked by
    transformers' validate_quantization_for_training().
    """
    method = _get_quantization_method_name(model)
    if method is not None and "BITNET" in method:
        return True

    # Fallback: scan for BitLinear layers (class name contains 'bitlinear' or
    # 'bitnet') — present when loaded via the default revision.
    for _, module in model.named_modules():
        cls = module.__class__.__name__.lower()
        if "bitlinear" in cls or ("bitnet" in cls and "linear" in cls):
            return True

    return False


def is_quantized_model(model: torch.nn.Module) -> bool:
    return bool(
        getattr(model, "is_quantized", False)
        or getattr(model, "hf_quantizer", None) is not None
        or getattr(model, "quantization_method", None) is not None
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    bitnet_detected: bool
    bitnet_path: Optional[str]   # hint for the user on how to reload
    auto_lora_applied: bool
    lora_target_modules: List[str]

    def to_dict(self):
        return {
            "quantized_detected": bool(self.quantized_detected),
            "bitnet_detected": bool(self.bitnet_detected),
            "bitnet_path": self.bitnet_path,
            "auto_lora_applied": bool(self.auto_lora_applied),
            "lora_target_modules": list(self.lora_target_modules),
        }


# ---------------------------------------------------------------------------
# LoRA target inference
# ---------------------------------------------------------------------------

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
        cls = module.__class__.__name__.lower()
        # Skip BitLinear — these are not trainable via LoRA in the native
        # BitNet checkpoint.  Only real nn.Linear layers can be targeted.
        if "bitlinear" in cls or ("bitnet" in cls and "linear" in cls):
            continue
        if isinstance(module, torch.nn.Linear):
            linear_like_leaf_names.add(leaf)
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            if "norm" in cls or "embed" in cls:
                continue
            linear_like_leaf_names.add(leaf)

    selected = [name for name in preferred if name in linear_like_leaf_names]
    if selected:
        return selected
    return sorted(linear_like_leaf_names)


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------

def resolve_scheduler_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            base = model.get_base_model()
            if isinstance(base, torch.nn.Module):
                return base
        except Exception:
            pass
    return model


# ---------------------------------------------------------------------------
# BitNet reload helper
# ---------------------------------------------------------------------------

_BITNET_PREQUANTIZED_HINT = """
  The model you loaded is a native BitNet checkpoint (packed ternary weights).
  transformers blocks training on this revision.

  For Falcon-E and Microsoft BitNet series models, load the pre-quantized
  bfloat16 revision instead and use onebitllms to inject trainable BitLinear
  layers:

      pip install onebitllms

      from transformers import AutoModelForCausalLM, AutoTokenizer
      from onebitllms import replace_linear_with_bitnet_linear

      tokenizer = AutoTokenizer.from_pretrained(model_id, revision="prequantized")
      model = AutoModelForCausalLM.from_pretrained(
          model_id,
          revision="prequantized",
          torch_dtype=torch.bfloat16,
          device_map="auto",
      )
      model = replace_linear_with_bitnet_linear(model)
      # Now pass model to DeepChaosScheduler and your trainer as normal.
      # After training, quantize back with:
      #   from onebitllms import quantize_to_1bit
      #   quantize_to_1bit(output_dir, quantized_output_dir)
"""


def apply_bitnet_linear_replacement(model: torch.nn.Module, verbose: bool = True) -> torch.nn.Module:
    """Replace linear layers with trainable BitLinear layers via onebitllms.

    Call this AFTER loading with revision="prequantized" and BEFORE passing
    the model to DeepChaosScheduler or any trainer.

    Raises RuntimeError if onebitllms is not installed.
    """
    try:
        from onebitllms import replace_linear_with_bitnet_linear  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "onebitllms is required for BitNet fine-tuning. "
            "Install with: pip install onebitllms"
        ) from exc

    model = replace_linear_with_bitnet_linear(model)
    if verbose:
        print("[lucky_pick_scheduler] BitLinear replacement applied (onebitllms).")
    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_model_for_training(
    model: torch.nn.Module,
    config: ModelPrepConfig | None = None,
) -> tuple[torch.nn.Module, ModelPrepReport]:
    cfg = config or ModelPrepConfig()

    bitnet = is_bitnet_model(model)
    quantized = is_quantized_model(model)
    auto_lora_applied = False
    lora_target_modules: List[str] = []
    bitnet_path: Optional[str] = None

    if bitnet:
        # Native BitNet checkpoint — transformers will hard-block the Trainer.
        # We cannot fix this at prepare time without reloading the model from
        # a different revision, which prepare_model_for_training can't do.
        # Emit a clear error with the exact steps to fix it.
        model_id = getattr(getattr(model, "config", None), "_name_or_path", None)
        if model_id:
            bitnet_path = f"{model_id} (revision='prequantized')"
        warnings.warn(
            f"\n\n[lucky_pick_scheduler] BitNet model detected.\n"
            f"{_BITNET_PREQUANTIZED_HINT}",
            UserWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "Cannot prepare a native BitNet checkpoint for training. "
            "Load with revision='prequantized' and call "
            "apply_bitnet_linear_replacement(model) before training. "
            "See the warning above for the full steps."
        )

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
        bitnet_detected=bool(bitnet),
        bitnet_path=bitnet_path,
        auto_lora_applied=bool(auto_lora_applied),
        lora_target_modules=lora_target_modules,
    )
    return model, report
