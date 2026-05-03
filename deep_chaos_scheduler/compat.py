"""Compatibility patches for remote-code models across transformers versions."""

from __future__ import annotations

from typing import Any, Callable, List


def apply_transformers_remote_code_compat(*, verbose: bool = True) -> List[str]:
    """Apply safe runtime patches for known remote-code/transformers mismatches.

    This function is idempotent and can be called multiple times.
    Returns the list of patch names that were applied in this call.
    """

    applied: List[str] = []

    import transformers.utils.generic as hf_generic

    if not hasattr(hf_generic, "OutputRecorder"):
        class OutputRecorder:  # pragma: no cover - simple compatibility shim
            def __init__(self, module_cls, index: int = 0):
                self.module_cls = module_cls
                self.index = index

        hf_generic.OutputRecorder = OutputRecorder  # type: ignore[attr-defined]
        applied.append("generic.OutputRecorder")

    if not hasattr(hf_generic, "check_model_inputs"):
        def check_model_inputs(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        hf_generic.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]
        applied.append("generic.check_model_inputs")

    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "_lps_orig_get_expanded_tied_weights_keys"):
        PreTrainedModel._lps_orig_get_expanded_tied_weights_keys = PreTrainedModel.get_expanded_tied_weights_keys  # type: ignore[attr-defined]

        def _patched_get_expanded_tied_weights_keys(self, *args, **kwargs):
            orig = PreTrainedModel._lps_orig_get_expanded_tied_weights_keys  # type: ignore[attr-defined]
            try:
                return orig(self, *args, **kwargs)
            except AttributeError as exc:
                # Some remote-code models expose _tied_weights_keys as list, while
                # newer transformers internals may temporarily expect dict-like mappings.
                if "'list' object has no attribute 'keys'" not in str(exc):
                    raise
                tied = getattr(self, "_tied_weights_keys", None)
                if isinstance(tied, dict):
                    return dict(tied)
                if isinstance(tied, (list, tuple, set)):
                    items = [item for item in tied if isinstance(item, str)]
                    return {item: item for item in items}
                return {}

        PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded_tied_weights_keys  # type: ignore[assignment]
        applied.append("modeling_utils.get_expanded_tied_weights_keys")

    if not hasattr(PreTrainedModel, "_lps_orig_mark_tied_weights_as_initialized"):
        PreTrainedModel._lps_orig_mark_tied_weights_as_initialized = PreTrainedModel.mark_tied_weights_as_initialized  # type: ignore[attr-defined]

        def _patched_mark_tied_weights_as_initialized(self, loading_info):
            tied = getattr(self, "all_tied_weights_keys", None)
            if isinstance(tied, (list, tuple, set)):
                items = [item for item in tied if isinstance(item, str)]
                setattr(self, "all_tied_weights_keys", {item: item for item in items})
            elif isinstance(tied, dict):
                # keep as-is
                pass
            elif tied is None:
                setattr(self, "all_tied_weights_keys", {})
            return PreTrainedModel._lps_orig_mark_tied_weights_as_initialized(self, loading_info)  # type: ignore[attr-defined]

        PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights_as_initialized  # type: ignore[assignment]
        applied.append("modeling_utils.mark_tied_weights_as_initialized")

    if verbose and applied:
        print(f"[deep_chaos_scheduler.compat] Applied patches: {', '.join(applied)}")
    return applied


def allow_quantized_training_in_trainer(*, verbose: bool = True) -> List[str]:
    """Patch Trainer quantized-model validation for unsupported quant wrappers.

    Some models (e.g. Falcon-E / Microsoft BitNet after replace_linear_with_bitnet_linear)
    are legitimately trainable but blocked by transformers' hard check in
    validate_quantization_for_training.  This is an explicit opt-in bypass.

    Patches three locations to be resilient against different transformers import orders:

    1. ``transformers.trainer_utils.validate_quantization_for_training`` — the
       canonical definition.
    2. ``transformers.trainer`` module-level name binding — importlib patch so
       the module attribute resolves to the no-op.
    3. ``Trainer.__init__.__globals__['validate_quantization_for_training']`` —
       the local name captured at the time trainer.py was imported.  This is the
       binding that actually fires at runtime.  It must be patched even if (1)
       and (2) are already done.

    This function is idempotent — calling it multiple times is safe.
    """

    applied: List[str] = []

    def _noop_validate_quantization_for_training(_model):
        return None

    import importlib

    # 1. Patch trainer_utils module attribute
    try:
        tu = importlib.import_module("transformers.trainer_utils")
        if getattr(tu, "validate_quantization_for_training", None) is not _noop_validate_quantization_for_training:
            if not hasattr(tu, "_lps_orig_validate_quantization_for_training"):
                setattr(tu, "_lps_orig_validate_quantization_for_training",
                        tu.validate_quantization_for_training)
            setattr(tu, "validate_quantization_for_training", _noop_validate_quantization_for_training)
            applied.append("transformers.trainer_utils.validate_quantization_for_training")
    except Exception:
        pass

    # 2. Patch transformers.trainer module attribute
    try:
        tr = importlib.import_module("transformers.trainer")
        if getattr(tr, "validate_quantization_for_training", None) is not _noop_validate_quantization_for_training:
            if not hasattr(tr, "_lps_orig_validate_quantization_for_training"):
                orig = getattr(tr, "validate_quantization_for_training", None)
                if orig is not None:
                    setattr(tr, "_lps_orig_validate_quantization_for_training", orig)
            setattr(tr, "validate_quantization_for_training", _noop_validate_quantization_for_training)
            applied.append("transformers.trainer.validate_quantization_for_training")
    except Exception:
        pass

    # 3. Patch Trainer.__init__.__globals__ — the binding that actually fires.
    #    transformers.trainer does:
    #      from .trainer_utils import validate_quantization_for_training
    #    which creates a local name in trainer.py's global namespace that is
    #    captured in Trainer.__init__.__globals__ at import time.  Patching the
    #    module attribute alone does NOT affect this binding.
    try:
        tr = importlib.import_module("transformers.trainer")
        trainer_cls = getattr(tr, "Trainer", None)
        if trainer_cls is not None:
            init_fn = getattr(trainer_cls, "__init__", None)
            if init_fn is not None:
                g = getattr(init_fn, "__globals__", None)
                if isinstance(g, dict):
                    if g.get("validate_quantization_for_training") is not _noop_validate_quantization_for_training:
                        orig = g.get("validate_quantization_for_training")
                        if orig is not None:
                            g["_lps_orig_validate_quantization_for_training"] = orig
                        g["validate_quantization_for_training"] = _noop_validate_quantization_for_training
                        applied.append(
                            "Trainer.__init__.__globals__.validate_quantization_for_training"
                        )
    except Exception:
        pass

    if verbose and applied:
        print(f"[deep_chaos_scheduler.compat] Patched trainer validation: {', '.join(applied)}")
    elif verbose:
        print("[deep_chaos_scheduler.compat] Trainer validation already patched (idempotent).")

    return applied


def patch_clip_grad_norm_disable_foreach(*, verbose: bool = True) -> bool:
    """Force ``torch.nn.utils.clip_grad_norm_`` to run with ``foreach=False``.

    Background: on some CUDA / driver / A100 combinations,
    ``torch._foreach_norm`` (the fused kernel used by the default grad-clip
    path) trips an XID 31 PDE MMU fault partway through training — surfacing
    as ``RuntimeError: CUDA error: an illegal memory access was encountered``
    inside ``clip_grad_norm_``. The single-tensor loop (foreach=False) is
    slightly slower but has been stable across the versions we've tested.

    Observed on: Gemma-4 (E4B), A100-80GB, torch 2.5.1 + CUDA 12.1.
    Trigger: ``trainer.train()`` crashes in ``_foreach_norm(device_grads, ...)``
    on one of the first few optimizer steps, not necessarily step 0.

    This patch is idempotent and also patches the reference held by
    ``accelerate.accelerator`` (which imports ``torch.nn.utils`` at module
    load time), since HuggingFace ``Trainer.clip_grad_norm_`` delegates to
    ``self.accelerator.clip_grad_norm_``.

    Returns True if the patch was applied this call, False if it was already
    applied.
    """
    import torch

    if getattr(torch.nn.utils.clip_grad_norm_, "_lps_disabled_foreach", False):
        if verbose:
            print("[deep_chaos_scheduler.compat] clip_grad_norm_ foreach already disabled.")
        return False

    _orig = torch.nn.utils.clip_grad_norm_

    def _clip_grad_norm_no_foreach(parameters, max_norm, norm_type=2.0,
                                   error_if_nonfinite=False, foreach=None):
        return _orig(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=False,
        )

    _clip_grad_norm_no_foreach._lps_disabled_foreach = True  # type: ignore[attr-defined]
    _clip_grad_norm_no_foreach._lps_orig = _orig  # type: ignore[attr-defined]

    torch.nn.utils.clip_grad_norm_ = _clip_grad_norm_no_foreach

    # Accelerate imports the symbol at module-load time, so patching
    # torch.nn.utils alone doesn't cover Trainer -> accelerator.clip_grad_norm_.
    try:
        import accelerate.accelerator as _accel_mod
        _accel_mod.torch.nn.utils.clip_grad_norm_ = _clip_grad_norm_no_foreach
    except Exception as exc:
        if verbose:
            print(f"[deep_chaos_scheduler.compat] accelerate patch skipped: {exc}")

    if verbose:
        print(
            "[deep_chaos_scheduler.compat] Patched clip_grad_norm_ -> foreach=False "
            "(A100 _foreach_norm XID-31 workaround)."
        )
    return True
