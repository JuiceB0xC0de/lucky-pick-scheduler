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
                    return list(tied.keys()) + [v for v in tied.values() if isinstance(v, str)]
                if isinstance(tied, (list, tuple, set)):
                    return [item for item in tied if isinstance(item, str)]
                return []

        PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded_tied_weights_keys  # type: ignore[assignment]
        applied.append("modeling_utils.get_expanded_tied_weights_keys")

    if verbose and applied:
        print(f"[lucky_pick_scheduler.compat] Applied patches: {', '.join(applied)}")
    return applied

