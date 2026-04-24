from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import TrainerCallback

from bol_scans import run_all


class BoLWandbCallback(TrainerCallback):
    """Run BoL scans at train start/end with optional CLI summary printing."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        run_pre_train: bool = True,
        run_post_train: bool = True,
        print_summary: bool = True,
        summary_top_k: int = 5,
        scan_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.run_pre_train = bool(run_pre_train)
        self.run_post_train = bool(run_post_train)
        self.print_summary = bool(print_summary)
        self.summary_top_k = int(summary_top_k)
        self.scan_kwargs = dict(scan_kwargs or {})
        self.pre_results: Optional[Dict[str, Any]] = None
        self.post_results: Optional[Dict[str, Any]] = None

    def _run_phase(self, phase: str) -> Dict[str, Any]:
        return run_all(
            self.model,
            self.tokenizer,
            phase=phase,
            print_summary=self.print_summary,
            summary_top_k=self.summary_top_k,
            **self.scan_kwargs,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.run_pre_train:
            return
        print("[BoL] Running pre-train scan...")
        self.pre_results = self._run_phase("pre")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.run_post_train:
            return
        print("[BoL] Running post-train scan...")
        self.post_results = self._run_phase("post")


class BoLPrintCallback(BoLWandbCallback):
    """Convenience alias with print_summary enabled by default."""

    def __init__(self, model, tokenizer, **kwargs):
        kwargs.setdefault("print_summary", True)
        super().__init__(model, tokenizer, **kwargs)
