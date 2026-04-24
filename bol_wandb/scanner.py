from __future__ import annotations

from typing import Any, Dict

from bol_scans import format_cli_summary, run_all


class BOLScanner:
    """Thin wrapper around bol_scans.run_all for pre/post diagnostics."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        print_summary: bool = True,
        summary_top_k: int = 5,
        **scan_kwargs: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.print_summary = bool(print_summary)
        self.summary_top_k = int(summary_top_k)
        self.scan_kwargs = dict(scan_kwargs)

    def run(self, phase: str, **overrides: Any) -> Dict[str, Any]:
        kwargs = dict(self.scan_kwargs)
        kwargs.update(overrides)
        return run_all(
            self.model,
            self.tokenizer,
            phase=phase,
            print_summary=self.print_summary,
            summary_top_k=self.summary_top_k,
            **kwargs,
        )

    @staticmethod
    def format_summary(results: Dict[str, Any], *, top_k: int = 5) -> str:
        return format_cli_summary(results, top_k=top_k)
