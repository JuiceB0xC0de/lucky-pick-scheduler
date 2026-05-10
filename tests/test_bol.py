"""CPU-only smoke tests for the bol_wandb / bol_scans public surface.

GPU-bound scan execution is not exercised here — we only verify imports,
callback construction, and that the scanner helper instantiates without
side effects. Heavy scans require a loaded LLM and should be run manually.
"""


def test_bol_wandb_imports():
    from bol_wandb import BOLScanner, BoLPrintCallback, BoLWandbCallback

    assert callable(BOLScanner)
    assert callable(BoLPrintCallback)
    assert callable(BoLWandbCallback)


def test_bol_scans_module_imports():
    import bol_scans

    assert hasattr(bol_scans, "run_all")
    assert hasattr(bol_scans, "format_results_for_cli")


def test_bol_print_callback_is_trainer_callback():
    from transformers import TrainerCallback

    from bol_wandb import BoLPrintCallback

    assert issubclass(BoLPrintCallback, TrainerCallback)
    # TrainerCallback hook surface is inherited.
    assert hasattr(BoLPrintCallback, "on_train_begin")
    assert hasattr(BoLPrintCallback, "on_train_end")
