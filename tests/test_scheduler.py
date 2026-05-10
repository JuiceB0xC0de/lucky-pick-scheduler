"""CPU-only smoke tests for the deep_chaos_scheduler public API.

These do not exercise GPU paths. They verify the package imports cleanly,
public symbols are wired through __init__, the config dataclass accepts
overrides, and the scheduler can be constructed against a tiny stub model
that mimics the `model.layers` decoder-stack shape.
"""

import torch
from torch import nn


def test_public_imports():
    from deep_chaos_scheduler import (
        AutoSchedulerConfig,
        DeepChaosConfig,
        DeepChaosScheduler,
        ModelPrepConfig,
        SchedulerBuildReport,
        apply_transformers_remote_code_compat,
        build_scheduler_stack,
        classify_model_parameters,
        infer_model_profile,
        prepare_model_for_training,
        resolve_transformer_layers,
    )

    assert callable(DeepChaosScheduler)
    assert callable(build_scheduler_stack)
    assert callable(resolve_transformer_layers)


def test_config_defaults_and_overrides():
    """Hoist is on by default; opting out is explicit."""
    from deep_chaos_scheduler import DeepChaosConfig

    cfg = DeepChaosConfig()
    assert cfg.sticky_interval == 50
    assert cfg.seed == 42
    assert cfg.use_layer_hoist is True
    assert cfg.hoist_stub_kind == "bias"

    cfg2 = DeepChaosConfig(sticky_interval=25, seed=199, use_layer_hoist=False)
    assert cfg2.sticky_interval == 25
    assert cfg2.seed == 199
    assert cfg2.use_layer_hoist is False


class _TinyDecoderLayer(nn.Module):
    """Mimics the surface a transformer decoder layer exposes — enough for
    resolve_transformer_layers to detect this as a block (`self_attn` +
    `mlp` attrs) and for scheduler bindings to introspect it."""

    def __init__(self, hidden: int = 16, heads: int = 4):
        super().__init__()
        self.hidden_size = hidden
        self.num_heads = heads
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.Linear(hidden * 2, hidden))

    def forward(self, x, *args, **kwargs):
        a, _ = self.self_attn(x, x, x, need_weights=False)
        return x + a + self.mlp(x)


class _TinyModel(nn.Module):
    def __init__(self, n_layers: int = 6, hidden: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([_TinyDecoderLayer(hidden) for _ in range(n_layers)])


def test_resolve_transformer_layers_finds_layers():
    from deep_chaos_scheduler import resolve_transformer_layers

    model = _TinyModel(n_layers=6)
    layers = resolve_transformer_layers(model)
    assert len(layers) == 6
    assert all(isinstance(layer, _TinyDecoderLayer) for layer in layers)


def test_scheduler_constructs_and_steps():
    """End-to-end smoke: build scheduler on a CPU stub, call .step(), confirm
    it returns a stats dict and reshuffles on a sticky boundary.

    Pinned to use_layer_hoist=False because the tiny stub uses
    nn.MultiheadAttention (packed Q/K/V) and nn.Sequential MLP, neither of
    which exposes the q_proj/up_proj/down_proj named submodules the hoist
    bindings code reads to size HoistStubs.  Hoist surgery on a real
    Qwen-style model is exercised by bench/bench_amd.py and
    bench/train_instrument.py.
    """
    from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler

    torch.manual_seed(0)
    model = _TinyModel(n_layers=6)
    cfg = DeepChaosConfig(
        sticky_interval=3,
        seed=7,
        announce_reshuffles=False,
        use_layer_hoist=False,
    )
    sched = DeepChaosScheduler(model, cfg)

    stats0 = sched.step(0)
    assert isinstance(stats0, dict)
    assert sched.last_shuffle_step == 0

    sched.step(1)
    sched.step(2)
    sched.step(3)
    assert sched.last_shuffle_step == 3
