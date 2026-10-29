import pytest

torch = pytest.importorskip("torch")
from torch import nn

from trainer.precision import (
    PrecisionPlan,
    _clone_grad_views,
    _should_convert_linear_to_fp8,
    apply_fp8_training,
    fp8_available,
    resolve_precision,
)


def test_bf16_plan():
    plan = resolve_precision("bfloat16", "cpu")
    assert plan.name == "bf16"
    assert plan.autocast_dtype == torch.bfloat16
    assert plan.use_fp8 is False


def test_fp8_fails_cleanly_when_unavailable(monkeypatch):
    monkeypatch.setattr("trainer.precision.fp8_available", lambda device="cuda": (False, "test unavailable"))
    with pytest.raises(RuntimeError, match="FP8 training requested but unavailable"):
        resolve_precision("fp8", "cuda")


def test_fp8_linear_filter_protects_tied_and_small_heads():
    hidden = nn.Linear(1280, 1280, bias=False)
    lm_head = nn.Linear(1280, 6400, bias=False)
    value_head = nn.Linear(1280, 1, bias=True)
    router = nn.Linear(1280, 4, bias=False)
    misaligned = nn.Linear(1280, 1000, bias=False)

    assert _should_convert_linear_to_fp8(hidden, "model.layers.0.self_attn.q_proj")
    assert not _should_convert_linear_to_fp8(lm_head, "lm_head")
    assert not _should_convert_linear_to_fp8(value_head, "value_head")
    assert not _should_convert_linear_to_fp8(router, "model.layers.0.mlp.gate")
    assert not _should_convert_linear_to_fp8(misaligned, "model.layers.0.mlp.up_proj")


def test_fp8_grad_view_clone_materializes_views():
    base = torch.randn(8, 8, requires_grad=True)
    transposed = base.t()
    cloned, untouched = _clone_grad_views((transposed, None))
    assert cloned._base is None
    assert torch.equal(cloned, transposed)
    assert untouched is None


def test_torchao_fp8_autograd_patch_clones_views(monkeypatch):
    torchao_linear = pytest.importorskip("torchao.float8.float8_linear")
    from trainer.precision import _patch_torchao_float8_autograd

    matmul = torchao_linear.matmul_with_hp_or_float8_args
    assert _patch_torchao_float8_autograd()
    assert _patch_torchao_float8_autograd()
    assert getattr(matmul.forward, "_whllm_clones_output", False)
    assert getattr(matmul.backward, "_whllm_clones_grads", False)


def test_fp8_conversion_enables_inner_dim_padding(monkeypatch):
    torchao_float8 = pytest.importorskip("torchao.float8")
    captured = {}

    def fake_convert(model, *, module_filter_fn=None, config=None):
        captured["pad_inner_dim"] = config.pad_inner_dim
        return model

    monkeypatch.setattr(torchao_float8, "convert_to_float8_training", fake_convert)
    model = nn.Sequential(nn.Linear(32, 32, bias=False))
    apply_fp8_training(model, PrecisionPlan("fp8", None, True, False, "test"))
    assert captured["pad_inner_dim"] is True
