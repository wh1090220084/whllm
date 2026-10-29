"""FP8/BF16 mixed-precision utilities for WHLLM.

The default training path uses BF16 autocast.  When ``--dtype fp8`` is used,
compatible hidden Linear layers are converted with ``torchao.float8`` while
the surrounding model, optimizer state, and activations remain in BF16. Tied
output heads and undersized/router matrices stay in high precision.

FP8 requires a CUDA device with compute capability >= 8.9 and the optional
``torchao`` package.  The module intentionally fails loudly instead of silently
falling back to BF16, so an experiment cannot be mislabeled as FP8.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class PrecisionPlan:
    name: str
    autocast_dtype: Optional[torch.dtype]
    use_fp8: bool
    use_grad_scaler: bool
    reason: str


def _is_cuda(device: str | torch.device) -> bool:
    return "cuda" in str(device)


def fp8_available(device: str | torch.device = "cuda") -> tuple[bool, str]:
    """Return whether the current runtime can perform FP8 training."""
    if not _is_cuda(device):
        return False, "FP8 requires a CUDA device"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if not hasattr(torch, "float8_e4m3fn"):
        return False, "this PyTorch build has no float8_e4m3fn dtype"
    try:
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:  # pragma: no cover - driver-specific
        return False, f"could not read CUDA capability: {exc}"
    if capability < (8, 9):
        return False, f"compute capability {capability} < (8, 9)"
    try:
        import torchao.float8  # noqa: F401
    except Exception as exc:
        return False, f"torchao.float8 is unavailable: {exc}"
    return True, "FP8 tensor cores and torchao.float8 are available"


def resolve_precision(name: str = "bfloat16", device: str | torch.device = "cuda") -> PrecisionPlan:
    """Resolve a user-facing precision name into a concrete training plan."""
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in {"bf16", "bfloat16"}:
        return PrecisionPlan("bf16", torch.bfloat16, False, False, "BF16 autocast")
    if normalized in {"fp16", "float16", "half"}:
        return PrecisionPlan("fp16", torch.float16, False, True, "FP16 autocast with GradScaler")
    if normalized in {"fp8", "fp8bf16", "bf16fp8"}:
        ok, reason = fp8_available(device)
        if not ok:
            raise RuntimeError(
                f"FP8 training requested but unavailable: {reason}. "
                "Install torchao and use a CUDA GPU with FP8 support, or use --dtype bfloat16."
            )
        return PrecisionPlan("fp8", torch.bfloat16, True, False, f"FP8 Linear + BF16 autocast ({reason})")
    raise ValueError(f"unsupported precision/dtype: {name!r}; choose bfloat16, float16, or fp8")


def autocast_context(plan: PrecisionPlan, device: str | torch.device = "cuda"):
    """Return the autocast context used around forward/loss computation."""
    if not _is_cuda(device) or plan.autocast_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=plan.autocast_dtype)


def build_grad_scaler(plan: PrecisionPlan, device: str | torch.device = "cuda"):
    """Build a scaler compatible with both newer and older PyTorch releases."""
    enabled = bool(plan.use_grad_scaler and _is_cuda(device))
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _should_convert_linear_to_fp8(module: torch.nn.Module, fqn: str) -> bool:
    """Return whether a Linear is safe and useful to convert to Float8Linear.

    torchao 0.15-0.18 returns a transposed view for the weight gradient of its
    custom FP8 matmul. A tied output head therefore triggers an autograd
    view+inplace error when embedding and head gradients accumulate. Output
    heads and routing matrices are also too small to benefit from FP8.
    """
    if not isinstance(module, torch.nn.Linear):
        return False
    leaf_name = fqn.rsplit(".", 1)[-1]
    if leaf_name in {"lm_head", "value_head"}:
        return False
    if fqn == "mlp.gate" or fqn.endswith(".mlp.gate"):
        return False
    out_features, in_features = module.weight.shape
    return out_features % 16 == 0 and in_features % 16 == 0


def _clone_grad_views(grads: tuple[Any, ...]) -> tuple[Any, ...]:
    """Clone autograd outputs that are views before they can be mutated in place."""
    return tuple(
        grad.clone()
        if isinstance(grad, torch.Tensor) and grad._base is not None
        else grad
        for grad in grads
    )


def _patch_torchao_float8_autograd() -> bool:
    """Materialize views returned by torchao's FP8 matmul autograd Function.

    torchao 0.15-0.18 returns ``res_bits.reshape(...)`` from the custom forward
    and reshaped/transposed views from its backward. On some PyTorch builds,
    later in-place operations (including gradient clipping) then raise:
    ``Output 0 of matmul_with_hp_or_float8_args is a view``. The shim follows
    the error's recommendation and clones both forward and backward outputs.

    Returns True when the compatibility patch is active, including repeat
    calls. Returns False when this torchao implementation is unavailable.
    """
    try:
        from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
    except (ImportError, AttributeError):
        return False

    current_forward = matmul_with_hp_or_float8_args.forward
    if not getattr(current_forward, "_whllm_clones_output", False):
        def cloned_forward(ctx, *args):
            output = current_forward(ctx, *args)
            return output.clone() if isinstance(output, torch.Tensor) else output

        cloned_forward._whllm_clones_output = True
        matmul_with_hp_or_float8_args.forward = staticmethod(cloned_forward)

    current_backward = matmul_with_hp_or_float8_args.backward
    if not getattr(current_backward, "_whllm_clones_grads", False):
        def cloned_backward(ctx, grad_output):
            return _clone_grad_views(current_backward(ctx, grad_output))

        cloned_backward._whllm_clones_grads = True
        matmul_with_hp_or_float8_args.backward = staticmethod(cloned_backward)
    return True


def apply_fp8_training(model: torch.nn.Module, plan: PrecisionPlan) -> torch.nn.Module:
    """Convert eligible hidden Linear layers to torchao Float8Linear in-place."""
    if not plan.use_fp8:
        return model
    try:
        from torchao.float8 import convert_to_float8_training
        from torchao.float8.config import Float8LinearConfig
        _patch_torchao_float8_autograd()
        # Dynamic token counts are often not divisible by 16. torchao must
        # pad scaled-mm dimensions or backward fails in torch._scaled_mm.
        fp8_config = Float8LinearConfig(pad_inner_dim=True)
    except Exception as exc:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "FP8 training requires `torchao` with the float8 training API. "
            "Install the optional dependency with `pip install -r requirements-fp8.txt`."
        ) from exc

    embedding_weight_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
    }

    def module_filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        # Also skip any Linear whose weight is tied to an embedding, even if
        # the model uses a non-standard output-head name.
        return _should_convert_linear_to_fp8(module, fqn) and id(module.weight) not in embedding_weight_ids

    try:
        return convert_to_float8_training(
            model,
            module_filter_fn=module_filter_fn,
            config=fp8_config,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to convert model to FP8 training: {exc}") from exc


def precision_summary(plan: PrecisionPlan) -> str:
    return f"precision={plan.name}; fp8={plan.use_fp8}; {plan.reason}"
