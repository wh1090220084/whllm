"""Muon/AdamW hybrid optimizer for WHLLM training.

Muon is applied to hidden-layer weight matrices. Embedding tables, output
heads, router gates, normalization weights, biases, and other non-matrix
parameters stay on AdamW, following the optimizer's intended parameter scope.
The wrapper exposes the normal Optimizer protocol so existing LR schedulers,
GradScaler usage, and checkpoint code continue to work.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW, Muon, Optimizer


_STATE_FORMAT = "whllm.muon_adamw.v1"
_ADAMW_MATRIX_MARKERS = (
    "embed_tokens.",
    "lm_head.",
    "value_head.",
    "reward_head.",
    ".mlp.gate.",
    ".router.",
)


def _normalize_named_parameters(
    parameters: Iterable[torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Return named parameters while accepting either names or bare tensors."""
    normalized: list[tuple[str, torch.Tensor]] = []
    for index, item in enumerate(parameters):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], torch.Tensor):
            name, parameter = item
        else:
            name, parameter = f"param_{index}", item
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f"optimizer parameter {name!r} is not a torch.Tensor")
        normalized.append((str(name), parameter))
    return normalized


def _uses_muon(name: str, parameter: torch.Tensor) -> bool:
    """Return whether a parameter belongs to Muon's hidden-matrix scope."""
    if parameter.ndim != 2:
        return False
    lowered = name.lower()
    return not any(marker in lowered for marker in _ADAMW_MATRIX_MARKERS)


class MuonAdamW(Optimizer):
    """Compose Muon and AdamW behind one Optimizer-compatible interface.

    Args:
        muon_params: Two-dimensional hidden-layer matrices for Muon.
        adamw_params: Parameters for AdamW, including embeddings and 1D tensors.
        muon_lr: Learning rate for Muon parameter groups.
        adamw_lr: Learning rate for AdamW; defaults to ``muon_lr``.
        weight_decay: AdamW weight decay.
        muon_weight_decay: Muon decoupled weight decay.

    The Muon group uses ``adjust_lr_fn="match_rms_adamw"`` so a learning rate
    tuned for AdamW can be reused under the RMS-matched Muon update rule.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.Tensor],
        adamw_params: Iterable[torch.Tensor],
        *,
        muon_lr: float,
        adamw_lr: float | None = None,
        weight_decay: float = 0.01,
        muon_weight_decay: float = 0.1,
    ) -> None:
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        if not muon_params and not adamw_params:
            raise ValueError("optimizer received no trainable parameters")
        if adamw_lr is None:
            adamw_lr = muon_lr

        self.muon = (
            Muon(
                muon_params,
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
            if muon_params
            else None
        )
        self.adamw = (
            AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)
            if adamw_params
            else None
        )

        # Optimizer.__init__ establishes scheduler/scaler protocol hooks. The
        # temporary group is replaced by the real inner-optimizer groups below.
        super().__init__(
            muon_params + adamw_params,
            defaults={"lr": muon_lr},
        )
        self._refresh_param_groups()

    @property
    def _inner_optimizers(self) -> tuple[Optimizer, ...]:
        return tuple(opt for opt in (self.muon, self.adamw) if opt is not None)

    def _refresh_param_groups(self) -> None:
        self.param_groups = [
            group
            for optimizer in self._inner_optimizers
            for group in optimizer.param_groups
        ]

    def step(self, closure: Any = None):
        """Advance each inner optimizer once."""
        if closure is not None:
            raise NotImplementedError("MuonAdamW does not support closure-based steps")
        results = [optimizer.step() for optimizer in self._inner_optimizers]
        return results[-1] if results else None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for both inner optimizers."""
        for optimizer in self._inner_optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return a versioned state containing both inner optimizer states."""
        return {
            "format": _STATE_FORMAT,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a state produced by this hybrid optimizer.

        AdamW-only checkpoint states cannot be mapped safely onto the new
        parameter groups and are rejected instead of silently resetting state.
        """
        if not isinstance(state_dict, dict) or state_dict.get("format") != _STATE_FORMAT:
            raise ValueError(
                "incompatible optimizer checkpoint: expected MuonAdamW state; "
                "restart training or resume model weights without optimizer state"
            )
        muon_state = state_dict.get("muon")
        adamw_state = state_dict.get("adamw")
        if (self.muon is None) != (muon_state is None):
            raise ValueError("checkpoint Muon group does not match this model")
        if (self.adamw is None) != (adamw_state is None):
            raise ValueError("checkpoint AdamW group does not match this model")
        if self.muon is not None:
            self.muon.load_state_dict(muon_state)
        if self.adamw is not None:
            self.adamw.load_state_dict(adamw_state)
        self._refresh_param_groups()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(muon_groups="
            f"{len(self.muon.param_groups) if self.muon is not None else 0}, "
            f"adamw_groups={len(self.adamw.param_groups) if self.adamw is not None else 0})"
        )


def build_muon_optimizer(
    parameters: Iterable[torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    *,
    lr: float,
    adamw_lr: float | None = None,
    weight_decay: float = 0.01,
    muon_weight_decay: float = 0.1,
) -> MuonAdamW:
    """Partition trainable parameters and build a Muon/AdamW optimizer.

    Args:
        parameters: ``named_parameters()`` or an iterable of trainable tensors.
        lr: Muon learning rate. When ``adamw_lr`` is omitted it is also used by
            AdamW because Muon is configured to match AdamW update RMS.
        adamw_lr: Optional separate AdamW learning rate.
        weight_decay: AdamW weight decay.
        muon_weight_decay: Muon decoupled weight decay.

    Returns:
        A scheduler- and GradScaler-compatible hybrid optimizer.
    """
    muon_params: list[torch.Tensor] = []
    adamw_params: list[torch.Tensor] = []
    seen: set[int] = set()
    for name, parameter in _normalize_named_parameters(parameters):
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        target = muon_params if _uses_muon(name, parameter) else adamw_params
        target.append(parameter)
    return MuonAdamW(
        muon_params,
        adamw_params,
        muon_lr=lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        muon_weight_decay=muon_weight_decay,
    )
