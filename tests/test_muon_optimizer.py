import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainer.optim import MuonAdamW, build_muon_optimizer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 8)
        self.model.q_proj = nn.Linear(8, 8, bias=False)
        self.model.norm = nn.Parameter(torch.ones(8))
        self.model.mlp = nn.Module()
        self.model.mlp.gate = nn.Linear(8, 4, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)


def _parameter_ids(parameters):
    return {id(parameter) for parameter in parameters}


def test_parameter_partition_uses_muon_only_for_hidden_matrices():
    model = ToyModel()
    optimizer = build_muon_optimizer(model.named_parameters(), lr=1e-3)

    assert isinstance(optimizer, MuonAdamW)
    assert optimizer.muon is not None
    assert optimizer.adamw is not None

    muon_ids = _parameter_ids(
        parameter for group in optimizer.muon.param_groups for parameter in group["params"]
    )
    adamw_ids = _parameter_ids(
        parameter for group in optimizer.adamw.param_groups for parameter in group["params"]
    )
    trainable_ids = _parameter_ids(model.parameters())
    assert muon_ids.isdisjoint(adamw_ids)
    assert muon_ids | adamw_ids == trainable_ids
    assert id(model.model.q_proj.weight) in muon_ids
    assert id(model.model.embed_tokens.weight) in adamw_ids
    assert id(model.model.norm) in adamw_ids
    assert id(model.model.mlp.gate.weight) in adamw_ids
    assert id(model.lm_head.weight) in adamw_ids


def test_step_scheduler_and_state_roundtrip():
    model = ToyModel()
    optimizer = build_muon_optimizer(model.named_parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    assert all(group["lr"] < 1e-3 for group in optimizer.param_groups)

    state = optimizer.state_dict()
    restored = build_muon_optimizer(model.named_parameters(), lr=1e-3)
    restored.load_state_dict(state)
    assert restored.state_dict()["format"] == "whllm.muon_adamw.v1"
    assert len(restored.param_groups) == len(optimizer.param_groups)


def test_rejects_legacy_adamw_state():
    model = ToyModel()
    optimizer = build_muon_optimizer(model.named_parameters(), lr=1e-3)
    legacy_state = AdamW(model.parameters(), lr=1e-3).state_dict()
    with pytest.raises(ValueError, match="incompatible optimizer checkpoint"):
        optimizer.load_state_dict(legacy_state)


def test_grad_scaler_protocol():
    model = ToyModel()
    optimizer = build_muon_optimizer(model.named_parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)

    loss = sum(parameter.square().mean() for parameter in model.parameters())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
