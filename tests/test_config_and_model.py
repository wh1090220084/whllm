import json
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForCausalLM

from model.model_wh import WhConfig, WhForCausalLM
from trainer.trainer_utils import load_tokenizer, sync_config_with_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def small_config():
    return WhConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=64,
        dropout=0.0,
        use_moe=False,
    )


def test_config_files_are_consistent():
    base = json.loads((ROOT / "configs" / "wh_0_6b.json").read_text(encoding="utf-8"))
    assert base["hidden_size"] == 1280
    assert base["num_hidden_layers"] == 24
    assert base["num_attention_heads"] == 16
    assert base["num_key_value_heads"] == 4
    assert base["head_dim"] == 80
    assert base["intermediate_size"] == 3744
    assert base["max_position_embeddings"] == 32768
    assert base["vocab_size"] == 6400
    assert (base["bos_token_id"], base["eos_token_id"], base["pad_token_id"]) == (1, 2, 0)
    assert base["use_moe"] is False

    moe = json.loads((ROOT / "configs" / "wh_0_6b_moe.json").read_text(encoding="utf-8"))
    cfg = WhConfig()
    assert moe["use_moe"] is True
    assert moe["num_hidden_layers"] == cfg.num_hidden_layers == 32
    assert moe["num_experts"] == cfg.num_experts == 8
    assert moe["num_experts_per_tok"] == cfg.num_experts_per_tok == 3
    assert moe["moe_intermediate_size"] == cfg.moe_intermediate_size == 464
    assert moe["vocab_size"] == cfg.vocab_size == 6400


def test_default_parameter_count_is_about_0_6b():
    cfg = WhConfig()
    with torch.device("meta"):
        model = WhForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 590_000_000 <= total <= 610_000_000, total


def test_tiny_forward_and_loss():
    cfg = small_config()
    model = WhForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_generation_respects_context_limit():
    cfg = small_config()
    cfg.max_position_embeddings = 8
    model = WhForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    output = model.generate(inputs=input_ids, max_new_tokens=16, do_sample=False, use_cache=False)
    assert output.shape[1] <= 8


def test_gradient_checkpointing_toggle():
    cfg = small_config()
    model = WhForCausalLM(cfg)
    model.gradient_checkpointing_enable()
    assert model.model.gradient_checkpointing is True
    model.gradient_checkpointing_disable()
    assert model.model.gradient_checkpointing is False


def test_auto_classes_load_saved_model_without_fallback_warning(tmp_path, caplog):
    model = WhForCausalLM(small_config())
    model.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["auto_map"] == {
        "AutoConfig": "model_wh.WhConfig",
        "AutoModelForCausalLM": "model_wh.WhForCausalLM",
    }

    with caplog.at_level(logging.WARNING, logger="transformers.configuration_utils"):
        config = AutoConfig.from_pretrained(tmp_path)
        loaded = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert isinstance(config, WhConfig)
    assert isinstance(loaded, WhForCausalLM)
    assert "to instantiate a model of type ``" not in caplog.text


def test_auto_tokenizer_does_not_fall_back_to_base_config(caplog):
    with caplog.at_level(logging.WARNING, logger="transformers.configuration_utils"):
        tokenizer = load_tokenizer(ROOT / "model")

    assert len(tokenizer) == max(tokenizer.get_vocab().values()) + 1
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (1, 2, 0)
    assert tokenizer.model_max_length == 32768
    assert "to instantiate a model of type ``" not in caplog.text


def test_moe_parameter_count_is_about_0_6b():
    cfg = WhConfig(hidden_size=1280, num_hidden_layers=32, use_moe=True)
    with torch.device("meta"):
        model = WhForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 590_000_000 <= total <= 610_000_000, total
    assert cfg.num_experts == 8
    assert cfg.num_experts_per_tok == 3
    assert cfg.moe_intermediate_size == 464


def test_tiny_moe_top3_forward_and_backward():
    cfg = small_config()
    cfg.use_moe = True
    cfg.num_experts = 8
    cfg.num_experts_per_tok = 3
    cfg.moe_intermediate_size = 32
    model = WhForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids, labels=input_ids)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    out.loss.backward()


class _FakeTokenizer:
    vocab_size = 129280
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def __len__(self):
        return 129280


def test_tokenizer_syncs_vocab_and_special_ids():
    cfg = WhConfig()
    sync_config_with_tokenizer(cfg, _FakeTokenizer())
    assert cfg.vocab_size == 129280
    assert (cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id) == (0, 1, 2)
