import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")

from dataset.lm_dataset import PretrainDataset, SFTDataset, resolve_jsonl_files
from trainer.trainer_utils import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")


def test_resolve_jsonl_files_recursively(tmp_path):
    _write_jsonl(tmp_path / "b.jsonl", "b")
    _write_jsonl(tmp_path / "nested" / "a.jsonl", "a")
    (tmp_path / "ignore.txt").write_text("not jsonl", encoding="utf-8")

    files = resolve_jsonl_files(tmp_path)
    assert [Path(path).name for path in files] == ["a.jsonl", "b.jsonl"]


def test_resolve_jsonl_files_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .jsonl files"):
        resolve_jsonl_files(tmp_path)


def test_pretrain_dataset_concatenates_multiple_files(tmp_path):
    _write_jsonl(tmp_path / "part-1.jsonl", "first sample")
    _write_jsonl(tmp_path / "part-2.jsonl", "second sample")
    tokenizer = load_tokenizer(ROOT / "model")

    dataset = PretrainDataset(str(tmp_path), tokenizer, max_length=16)
    assert len(dataset) == 2
    input_ids, labels = dataset[0]
    assert input_ids.shape == labels.shape == (16,)
    assert labels[-1].item() == -100


def test_sft_accepts_messages_and_conversations(tmp_path):
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    (tmp_path / "messages.jsonl").write_text(
        json.dumps({"messages": conversation}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "both.jsonl").write_text(
        json.dumps({"messages": conversation, "conversations": conversation, "extra": "ignored"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "mixed-fields.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "tools", "tools": "[]"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi", "tool_calls": "[]"},
                ]
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    tokenizer = load_tokenizer(ROOT / "model")

    dataset = SFTDataset(str(tmp_path), tokenizer, max_length=64)
    assert len(dataset) == 3
    assert dataset.samples.column_names == ["conversations"]
    for index in range(len(dataset)):
        input_ids, labels = dataset[index]
        assert input_ids.shape == labels.shape == (64,)
