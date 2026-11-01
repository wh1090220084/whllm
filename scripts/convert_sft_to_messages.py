#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 Alpaca/ShareGPT 风格 SFT JSONL 流式转换为统一的 messages 格式。

逐行识别 Alpaca（history/instruction/input/output）、conversations 或 messages
记录，展开多轮历史并统一角色为 system/user/assistant/tool。普通 SFT 默认删除
reasoning_content，随后原子写入目标 JSONL，失败时不会留下半成品。

用法：
    python scripts/convert_sft_to_messages.py INPUT.jsonl OUTPUT.jsonl

默认输出 {"messages": [...]}；--output-field conversations 可兼容本项目当前
SFT 加载器。--preserve-metadata 保留非对话顶层字段，--dry-run 只校验不写文件，
--max-records N 用于小样本试跑，--overwrite 才允许替换已有输出。

工具记录默认保留 tool 角色及 tools/tool_calls 字段；--tool-mode skip 可排除工具记录。
--on-invalid skip 可跳过无法修复的坏样本并在汇总中报告数量。

依赖：Python 3.10+ 标准库。文件按行流式处理，内存占用与单条记录大小相关。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

Message = dict[str, Any]


class ConversionError(ValueError):
    """表示输入记录不满足可转换的 SFT 结构。"""


def _as_text(value: Any, field: str) -> str:
    """转换文本字段；None 视为空串，复杂类型直接报错以免静默改变语义。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConversionError(f"字段 {field!r} 必须是字符串，实际为 {type(value).__name__}")
    return value


def _normalize_role(role: Any) -> str:
    """把常见对话角色映射到统一角色集合。"""
    if not isinstance(role, str):
        raise ConversionError(f"role 必须是字符串，实际为 {type(role).__name__}")
    mapping = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "model": "assistant",
        "tool": "tool",
        "function": "tool",
    }
    normalized = mapping.get(role.strip().lower())
    if normalized is None:
        raise ConversionError(f"不支持的角色: {role!r}")
    return normalized


def _make_message(
    role: Any,
    content: Any,
    *,
    reasoning: Any = None,
    keep_reasoning: bool = False,
    field: str = "content",
) -> Message:
    message = {"role": _normalize_role(role), "content": _as_text(content, field)}
    if keep_reasoning:
        reasoning_text = _as_text(reasoning, "reasoning_content")
        if reasoning_text.strip():
            message["reasoning_content"] = reasoning_text
    return message


def _append_history(messages: list[Message], history: Any) -> None:
    """按原顺序展开 Alpaca history；每项应为 [user_text, assistant_text]。"""
    if history is None:
        return
    if not isinstance(history, list):
        raise ConversionError(f"history 必须是列表，实际为 {type(history).__name__}")
    for index, pair in enumerate(history):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConversionError(f"history[{index}] 必须是长度为 2 的 user/assistant 列表")
        messages.append(_make_message("user", pair[0], field=f"history[{index}][0]"))
        messages.append(_make_message("assistant", pair[1], field=f"history[{index}][1]"))

def _convert_alpaca(record: dict[str, Any], *, keep_reasoning: bool, stats: Counter[str]) -> list[Message]:
    messages: list[Message] = []
    _append_history(messages, record.get("history"))

    instruction = _as_text(record.get("instruction"), "instruction")
    user_input = _as_text(record.get("input"), "input")
    user_parts = [part for part in (instruction, user_input) if part.strip()]
    if not user_parts:
        raise ConversionError("instruction 和 input 均为空，无法构造最后一个 user 消息")
    if instruction.strip() and user_input.strip():
        stats["alpaca_instruction_merged"] += 1

    output = _as_text(record.get("output"), "output")
    messages.append(_make_message("user", "\n\n".join(user_parts), field="instruction/input"))
    messages.append(_make_message("assistant", output, field="output"))
    return messages


def _convert_message_list(
    items: Any,
    *,
    keep_reasoning: bool,
    tool_mode: str,
    stats: Counter[str],
    source_name: str,
) -> list[Message]:
    if not isinstance(items, list):
        raise ConversionError(f"{source_name} 必须是列表，实际为 {type(items).__name__}")

    messages: list[Message] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ConversionError(f"{source_name}[{index}] 必须是对象")
        role = _normalize_role(item.get("role"))
        content = item.get("content")
        reasoning = item.get("reasoning_content")
        has_tools = bool(item.get("tools"))
        has_tool_calls = bool(item.get("tool_calls"))
        if role == "tool":
            stats["tool_messages"] += 1
        if has_tools:
            stats["messages_with_tools"] += 1
        if has_tool_calls:
            stats["messages_with_tool_calls"] += 1
        if tool_mode == "skip" and (role == "tool" or has_tools or has_tool_calls):
            raise ConversionError("记录包含 tool 角色或 tools/tool_calls 字段；按 --tool-mode skip 跳过")

        if isinstance(reasoning, str) and reasoning.strip():
            stats["reasoning_nonempty"] += 1
            if not keep_reasoning:
                stats["reasoning_dropped"] += 1
        message = _make_message(
            role,
            content,
            reasoning=reasoning,
            keep_reasoning=keep_reasoning,
            field=f"{source_name}[{index}].content",
        )
        if tool_mode == "preserve":
            if role == "system" and has_tools:
                message["tools"] = item["tools"]
            if role == "assistant" and has_tool_calls:
                message["tool_calls"] = item["tool_calls"]
        messages.append(message)
    return messages

def _detect_format(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "messages"
    if "conversations" in record:
        return "conversations"
    if any(key in record for key in ("instruction", "input", "output", "history")):
        return "alpaca"
    raise ConversionError("无法识别记录格式：缺少 messages/conversations/Alpaca 字段")


def convert_record(
    record: Any,
    *,
    input_format: str = "auto",
    keep_reasoning: bool = False,
    tool_mode: str = "preserve",
    stats: Counter[str] | None = None,
) -> list[Message]:

    """把单条 SFT 记录转换为统一消息列表。

    auto 模式逐条识别格式。空 assistant content 会保留，因为是否丢弃属于数据
    质量策略；但记录必须至少包含一个 assistant 且以 assistant 结束。
    """
    if not isinstance(record, dict):
        raise ConversionError(f"JSONL 记录必须是对象，实际为 {type(record).__name__}")
    if tool_mode not in {"preserve", "skip"}:
        raise ValueError(f"unsupported tool_mode: {tool_mode}")
    stats = stats if stats is not None else Counter()
    detected = _detect_format(record) if input_format == "auto" else input_format
    stats[f"format_{detected}"] += 1

    if detected == "alpaca":
        messages = _convert_alpaca(record, keep_reasoning=keep_reasoning, stats=stats)
    elif detected == "conversations":
        messages = _convert_message_list(
            record.get("conversations"),
            keep_reasoning=keep_reasoning,
            tool_mode=tool_mode,
            stats=stats,
            source_name="conversations",
        )
    elif detected == "messages":
        messages = _convert_message_list(
            record.get("messages"),
            keep_reasoning=keep_reasoning,
            tool_mode=tool_mode,
            stats=stats,
            source_name="messages",
        )
    else:
        raise ConversionError(f"未知输入格式: {input_format!r}")

    if not messages:
        raise ConversionError("转换后没有消息")
    if messages[0]["role"] == "assistant":
        raise ConversionError("对话不能以 assistant 消息开头")
    if messages[-1]["role"] != "assistant":
        raise ConversionError("最后一轮必须以 assistant 消息结束")
    if not any(message["role"] == "assistant" for message in messages):
        raise ConversionError("记录中缺少 assistant 消息")

    stats["messages_total"] += len(messages)
    stats["assistant_messages"] += sum(message["role"] == "assistant" for message in messages)
    stats["empty_content_messages"] += sum(not message["content"].strip() for message in messages)
    return messages


def _output_record(messages: list[Message], output_field: str, metadata: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {output_field: messages}
    record.update(metadata)
    return record


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                yield line_number, None
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"{path}:{line_number}: JSON 解析失败: {exc}") from exc

def convert_file(
    input_path: Path,
    output_path: Path | None,
    *,
    input_format: str = "auto",
    output_field: str = "messages",
    keep_reasoning: bool = False,
    tool_mode: str = "preserve",
    on_invalid: str = "fail",
    preserve_metadata: bool = False,
    overwrite: bool = False,
    max_records: int = 0,
    progress_every: int = 100_000,
) -> Counter[str]:
    """流式转换一个 JSONL 文件并返回统计信息。

    output_path 为 None 时只做 dry-run 校验。写入使用同目录临时文件并在成功后
    原子替换，因此失败不会把已有目标文件改坏。on_invalid='skip' 时只跳过无法
    修复的单条记录并打印前 20 个错误位置，其余记录继续转换。
    """
    if input_format not in {"auto", "alpaca", "conversations", "messages"}:
        raise ValueError(f"unsupported input_format: {input_format}")
    if output_field not in {"messages", "conversations"}:
        raise ValueError(f"unsupported output_field: {output_field}")
    if tool_mode not in {"preserve", "skip"}:
        raise ValueError(f"unsupported tool_mode: {tool_mode}")
    if on_invalid not in {"fail", "skip"}:
        raise ValueError(f"unsupported on_invalid: {on_invalid}")
    if max_records < 0:
        raise ValueError("max_records 不能为负数")
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if output_path is not None:
        if output_path.resolve() == input_path.resolve():
            raise ValueError("输出路径不能与输入路径相同")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"输出文件已存在；使用 --overwrite 才会替换: {output_path}")

    stats: Counter[str] = Counter()
    started = time.monotonic()
    temp_path: Path | None = None
    output_handle = None

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        output_handle = temp_path.open("w", encoding="utf-8", newline="\n")

    try:
        for line_number, record in _iter_jsonl(input_path):
            if record is None:
                stats["blank_lines"] += 1
                continue
            stats["seen_records"] += 1
            try:
                messages = convert_record(
                    record,
                    input_format=input_format,
                    keep_reasoning=keep_reasoning,
                    tool_mode=tool_mode,
                    stats=stats,
                )
            except ConversionError as exc:
                location = f"{input_path}:{line_number}"
                if on_invalid == "skip":
                    stats["skipped_records"] += 1
                    if stats["skipped_records"] <= 20:
                        print(f"[SKIP] {location}: {exc}", file=sys.stderr)
                    continue
                raise ConversionError(f"{location}: {exc}") from exc

            stats["records"] += 1
            if output_handle is not None:
                metadata: dict[str, Any] = {}
                if preserve_metadata and isinstance(record, dict):
                    core_fields = {"instruction", "input", "output", "history", "conversations", "messages"}
                    metadata = {key: value for key, value in record.items() if key not in core_fields}
                output_handle.write(
                    json.dumps(
                        _output_record(messages, output_field, metadata),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                output_handle.write("\n")

            if progress_every and stats["seen_records"] % progress_every == 0:
                elapsed = time.monotonic() - started
                rate = stats["seen_records"] / elapsed if elapsed else 0.0
                print(f"seen={stats['seen_records']:,} written={stats['records']:,} rate={rate:,.0f} records/s", file=sys.stderr)
            if max_records and stats["seen_records"] >= max_records:
                stats["partial"] = 1
                break

        if output_handle is not None:
            output_handle.flush()
            os.fsync(output_handle.fileno())
            output_handle.close()
            output_handle = None
            assert temp_path is not None and output_path is not None
            os.replace(temp_path, output_path)
            temp_path = None
            stats["output_bytes"] = output_path.stat().st_size
    except BaseException:
        if output_handle is not None:
            output_handle.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise

    stats["elapsed_seconds"] = int(time.monotonic() - started)
    return stats

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Alpaca/ShareGPT SFT JSONL to unified messages")
    parser.add_argument("input", type=Path, help="input JSONL file")
    parser.add_argument("output", nargs="?", type=Path, help="output JSONL file; optional with --dry-run")
    parser.add_argument("--input-format", choices=("auto", "alpaca", "conversations", "messages"), default="auto")
    parser.add_argument("--output-field", choices=("messages", "conversations"), default="messages")
    parser.add_argument("--reasoning", choices=("drop", "keep"), default="drop")
    parser.add_argument(
        "--tool-mode",
        choices=("preserve", "skip"),
        default="preserve",
        help="preserve tool role and tools/tool_calls fields, or skip tool-related records",
    )
    parser.add_argument(
        "--on-invalid",
        choices=("fail", "skip"),
        default="fail",
        help="fail fast, or skip malformed records and report their line numbers",
    )
    parser.add_argument("--preserve-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("非 dry-run 模式必须提供 output 路径")
    if args.max_records < 0 or args.progress_every < 0:
        parser.error("--max-records 和 --progress-every 不能为负数")

    try:
        stats = convert_file(
            args.input,
            None if args.dry_run else args.output,
            input_format=args.input_format,
            output_field=args.output_field,
            keep_reasoning=args.reasoning == "keep",
            tool_mode=args.tool_mode,
            on_invalid=args.on_invalid,
            preserve_metadata=args.preserve_metadata,
            overwrite=args.overwrite,
            max_records=args.max_records,
            progress_every=args.progress_every,
        )
    except (ConversionError, FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    mode = "dry-run" if args.dry_run else f"written to {args.output}"
    print(f"[OK] {mode}")
    print(f"records={stats['records']:,} messages={stats['messages_total']:,} elapsed={stats['elapsed_seconds']}s")
    formats = {key.removeprefix("format_"): value for key, value in stats.items() if key.startswith("format_")}
    print(f"formats={formats}")
    print(
        f"reasoning_nonempty={stats.get('reasoning_nonempty', 0):,} dropped={stats.get('reasoning_dropped', 0):,} "
        f"empty_content_messages={stats.get('empty_content_messages', 0):,}"
    )
    print(
        f"skipped_records={stats.get('skipped_records', 0):,} " 
        f"tool_messages={stats.get('tool_messages', 0):,} " 
        f"messages_with_tools={stats.get('messages_with_tools', 0):,} " 
        f"messages_with_tool_calls={stats.get('messages_with_tool_calls', 0):,}"
    )
    if stats["partial"]:
        print("[WARN] 本次只处理了 --max-records 指定的前 N 条，输出不是完整数据集", file=sys.stderr)
    if "output_bytes" in stats:
        print(f"output_bytes={stats['output_bytes']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
