#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 JSONL 文件的前 N 行并打印。支持中文路径。

用法示例：
    python preview_jsonl.py data.jsonl
    python preview_jsonl.py 数据/aa.jsonl -n 10
    python preview_jsonl.py aa.jsonl -n 3 --raw
    python preview_jsonl.py aa.jsonl -n 5 --skip-empty --max-chars 500
    python preview_jsonl.py aa.jsonl --encoding gbk
"""

import argparse
import json
import sys
from pathlib import Path


def _force_utf8_stdio() -> None:
    """让标准输出/错误流使用 UTF-8，避免打印中文时崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Python < 3.7 或受限环境忽略即可
            pass


def preview_jsonl(
    path: str,
    n: int = 5,
    raw: bool = False,
    skip_empty: bool = True,
    max_chars: int = 0,
    encoding: str = "utf-8",
) -> None:
    """
    path:       JSONL 文件路径（支持中文）
    n:          读取前 n 行（按非空行计数）
    raw:        True 则直接打印原始行，不解析 JSON
    skip_empty: 是否跳过空行
    max_chars:  单行最大打印字符数，0 表示不截断
    encoding:   文件编码，默认 utf-8
    """
    p = Path(path)

    if not p.exists():
        print(f"[ERROR] 文件不存在：{p}", file=sys.stderr)
        print(f"[INFO]  当前工作目录：{Path.cwd()}", file=sys.stderr)
        sys.exit(1)

    if not p.is_file():
        print(f"[ERROR] 不是普通文件：{p}", file=sys.stderr)
        sys.exit(1)

    count = 0
    try:
        with p.open("r", encoding=encoding, errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if skip_empty and not line.strip():
                    continue

                count += 1
                print(f"===== line {lineno} (sample #{count}) =====")

                out = line
                if not raw:
                    try:
                        obj = json.loads(line)
                        out = json.dumps(obj, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] 第 {lineno} 行不是合法 JSON：{e}", file=sys.stderr)
                        out = line

                if max_chars and len(out) > max_chars:
                    out = out[:max_chars] + f"\n... [已截断，原长度 {len(out)}]"

                print(out)
                print()

                if count >= n:
                    break
    except UnicodeDecodeError as e:
        print(f"[ERROR] 按 {encoding} 解码失败：{e}", file=sys.stderr)
        print("[INFO]  可尝试 --encoding gbk 或 --encoding utf-8-sig", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] 读取文件失败：{e}", file=sys.stderr)
        sys.exit(1)

    if count == 0:
        print("[INFO] 没有读到有效行。", file=sys.stderr)


def main() -> None:
    _force_utf8_stdio()

    parser = argparse.ArgumentParser(description="读取 JSONL 文件前 N 行（支持中文路径）")
    parser.add_argument("path", help="JSONL 文件路径")
    parser.add_argument("-n", "--num", type=int, default=5, help="读取前几行，默认 5")
    parser.add_argument("--raw", action="store_true", help="不解析 JSON，直接打印原始行")
    parser.add_argument("--keep-empty", action="store_true", help="不跳过空行")
    parser.add_argument("--max-chars", type=int, default=0, help="单行最大打印字符数，0 不截断")
    parser.add_argument("--encoding", default="utf-8",
                        help="文件编码，默认 utf-8；常见可选 gbk / utf-8-sig")
    args = parser.parse_args()

    preview_jsonl(
        path=args.path,
        n=args.num,
        raw=args.raw,
        skip_empty=not args.keep_empty,
        max_chars=args.max_chars,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()