"""转换脚本核心规则的最小回归测试。"""

import unittest
from collections import Counter

from scripts.convert_sft_to_messages import ConversionError, convert_file, convert_record


class ConvertSftToMessagesTest(unittest.TestCase):
    def test_alpaca_history_is_expanded_in_order(self):
        record = {
            "instruction": "请简洁回答",
            "input": "清朝八旗有多少旗人？",
            "output": "约 200 万人。",
            "history": [["写一首诗。", "秋风起。"], ["推荐歌曲。", "Back in Black。"]],
        }

        messages = convert_record(record)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "写一首诗。"},
                {"role": "assistant", "content": "秋风起。"},
                {"role": "user", "content": "推荐歌曲。"},
                {"role": "assistant", "content": "Back in Black。"},
                {"role": "user", "content": "请简洁回答\n\n清朝八旗有多少旗人？"},
                {"role": "assistant", "content": "约 200 万人。"},
            ],
        )

    def test_conversations_roles_are_normalized_and_reasoning_is_dropped(self):
        record = {
            "conversations": [
                {"role": "human", "content": "问题", "reasoning_content": ""},
                {"role": "gpt", "content": "答案", "reasoning_content": "内部推理"},
            ]
        }
        stats = Counter()

        messages = convert_record(record, stats=stats)

        self.assertEqual(
            messages,
            [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}],
        )
        self.assertEqual(stats["reasoning_nonempty"], 1)
        self.assertEqual(stats["reasoning_dropped"], 1)

    def test_reasoning_can_be_kept_on_assistant_messages(self):
        record = {
            "messages": [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "答案", "reasoning_content": "推理"},
            ]
        }

        messages = convert_record(record, keep_reasoning=True)

        self.assertEqual(messages[-1]["reasoning_content"], "推理")

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ConversionError):
            convert_record(
                {
                    "messages": [
                        {"role": "critic", "content": "未知角色"},
                        {"role": "assistant", "content": "答案"},
                    ]
                }
            )

    def test_record_must_end_with_assistant(self):
        with self.assertRaises(ConversionError):
            convert_record({"messages": [{"role": "user", "content": "没有回答"}]})


    def test_tool_messages_and_fields_are_preserved(self):
        record = {
            "conversations": [
                {"role": "system", "content": "", "tools": "[{\"name\": \"calc\"}]"},
                {"role": "user", "content": "calculate"},
                {"role": "assistant", "content": "call tool", "tool_calls": "[{\"name\": \"calc\"}]"},
                {"role": "tool", "content": "result 42"},
                {"role": "assistant", "content": "answer is 42"},
            ]
        }

        messages = convert_record(record)

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[0]["tools"], "[{\"name\": \"calc\"}]")
        self.assertEqual(messages[2]["tool_calls"], "[{\"name\": \"calc\"}]")

    def test_tool_records_can_be_skipped(self):
        record = {
            "messages": [
                {"role": "user", "content": "calculate"},
                {"role": "assistant", "content": "call tool", "tool_calls": "[]"},
                {"role": "tool", "content": "result 42"},
                {"role": "assistant", "content": "answer is 42"},
            ]
        }

        with self.assertRaises(ConversionError):
            convert_record(record, tool_mode="skip")

    def test_invalid_records_can_be_skipped_during_file_conversion(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "output.jsonl"
            valid = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
            invalid = {"instruction": "", "input": "", "output": "missing user"}
            input_path.write_text(
                json.dumps(valid, ensure_ascii=False) + "\n" + json.dumps(invalid, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            stats = convert_file(input_path, output_path, on_invalid="skip", progress_every=0)

            self.assertEqual(stats["records"], 1)
            self.assertEqual(stats["skipped_records"], 1)
            self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 1)

if __name__ == "__main__":
    unittest.main()
