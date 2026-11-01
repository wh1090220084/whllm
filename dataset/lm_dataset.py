from torch.utils.data import Dataset
import torch
import json
import os
import random
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_jsonl_files(data_path):
    """Return one JSONL file or all JSONL files below a dataset directory."""
    path = Path(data_path).expanduser()
    if path.is_dir():
        files = sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() == ".jsonl"),
            key=lambda candidate: str(candidate).lower(),
        )
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    if not files:
        raise FileNotFoundError(f"no .jsonl files found under dataset path: {path}")
    return [str(file_path) for file_path in files]


def _normalize_conversations(value):
    """Normalize common chat JSON shapes to a list of message dictionaries."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        for key in ("messages", "conversations"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list):
        raise ValueError(f"conversation field must be a list, got {type(value).__name__}")

    normalized = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"chat message must be an object, got {type(message).__name__}")
        item = dict(message)
        content = item.get("content", "")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                content = str(content)

        def json_text(value):
            if value is None or value == [] or value == {}:
                return ""
            if isinstance(value, str):
                return "" if value.strip() in {"", "[]", "{}", "null"} else value
            return json.dumps(value, ensure_ascii=False)

        reasoning = item.get("reasoning_content", "")
        if reasoning is None:
            reasoning = ""
        elif not isinstance(reasoning, str):
            reasoning = json_text(reasoning)

        # Keep a fixed message schema so Arrow does not infer incompatible
        # structs from rows that include or omit tools/tool_calls.
        normalized.append({
            "role": str(item.get("role", "")),
            "content": content,
            "reasoning_content": reasoning,
            "tools": json_text(item.get("tools")),
            "tool_calls": json_text(item.get("tool_calls")),
        })
    return normalized


def _conversations_from_record(record, keys=("conversations", "messages", "conversation")):
    for key in keys:
        value = record.get(key)
        if value:
            conversations = _normalize_conversations(value)
            if conversations:
                return conversations
    return []


def _iter_normalized_records(data_file, file_mtime_ns, file_size):
    del file_mtime_ns, file_size  # Included in the cache fingerprint.
    with open(data_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_file}:{line_number}: invalid JSON: {exc}") from exc
            conversations = _conversations_from_record(record)
            if not conversations:
                raise ValueError(f"{data_file}:{line_number}: no messages/conversations field")
            yield {"conversations": conversations}


def _load_conversation_dataset(data_files):
    """Load heterogeneous conversation JSONL shards with one stable schema."""
    parts = []
    for data_file in data_files:
        stat = Path(data_file).stat()
        part = HFDataset.from_generator(
            _iter_normalized_records,
            gen_kwargs={
                "data_file": data_file,
                "file_mtime_ns": stat.st_mtime_ns,
                "file_size": stat.st_size,
            },
        )
        parts.append(part)
    return concatenate_datasets(parts)


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是WHLLM，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是WHLLM，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are WHLLM, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are WHLLM, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_files = resolve_jsonl_files(data_path)
        self.samples = load_dataset('json', data_files=self.data_files, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        valid_length = len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - valid_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # Mask by position rather than token ID: some tokenizers intentionally
        # use the same ID for padding and EOS, and EOS must remain a target.
        labels[valid_length:] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_files = resolve_jsonl_files(jsonl_path)
        self.samples = _load_conversation_dataset(self.data_files)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.data_files = resolve_jsonl_files(file_path)
        self.samples = load_dataset('json', data_files=self.data_files, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.data_files = resolve_jsonl_files(jsonl_path)
        self.samples = _load_conversation_dataset(self.data_files)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.data_files = resolve_jsonl_files(jsonl_path)
        for data_file in self.data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        conversations = _conversations_from_record(record)
                        if not conversations:
                            raise ValueError(f"{data_file}: record has no messages/conversations field")
                        record["conversations"] = conversations
                        self.samples.append(record)

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass