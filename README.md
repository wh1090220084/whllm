# WHLLM

A full-stack training project for a **0.6B decoder-only LLM** with **32k context**, **BF16 training**, and optional **FP8 + BF16 mixed training**.

## Status

This repository is an implementation scaffold and reference training stack, not a released pretrained model. No datasets or model weights are bundled.

The requested names **Qwen3.8** and **DeepSeekV4.1** do not have a public, reproducible architecture specification that can be copied verbatim. This project therefore uses verifiable, widely adopted modern LLM design patterns (Pre-Norm RMSNorm, GQA, QK-Norm, SwiGLU, RoPE/YaRN, optional top-k MoE) and keeps the model config explicit so a future architecture can be substituted without changing the training interface.

## Model Blueprint

| Item | Default |
|---|---:|
| Parameters | ~599.4M dense; ~595.8M MoE total; ~310.7M MoE active |
| Layers | 32 |
| Hidden size | 1280 |
| Attention | GQA, 16 Q heads / 4 KV heads |
| Head dimension | 80 |
| FFN | SwiGLU, intermediate size 3744 |
| Vocabulary | 6400 (tied embeddings) |
| Context | 32768 positions |
| RoPE theta | 1,000,000 |
| Normalization | RMSNorm + QK-Norm |
| MoE variant | 8 experts / top-3, expert width 464 |
| Default train precision | BF16 |
| Optional train precision | FP8 Linear + BF16 surrounding computation |

The exact config is in [`configs/wh_0_6b.json`](configs/wh_0_6b.json). A MoE variant is in [`configs/wh_0_6b_moe.json`](configs/wh_0_6b_moe.json).

## Project Layout

```text
model/                 WH model definition, LoRA, tokenizer files
dataset/               Pretrain/SFT/DPO/RLAIF/Agent RL datasets
trainer/               Pretrain, SFT, LoRA, DPO, PPO, GRPO/CISPO, Agent RL, distillation
trainer/precision.py   BF16/FP16/FP8 precision plan
trainer/rollout_engine.py
scripts/               WebUI, OpenAI-compatible API, conversion, ToolCall evaluation
configs/               Dense and MoE model configs
data/                  Put downloaded JSONL data here
tests/                 Smoke tests and static checks
```

## Installation

```bash
pip install -r requirements.txt
# Optional, only for --dtype fp8:
pip install -r requirements-fp8.txt
```

FP8 requires a CUDA GPU with compute capability >= 8.9 (for example L40/H100-class hardware) and the `torchao` FP8 training API. If the runtime cannot provide FP8, the code fails with a clear error instead of silently running BF16.

## Training

Run training scripts from the `trainer` directory so relative data and checkpoint paths resolve correctly.

All training entry points use a hybrid optimizer: Muon updates hidden-layer weight
matrices, while embeddings, output heads, router gates, norms, biases, and other
non-matrix parameters remain on AdamW. This requires PyTorch 2.9 or newer. Optimizer
states saved by the previous AdamW-only implementation are not resume-compatible.

### Pretrain

```bash
cd trainer
torchrun --nproc_per_node 8 train_pretrain.py \
  --data_path ../data/pretrain.jsonl \
  --max_seq_len 32768 \
  --gradient_checkpointing 1
```

### SFT

```bash
cd trainer
torchrun --nproc_per_node 8 train_full_sft.py \
  --data_path ../data/sft.jsonl \
  --max_seq_len 32768 \
  --gradient_checkpointing 1
```

### BF16 default / FP8 mixed

BF16 is the default:

```bash
python train_full_sft.py --dtype bfloat16
```

FP8 + BF16 mixed training:

```bash
python train_full_sft.py --dtype fp8 --gradient_checkpointing 1
```

In FP8 mode, eligible Linear layers are converted by `torchao.float8.convert_to_float8_training`; autocast and the surrounding activations remain BF16. FP8 is optional and should be benchmarked against BF16 for both throughput and convergence.

### Other Stages

```bash
# LoRA
python train_lora.py --dtype bfloat16

# DPO
python train_dpo.py --dtype bfloat16

# PPO (actor + critic + reference + reward model)
python train_ppo.py --dtype bfloat16

# GRPO or CISPO
python train_grpo.py --dtype bfloat16 --loss_type cispo

# Multi-turn tool/agent RL
python train_agent.py --dtype bfloat16

# Distillation
python train_distillation.py --dtype bfloat16
```

For RL, `--rollout_engine torch` is the simplest path. `--rollout_engine sglang` requires a running SGLang server and shared checkpoint directory.

## Environment Troubleshooting

`ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'`
means the installed `torchao` is newer than the installed PyTorch; it is not a
WH model error. Install the project environment first:

```bash
pip install -r requirements.txt
```

If FP8 is not needed, remove the optional package with `pip uninstall -y torchao`.
If FP8 is needed, install a CUDA-matched PyTorch 2.9+ build first, then run
`pip install -r requirements-fp8.txt`.

If FP8 reports `Output 0 of matmul_with_hp_or_float8_args is a view and is
being modified inplace`, the current FP8 path clones both custom forward and
backward outputs and keeps tied/output-head layers in BF16. If a custom
`torchao` build still fails, use `--dtype bfloat16` and report the exact
torch/torchao versions.

FP8 also enables torchao `pad_inner_dim` because dynamic batch-token counts are
often not divisible by 16. Padding avoids `torch._scaled_mm` shape failures and
may increase peak temporary FP8 memory.

## Data Formats

Pretrain:

```json
{"text": "text for next-token prediction"}
```

SFT / RLAIF:

```json
{"conversations": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

DPO:

```json
{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Agent RL additionally uses a `gt` field and tool definitions. The chat template supports `tools`, `tool_calls`, `<tool_call>`, `<tool_response>`, ` thinking`, and `open_thinking`.

## Inference

```bash
# Native PyTorch weights
python eval_llm.py --load_from ./model --weight full_sft --max_new_tokens 4096

# Transformers-format export
python scripts/convert_model.py
python eval_llm.py --load_from ./export/whllm
```

WH exports include `auto_map` metadata. When loading an exported checkpoint from
another process without importing the local `model` package first, pass
`trust_remote_code=True` to `AutoConfig`, `AutoModelForCausalLM`, and
`AutoTokenizer`.

OpenAI-compatible API:

```bash
cd scripts
python serve_openai_api.py --load_from ../model --weight full_sft
```

The API exposes `/v1/chat/completions`, including streaming content, `reasoning_content`, and tool calls.

Streamlit UI (the script scans its own directory for exported models):

```bash
cd scripts
cp -r ../export/whllm ./whllm-export
streamlit run web_demo.py
```

## 32k Context Notes

- `max_position_embeddings` is 32768.
- Generation is capped so prompt tokens plus generated tokens do not exceed 32768.
- Training defaults enable activation gradient checkpointing; disable it with `--gradient_checkpointing 0` when you have enough VRAM.
- Long-context training still requires careful memory planning. This scaffold provides DDP, gradient accumulation, and checkpointing, but does not claim production-grade sequence parallelism or asynchronous RL.
- The included tokenizer is derived from the reference project and has a 6400-token vocabulary; a different tokenizer requires matching embedding weights and retraining.

## Tests

```bash
pytest
```

The tests are designed to be run in an environment with PyTorch and Transformers installed. Small-test configurations avoid requiring a GPU or downloaded weights. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the evidence collected in this workspace and its remaining gaps.

## Attribution

The initial full-stack structure was inspired by the open-source **MiniMind** project. The repository retains its Apache-2.0 license and attribution; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
