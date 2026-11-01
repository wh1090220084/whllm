# Validation Record

## Environment Used

- Python 3.12.12
- PyTorch 2.13.0+cu126
- Transformers 5.5.0
- CUDA available
- Local GPU compute capability: 8.6
- `torchao`: not installed
- `pytest`: not installed in this environment

The project metadata targets `transformers>=4.51,<5`; the available local environment uses Transformers 5.5.0. The tested custom model APIs remained compatible, but production runs should pin a tested Transformers version.

## Passed Checks

### Static Compilation

```bash
python -m compileall -q .
```

Passed for the full project.

### Model Smoke Test

- Dense default config parameter count: **599.415M**
- Tiny forward/loss/backward test: passed
- Generation context cap: passed
- Gradient checkpointing backward: passed
- GQA head validation: passed

### Dataset Smoke Test

Temporary JSONL files were loaded through:

- `PretrainDataset`
- `SFTDataset`
- `DPODataset`

Multiple JSONL files under one directory are discovered recursively, sorted, and
concatenated. A temporary two-file directory was loaded into one `PretrainDataset`.
SFT shards using either `messages` or `conversations`, including files with both and
extra columns, were normalized into one conversation dataset. The real
`data/pretrain` directory currently contains three JSONL shards; the large shards
were enumerated but not fully loaded during validation.

Tokenizer loaded successfully with 6400 tokens; BOS/EOS/PAD are 1/2/0.

### Training Loop Smoke Tests

The following completed one CPU step with a reduced 64-hidden/2-layer model:

- `train_pretrain.py`
- `train_full_sft.py`
- `train_lora.py`
- `train_dpo.py`
- `train_distillation.py`

### Export and Inference Smoke Tests

- `WhForCausalLM.save_pretrained` / `AutoModelForCausalLM.from_pretrained` round-trip: passed
- Dense WH -> Qwen3-compatible export: passed for a tiny test model
- OpenAI-compatible export directory creation: passed
- `eval_llm.py` with a tiny exported model and the WH streaming path: passed

### CLI Import Checks

`--help` checks passed for all pretraining, SFT, LoRA, DPO, PPO, GRPO, Agent RL, distillation, inference, conversion, tool-call evaluation, and API entry points.

### Muon Optimizer Migration (2026-10-29)

- Shared `MuonAdamW` parameter partition: passed on a toy model; hidden matrices
  use Muon, while embeddings, output heads, router gates, norms, and non-matrix
  parameters use AdamW.
- One-step update with `CosineAnnealingLR`: passed.
- Optimizer state save/load round-trip: passed.
- Legacy AdamW-only optimizer state rejection: passed with a clear error.
- CPU `GradScaler` unscale/step/update protocol: passed.
- Static compilation and imports of all eight training entry points: passed.
- Full GPU/FP8 training and convergence comparison: not run in this workspace.
- Default MoE profile: 32 layers, hidden 1280, 8 experts, top-3, expert width
  464, vocabulary 6400 gives ~595.8M total and ~310.7M active parameters.
- Dense profile: 32 layers with the same vocabulary gives ~599.4M.
- FP8 conversion filter: inspected against torchao 0.15.0 and 0.18.0 source;
  tied output heads and router/small matrices are excluded from Float8Linear.
- FP8 autograd view shim: reproduced the exact forward view+inplace error with
  torchao 0.18.0 CPU emulation; cloning forward and backward outputs fixed it.
- FP8 dynamic-shape guard: `pad_inner_dim=True` is passed to torchao so
  non-16-aligned token counts do not fail in `torch._scaled_mm` backward.

## Not Yet Proven

- Full 0.6B training on GPU was not executed in this workspace.
- FP8 numerical convergence and throughput were not benchmarked because `torchao` and an FP8-capable compute capability were unavailable locally.
- PPO/GRPO/Agent RL were not run end-to-end because they require the external reward model and suitable GPU memory.
- No full benchmark or production load test was performed.
- The repository test suite is written for `pytest`, but pytest was not installed in the validation environment; direct smoke tests were used instead.
