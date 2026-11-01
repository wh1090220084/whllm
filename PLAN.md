# Implementation Plan

## Authorization and Target

- User requested a new project under `D:\研究项目` whose name starts with `wh`.
- Target: `D:\研究项目\whllm`.
- The existing upstream reference project is only a read-only source baseline; no weights or large datasets are copied.

## Scope

1. Define a ~0.6B dense decoder with 32k context and modern components: GQA, QK-Norm, RMSNorm, SwiGLU, RoPE/YaRN, optional MoE.
2. Preserve the reference directory and training-script structure.
3. Provide Pretrain, SFT, LoRA, DPO, PPO, GRPO/CISPO, Agent RL, distillation, rollout, inference, API, WebUI, conversion, and ToolCall evaluation entry points.
4. Add BF16/FP16/FP8 precision planning, optional `torchao.float8`, and explicit failure when FP8 is unavailable.
5. Add gradient checkpointing and context caps for 32k-context training.
6. Provide config files, tests, and operator documentation.

## Exclusions

- No actual pretrained weights, 0.6B checkpoint, or large dataset is bundled.
- No claim of exact parity with non-public Qwen3.8/DeepSeekV4.1 architectures.
- No production deployment hardening, authentication, multi-tenant serving, or asynchronous RL buffer.
- No full-scale multi-node 32k benchmark is run in this workspace.

## Acceptance Criteria

- All Python files compile under the available interpreter.
- Config parameter count is approximately 0.6B (expected ~599.4M).
- A small CPU/GPU model smoke test can forward and produce logits.
- Pretrain/SFT/LoRA/DPO/PPO/GRPO/Agent RL scripts expose `--dtype fp8` and `--gradient_checkpointing`.
- FP8 request fails clearly when dependencies/hardware are unavailable.
- `pytest` passes in a dependency-complete environment.

## Risks and Rollback

- 32k full training is memory intensive and requires tuning of batch size, attention backend, and checkpointing.
- FP8 numerics and kernel support vary by GPU/driver/torchao version.
- Remote reward-model code may need Transformers compatibility patches; the supplied `trainer/trainer_utils.py` includes the known InternLM2/rope/cache guards.
- Rollback is deletion of the new project directory; no existing upstream reference files are modified.


## Muon Optimizer Migration (2026-10-29)

### Authorization and Target

- User approved replacing the training optimizer with Muon on 2026-10-29.
- Target: all optimizer construction sites under `trainer/` and the shared optimizer helper.
- Model weights, datasets, checkpoint model format, and inference paths are out of scope.

### Scope

1. Add a shared hybrid optimizer: Muon for two-dimensional hidden-layer matrices and AdamW for embeddings, output heads, router gates, norms, biases, and other non-matrix parameters.
2. Use Muon's `adjust_lr_fn="match_rms_adamw"` so existing learning-rate schedules remain meaningful.
3. Replace AdamW construction in Pretrain, Full SFT, LoRA, DPO, Distillation, PPO actor/critic, GRPO, and Agent RL entry points.
4. Preserve scheduler, gradient-scaler, checkpoint, DDP, and FP8 call paths through a scheduler- and scaler-compatible optimizer wrapper.

### Assumptions

- Training requires PyTorch with native `torch.optim.Muon`; the dependency floor will be raised accordingly.
- Existing AdamW-tuned learning rates remain the initial values. Any full-scale convergence change must be measured separately.
- Old checkpoint optimizer states are not resume-compatible because the parameter-group layout and update rule change.

### Acceptance Criteria

- Every training entry constructs its optimizer through the shared Muon helper.
- No trainable parameter is assigned to both Muon and AdamW, and every trainable parameter is assigned exactly once.
- Muon receives only eligible 2D matrices; excluded matrix parameters remain on AdamW.
- A CPU smoke test performs one update, exercises `CosineAnnealingLR`, and round-trips optimizer state.
- All Python files compile, and available tests pass.

### Risks and Rollback

- Muon may change convergence speed and stability; it requires a controlled training comparison before any quality claim.
- The wrapper must remain compatible with GradScaler and FP8; focused tests cover its optimizer protocol, while full GPU training is not run in this workspace.
- Rollback is restoring the previous AdamW construction lines and dependency floor; saved model weights remain usable.
- Rollback owner: project maintainer.


## 0.6B MoE Design (2026-10-30)

### Authorization and Target

- User requested a 0.6B-total MoE with 8 experts and top-3 activation.
- Target: default MoE configuration and architecture documentation.

### Design

- Keep the 1280 hidden size and 32 decoder layers.
- Use 8 independent SwiGLU experts per layer and activate 3 experts per token.
- Set each expert intermediate size to 464 to fit the total parameter budget.
- Keep the router gate and dense attention/embedding parameters unchanged.

### Acceptance Criteria

- Dense parameter count remains approximately 599.4M.
- MoE total parameter count remains within 590M-610M.
- MoE config explicitly records 8 experts and top-3 routing.
- A tiny MoE forward/backward smoke test passes with top-3 expert selection.
- Documentation distinguishes total parameters from activated parameters.

### Risks and Rollback

- The 0.6B total budget reduces each expert's width; quality must be validated by training, not inferred from parameter count.
- Top-3 increases active compute versus top-1 but is still sparse versus running all 8 experts.
- Rollback restores the prior MoE width and expert count in the config defaults.


## DeepSeek Tokenizer + 0.6B MoE Migration (2026-10-30)

### Authorization and Target

- User requested implementation of the analyzed DeepSeek-tokenizer design.
- Target: model configs, default model profile, training defaults, tests, and docs.

### Selected Design

- Vocabulary: 129280; BOS/EOS/PAD IDs: 0/1/2.
- Default MoE profile: 30 layers, hidden 1280, 8 experts, top-3, expert width 336.
- Expected total parameters: approximately 598.4M.
- Dense compatibility profile: 24 layers with the same vocabulary, approximately 608.9M.

### Exclusions

- Do not copy DeepSeek V4.1's 40-layer/384-expert architecture into WHLLM.
- Do not invent a chat template that was not supplied with the tokenizer.
- Old 6400-vocabulary checkpoints and optimizer states are not resume-compatible.

### Validation

- Parse and load the new tokenizer.
- Verify vocabulary size, BOS/EOS/PAD IDs, context length, and EOS label preservation.
- Verify dense and MoE parameter counts on meta device.
- Compile and import affected model/training/inference modules.


### Current Tokenizer Decision (2026-10-30)

- The DeepSeek-tokenizer experiment was reverted.
- Active profile: 6400-token vocabulary with BOS/EOS/PAD 1/2/0.
- Default MoE remains 32 layers, 8 experts, top-3, expert width 464.

