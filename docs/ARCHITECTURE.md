# Architecture

## Dense 0.6B Decoder

The default blueprint is a decoder-only Transformer:

- Pre-Norm blocks: RMSNorm -> attention -> residual -> RMSNorm -> FFN -> residual
- Grouped-query attention: 16 query heads, 4 key/value heads, head dimension 80
- QK-Norm applied to each query/key head
- SwiGLU gated FFN with intermediate size 3744
- RoPE with theta 1e6 and optional YaRN scaling
- Tied input/output embeddings with vocabulary size 6400
- 32 layers and 32,768 position embeddings
- Optional top-k MoE replacing the dense FFN: 8 experts, top-3 routing,
  per-expert intermediate size 464. This keeps total parameters near 0.6B;
  only about 310.7M parameters are activated per token.

## Optimization

All training entry points use `trainer.optim.build_muon_optimizer`. Muon handles
two-dimensional hidden-layer matrices with `adjust_lr_fn="match_rms_adamw"` so the
existing learning-rate schedules remain applicable. Embedding tables, output heads,
router gates, normalization weights, biases, and other non-matrix parameters stay on
AdamW. A versioned wrapper preserves scheduler, GradScaler, and checkpoint protocol
compatibility while keeping the two update rules separate.

## Precision Modes

- `bfloat16`: BF16 autocast without gradient scaling.
- `float16`: FP16 autocast with `GradScaler`.
- `fp8`: `torchao.float8.convert_to_float8_training` converts eligible Linear modules to FP8; surrounding autocast is BF16.

FP8 is intentionally opt-in. The project does not silently downgrade a requested FP8 run to BF16.
Output heads, tied embedding weights, MoE router gates, and non-16-aligned matrices
remain in BF16. This avoids the tied-weight form of torchao's FP8 gradient-view
limitation and keeps small routing/output layers in a numerically safer path.
When FP8 is enabled, a compatibility shim clones both torchao custom-matmul
forward outputs and backward gradients before any in-place accumulation,
clipping, or optimizer update. `pad_inner_dim=True` automatically pads dynamic
token dimensions to the multiple-of-16 requirement of `torch._scaled_mm`.

## Long Context

`WhForCausalLM.generate` caps prompt plus new tokens at `max_position_embeddings`. Training scripts expose `--gradient_checkpointing` and enable it by default to reduce activation memory for 32k sequences. Attention uses PyTorch SDPA when no padding mask is supplied; padded paths fall back to the explicit attention-mask implementation.

## Extension Points

- Replace `WhAttention` with MLA or another attention implementation.
- Replace the top-k router with auxiliary-loss-free routing or expert parallel dispatch.
- Add FSDP/sequence parallelism for larger multi-node runs.
- Replace the mock reward model or mock tools with real services.
