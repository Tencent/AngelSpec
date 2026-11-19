# Memory and throughput options

These options are off by default.

## Activation checkpointing

`training.gradient_checkpointing=true` recomputes each DFlash-family decoder layer during
backward instead of keeping its activations (DSpark and DFly2: offline training only). MTP has
its own checkpointing.

## Chunked losses and projections

| Option | Drafters | Effect |
|---|---|---|
| `training.dflash_loss_checkpoint_chunks` + `ANGELSPEC_DFLASH_LOSS_CHUNK` | DFlash family | Recomputes the LM-head projection and loss per group of draft blocks in backward, instead of keeping vocabulary-sized logits |
| `training.mtp_loss_checkpoint_chunks` + `ANGELSPEC_MTP_LOSS_CHUNK` | MTP | The same for MTP's projection and loss |
| `training.mtp_argmax_chunk_size` | MTP | Projects at most this many rows to the vocabulary at once for on-policy predictions |
| `training.mtp_alignment_chunk_size` | MTP | Normalizes the shifted teacher and draft hidden states in token chunks |
| `training.dfly_context_chunk_size` / `training.dfly_context_checkpoint` | DFly | Runs the context projection and fusion in token chunks, and recomputes them in backward |

Chunk sizes go through `training.train_env_vars`:

```yaml
training:
  dflash_loss_checkpoint_chunks: true
  train_env_vars: '{"ANGELSPEC_DFLASH_LOSS_CHUNK": "128"}'
```

Chunking changes GEMM shapes, so floating-point rounding can differ.
`dfly_context_checkpoint` without chunks leaves training bitwise unchanged;
`configs/vllm_hy_v4_flash_dfly_256k.yaml` enables it.

## Data-parallel balance

`training.packing_balance_dp: true` assigns online packed rows to DP ranks by token count
within each optimizer step, so the ranks carry similar loads.

## Inference request length

`inference.vllm.max_model_len` sets the length of the target's requests independently of
`training.max_seq_length`, the packed row length. Packing documents into a long row does not
mean the target can serve one document that long.
