# Long-Sequence Training (USP)

Training on very long contexts (128k tokens and beyond) exceeds the memory of a single GPU for
both activations and attention. AngelSpec supports **Ulysses sequence parallelism (USP)** to shard
a long sequence across GPUs so each holds only a slice.

## When to use it

Reach for USP when `max_seq_length` is large enough that a single sequence does not fit on one GPU.
For moderate lengths, standard data-parallel training is simpler and faster.

## Configuration

USP is enabled through the sequence-parallel and attention-backend settings:

```yaml
training:
  attention_backend: usp
  sp_ulysses_size: 8      # shard each sequence across 8 ranks
  sp_ring_size: 1
  micro_batch_size: 1
```

- `sp_ulysses_size` — the number of ranks a single sequence is split across.
- `attention_backend: usp` — uses the FlexAttention-based long-sequence path with sequence
  parallelism.

## How it works

Each sequence-parallel rank holds one contiguous slice of the sequence. Attention is computed with
sequence parallelism so that every rank still attends over the full context, while activations and
the attention computation are distributed across ranks. This keeps per-GPU memory bounded as the
sequence grows.

For [MTP](../concepts/mtp.md), the long-sequence attention path runs the first attention block
through compiled FlexAttention (linear rather than quadratic memory in sequence length), with USP
layered on top to shard across GPUs. Full-vocabulary loss is computed in chunks to keep the
logit projection within memory.

## Notes

- MTP requires `usp_local_shard: true` for all USP runs because it shards the full sequence inside
  the model forward. MTP currently supports pure Ulysses only (`sp_ring_size: 1`).
- MTP can combine USP with [sequence packing](sequence_packing.md). DFlash packing and USP remain
  mutually exclusive.
- Long-sequence runs are sensitive to memory settings; start from a working example config and
  scale sequence length gradually.
