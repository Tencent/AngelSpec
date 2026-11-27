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
  usp_local_shard: true
  fsdp_strategy: FULL_SHARD
  hsdp_shard_dp: 2
  hsdp_shard_sp: 8
  micro_batch_size: 1     # required under USP
```

- `sp_ulysses_size` — the number of ranks a single sequence is split across.
- `attention_backend: usp` — uses the FlexAttention-based long-sequence path with sequence
  parallelism.
- `usp_local_shard: true` — required for USP; each SP rank receives the full sample and
  slices in-forward.
- `fsdp_strategy: FULL_SHARD` — enables native FSDP2 HSDP with USP ranks replicated and data
  parallel ranks sharded.
- `hsdp_shard_dp` — number of data-parallel ranks included in each HSDP shard group.
- `hsdp_shard_sp` — number of sequence-parallel ranks folded into each HSDP shard group.
- `hsdp_shard_size` remains available as a legacy shorthand.

## SP-leader Mooncake fetching

With `usp_local_shard: true`, `sp_leader_mooncake_fetch: true` makes only SP rank 0 read each
sample from Mooncake and send it to its SP group — the full sample for MTP, each rank's sequence
shard for the DFlash family — cutting Mooncake traffic by about the SP size. It suits high-SP
long-context runs. With `prefetch_depth > 0` the leader reads ahead
into host memory: about 9 GiB for 256k-token Hy4 flash MTP samples at `prefetch_depth: 6`.

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
- DFlash-family USP also requires `usp_local_shard: true` and `sp_ring_size: 1`.
- USP requires `micro_batch_size: 1`; the controller fans each sample to the SP group.
- Both MTP and the DFlash family can combine USP with [sequence packing](sequence_packing.md).
- Long-sequence runs are sensitive to memory settings; start from a working example config and
  scale sequence length gradually.
