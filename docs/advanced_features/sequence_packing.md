# Sequence Packing

When a dataset mixes short and long samples, padding each micro-batch to its longest sequence
wastes compute on padding tokens. **Sequence packing** (available for DFlash and MTP) packs
multiple short samples into one fixed-length sequence, with document-aware masking so packed
documents do not attend across each other.

## What it does

Multiple samples are concatenated into a single `max_seq_length` row. A document-aware block mask
and document-local RoPE keep each packed document independent — attention never crosses a document
boundary, and positions restart per document. The result is far less padding and higher effective
throughput.

Packing changes the training recipe (fixed per-sequence anchor budget, token-weighted sample
importance), so loss and accuracy curves are **not directly comparable** to non-packed runs; the
benefit is efficiency (throughput and padding), not convergence quality. Recalibrate the LR and
step schedule when switching packing on.

## Configuration

```yaml
training:
  dflash_packing: true
  prefetch_depth: 0            # required: packing needs per-step dispatch logic
  max_seq_length: 4096         # per-row token budget
  draft_accumulation_steps: 32 # also the number of packed rows per rank per step
```

Optionally, `dflash_packing_token_weighted_loss: true` weights each packed row by its
supervised-token count instead of equally (this materializes all rows for the step first).

By default the controller dispatches partial rows as soon as it has one non-empty row per rank.
For inference-bound workloads, an optional fill guard can wait briefly for denser rows:

```yaml
training:
  packing_min_fill_ratio: 0.8
  packing_max_wait_seconds: 5.0
```

The threshold applies to the total tokens across the candidate dispatch, not to each individual
row. If the timeout expires, AngelSpec flushes the partial rows with a warning instead of waiting
forever. The default `packing_min_fill_ratio: 0.0` preserves immediate dispatch.

For MTP, use `mtp_packing: true`. MTP packing can be combined with USP when every SP rank receives
the full packed row and slices it locally:

```yaml
training:
  mtp_packing: true
  attention_backend: usp
  usp_local_shard: true
  sp_ulysses_size: 4
  sp_ring_size: 1
```

## How it stays deadlock-free

Packing is done in the controller, not the trainer. The controller packs samples into rows and
guarantees that **every data-parallel rank receives the same number of rows each step**. Because
FSDP's per-forward all-gather and per-step gradient reduction then happen the same number of times
on every rank, collective communication stays aligned and cannot deadlock. The trainer only
reassembles the rows the controller dispatched; it does not pack.

## Limitations

- DFlash packing is mutually exclusive with [USP](long_sequence_usp.md).
- MTP packing supports USP only with `usp_local_shard: true`; USP sharded-data pre-sharding is
  incompatible with packing.
- Every input sample must fit within `max_seq_length`. AngelSpec fails fast instead of silently
  dropping an oversized sample or waiting forever for an impossible row.
