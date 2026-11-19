# Offline training

Offline training reads target hidden states from disk instead of running the target, so a
captured dataset can train many drafters. It is on when `offline.manifest_path` is set;
online training is unchanged otherwise.

## Supported drafters

| Draft | Offline training | Packing | SP |
|---|---|---|---|
| DFlash (including gated-sum layer selection) | Yes | Yes | SP1 / Ulysses |
| DFly | Yes | Yes | SP1 / Ulysses |
| DFlare | Yes, GQA only | Yes | SP1 / Ulysses |
| DSpark | Yes | Yes | SP1 / Ulysses |
| DFly2 | Yes; the frozen bundle carries the target layers it reads | Yes | SP1 |
| MTP (text) | Yes | `training.mtp_packing` | SP1 / Ulysses |
| Eagle3 | No | — | — |

Offline training uses micro-batch 1, FULL_SHARD and at least two training GPUs. OPD and
exact training resume (`training.load_path`) are rejected. Metrics go to a local JSONL file
plus the usual W&B, SwanLab and TensorBoard flags.

## Capturing hidden states

Both ways below write the same dump format.

**While training online.** Add `online_dump` to an online recipe; the run writes every
sample it trains on. Capture is lossless: a slow writer throttles dispatch, and a capture
failure fails the run.

```bash
python3 -m angelspec.train_entry \
  --config configs/vllm_hy_v4_flash_mtp_256k.yaml \
  online_dump.enabled=true \
  online_dump.dump_path=/path/to/hidden \
  online_dump.frozen_target_path=/path/to/frozen \
  inference.last_hidden_states_prenorm=true
```

**Capture only.** Add `online_dump.dump_only=true` to the same command: the run starts the
inference engines but no trainer, and passes over the dataset `online_dump.capture_epochs`
times.

| Option | Default | Meaning |
|---|---|---|
| `online_dump.capture_epochs` | 1 | Passes over the dataset; repeated samples are deduplicated |
| `online_dump.validation_fraction` | 0.05 | Share of samples put in the validation split |
| `online_dump.max_pending_bytes` | 32 GiB | Copied samples waiting for a file writer |
| `online_dump.reader_threads` / `writer_threads` | 8 / 16 | Mooncake readers and file writers per writer actor |
| `online_dump.writers` | one per inference node | Writer actors |
| `online_dump.writer_node_ip` | — | Run the writers on one CPU node instead |
| `online_dump.verify_on_publish` | false | Rehash every file before publishing |

Capture needs pre-tokenized input, `train_with_decode=false`, `usp_local_shard=true` under
USP and pre-final-norm last hidden states. Image and video capture needs vLLM and a draft
config with the Hy4 flash position recipe; it stores the metadata that rebuilds M-RoPE
positions offline, not the media. Samples below `dataset.min_loss_tokens` are not written.

A run publishes `manifest.json` when it ends cleanly. An interrupted run stays
unpublished; inspect it, then publish it explicitly:

```bash
python3 -m angelspec.offline_entry status --manifest /path/to/hidden --verify
python3 -m angelspec.offline_entry finalize --manifest /path/to/hidden
```

## Frozen target assets

Offline training needs the target's embedding, LM head and final norm, plus the target
layers a drafter reads: DFly2's attention and DSA indexer layers, the MTP fusion layer with
`mtp_fusion_init: target`, or the layer an MTP drafter initializes from. Capture exports
this bundle to `online_dump.frozen_target_path` from the draft config. For other data,
export it yourself with the draft config you will train:

```bash
python3 -m angelspec.offline_entry export-assets \
  --config configs/offline_dfly_hy_v4_flash_smoke.yaml \
  model.target_model_path=/path/to/hy-v4-flash \
  offline.frozen_target_path=/path/to/frozen
```

Training refuses a bundle that lacks the layers its drafter reads; export a new bundle
rather than editing one.

## Training from a dump

```bash
export OFFLINE_HIDDEN_PATH=/path/to/hidden
export OFFLINE_FROZEN_TARGET_PATH=/path/to/frozen
export OFFLINE_OUTPUT_DIR=/path/to/output

python3 -m angelspec.train_entry --config configs/offline_dfly_hy_v4_flash_smoke.yaml
```

`configs/offline_mtp_hy_v4_flash_smoke.yaml` and `configs/offline_dfly_qwen3_8b_smoke.yaml`
work the same way. DFly2 trains offline from its online recipe:

```bash
python3 -m angelspec.train_entry \
  --config configs/vllm_hy_v4_flash_dfly2_256k.yaml \
  offline.manifest_path="$OFFLINE_HIDDEN_PATH" \
  offline.frozen_target_path="$OFFLINE_FROZEN_TARGET_PATH" \
  output_dir="$OFFLINE_OUTPUT_DIR"
```

| Option | Meaning |
|---|---|
| `offline.format` | `native` (default) or `dfly` for an imported DFly dump |
| `offline.eval_samples` | Validation samples evaluated at each eval; 0 when the dump has none |
| `offline.verify_hashes` | Hash each file on read (default true) |
| `offline.row_plan_path` | Replay a fixed row order (below) |
| `training.prefetch_depth` | Rows read ahead on CPU; 0 disables prefetch |

A dump can train other drafters of the same target as long as it holds the layers they
read. The run writes `offline_run.json`, `offline_metrics.jsonl`, checkpoints and
`draft_hf/`.

### Packing

`training.dflash_packing=true` (`training.mtp_packing=true` for MTP) packs samples into
`max_seq_length` rows without splitting them; a sample longer than a row is an error. The
global batch then counts rows, and `dflash_num_anchors` applies per row.
`training.packing_min_fill_ratio` must be 0 offline.

### Sequence parallelism

For example, the eight-GPU smoke recipe at SP2 keeps a global batch of 8 rows with:

```text
training.attention_backend=usp
training.sp_ulysses_size=2
training.sp_ring_size=1
training.usp_local_shard=true
training.sp_leader_mooncake_fetch=false
training.hsdp_shard_dp=4
training.hsdp_shard_sp=2
training.draft_accumulation_steps=2
training.dflash_packing=true
```

SP must divide the training GPU count and the anchor count, and the row length must be
divisible by SP. Ring SP is not supported.

### Replaying the online order

Each record keeps its `provenance`, including the dispatch, DP rank and packed row it
trained in. To replay the online step order, write `{"rows": [[sample_id, ...], ...]}` in
dispatch order and set `offline.row_plan_path` with `dataset.shuffle_dataset=false`. A row
plan may only reference training-split samples, each once.

## MTP

MTP capture stores only the final decoder layer, and a DFlash-family dump can also train
MTP. MTP needs the real target final norm, and under USP `usp_local_shard=true`. With
`mtp_init_from_mtp_layer` or `mtp_init_from_target_layer`, export the bundle with the same
MTP config so it holds that layer.

## Importing a DFly dump

```bash
python3 -m angelspec.offline_entry verify \
  --manifest "$OFFLINE_HIDDEN_PATH" --format dfly \
  --tokenized-path "$OFFLINE_TOKENIZED_PATH" \
  --hidden-size 3072 --context-layers 2 --vocab-size 120832

python3 -m angelspec.train_entry \
  --config configs/offline_dfly_hy_v4_flash_smoke.yaml \
  offline.format=dfly offline.tokenized_path="$OFFLINE_TOKENIZED_PATH"
```

## Dump format

A native dump (`angelspec_hidden_v2`) holds one safetensors file per sample with
`input_ids`, `loss_mask` and `hidden_layer_{i}` for each captured decoder layer `i` (its
output before the final norm), plus per-sample receipts and an immutable `manifest.json`
that records the capture recipe and checksums. Drafters select layers by ID, so a dump may
hold more layers than a drafter reads. `angelspec.data.offline.HiddenWriter` writes the
same format from your own pipeline:

```python
from angelspec.data.offline import HiddenWriter, layer_recipe

recipe = {
    **layer_recipe([39, 40, 41], target_num_hidden_layers=42, hidden_size=3072),
    "target_metadata_sha256": source_target_identity,
    "vocab_size": 120832,
    "storage_dtype": "bfloat16",
    "tokenizer": tokenizer_identity,
    "mask_policy": mask_policy_identity,
}
writer = HiddenWriter(output_dir, recipe)
writer.write(sample_id, tensors, split="train")
writer.publish(expected_sample_ids)
```
