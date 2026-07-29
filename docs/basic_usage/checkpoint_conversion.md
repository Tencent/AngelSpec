# Checkpoint Conversion

Training checkpoints are saved in a sharded (FSDP2 / DCP) format. To serve a draft model, convert
it to HuggingFace format.

## FSDP → HuggingFace

```bash
python tools/convert_to_hf.py --input-dir ./outputs/my_run/iter_0010000/
```

This produces a standard HuggingFace directory that can be loaded by a serving engine. The draft
architecture is read from its config, so DFlash, DFlare, DSpark, and Eagle3 checkpoints all
convert with the same command (their extra modules — for example the fusion weights or prediction
heads — are preserved).

MTP checkpoints use a dedicated converter that unfuses the expert weights and writes the MTP
layer where the serving engine expects it; see [MTP](../concepts/mtp.md).

## Vocabulary pruning

Vocabulary pruning shrinks the draft model's `lm_head` to a smaller token set and emits the
`d2t` / `t2d` mappings between draft and target vocabularies. It can be applied two ways:

- **Pre-pruning** — set `draft_vocab_size` in the training config. The checkpoint already contains
  the pruned `lm_head` and the mappings, so the basic conversion command is enough.
- **Post-pruning** — train with the full vocabulary, then pass `--prune-vocab` at conversion time
  along with a representative dataset to compute token frequencies:

```bash
python tools/convert_to_hf.py \
    --input-dir ./outputs/my_run/iter_0010000/ \
    --prune-vocab \
    --dataset-path <dataset> \
    --draft-vocab-size 32000 \
    --tokenizer Qwen/Qwen3-8B \
    --chat-template qwen \
    --prompt-key conversations
```

Pass `--cache-dir ./cache` to reuse the tokenized dataset cache from training.
