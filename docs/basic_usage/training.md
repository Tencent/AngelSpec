# Training

## Launching a run

Training is launched through the `angelspec.train_entry` module, driven by a YAML config. The
example scripts wrap this call:

```bash
./examples/qwen3-8b-dfly/run.sh
```

Under the hood each script runs something like:

```bash
python3 -m angelspec.train_entry --config configs/vllm_qwen3_8b_dfly.yaml
```

Any config value can be overridden on the command line using dotted keys:

```bash
python3 -m angelspec.train_entry --config configs/sglang_qwen3_8b.yaml \
    training.learning_rate=5e-5 training.num_train_steps=500
```

Multiple `--config` files merge left to right, and CLI overrides win over both. The fully
resolved config is saved to `output_dir/config.yaml` for reproducibility.

## Config at a glance

A config is grouped into sections; the full field list and defaults live in
`angelspec/config/train_config.py`. The essentials:

```yaml
dataset:
  chat_template: qwen3
  train_data_path: /path/to/train.jsonl
  max_seq_length: 8192

model:
  target_model_path: Qwen/Qwen3-8B
  target_model_backend: vllm             # vllm | sglang | hf
  draft_model_config: angelspec/config/dfly_qwen3_8b_draft_config.json

training:
  num_epochs: 1
  micro_batch_size: 2
  learning_rate: 1e-4
  training_num_gpus_per_node: 2

inference:
  inference_num_gpus: 2

logging:
  report_to: wandb                      # none | wandb | tensorboard
```

The draft architecture is chosen by `model.draft_model_config` (see
[The Draft-Model Family](../concepts/draft_model_family.md)).

## Resume vs. continual training

Both modes use `training.load_path`, but restore different state:

| Goal | `load_path` | `continual_training` | Restored |
|------|-------------|----------------------|----------|
| Resume an interrupted run | required | `false` (default) | model, optimizer, LR scheduler, RNG, step |
| Start a new run from existing weights | required | `true` | model weights only |

Resume the same run (keep the same `output_dir`):

```yaml
training:
  load_path: /path/to/run/checkpoints
output_dir: /path/to/run
```

Start a fresh run from existing weights:

```yaml
training:
  load_path: /path/to/old_run/checkpoints
  continual_training: true
  learning_rate: 1e-5
output_dir: /path/to/new_run
```

## Learning-rate schedule

The default schedule is Warmup-Stable-Decay (WSD): a warmup phase, a stable phase, and a decay
phase. Key knobs: `learning_rate`, `warmup_ratio`, `wsd_decay_ratio`, `wsd_decay_style`, and
`min_lr`. Eagle3 runs use a cosine schedule by default.

## Optimizer

AdamW (with fp32 master weights) is the default. Setting `optimizer_type: muon` uses
Momentum-Orthogonalized Newton-Schulz updates for the 2-D weight matrices while keeping norms,
embeddings, and the LM head on AdamW. This is available for the DFlash family.

## Checkpointing

- `save_interval` — steps between checkpoints; `save_per_epoch` — also save each epoch.
- `max_checkpoints` — keep only the *N* most recent (`0` keeps all).
- Checkpoints are written under `output_dir/checkpoints`. Convert them for serving with
  [Checkpoint Conversion](../basic_usage/checkpoint_conversion.md).

## Evaluation

Set `dataset.eval_data_path` and `eval_interval` to run periodic evaluation. AngelSpec can also
run **online evaluation** — periodically serving the current draft in a spec-decode engine and
logging the real acceptance length/rate — via the `online_eval` config section.

## Experiment tracking

Logging is off by default (`report_to: none`). Set `report_to: wandb` (and provide an API key)
or `report_to: tensorboard` to enable it.
