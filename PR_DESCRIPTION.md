# [Refactor] Promote DFly to a first-class DFlash-family drafter and add an end-to-end multi-step TV loss

> **Title:** `refactor(drafter): promote DFly to a standalone DFlash-family drafter + add e2e multi-step TV loss`


## Summary

This PR promotes **DFly** from an architecture *variant*
that piggy-backed on the DSpark code path into a standalone, first-class member
of the DFlash drafter family. It gets its own config, model, training wrapper,
and trainer, and no longer depends on DSpark in any way.

Alongside the refactor, this PR:

- Removes the dead `treeflash_dspark_dflare` drafter and its test.
- Slims down DSpark by moving the shared TreeFlash hidden-states correction (and
  related knobs) out of `dspark.py` and into `dfly.py`, where it now belongs.
- Adds an optional **end-to-end multi-step TV loss** (`γ`-step MTP) to the
  DFlash composable loss, replacing the now-unused KL/LK temperature knobs.

The net effect is a cleaner architecture graph (DFly no longer "rides" DSpark),
less coupling between drafters, and a large reduction in DSpark's surface area
(~+367 / −933 lines overall).

## Motivation

Previously, DFly was selected via `DSparkConfig` + `model_arch == "dfly"` and was
dispatched through `DSparkTrainer` / `DSparkModel`. This meant:

- DFly's behavior was implicit and hard to discover (hidden behind a string flag).
- DSpark carried a lot of machinery (hidden-states correction, position-adaptive
  alpha, etc.) that only DFly actually used.
- The `auto` dispatch logic had brittle special-case branches keyed on
  `model_arch` string comparison.

Making DFly a proper config/model/trainer triple removes the cross-routing,
makes the dispatch type-based, and lets each drafter own only what it needs.

## Changes

### New — DFly as a first-class drafter

- **`angelspec/models/dfly.py`** (new): `DFlyModel` training wrapper. Subclasses
  `DFlashModel` and overrides `_compute_draft_logits` to apply the optional
  TreeFlash hidden-states correction (formula (1)) before the LM head.
- **`angelspec/training/dfly_trainer.py`** (new): `DFlyTrainer`, a thin subclass
  of `DFlashTrainer` that only overrides the two model-build seams
  (`_build_draft_model` / `_build_training_wrapper`). Reads the `dflash_*`
  hyperparameter namespace.
- **`angelspec/models/draft/dfly.py`**: now defines its own `DFlyConfig`
  (extends `DFlashConfig`, `model_type = "qwen3"`) and owns the
  `HiddenStatesCorrection` module / `build_hidden_correction` helper (moved here
  from `dspark.py`). `DFlyDraftModel.config_class` is now `DFlyConfig`.

### Dispatch / registration

- **`angelspec/models/draft/auto.py`**: register `DFlyConfig → DFlyDraftModel`
  and architecture `"Qwen3DFlyModel" → DFlyConfig`. Removed the
  `model_arch == "dfly"` and `model_arch == "dflare"` (TreeFlash) special-case
  branches.
- **`angelspec/training/trainer_actor.py`**: add a `DFlyConfig` dispatch branch.
  Since both `DSparkConfig` and `DFlyConfig` subclass `DFlashConfig`, they are
  checked before the `DFlashConfig` branch.
- **`angelspec/models/__init__.py`** / **`angelspec/models/draft/__init__.py`**:
  export `DFlyModel`; drop the `TreeflashDSparkDFlareDraftModel` export.

### DSpark slim-down

- **`angelspec/models/draft/dspark.py`**: removed the hidden-states correction,
  `PositionAdaptiveAlpha`, position-adaptive Markov knobs, and related
  parameters — DSpark now only carries the Markov head and confidence head.
- **`angelspec/models/dspark.py`**: corresponding wrapper cleanup.

### End-to-end multi-step TV loss

- **`angelspec/models/dflash.py`**: add `_compute_e2e_tv_loss`, an independent
  γ-step MTP TV term added on top of the total loss (not mutually exclusive with
  KL/LK), gated on `e2e_tv_loss_weight > 0` and the presence of target
  `last_hidden_states`. Emits `e2e_tv_loss` in `loss_components`.

      L_e2e = 1 - (1/γ) * Σ_{j=1..γ} Π_{i=1..j} α_i

- **`angelspec/config/train_config.py`**: add `dflash_e2e_tv_loss_weight`
  (default `0.0`, disabled) and `DatasetConfig.num_proc` (default `64`); remove
  the now-unused `dflash_kl_temperature`, `dflash_kl_topk_renormalize`, and
  `dflash_lk_temperature`.

### Removals

- **`angelspec/models/draft/treeflash_dspark_dflare.py`** (deleted).
- **`tests/test_treeflash.py`** (deleted).
- **`angelspec/config/dflare_dspark_treeflash_qwen3_4b_draft_config.json`** (deleted).

### Configs

- **`angelspec/config/dfly_*_draft_config.json`**: switch from
  `architectures: ["DSparkDraftModel"]` / `model_type: "dspark"` /
  `model_arch: "dfly"` to `architectures: ["Qwen3DFlyModel"]` /
  `model_type: "qwen3"`; drop the DSpark-only `markov_rank` /
  `enable_confidence_head` / `confidence_head_with_markov` keys.
- **`configs/vllm_qwen3_8b_dfly.yaml`**, **`configs/vllm_hy3_dfly.yaml`**: fix
  `lm_head_key` to `lm_head.weight`, migrate `dspark_*` hyperparameters to the
  `dflash_*` namespace, and set up the two-stage loss schedule (cold-start
  lk-loss → final `e2e_tv_loss`).

### Docs & tests

- **`docs/concepts/dfly.md`**: document DFly as an independent `DFlyConfig` /
  `Qwen3DFlyModel` / `DFlyTrainer` drafter; update the comparison table.
- **`docs/concepts/dspark.md`**, **`dflash.md`**, **`draft_model_family.md`**:
  minor updates reflecting the moved correction module and the new loss term.
- **`tests/test_dfly.py`**: updated to build via `DFlyConfig` and exercise the
  `DFlyModel` wrapper; asserts a plain `DSparkConfig` still routes to
  `DSparkDraftModel` (no cross-routing).

## Compatibility / migration notes

- **Breaking config change:** existing DFly checkpoints/configs using
  `architectures: ["DSparkDraftModel"]` + `model_arch: "dfly"` must be migrated
  to `architectures: ["Qwen3DFlyModel"]` (see updated `dfly_*_draft_config.json`).
- The removed `dflash_kl_temperature` / `dflash_kl_topk_renormalize` /
  `dflash_lk_temperature` training args are no longer accepted.
- `treeflash_dspark_dflare` is gone; any references must be removed.

## Testing

- `tests/test_dfly.py` covers auto-dispatch, model structure (shared-KV layers,
  re-added `context_proj`, inherited fusion), the zero-init identity of the
  hidden correction, a tiny forward through `DFlyModel` (finite loss, correct
  `loss_components`, correction actually affects the loss), and state-dict keys.
