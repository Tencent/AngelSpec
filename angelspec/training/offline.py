import copy
import json
from pathlib import Path

import torch

from angelspec.data.offline import (
    HiddenDataset,
    OfflineBatchFetcher,
    OfflineRowPlan,
    atomic_json,
    child_path,
    file_hash,
    prefetch_offline,
    read_json,
)


def load_row_plan(args):
    path = getattr(args, "offline_row_plan_path", None)
    if not path:
        return None
    rows = read_json(Path(path)).get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
        raise ValueError('offline.row_plan_path must contain {"rows": [[sample_id, ...], ...]}')
    return rows


def open_dataset(args):
    path = Path(args.offline_manifest_path)
    if path.is_file():
        if path.name != "manifest.json":
            raise ValueError("offline.manifest_path must name manifest.json or its directory")
        path = path.parent
    return HiddenDataset(
        path,
        format=args.offline_format,
        verify_hashes=args.offline_verify_hashes,
        tokenized_path=args.offline_tokenized_path,
    )


def verify_frozen_target(path, expected_identity):
    root = Path(path)
    metadata = read_json(root / "frozen_assets.json")
    if metadata.get("frozen_components_only") is not True:
        raise ValueError("Expected an explicit frozen target component bundle")
    if not expected_identity or metadata["source_target_metadata_sha256"] != expected_identity:
        raise ValueError("Frozen target identity differs from captured teacher")
    files = metadata["bundle_files_sha256"]
    if "config.json" not in files or "model.safetensors.index.json" not in files:
        raise ValueError("Frozen bundle lacks config/index integrity records")
    index = read_json(root / "model.safetensors.index.json")
    for key in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        if index["weight_map"].get(key) not in files:
            raise ValueError(f"Frozen bundle missing verified component: {key}")
    for name, digest in files.items():
        if file_hash(child_path(root, name)) != digest:
            raise ValueError(f"Frozen target integrity mismatch: {name}")
    return read_json(root / "config.json")


def validate_offline_args(args):
    if getattr(args, "eval_datasets", None):
        raise ValueError("dataset.eval_datasets requires the online hidden-state pipeline")
    unsupported = (
        "dflash_opd_enabled",
        "train_with_decode",
        "enable_opd_score_engine",
        "online_eval_enabled",
        "continual_training",
        "defer_tokenization",
        "dynamic_loss_mask",
        "sp_leader_mooncake_fetch",
        "debug_inference_only",
        "colocate",
    )
    for name in unsupported:
        if getattr(args, name, False):
            raise ValueError(f"Offline DFlash v1 does not support {name}")
    if args.dflash_packing and args.mtp_packing:
        raise ValueError("dflash_packing and mtp_packing are mutually exclusive")
    if args.attention_backend not in ("sdpa", "flex_attention", "usp"):
        raise ValueError("Offline attention_backend must be sdpa, flex_attention or usp")
    if args.attention_backend == "usp":
        from angelspec.train_entry import _resolve_batch_size, _validate_usp_args
        from angelspec.utils.usp import validate_mtp_usp_layout

        _resolve_batch_size(args)
        _validate_usp_args(args)
        validate_mtp_usp_layout(
            attention_backend="usp",
            usp_local_shard=args.usp_local_shard,
            sp_ring_size=args.sp_ring_size,
        )
    elif args.sp_ulysses_size != 1 or args.sp_ring_size != 1 or args.usp_local_shard:
        raise ValueError("flex_attention requires SP1 and usp_local_shard=false")
    if getattr(args, "dflash_packing_token_weighted_loss", False) and not args.dflash_packing:
        raise ValueError("Token-weighted packing loss requires dflash_packing=true")
    if (args.dflash_packing or args.mtp_packing) and getattr(args, "packing_min_fill_ratio", 0):
        raise ValueError(
            "Offline packing closes every epoch's tail; packing_min_fill_ratio must be 0"
        )
    if args.micro_batch_size != 1:
        raise ValueError("Offline DFlash v1 requires micro_batch_size=1")
    if args.fsdp_strategy.upper() != "FULL_SHARD":
        raise ValueError("Offline DFlash v1 requires FULL_SHARD")
    if getattr(args, "load_path", None):
        raise ValueError(
            "Offline exact resume is not implemented yet; use draft_init_path for a fresh warm start"
        )
    if getattr(args, "eval_data_path", None):
        raise ValueError(
            "Offline evaluation uses the manifest validation split, not eval_data_path"
        )
    if not args.offline_frozen_target_path or not args.draft_model_config:
        raise ValueError("Offline training requires frozen_target_path and draft_model_config")
    if not args.output_dir or args.draft_accumulation_steps < 1:
        raise ValueError("Set a fresh output_dir and positive draft_accumulation_steps")
    if getattr(args, "online_eval_num_gpus", 0):
        raise ValueError("Online serving evaluation is not part of offline v1")
    if args.training_num_nodes * args.training_num_gpus_per_node < 2:
        raise ValueError("Offline FSDP v1 requires at least two training GPUs")
    if args.save_interval < 0:
        raise ValueError("save_interval cannot be negative")
    if args.prefetch_depth < 0:
        raise ValueError("prefetch_depth cannot be negative")
    if args.offline_eval_samples < 0 or args.eval_interval < 1:
        raise ValueError("Invalid offline evaluation settings")
    for key, expected in (
        ("embedding_key", "model.embed_tokens.weight"),
        ("lm_head_key", "lm_head.weight"),
        ("norm_key", "model.norm.weight"),
    ):
        if getattr(args, key, expected) != expected:
            raise ValueError(f"Offline frozen bundle currently requires {key}={expected}")


def check_frozen_target_weights(path, draft_config):
    from angelspec.data.offline_assets import target_weight_prefixes

    for name in ("target_mlp_init_layers", "target_mlp_dense_init_layers"):
        if getattr(draft_config, name, None) is not None:
            raise ValueError(
                f"Offline training cannot run {name}: the frozen bundle holds no target experts"
            )
    advice = (
        "the frozen bundle lacks target layers this draft reads; export it with this draft "
        "config (offline_entry export-assets or an online capture)"
    )
    try:
        needed = target_weight_prefixes(path, draft_config)
    except ValueError as error:
        raise ValueError(f"{advice}: {error}") from error
    if not needed:
        return
    exported = read_json(Path(path) / "frozen_assets.json").get("target_weight_prefixes", [])
    missing = sorted(set(needed) - set(exported))
    if missing:
        raise ValueError(f"{advice}: {missing}")


def configure_model(args, dataset, draft_config, target_config):
    from angelspec.data.positions import position_recipe_from_config
    from angelspec.training.position_config import configure_draft_positions

    configure_draft_positions(draft_config, target_config)
    expected_position = dataset.recipe.get("position_recipe_fingerprint")
    position_recipe = position_recipe_from_config(draft_config)
    if expected_position is not None and (
        position_recipe is None or position_recipe.identity != expected_position
    ):
        raise ValueError("Offline position recipe differs from the captured dataset")
    from angelspec.models.draft.mtp import MTPConfig
    from angelspec.models.draft.offline_family import (
        offline_layer_ids,
        offline_training_knobs,
    )

    if isinstance(draft_config, MTPConfig):
        from angelspec.training.offline_mtp import configure_offline_mtp

        return configure_offline_mtp(args, dataset, draft_config, target_config)
    text_target = target_config.get("text_config", target_config)
    if getattr(args, "mtp_packing", False):
        raise ValueError("DFlash-family uses dflash_packing, not mtp_packing")
    if getattr(args, "attention_backend", "flex_attention") not in ("flex_attention", "usp"):
        raise ValueError("Offline DFlash-family requires flex_attention or usp")
    family, context_count, anchors = offline_training_knobs(args, draft_config)
    check_frozen_target_weights(args.offline_frozen_target_path, draft_config)
    layers = offline_layer_ids(draft_config)
    draft_config.target_layer_ids = layers
    if getattr(draft_config, "draft_vocab_size", draft_config.vocab_size) not in (
        None,
        draft_config.vocab_size,
    ):
        raise ValueError("Offline DFlash v1 does not support vocabulary pruning")
    if len(layers) != context_count:
        raise ValueError("Draft context layer count differs from num_target_layers")
    final_id = text_target["num_hidden_layers"] - 1
    final_from_last = final_id in layers
    if final_from_last and layers[-1] != final_id:
        raise ValueError("The reused final hidden must be the last context layer")
    hidden_layers = layers[:-1] if final_from_last else layers
    if not hidden_layers:
        raise ValueError("Offline DFlash v1 requires at least one auxiliary hidden layer")
    bind_dump_layers(args, dataset, hidden_layers, final_id, text_target)
    if draft_config.target_hidden_size != text_target["hidden_size"]:
        raise ValueError("Draft target hidden width differs from frozen target")
    if draft_config.vocab_size != text_target["vocab_size"]:
        raise ValueError("Draft vocabulary differs from frozen target")
    if draft_config.target_num_hidden_layers != text_target["num_hidden_layers"]:
        raise ValueError("Draft target layer count differs from frozen target")
    if draft_config.hidden_size != text_target["hidden_size"]:
        raise ValueError("Offline DFlash v1 requires matching draft/target embedding widths")
    if args.min_loss_tokens < 2 * args.dflash_block_size:
        raise ValueError("min_loss_tokens must be at least twice dflash_block_size")
    if not 1 <= anchors or args.dflash_block_size < 2:
        raise ValueError("Invalid anchor/block budget")
    if anchors % getattr(args, "sp_ulysses_size", 1):
        raise ValueError("Anchor count must be divisible by SP")
    args.offline_model_family = family
    for record in dataset.records:
        if record["seq_len"] > args.max_seq_length or (
            record["seq_len"] == args.max_seq_length and not getattr(args, "dflash_packing", False)
        ):
            raise ValueError("Stored samples exceed max_seq_length; refusing hidden truncation")
    if getattr(args, "dflash_packing", False) and args.max_seq_length % args.sp_ulysses_size:
        raise ValueError("Packed max_seq_length must divide evenly across SP ranks")
    args.dflash_final_layer_from_last_hidden = final_from_last
    args.offline_hidden_size = text_target["hidden_size"]
    args.offline_vocab_size = text_target["vocab_size"]
    args.target_model_path = str(Path(args.offline_frozen_target_path).resolve())
    args.draft_model_config_obj = draft_config


def bind_dump_layers(args, dataset, aux_layers, final_id, text_target):
    """Select the dump layers a draft reads: aux context in order, then the final teacher."""
    dataset.bind_final_layer(final_id)
    missing = sorted((set(aux_layers) | {final_id}) - set(dataset.layer_ids))
    if missing:
        raise ValueError(
            f"Offline dump lacks target layers {missing}; it stores {dataset.layer_ids}"
        )
    if dataset.hidden_size != text_target["hidden_size"]:
        raise ValueError("Stored hidden width differs from target")
    if dataset.recipe.get("vocab_size", text_target["vocab_size"]) != text_target["vocab_size"]:
        raise ValueError("Stored vocabulary differs from target")
    args.offline_aux_layer_ids = list(aux_layers)
    args.offline_final_layer_id = final_id
    args.offline_context_layers = len(aux_layers)


def attach_offline_data(trainer, manifest_identity):
    from angelspec.data.positions import position_recipe_from_config

    args = trainer.args
    dataset = open_dataset(args)
    if dataset.identity != manifest_identity:
        raise ValueError("Offline manifest changed between driver and worker")
    dataset.bind_final_layer(args.offline_final_layer_id)
    needs_norm = (
        getattr(args, "offline_model_family", "") == "mtp"
        or getattr(trainer, "_distill_enabled", False)
        or getattr(trainer, "l1_loss_alpha", 0) > 0
        or (hasattr(trainer, "dflash") and trainer.dflash._extra_distill_needed())
    )
    if needs_norm and trainer.target_lm_head.norm is None:
        raise ValueError("Offline distillation requires the real target final norm")
    fetcher = OfflineBatchFetcher(
        dataset,
        rank=trainer.dp_rank,
        world_size=trainer.dp_size,
        seed=args.seed,
        max_length=args.max_seq_length,
        hidden_size=args.offline_hidden_size,
        aux_layer_ids=args.offline_aux_layer_ids,
        final_layer_id=args.offline_final_layer_id,
        vocab_size=args.offline_vocab_size,
        min_loss_tokens=args.min_loss_tokens,
        shuffle=args.shuffle_dataset,
        packing=getattr(args, "dflash_packing", False) or getattr(args, "mtp_packing", False),
        model_family=getattr(args, "offline_model_family", "dflash"),
        position_recipe=position_recipe_from_config(
            getattr(trainer.args, "draft_model_config_obj", None)
        ),
        row_plan=load_row_plan(args),
    )
    fetcher.seek(0)
    trainer._offline_fetcher = fetcher
    trainer.data_fetcher = fetcher
    if args.prefetch_depth:
        trainer.data_fetcher = prefetch_offline(fetcher, args.prefetch_depth)
    return len(fetcher.records)


@torch.no_grad()
def evaluate_offline(trainer):
    args = trainer.args
    fetcher = trainer._offline_fetcher
    records = [r for r in fetcher.dataset.records if r["split"] == "validation"]
    records = records[: args.offline_eval_samples]
    if not records:
        return {}
    metrics = []
    device = torch.cuda.current_device()
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(args.seed)
            plan = OfflineRowPlan(
                records, max_length=args.max_seq_length, packing=fetcher.packing, shuffle=False
            )
            for row in plan.rows_for_epoch(0):
                batch = fetcher.load_row([records[index] for index in row])
                metrics.append(trainer.eval_forward(batch))
        return trainer._aggregate_eval_metrics(metrics)
    finally:
        trainer.model.train(was_training)


def run_offline_training(args):
    import ray
    from ray.util.placement_group import remove_placement_group

    from angelspec import AutoDraftModelConfig
    from angelspec.ray.placement_group import (
        allocate_train_group,
        create_placement_groups,
    )
    from angelspec.training.trainer_actor import TrainerActor
    from angelspec.utils.logging import init_tracking, logger

    args = copy.deepcopy(args)
    validate_offline_args(args)
    dataset = open_dataset(args)
    target_config = verify_frozen_target(
        args.offline_frozen_target_path, dataset.recipe.get("target_metadata_sha256")
    )
    draft_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    configure_model(args, dataset, draft_config, target_config)
    train_count = sum(r["split"] == "train" for r in dataset.records)
    if not train_count:
        raise ValueError("Offline dataset has no training split")
    if args.offline_eval_samples and not any(r["split"] == "validation" for r in dataset.records):
        raise ValueError(
            "No validation split; explicitly set offline.eval_samples=0 to disable eval"
        )
    world = args.training_num_nodes * args.training_num_gpus_per_node
    sp_size = args.sp_ulysses_size if args.attention_backend == "usp" else 1
    args.dp_size = world // sp_size
    args.sp_size = sp_size
    args.global_batch_size = args.dp_size * args.draft_accumulation_steps
    packing = args.dflash_packing or args.mtp_packing
    row_plan = OfflineRowPlan(
        [r for r in dataset.records if r["split"] == "train"],
        max_length=args.max_seq_length,
        packing=packing,
        shuffle=args.shuffle_dataset,
        seed=args.seed,
        row_plan=load_row_plan(args),
    )
    if args.num_train_steps is None:
        args.num_train_steps = row_plan.row_budget(args.num_epochs) // args.global_batch_size
    if args.num_train_steps < 1:
        raise ValueError("Offline training requires at least one complete optimizer step")
    if args.lr_total_steps is None:
        args.lr_total_steps = args.num_train_steps
    if args.lr_total_steps < args.num_train_steps:
        raise ValueError("lr_total_steps must cover num_train_steps")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    if (output / "offline_run.json").exists() or (
        checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    ):
        raise FileExistsError("Use a fresh offline output directory")
    atomic_json(
        output / "offline_run.json",
        {
            "manifest_sha256": dataset.identity,
            "manifest": str(dataset.root),
            "seed": args.seed,
            "world_size": world,
            "global_batch": args.global_batch_size,
            "sp_size": sp_size,
            "dp_size": args.dp_size,
            "model_family": args.offline_model_family,
            "batch_unit": "packed_rows" if packing else "samples",
            "packing": packing,
            "rows_in_epoch_zero": len(row_plan.rows_for_epoch(0)),
            "packing_token_weighted_loss": args.dflash_packing_token_weighted_loss,
            "steps": args.num_train_steps,
            "sampling": (
                "explicit row plan"
                if row_plan.explicit_rows is not None
                else "seed+epoch shuffle" if args.shuffle_dataset else "manifest order"
            ),
            "row_plan_sha256": (
                file_hash(Path(args.offline_row_plan_path))
                if row_plan.explicit_rows is not None
                else None
            ),
            "context_layer_ids": args.offline_aux_layer_ids,
            "teacher_layer_id": args.offline_final_layer_id,
            "epoch_tail": "cyclic continuation across optimizer steps",
            "resume_supported": False,
            "eval_samples": args.offline_eval_samples,
        },
    )
    args.debug_train_only = True
    args.inference_num_gpus = 0
    args.per_dp_rank_batch_size = 1
    init_tracking(args)
    pgs = create_placement_groups(args)
    group = None
    try:
        group = allocate_train_group(
            args,
            args.training_num_nodes,
            args.training_num_gpus_per_node,
            pgs["training"],
            training_class=TrainerActor,
        )
        ray.get(group.async_init(args, role="training", mooncake_config=None))
        actors = group._actor_handlers
        ray.get([actor.set_offline_data.remote(dataset.identity) for actor in actors])
        for step in range(args.num_train_steps):
            results = group.train_from_queue(step, args.draft_accumulation_steps)
            metrics = results[0]
            if args.offline_eval_samples and (
                (step + 1) % args.eval_interval == 0 or step + 1 == args.num_train_steps
            ):
                evaluation = ray.get([actor.eval_offline.remote() for actor in actors])
                metrics.update(evaluation[0])
            with open(output / "offline_metrics.jsonl", "a") as stream:
                stream.write(json.dumps({"step": step + 1, **metrics}) + "\n")
            log_offline_metrics(args, metrics, step + 1)
            logger.info("Offline step %d/%d: %s", step + 1, args.num_train_steps, metrics)
            if (args.save_interval and (step + 1) % args.save_interval == 0) or (
                step + 1 == args.num_train_steps
            ):
                group.save_model(step, force_sync=True)
        group.save_draft_model_for_serving(str(output / "draft_hf"))
        ray.get([actor.close.remote() for actor in actors])
        atomic_json(output / "offline_complete.json", {"steps": args.num_train_steps})
    finally:
        if group is not None:
            for actor in group._actor_handlers:
                ray.kill(actor, no_restart=True)
        remove_placement_group(pgs["training"][0])


def log_offline_metrics(args, metrics, step):
    values = {**metrics, "train/step": step}
    if getattr(args, "use_wandb", False):
        import wandb

        if wandb.run is not None:
            wandb.log(values)
    if getattr(args, "use_swanlab", False):
        from angelspec.utils import swanlab

        swanlab.log(values, step=step)
    if getattr(args, "use_tensorboard", False):
        from angelspec.utils.logging import get_tb_writer

        writer = get_tb_writer()
        if writer is not None:
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, step)
