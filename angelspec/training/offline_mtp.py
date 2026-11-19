from pathlib import Path

import torch

from angelspec.data.offline import read_json
from angelspec.data.offline_assets import mtp_init_prefix


def configure_offline_mtp(args, dataset, config, target):
    text_target = target.get("text_config", target)
    if args.dflash_packing:
        raise ValueError("MTP uses training.mtp_packing, not dflash_packing")
    if args.ttt_length < 1:
        raise ValueError("MTP requires positive ttt_length")
    if args.min_loss_tokens < args.ttt_length + 2:
        raise ValueError("MTP min_loss_tokens must be at least ttt_length+2")
    if args.attention_backend not in ("sdpa", "flex_attention", "usp"):
        raise ValueError("Unsupported offline MTP attention backend")
    if args.compile_model:
        raise ValueError("Offline MTP compile_model is not wired; use existing kernel compilation")
    if getattr(args, "mtp_opd_tree_forward", False):
        raise ValueError("Offline MTP does not support target score-engine OPD")
    if args.ploss_weights is not None and len(args.ploss_weights) != args.ttt_length:
        raise ValueError("ploss_weights must match ttt_length")
    if (
        config.hidden_size != text_target["hidden_size"]
        or config.target_hidden_size != text_target["hidden_size"]
    ):
        raise ValueError("MTP draft and target hidden widths must match")
    if config.vocab_size != text_target["vocab_size"]:
        raise ValueError("MTP vocabulary differs from frozen target")
    if not config.tie_lm_head:
        raise ValueError("Offline MTP requires the frozen target LM head")
    if getattr(config, "draft_vocab_size", None) not in (None, config.vocab_size):
        raise ValueError("Offline MTP does not support vocabulary pruning")
    from angelspec.training.offline import bind_dump_layers

    bind_dump_layers(args, dataset, [], text_target["num_hidden_layers"] - 1, text_target)
    for record in dataset.records:
        if record["seq_len"] > args.max_seq_length or (
            record["seq_len"] == args.max_seq_length and not args.mtp_packing
        ):
            raise ValueError(
                "Stored MTP samples exceed max_seq_length; refusing hidden truncation"
            )
    prefix = mtp_init_prefix(args, config)
    if prefix and args.draft_init_path:
        raise ValueError("Choose target-layer initialization or draft_init_path, not both")
    if prefix:
        metadata = read_json(Path(args.offline_frozen_target_path) / "frozen_assets.json")
        if metadata.get("mtp_init_prefix") != prefix:
            raise ValueError(
                "Export frozen assets with the requested MTP initialization layer first"
            )
    if config.mlp_type == "moe":
        world = args.training_num_nodes * args.training_num_gpus_per_node
        shard_size = args.hsdp_shard_size if args.attention_backend == "usp" else world
        if config.num_experts % shard_size:
            raise ValueError("FSDP shard size must divide MTP num_experts")
    args.offline_model_family = "mtp"
    args.offline_hidden_size = text_target["hidden_size"]
    args.offline_vocab_size = text_target["vocab_size"]
    args.last_hidden_states_prenorm = True
    args.target_model_path = str(Path(args.offline_frozen_target_path).resolve())
    args.draft_model_config_obj = config


def initialize_offline_mtp(trainer, draft, config, target_path):
    import torch.distributed as dist

    if dist.get_rank() != 0:
        return
    prefix = mtp_init_prefix(trainer.args, config)
    if prefix:
        _, source, layer, _ = prefix.split(".")
        trainer._load_mtp_from_target(draft, target_path, int(layer), source=source)
    elif trainer.args.draft_init_path:
        from angelspec.training.dflash_trainer import DFlashTrainer

        embedding = draft.embed_tokens.weight.detach().clone()
        DFlashTrainer._load_draft_init_weights(trainer, draft)
        if not torch.equal(embedding, draft.embed_tokens.weight):
            raise ValueError("Offline MTP warm-start embedding differs from frozen target")
