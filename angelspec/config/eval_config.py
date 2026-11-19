import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def named_eval_datasets(args):
    datasets = getattr(args, "eval_datasets", None)
    if not datasets:
        return ()
    if getattr(args, "eval_data_path", None):
        raise ValueError("Set only one of dataset.eval_data_path and dataset.eval_datasets")
    if getattr(args, "offline_manifest_path", None):
        raise ValueError("dataset.eval_datasets requires the online hidden-state pipeline")
    if isinstance(datasets, DictConfig):
        datasets = OmegaConf.to_container(datasets, resolve=True)
    if not isinstance(datasets, Mapping):
        raise ValueError("dataset.eval_datasets must map dataset names to paths")
    result = []
    for name, path in datasets.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[\w.-]{1,64}", name)
            or name in (".", "..")
        ):
            raise ValueError("Eval dataset names must use 1-64 letters, digits, _, . or -")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Eval dataset {name!r} requires a nonempty path")
        result.append((name, path))
    return tuple(sorted(result))


# The controller prefixes every eval sample's data_id with this.
EVAL_DATA_ID_PREFIX = "eval_"


def eval_max_seq_length(args):
    return getattr(args, "eval_max_seq_length", None) or getattr(args, "max_seq_length", None)


def engine_max_seq_length(args):
    lengths = [n for n in (getattr(args, "max_seq_length", None), eval_max_seq_length(args)) if n]
    return max(lengths, default=None)


def named_eval_cache_key(args, info):
    def identity(path):
        path = Path(os.path.expanduser(path))
        if path.is_file():
            stat = path.stat()
            return [(str(path.resolve()), stat.st_size, stat.st_mtime_ns)]
        if path.is_dir():
            return [
                (str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(path.rglob("*"))
                if p.is_file()
            ]
        return str(path)

    fields = (
        "target_model_path",
        "draft_model_config",
        "max_seq_length",
        "dp_size",
        "sp_ulysses_size",
        "sp_ring_size",
        "usp_local_shard",
        "attention_backend",
        "training_num_nodes",
        "training_num_gpus_per_node",
        "ttt_length",
        "inference_engine_type",
        "store_last_hidden_states",
        "aux_hidden_states_layers",
        "last_hidden_states_prenorm",
        "embedding_key",
        "lm_head_key",
        "norm_key",
        "chat_template",
        "prompt_key",
        "eval_prompt_key",
        "last_turn_loss_only",
        "defer_tokenization",
        "dynamic_loss_mask",
        "min_loss_tokens",
        "drop_overlength",
        "vllm_enable_multimodal",
        "vllm_mm_processor_kwargs",
        "sglang_enable_multimodal",
    )
    settings = {key: getattr(args, key, None) for key in fields}
    settings["max_seq_length"] = eval_max_seq_length(args)
    from angelspec.data.megatron_bin import resolve_image_newline_rewrite
    from angelspec.data.positions import position_recipe_from_config

    recipe = position_recipe_from_config(getattr(args, "draft_model_config_obj", None))
    if recipe is not None:
        settings["position_recipe"] = recipe.identity
    newline_rewrite = resolve_image_newline_rewrite(args)
    if newline_rewrite is not None:
        settings["image_newline_rewrite"] = list(newline_rewrite)
    for key, value in settings.items():
        if OmegaConf.is_config(value):
            settings[key] = OmegaConf.to_container(value, resolve=True)
    draft = settings.get("draft_model_config")
    if draft and Path(draft).is_file():
        settings["draft_content"] = hashlib.sha256(Path(draft).read_bytes()).hexdigest()
    target = settings.get("target_model_path")
    if target and (Path(target) / "config.json").is_file():
        settings["target_config"] = hashlib.sha256(
            (Path(target) / "config.json").read_bytes()
        ).hexdigest()
    recipe = {
        "version": 1,
        "dataset": info,
        "input_identity": identity(info["path"]),
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[
        :24
    ]
