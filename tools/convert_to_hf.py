"""
Convert FSDP checkpoint to HuggingFace format with optional vocabulary pruning.

Usage:
    # Basic conversion (no vocab pruning):
    python tools/convert_to_hf.py --input-dir <checkpoint_dir>

    # With target model (loads correct embed_tokens + auto-generates config):
    python tools/convert_to_hf.py --input-dir <checkpoint_dir> \
        --target-model-path moonshotai/Kimi-K2.5 --trust-remote-code

    # Analyze token coverage to choose draft vocab size:
    python tools/convert_to_hf.py \
        --input-dir <checkpoint_dir> \
        --target-model-path moonshotai/Kimi-K2.5 --trust-remote-code \
        --analyze-vocab \
        --dataset-path <your_dataset> \
        --tokenizer moonshotai/Kimi-K2.5 \
        --chat-template kimi-k25-vlm \
        --prompt-key messages

    # With vocab pruning (reuses tokenized dataset cache from training):
    python tools/convert_to_hf.py \
        --input-dir <checkpoint_dir> \
        --prune-vocab \
        --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
        --draft-vocab-size 32000 \
        --tokenizer Qwen/Qwen3-8B \
        --chat-template qwen \
        --prompt-key conversations

Options:
    --input-dir     Path to FSDP checkpoint directory (required)
    --output-dir    Output directory (default: {input_dir}_hf)
    --config        Path to draft model config.json (default: {input_dir}/config.json)
    --target-model-path  Target model (HF hub id or path) — auto-generates config.json if missing
    --dtype         Output dtype (float16, bfloat16, float32). Use float16 for FP8 target models
    -f, --force     Overwrite output directory if exists
    --analyze-vocab Analyze token coverage at various draft vocab sizes (no conversion)
    --prune-vocab   Enable vocabulary pruning (requires --dataset-path and --draft-vocab-size)
    --dataset-path  Dataset for vocabulary pruning
    --draft-vocab-size  Draft vocabulary size
"""

import argparse
import json
import logging
import os
import pickle
import time
from collections import Counter
from typing import Optional

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import save_file
from tqdm import tqdm
from typing_extensions import override

from angelspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "version.txt")


def _get_angelspec_version() -> str:
    try:
        with open(_VERSION_FILE) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


# ── FSDP loading helpers ─────────────────────────────────────────────────────


class _UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class _WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = _UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            logger.debug("Found %s in torch_dist ckpt", k)
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


# ── Export fixups ────────────────────────────────────────────────────────────

# vLLM checkpoints use `layers.0.xxx` for the single decoder layer,
# while our training code uses `midlayer.xxx`.
_WEIGHT_KEY_REMAP = [("midlayer.", "layers.0.")]


def _remap_weight_keys(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for k, v in tensors.items():
        new_key = k
        for old_prefix, new_prefix in _WEIGHT_KEY_REMAP:
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix) :]
                break
        remapped[new_key] = v
    return remapped


def _fixup_export_config(raw_config: dict, export_for_vllm: bool = False) -> dict:
    """Apply model-type-specific fixups to the exported config."""
    config = json.loads(json.dumps(raw_config))

    if export_for_vllm and config.get("model_type") == "kimi_k2":
        # vLLM uses 1-indexed layer IDs for kimi_k2
        key = "eagle_aux_hidden_state_layer_ids"
        if key in config:
            config[key] = [x + 1 for x in config[key]]
        eagle_cfg = config.get("eagle_config")
        if eagle_cfg and key in eagle_cfg:
            eagle_cfg[key] = [x + 1 for x in eagle_cfg[key]]

    return config


# ── Core conversion ─────────────────────────────────────────────────────────


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _generate_config_from_target(
    target_model_path: str, output_path: str, trust_remote_code: bool = False
) -> str:
    from angelspec.config.utils import generate_draft_model_config

    logger.info("Auto-generating draft model config from %s", target_model_path)
    config_dict = generate_draft_model_config(
        target_model_path=target_model_path,
        template_config_path=None,
        cache_dir=None,
    )
    config_dict["tie_word_embeddings"] = False
    if not config_dict.get("draft_vocab_size"):
        config_dict["draft_vocab_size"] = config_dict.get("vocab_size")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Saved generated config to %s", output_path)
    return output_path


def _resolve_config_path(
    input_dir: str,
    config_arg: Optional[str],
    target_model_path: Optional[str],
    trust_remote_code: bool = False,
) -> str:
    if config_arg:
        return config_arg
    config_path = os.path.join(input_dir, "config.json")
    if os.path.isfile(config_path):
        return config_path
    if target_model_path:
        return _generate_config_from_target(target_model_path, config_path, trust_remote_code)
    raise FileNotFoundError(
        f"config.json not found in {input_dir}. Provide --config or --target-model-path to auto-generate one."
    )


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=_WrappedStorageReader(input_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _extract_model_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    model_state = {}
    skipped_keys = []

    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if "draft_model." not in k:
            skipped_keys.append(k)
            continue
        new_key = k.split("draft_model.")[-1]
        if new_key in ("t2d", "d2t"):
            continue
        model_state[new_key] = v

    logger.info(
        "Extracted %d model weight keys (skipped %d non-draft keys)",
        len(model_state),
        len(skipped_keys),
    )
    for k, v in model_state.items():
        logger.debug("  %s: shape=%s, dtype=%s", k, list(v.shape), v.dtype)

    return model_state


def _prepare_export_tensors(hf_model, export_for_vllm: bool) -> dict[str, torch.Tensor]:
    tensors = hf_model.state_dict()
    if export_for_vllm:
        logger.info("Exporting vLLM-compatible checkpoint keys")
        return _remap_weight_keys(tensors)
    logger.info("Exporting native checkpoint keys")
    return dict(tensors)


def _save_without_vocab_pruning(
    hf_model, output_dir: str, raw_config: dict, vocab_size: int, export_for_vllm: bool = False
) -> None:
    version = _get_angelspec_version()
    tensors = _prepare_export_tensors(hf_model, export_for_vllm)
    save_file(
        tensors,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"angelspec_version": version},
    )

    export_config = _fixup_export_config(raw_config, export_for_vllm=export_for_vllm)
    export_config["draft_vocab_size"] = vocab_size
    export_config["_angelspec_version"] = version
    actual_dtype = next(iter(tensors.values())).dtype
    export_config["torch_dtype"] = str(actual_dtype).replace("torch.", "")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(export_config, f, indent=2)

    logger.info(
        "Model saved to %s (no vocab pruning, draft_vocab_size=%d)",
        output_dir,
        vocab_size,
    )


def _save_with_vocab_pruning(
    hf_model,
    output_dir: str,
    raw_config: dict,
    vocab_size: int,
    draft_vocab_size: int,
    d2t: torch.Tensor,
    t2d: torch.Tensor,
    export_for_vllm: bool = False,
) -> None:
    tensors = _prepare_export_tensors(hf_model, export_for_vllm)
    tensors["t2d"] = t2d
    tensors["d2t"] = d2t

    lm_head_key = "lm_head.weight"
    if lm_head_key in tensors:
        current_size = tensors[lm_head_key].shape[0]
        if current_size == vocab_size and current_size != draft_vocab_size:
            # lm_head is full vocab — prune to draft vocab using d2t mapping
            logger.info(
                "Trimming lm_head from %d to %d using d2t mapping",
                current_size,
                draft_vocab_size,
            )
            tensors[lm_head_key] = tensors[lm_head_key][d2t]
        elif current_size == draft_vocab_size:
            raise ValueError(
                f"lm_head is already pruned to draft_vocab_size ({current_size}). "
                f"This model was trained with vocabulary pruning, so the lm_head weight "
                f"ordering is tied to the training data's token mapping. "
                f"Post-training re-pruning with a different dataset is not supported. "
                f"Retrain with draft_vocab_size == vocab_size (full vocab) for post-training pruning."
            )
        else:
            logger.warning(
                "lm_head size (%d) matches neither vocab_size (%d) nor draft_vocab_size (%d), skipping trim",
                current_size,
                vocab_size,
                draft_vocab_size,
            )

    version = _get_angelspec_version()
    save_file(
        tensors,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"angelspec_version": version},
    )

    export_config = _fixup_export_config(raw_config, export_for_vllm=export_for_vllm)
    export_config["draft_vocab_size"] = draft_vocab_size
    export_config["_angelspec_version"] = version
    actual_dtype = next(iter(tensors.values())).dtype
    export_config["torch_dtype"] = str(actual_dtype).replace("torch.", "")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(export_config, f, indent=2)

    logger.info(
        "Model saved to %s  (draft_vocab_size=%d, vocab_size=%d, d2t=%s, t2d=%s)",
        output_dir,
        draft_vocab_size,
        vocab_size,
        list(d2t.shape),
        list(t2d.shape),
    )


def _count_token_frequencies(prompts: list[dict]) -> Counter:
    """Count token frequencies from tokenized prompts using packed_loss_mask."""
    import numba
    import numpy as np

    from angelspec.data.utils import unpack_loss_mask

    @numba.njit(cache=True)
    def _histogram(ids, mask, counts):
        for i in range(len(ids)):
            if mask[i] == 1:
                counts[ids[i]] += 1

    all_ids = []
    all_masks = []
    for item in tqdm(prompts, desc="Preparing token data"):
        input_ids = item["input_ids"].squeeze()

        if "loss_mask" in item and item["loss_mask"] is not None:
            loss_mask = item["loss_mask"].squeeze()
        elif "packed_loss_mask" in item and item["packed_loss_mask"] is not None:
            loss_mask = unpack_loss_mask(item["packed_loss_mask"])
        else:
            loss_mask = torch.ones_like(input_ids)

        all_ids.append(input_ids.numpy().astype(np.int64))
        all_masks.append(loss_mask.numpy().astype(np.int64))

    flat_ids = np.concatenate(all_ids)
    flat_masks = np.concatenate(all_masks)

    counts = np.zeros(int(flat_ids.max()) + 1, dtype=np.int64)
    _histogram(flat_ids, flat_masks, counts)

    nonzero = np.nonzero(counts)[0]
    return Counter(dict(zip(nonzero.tolist(), counts[nonzero].tolist())))


def _load_tokenized_prompts(
    dataset_path: str,
    tokenizer: str,
    chat_template: str,
    prompt_key: str,
    max_seq_length: int,
    cache_dir: Optional[str],
) -> list:
    from angelspec.data.dataset import load_conversation_dataset

    args_ns = argparse.Namespace(
        train_data_path=dataset_path,
        target_model_path=tokenizer,
        chat_template=chat_template,
        prompt_key=prompt_key,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir or "./cache",
        train_with_decode=False,
    )
    return load_conversation_dataset(args_ns)


def _analyze_vocab_coverage(
    token_dict: Counter,
    vocab_size: int,
    candidate_sizes: Optional[list[int]] = None,
) -> None:
    total_freq = sum(token_dict.values())
    unique_tokens = len(token_dict)
    sorted_freqs = [freq for _, freq in token_dict.most_common()]

    if candidate_sizes is None:
        candidate_sizes = sorted(
            set([4096, 8192, 16384, 32000, 48000, 64000, 96000, unique_tokens, vocab_size])
        )

    from itertools import accumulate

    cumsum = list(accumulate(sorted_freqs))

    print(f"\n{'=' * 65}")
    print("  Vocab Coverage Analysis")
    print(f"{'=' * 65}")
    print(f"  Target vocab size:  {vocab_size:>10,}")
    print(f"  Unique tokens seen: {unique_tokens:>10,}")
    print(f"  Total token count:  {total_freq:>10,}")
    print(f"{'─' * 65}")
    print(f"  {'draft_vocab_size':>18s}  {'coverage':>10s}  {'tokens dropped':>15s}")
    print(f"{'─' * 65}")

    for size in candidate_sizes:
        if size < 1:
            continue
        idx = min(size, len(cumsum)) - 1
        coverage = cumsum[idx] / total_freq if total_freq > 0 else 0.0
        dropped = max(0, unique_tokens - size)
        marker = " ◄" if size == unique_tokens else ""
        print(f"  {size:>18,}  {coverage:>9.2%}  {dropped:>15,}{marker}")

    print(f"{'=' * 65}")
    print(
        "\n  Tip: pick the smallest size with acceptable coverage (≥99% is typical).\n"
        "  Tokens not in the top-N are unmatchable during speculative decoding.\n"
    )


def _convert_fsdp_to_hf(
    config_path: str,
    input_dir: str,
    output_dir: str,
    target_model_path: Optional[str] = None,
    embedding_key: str = "model.embed_tokens",
    export_for_vllm: bool = False,
    output_dtype: Optional[torch.dtype] = None,
    prune_vocab: bool = False,
    dataset_path: Optional[str] = None,
    draft_vocab_size: Optional[int] = None,
    tokenizer: Optional[str] = None,
    chat_template: Optional[str] = None,
    prompt_key: str = "conversations",
    max_seq_length: int = 32768,
    cache_dir: Optional[str] = None,
) -> None:
    logger.info("Loading FSDP model from %s", input_dir)
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    logger.info("FSDP model loaded in %.2f sec", time.time() - t)

    model_state = _extract_model_weights(state_dict)
    if not model_state:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    config = AutoDraftModelConfig.from_file(config_path)
    hf_model = AutoEagle3DraftModel.from_config(config)

    ckpt_dtype = Counter(v.dtype for v in model_state.values()).most_common(1)[0][0]
    final_dtype = output_dtype or ckpt_dtype
    logger.info("Checkpoint dtype: %s, output dtype: %s", ckpt_dtype, final_dtype)
    hf_model = hf_model.to(ckpt_dtype)

    missing, unexpected = hf_model.load_state_dict(model_state, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    # Optionally override embed_tokens from target model.
    # Useful for older checkpoints where embed_tokens may not have been saved correctly.
    if target_model_path:
        logger.info("Loading embed_tokens from target model: %s", target_model_path)
        embed_key = embedding_key + ".weight"
        try:
            hf_model.load_embedding(target_model_path, embedding_key=embed_key)
        except KeyError as e:
            raise ValueError(
                f"Embedding key '{embed_key}' not found in target model. "
                f"Use --embedding-key to specify the correct key prefix."
            ) from e
        logger.info(
            "Loaded embed_tokens: shape=%s, dtype=%s",
            list(hf_model.embed_tokens.weight.shape),
            hf_model.embed_tokens.weight.dtype,
        )

    if output_dtype and output_dtype != ckpt_dtype:
        logger.info("Casting model from %s to %s", ckpt_dtype, output_dtype)
        hf_model = hf_model.to(output_dtype)

    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        raw_config = json.load(f)
    vocab_size = raw_config["vocab_size"]
    config_draft_vocab_size = raw_config.get("draft_vocab_size") or vocab_size

    if not prune_vocab:
        if config_draft_vocab_size != vocab_size:
            raise ValueError(
                f"draft_vocab_size ({config_draft_vocab_size}) != vocab_size ({vocab_size}) "
                f"in {config_path}. This model was trained with vocabulary pruning. "
                f"Use --prune-vocab to generate t2d/d2t token mappings during conversion."
            )
        _save_without_vocab_pruning(
            hf_model, output_dir, raw_config, vocab_size, export_for_vllm=export_for_vllm
        )
        return

    # ── Vocab pruning ────────────────────────────────────────────────────
    assert dataset_path is not None and draft_vocab_size is not None
    assert tokenizer is not None and chat_template is not None
    from angelspec.data.preprocessing import process_token_dict_to_mappings

    logger.info(
        "Vocab pruning: vocab_size=%d, draft_vocab_size=%d",
        vocab_size,
        draft_vocab_size,
    )

    prompts = _load_tokenized_prompts(
        dataset_path, tokenizer, chat_template, prompt_key, max_seq_length, cache_dir
    )
    logger.info("Loaded %d tokenized samples", len(prompts))

    token_dict = _count_token_frequencies(prompts)
    logger.info(
        "Found %d unique tokens from %d total tokens",
        len(token_dict),
        sum(token_dict.values()),
    )

    d2t, t2d = process_token_dict_to_mappings(token_dict, draft_vocab_size, vocab_size)
    _save_with_vocab_pruning(
        hf_model,
        output_dir,
        raw_config,
        vocab_size,
        draft_vocab_size,
        d2t,
        t2d,
        export_for_vllm=export_for_vllm,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FSDP checkpoint to HuggingFace format with optional vocabulary pruning"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to FSDP checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {input_dir}_hf)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the draft model config JSON. Defaults to {input_dir}/config.json",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default=None,
        help="Target model (HF hub id or local path). "
        "Used to load embed_tokens and auto-generate config.json if missing",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens",
        help="Key prefix for embedding weights in target model (default: model.embed_tokens)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading target model config/tokenizer",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Export vLLM-compatible checkpoint keys. Defaults to native keys for sglang.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype for model weights. If not set, uses the checkpoint's native dtype. "
        "Use float16 when the target model runs FP8 inference (hidden states are float16)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output directory if exists",
    )

    vocab = parser.add_argument_group("vocab pruning (requires --prune-vocab)")
    vocab.add_argument(
        "--analyze-vocab",
        action="store_true",
        help="Analyze token coverage at various draft vocab sizes and exit (no conversion). "
        "Requires --dataset-path, --tokenizer, --chat-template",
    )
    vocab.add_argument(
        "--prune-vocab",
        action="store_true",
        help="Enable vocabulary pruning. Requires --dataset-path, --draft-vocab-size, --tokenizer, --chat-template",
    )
    vocab.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset for vocabulary pruning (HuggingFace hub id, local directory, or jsonl file)",
    )
    vocab.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="Draft vocabulary size",
    )
    vocab.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name/path for tokenizing conversations",
    )
    vocab.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Chat template name (e.g. qwen, llama3, llama4)",
    )
    vocab.add_argument(
        "--prompt-key",
        type=str,
        default="conversations",
        help="Column name for conversations in dataset (default: conversations)",
    )
    vocab.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
        help="Max sequence length for tokenization (default: 32768). "
        "Should match training config to reuse tokenization cache",
    )
    vocab.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for tokenized dataset (reuses training cache if available)",
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.analyze_vocab or args.prune_vocab:
        if not args.dataset_path:
            raise ValueError(
                "--dataset-path is required when --prune-vocab or --analyze-vocab is set"
            )
        if not args.tokenizer:
            raise ValueError(
                "--tokenizer is required when --prune-vocab or --analyze-vocab is set"
            )
        if not args.chat_template:
            raise ValueError(
                "--chat-template is required when --prune-vocab or --analyze-vocab is set"
            )

    if args.prune_vocab and not args.draft_vocab_size:
        raise ValueError("--draft-vocab-size is required when --prune-vocab is set")

    if args.chat_template:
        from angelspec.data.template import TEMPLATE_REGISTRY

        available = TEMPLATE_REGISTRY.get_all_template_names()
        if args.chat_template not in available:
            raise ValueError(
                f"Chat template '{args.chat_template}' not found. Available: {available}"
            )


if __name__ == "__main__":
    args = _parse_args()
    _validate_args(args)

    input_dir = args.input_dir.rstrip("/")

    if args.analyze_vocab:
        config_path = _resolve_config_path(
            input_dir, args.config, args.target_model_path, args.trust_remote_code
        )
        with open(config_path) as f:
            raw_config = json.load(f)
        vocab_size = raw_config["vocab_size"]

        prompts = _load_tokenized_prompts(
            args.dataset_path,
            args.tokenizer,
            args.chat_template,
            args.prompt_key,
            args.max_seq_length,
            args.cache_dir,
        )
        logger.info("Loaded %d tokenized samples", len(prompts))

        token_dict = _count_token_frequencies(prompts)
        _analyze_vocab_coverage(token_dict, vocab_size)
        raise SystemExit(0)

    output_dir = args.output_dir or f"{input_dir}_hf"
    config_path = _resolve_config_path(
        input_dir, args.config, args.target_model_path, args.trust_remote_code
    )

    if os.path.exists(output_dir) and not args.force:
        raise ValueError(f"Output directory {output_dir} already exists. Use -f to overwrite.")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    output_dtype = dtype_map[args.dtype] if args.dtype else None

    model_dir = _detect_model_dir(input_dir)
    _convert_fsdp_to_hf(
        config_path=config_path,
        input_dir=model_dir,
        output_dir=output_dir,
        target_model_path=args.target_model_path,
        embedding_key=args.embedding_key,
        export_for_vllm=args.vllm,
        output_dtype=output_dtype,
        prune_vocab=args.prune_vocab,
        dataset_path=args.dataset_path,
        draft_vocab_size=args.draft_vocab_size,
        tokenizer=args.tokenizer,
        chat_template=args.chat_template,
        prompt_key=args.prompt_key,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )
