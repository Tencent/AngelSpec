# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
import json
import logging as _logging
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from angelspec.data.megatron_bin import (
    _rewrite_image_newlines,
    resolve_image_newline_rewrite,
)
from angelspec.data.parse import create_parser, has_thinking_content
from angelspec.data.positions import IMAGE_NEWLINE_SLOTS_KEY
from angelspec.data.preprocessing import (
    _normalize_conversation,
    preprocess_conversations,
)
from angelspec.data.template import TEMPLATE_REGISTRY
from angelspec.data.utils import (
    DSV41_IMAGE,
    HY4_IMAGE_END,
    HY4_IMAGE_NEWLINE,
    HY4_IMAGE_PAD,
    HY4_IMAGE_START,
    HY4_VIDEO_END,
    HY4_VIDEO_PAD,
    HY4_VIDEO_START,
    Hy4VideoLayout,
    estimate_row_count,
    expand_image_tokens,
    extract_media_urls,
    flatten_multimodal_content,
    load_hf_dataset,
    render_dsv41_images,
    render_hy4_media,
)
from angelspec.utils.hf_config_registry import register_extra_hf_configs
from angelspec.utils.logging import logger
from angelspec.utils.processing import load_tokenizer

_logging.getLogger("transformers_modules").setLevel(_logging.ERROR)

_worker_state = {}


def _init_tokenize_worker(
    tokenizer_path,
    trust_remote_code,
    chat_template_name,
    last_turn_loss_only=False,
    min_loss_tokens=0,
    drop_overlength=False,
    mm_processor_kwargs=None,
    image_newline_as_placeholder=False,
    min_seq_length=0,
):
    """Initializer for each worker process — loads tokenizer once."""
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    # A forked worker inherits the parent's OpenMP state; once the parent has run a
    # parallel torch op, the first one here (e.g. zeros over > 32k tokens) deadlocks.
    torch.set_num_threads(1)
    _worker_state["tokenizer"] = load_tokenizer(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["parser"] = create_parser(_worker_state["tokenizer"], _worker_state["template"])
    _worker_state["preprocess"] = preprocess_conversations
    _worker_state["last_turn_loss_only"] = last_turn_loss_only
    _worker_state["min_loss_tokens"] = min_loss_tokens
    _worker_state["drop_overlength"] = drop_overlength
    _worker_state["min_seq_length"] = min_seq_length
    _worker_state["mm_processor_kwargs"] = mm_processor_kwargs
    _worker_state["image_newline_as_placeholder"] = image_newline_as_placeholder
    _worker_state["image_processor"] = None
    _worker_state["video_layout"] = None
    _worker_state["model_path"] = tokenizer_path
    _worker_state["dsv41_vision"] = None
    if _worker_state["template"].image_layout == "dsv41":
        with open(os.path.join(tokenizer_path, "config.json")) as f:
            _worker_state["dsv41_vision"] = json.load(f)["vision_config"]
    if mm_processor_kwargs is not None:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                tokenizer_path, trust_remote_code=trust_remote_code
            )
            _worker_state["image_processor"] = getattr(processor, "image_processor", None)
        except Exception:
            pass


def _resolve_last_turn_loss_only(messages):
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        return has_thinking_content(messages)
    return bool(ltlo)


def _hy4_image_newlines_to_placeholder(input_ids, tokenizer):
    """Put the image placeholder in Hy4 image-row newline slots, the video placeholder in
    video frames' ones, as the megatron loader does when the vision tower emits each row's
    newline embedding.

    Returns the rewritten ids and the rewritten indices, which position
    reconstruction needs to recover the row layout.
    """
    ids = np.asarray(input_ids)
    # preprocess_conversations stores one sample as [1, seq_len], while the
    # shared Megatron helper expects a flat token sequence.  Passing the 2-D
    # array directly makes np.nonzero(...)[0] return batch coordinates (all
    # zero) instead of token offsets.
    rewritten, slots = _rewrite_image_newlines(
        ids.reshape(-1),
        tokenizer.convert_tokens_to_ids(HY4_IMAGE_START),
        tokenizer.convert_tokens_to_ids(HY4_IMAGE_END),
        tokenizer.convert_tokens_to_ids(HY4_IMAGE_NEWLINE),
        tokenizer.convert_tokens_to_ids(HY4_IMAGE_PAD),
        video=(
            tokenizer.convert_tokens_to_ids(HY4_VIDEO_START),
            tokenizer.convert_tokens_to_ids(HY4_VIDEO_END),
            tokenizer.convert_tokens_to_ids(HY4_VIDEO_PAD),
        ),
    )
    return rewritten.reshape(ids.shape), slots


def _hy4_video_layout():
    if _worker_state.get("video_layout") is None:
        _worker_state["video_layout"] = Hy4VideoLayout(
            _worker_state["model_path"], _worker_state.get("mm_processor_kwargs")
        )
    return _worker_state["video_layout"]


def _tokenize_single(args):
    """Worker function — normalize, expand VL tokens, and tokenize one sample."""
    raw_messages, max_length, train_with_decode, extra = args
    messages = _normalize_conversation(raw_messages)
    template = _worker_state["template"]
    multimodal_inputs = extract_media_urls(messages)
    image_processor = _worker_state.get("image_processor")
    conversation, is_preformatted = messages, False
    extra_kwargs = {k: [v] for k, v in (extra or {}).items()}
    if template.image_layout == "hy4" and multimodal_inputs:
        if image_processor is None:
            raise ValueError(
                "Hy4 flash media samples need inference.vllm.enable_multimodal and "
                "inference.vllm.mm_processor_kwargs, whose pixel bounds size the grids"
            )
        conversation, multimodal_inputs = render_hy4_media(
            _worker_state["parser"],
            messages,
            image_processor,
            _worker_state.get("mm_processor_kwargs"),
            _hy4_video_layout() if multimodal_inputs.get("videos") else None,
            add_generation_prompt=train_with_decode,
            **(extra or {}),
        )
        if conversation is None:
            return None
        is_preformatted, extra_kwargs = True, {}
    elif template.image_layout == "dsv41" and multimodal_inputs:
        if multimodal_inputs.get("videos"):
            raise NotImplementedError(
                "DeepSeek-V4.1 conversation samples with video are not supported"
            )
        conversation, images = render_dsv41_images(
            _worker_state["parser"],
            messages,
            _worker_state["dsv41_vision"],
            add_generation_prompt=train_with_decode,
            **(extra or {}),
        )
        if conversation is None:
            return None
        is_preformatted, extra_kwargs = True, {}
        multimodal_inputs = {"images": images}
    else:
        flatten_multimodal_content(messages, template.image_placeholder)
        if image_processor and multimodal_inputs:
            image_paths = multimodal_inputs.get("images") or []
            expand_image_tokens(
                messages,
                image_paths,
                template.image_placeholder,
                image_processor,
                _worker_state.get("mm_processor_kwargs"),
            )
    processed = _worker_state["preprocess"](
        _worker_state["tokenizer"],
        [conversation],
        template,
        max_length=max_length,
        is_preformatted=is_preformatted,
        include_attention_mask=False,
        use_packed_loss_mask=True,
        add_generation_prompt=train_with_decode,
        return_formatted_text=True,
        last_turn_loss_only=_resolve_last_turn_loss_only(messages),
        min_loss_tokens=_worker_state.get("min_loss_tokens", 0),
        drop_overlength=_worker_state.get("drop_overlength", False),
        min_length=_worker_state.get("min_seq_length", 0),
        **extra_kwargs,
    )
    if not processed["input_ids"]:
        return None
    input_ids = processed["input_ids"][0]
    newline_slots = []
    if is_preformatted and (multimodal_inputs.get("images") or multimodal_inputs.get("videos")):
        # Truncation must not cut into a medium, or vLLM cannot match its placeholders.
        tokenizer = _worker_state["tokenizer"]
        bounds = (
            [DSV41_IMAGE] if template.image_layout == "dsv41" else [HY4_IMAGE_END, HY4_VIDEO_END]
        )
        ids = torch.as_tensor(input_ids)
        for token in bounds:
            if int((ids == tokenizer.convert_tokens_to_ids(token)).sum()) != conversation.count(
                token
            ):
                return None
        if _worker_state.get("image_newline_as_placeholder"):
            input_ids, newline_slots = _hy4_image_newlines_to_placeholder(input_ids, tokenizer)
    return {
        "input_ids": input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
        "packed_loss_mask": processed["packed_loss_mask"][0],
        "formatted_prompt": processed["formatted_text"][0],
        "multimodal_inputs": multimodal_inputs,
        "image_newline_slots": newline_slots,
    }


def _init_format_worker(
    tokenizer_path, trust_remote_code, chat_template_name, last_turn_loss_only=False
):
    _logging.getLogger("transformers_modules").setLevel(_logging.ERROR)
    torch.set_num_threads(1)  # see _init_tokenize_worker
    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    _worker_state["template"] = TEMPLATE_REGISTRY.get(chat_template_name)
    _worker_state["parser"] = create_parser(tokenizer, _worker_state["template"])
    _worker_state["last_turn_loss_only"] = last_turn_loss_only


def _format_single(args):
    """Worker function — normalize, extract media, format text (skip tokenization)."""
    raw_messages, _, train_with_decode, _ = args
    messages = _normalize_conversation(raw_messages)
    template = _worker_state["template"]
    multimodal_inputs = extract_media_urls(messages)
    flatten_multimodal_content(messages, template.image_placeholder)

    result = {}
    ltlo = _worker_state.get("last_turn_loss_only", False)
    if ltlo == "auto":
        result["has_thinking"] = has_thinking_content(messages)

    parser = _worker_state["parser"]
    formatted = parser.format(
        messages, add_generation_prompt=train_with_decode, expand_media_tokens=False
    )
    if not formatted:
        return None
    result["formatted_prompt"] = formatted
    result["multimodal_inputs"] = multimodal_inputs
    return result


def _build_entry(data_id, result, defer_tokenization):
    """Convert a worker result into a stored dataset entry (or None to skip)."""
    if result is None:
        return None
    metadata = {}
    if "has_thinking" in result:
        metadata["has_thinking"] = result["has_thinking"]
    if result.get("image_newline_slots"):
        metadata[IMAGE_NEWLINE_SLOTS_KEY] = result["image_newline_slots"]

    multimodal_inputs = result.get("multimodal_inputs")
    entry = {
        "data_id": data_id,
        "metadata": metadata,
        "multimodal_inputs": multimodal_inputs,
        "formatted_prompt": result["formatted_prompt"],
    }

    if not defer_tokenization:
        input_ids = result["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        entry["input_ids"] = input_ids
        entry["packed_loss_mask"] = result["packed_loss_mask"]

    return entry


def _iter_raw_samples(hf_dataset, prompt_key, total_estimate):
    """Yield (data_id, raw_prompt, extra) lazily from the source dataset.

    Streams the source dataset so raw samples are never all held in memory at
    once. The yielded tuple matches what the worker fns consume.
    """
    seen_ids = set()
    for idx, sample in enumerate(tqdm(hf_dataset, desc="Loading samples", total=total_estimate)):
        raw_prompt = sample.get(prompt_key, "")
        if not isinstance(raw_prompt, list):
            raise ValueError(
                f"Expected conversation format (list of messages) for sample {idx}, "
                f"got {type(raw_prompt)}"
            )
        data_id = sample.get("id", f"sample_{idx}")
        # Results are matched to samples by data_id, so a repeated source id (e.g.
        # several rollouts of one prompt) must not collapse into one entry.
        if data_id in seen_ids:
            data_id = f"{data_id}#{idx}"
        seen_ids.add(data_id)
        extra = {}
        for key in ("tools", "reasoning_effort"):
            val = sample.get(key)
            if val:
                extra[key] = val
        yield data_id, raw_prompt, extra


def load_conversation_dataset(args):
    """Load conversation dataset and optionally tokenize for training.

    When defer_tokenization=True, only applies the chat template to produce
    formatted text — no tokenizer is loaded and no input_ids/loss_mask are
    generated. The inference engine handles tokenization and media token
    expansion; loss mask is computed at training time from the engine's
    actual input_ids.

    When defer_tokenization=False (default), fully tokenizes and produces
    input_ids + packed_loss_mask for the input_ids engine path.

    Returns list of dicts. Fields depend on mode:
        defer_tokenization=True:  data_id, formatted_prompt, multimodal_inputs, metadata
        defer_tokenization=False: data_id, input_ids, packed_loss_mask, formatted_prompt, multimodal_inputs, metadata
    """
    prompt_key = getattr(args, "prompt_key", "text")
    chat_template_name = getattr(args, "chat_template", None)
    # Reserve 1 token: the inference engine must generate >=1 token to extract
    # hidden states, so a prompt of exactly max_seq_length is rejected
    # (prompt_len + 1 > max_model_len). Truncating to max_seq_length - 1 keeps
    # the vast majority of samples out of the engine-side length skip guard.
    max_length = args.max_seq_length - 1
    defer_tokenization = getattr(args, "defer_tokenization", False)
    min_seq_length = int(getattr(args, "min_seq_length", 0) or 0)
    if min_seq_length and defer_tokenization:
        raise ValueError(
            "dataset.min_seq_length needs token counts, which defer_tokenization leaves to the engine"
        )
    enable_multimodal = getattr(args, "vllm_enable_multimodal", False) or getattr(
        args, "sglang_enable_multimodal", False
    )

    logger.info(f"Max sequence length allowed for training: {max_length}")

    if not chat_template_name:
        raise ValueError("chat_template must be set for load_conversation_dataset")

    custom_template = TEMPLATE_REGISTRY.get(chat_template_name)
    hf_dataset = load_hf_dataset(args.train_data_path, keep_columns=(prompt_key,))

    dataset_name = os.path.basename(args.train_data_path)
    file_stat = ""
    if os.path.isfile(args.train_data_path):
        st = os.stat(args.train_data_path)
        file_stat = f"-{st.st_size}-{st.st_mtime}"
    last_turn_loss_only_flag = getattr(args, "last_turn_loss_only", False)
    train_with_decode = getattr(args, "train_with_decode", False)
    min_loss_tokens_val = getattr(args, "min_loss_tokens", 0)
    drop_overlength_flag = getattr(args, "drop_overlength", False)

    mm_processor_kwargs = None
    if enable_multimodal and not defer_tokenization:
        mm_kw = getattr(args, "vllm_mm_processor_kwargs", None)
        if mm_kw:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(mm_kw, DictConfig):
                mm_kw = OmegaConf.to_container(mm_kw, resolve=True)
            mm_processor_kwargs = mm_kw
    image_newline_as_placeholder = (
        enable_multimodal
        and not defer_tokenization
        and custom_template.image_layout == "hy4"
        and resolve_image_newline_rewrite(args) is not None
    )

    cache_params = (
        f"{dataset_name}-{args.train_data_path}{file_stat}-{args.target_model_path}"
        f"-{max_length}-{chat_template_name}-{prompt_key}-ltlo={last_turn_loss_only_flag}"
        f"-defer={defer_tokenization}-decode={train_with_decode}"
        f"-mlt={min_loss_tokens_val}-drop={drop_overlength_flag}"
        f"-mm={enable_multimodal}-img={custom_template.image_layout}"
    )
    # Pixel bounds set the image token counts, and the newline rewrite sets the
    # image row layout; both are baked into the tokenized samples.
    if mm_processor_kwargs:
        cache_params += f"-mmkw={json.dumps(mm_processor_kwargs, sort_keys=True, default=str)}"
    if image_newline_as_placeholder:
        # v2 records flattened token offsets for the [1, seq_len] tensors
        # returned by preprocess_conversations.  Do not reuse v1 caches, whose
        # image newline slots were batch coordinates (all zero).
        cache_params += "-newline_slots=placeholder-v2"
    if min_seq_length:
        # Only when set, so caches built before min_seq_length existed keep their keys.
        cache_params += f"-minlen={min_seq_length}"
    cache_key = hashlib.md5(cache_params.encode()).hexdigest()
    cache_dir = os.path.join(getattr(args, "cache_dir", "./cache"), "tokenized_dataset")
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(cache_path):
        logger.info(f"Loading dataset from cache: {cache_path}")
        try:
            prompts = torch.load(cache_path, weights_only=False)
            logger.info(f"Loaded {len(prompts)} cached samples")
            return prompts
        except Exception as exc:
            corrupt_path = f"{cache_path}.corrupt.{int(time.time())}"
            os.replace(cache_path, corrupt_path)
            logger.warning(f"Corrupt dataset cache moved to {corrupt_path}; rebuilding: {exc}")

    mode_label = "Formatting" if defer_tokenization else "Tokenizing"
    logger.info(f"{mode_label} dataset (cache will be saved to {cache_path})")

    register_extra_hf_configs()

    total_estimate = estimate_row_count(args.train_data_path)
    num_proc = getattr(args, "num_proc", 64)

    stream_loading = getattr(args, "stream_dataset_loading", False)

    last_turn_loss_only = getattr(args, "last_turn_loss_only", False)
    if defer_tokenization:
        worker_init = _init_format_worker
        worker_initargs = (args.target_model_path, True, chat_template_name, last_turn_loss_only)
        worker_fn = _format_single
        desc = "Processing dataset"
    else:
        if last_turn_loss_only:
            logger.info(
                f"last_turn_loss_only={last_turn_loss_only}: loss mask will only cover the last assistant turn"
            )
        min_loss_tokens = getattr(args, "min_loss_tokens", 0)
        drop_overlength = getattr(args, "drop_overlength", False)
        worker_init = _init_tokenize_worker
        worker_initargs = (
            args.target_model_path,
            True,
            chat_template_name,
            last_turn_loss_only,
            min_loss_tokens,
            drop_overlength,
            mm_processor_kwargs if enable_multimodal else None,
            image_newline_as_placeholder,
            min_seq_length,
        )
        worker_fn = _tokenize_single
        desc = "Processing dataset"

    if stream_loading:
        prompts, skipped = _load_streaming(
            hf_dataset,
            prompt_key,
            total_estimate,
            max_length,
            train_with_decode,
            num_proc,
            worker_init,
            worker_initargs,
            worker_fn,
            desc,
            defer_tokenization,
            buffer_size=getattr(args, "stream_buffer_size", 4096),
        )
    else:
        prompts, skipped = _load_materialized(
            hf_dataset,
            prompt_key,
            total_estimate,
            mode_label,
            max_length,
            train_with_decode,
            num_proc,
            worker_init,
            worker_initargs,
            worker_fn,
            desc,
            defer_tokenization,
        )

    if skipped:
        reasons = "empty source, zero loss mask"
        if getattr(args, "drop_overlength", False):
            reasons += f", or > {max_length} tokens (drop_overlength)"
        if min_seq_length:
            reasons += f", or < {min_seq_length} tokens (min_seq_length)"
        logger.warning(f"Skipped {skipped} samples ({reasons})")

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(prompts, tmp_path)
    os.replace(tmp_path, cache_path)
    logger.info(f"Saved {len(prompts)} samples to cache: {cache_path}")

    return prompts


def load_training_dataset(args):
    """Dispatch to the right loader based on data format.

    Routes to :func:`load_megatron_bin_dataset` when ``train_data_path`` points
    at a Megatron ``.bin``/``.idx`` dataset (auto-detected) or when
    ``args.data_format == "megatron_bin"``; otherwise to
    :func:`load_conversation_dataset`. Both return the same list[dict] schema.
    """
    from angelspec.data.megatron_bin import (
        is_megatron_bin_path,
        load_megatron_bin_blend,
        load_megatron_bin_dataset,
    )

    if getattr(args, "bin_data_list", None):
        logger.info("Using Megatron .bin/.idx weighted blend loader (bin_data_list)")
        return load_megatron_bin_blend(args)

    fmt = getattr(args, "data_format", None)
    path = getattr(args, "train_data_path", None)
    if fmt == "megatron_bin" or (fmt in (None, "auto") and is_megatron_bin_path(path)):
        logger.info("Using Megatron .bin/.idx loader (offline-packed SFT data)")
        return load_megatron_bin_dataset(args)
    return load_conversation_dataset(args)


def _load_materialized(
    hf_dataset,
    prompt_key,
    total_estimate,
    mode_label,
    max_length,
    train_with_decode,
    num_proc,
    worker_init,
    worker_initargs,
    worker_fn,
    desc,
    defer_tokenization,
):
    """Materialize all raw samples, then process in one parallel pass."""
    raw_samples = list(_iter_raw_samples(hf_dataset, prompt_key, total_estimate))

    logger.info(
        f"Loaded {len(raw_samples)} samples, {mode_label.lower()} with {num_proc} workers..."
    )

    work_items = [
        (raw_prompt, max_length, train_with_decode, extra) for _, raw_prompt, extra in raw_samples
    ]

    if num_proc <= 1:
        worker_init(*worker_initargs)
        results = [worker_fn(item) for item in tqdm(work_items, desc=desc)]
    else:
        with mp.Pool(num_proc, initializer=worker_init, initargs=worker_initargs) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_fn, work_items, chunksize=64),
                    total=len(work_items),
                    desc=desc,
                )
            )

    prompts = []
    skipped = 0
    for (data_id, _, _), result in zip(raw_samples, results):
        entry = _build_entry(data_id, result, defer_tokenization)
        if entry is None:
            skipped += 1
            continue
        prompts.append(entry)

    return prompts, skipped


def _load_streaming(
    hf_dataset,
    prompt_key,
    total_estimate,
    max_length,
    train_with_decode,
    num_proc,
    worker_init,
    worker_initargs,
    worker_fn,
    desc,
    defer_tokenization,
    buffer_size=4096,
):
    """Streaming variant: lazily feed workers in bounded windows.

    Raw samples, work items and results are never fully materialized. Only a
    bounded window (``buffer_size``) of in-flight items plus the final
    ``prompts`` list live in memory, keeping the host-memory peak low for very
    large / long-context datasets. Produces byte-identical entries to
    :func:`_load_materialized` (same worker fns, so ``input_ids`` is populated
    when ``defer_tokenization`` is False).
    """
    logger.info(f"Streaming dataset load with {num_proc} workers (buffer_size={buffer_size})...")

    raw_iter = _iter_raw_samples(hf_dataset, prompt_key, total_estimate)

    prompts = []
    skipped = 0

    if num_proc <= 1:
        worker_init(*worker_initargs)
        for data_id, raw_prompt, extra in raw_iter:
            item = (raw_prompt, max_length, train_with_decode, extra)
            entry = _build_entry(data_id, worker_fn(item), defer_tokenization)
            if entry is None:
                skipped += 1
            else:
                prompts.append(entry)
        return prompts, skipped

    # Chunk the lazy stream so imap's internal prefetch stays bounded: submit at
    # most ``buffer_size`` items, drain them, then submit the next window.
    chunk = max(64, buffer_size // max(1, num_proc))
    pbar = tqdm(total=total_estimate, desc=desc)
    with mp.Pool(num_proc, initializer=worker_init, initargs=worker_initargs) as pool:
        while True:
            data_ids = []
            work_items = []
            for data_id, raw_prompt, extra in raw_iter:
                data_ids.append(data_id)
                work_items.append((raw_prompt, max_length, train_with_decode, extra))
                if len(work_items) >= buffer_size:
                    break

            if not work_items:
                break

            for data_id, result in zip(
                data_ids, pool.imap(worker_fn, work_items, chunksize=chunk)
            ):
                entry = _build_entry(data_id, result, defer_tokenization)
                if entry is None:
                    skipped += 1
                else:
                    prompts.append(entry)
                pbar.update(1)

            del data_ids, work_items

    pbar.close()
    return prompts, skipped
