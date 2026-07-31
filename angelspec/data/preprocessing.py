# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import Counter
from typing import Dict, List, Tuple, Union

import numba
import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None

from angelspec.data.parse import create_parser
from angelspec.data.template import ChatTemplate
from angelspec.data.utils import (
    pack_loss_mask,
    serialize_packed_loss_mask,
    unpack_loss_mask,
)

Conversation = List[Dict[str, str]]

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}


def _normalize_conversation(conversation: Conversation) -> Conversation:
    """
    Normalize conversation format to use role/content keys.
    Handles ShareGPT format (from/value) and converts to standard format (role/content).
    """
    if not conversation:
        return conversation

    first_msg = conversation[0]
    if "role" in first_msg and "content" in first_msg:
        return conversation

    if "from" in first_msg and "value" in first_msg:
        normalized = []
        for msg in conversation:
            role = ROLE_MAPPING.get(msg["from"], msg["from"])
            entry = {"role": role, "content": msg["value"]}
            for field in ("thinking", "thinking_content", "reasoning_content", "reasoning"):
                if msg.get(field):
                    entry["reasoning_content"] = msg[field]
                    break
            normalized.append(entry)
        return normalized

    return conversation


# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Conversation], List[str]],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    is_preformatted: bool = False,
    include_attention_mask: bool = True,
    use_packed_loss_mask: bool = True,
    add_generation_prompt: bool = False,
    return_formatted_text: bool = False,
    last_turn_loss_only: bool = False,
    min_loss_tokens: int = 0,  # DFlash: skip sequences with too few supervised tokens for anchor sampling
    drop_overlength: bool = False,  # drop (not truncate) samples whose token count exceeds max_length
    **kwargs,
) -> Dict[str, List]:
    """
    Preprocess a batch of ShareGPT style conversations or pre-formatted text.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        conversations: A list of conversations (if is_preformatted=False) or
                      a list of pre-formatted text strings (if is_preformatted=True).
        chat_template: The chat template to use for formatting/identifying spans.
        max_length: The maximum length of the tokenized input.
        is_preformatted: Whether the input is already formatted text strings.
        include_attention_mask: Whether to include attention_mask in the output.
                               If False, attention_mask can be generated as ones on the inference side.
        use_packed_loss_mask: If True, store loss_mask as packed string (compact).
                             If False, store as tensor (legacy behavior).
        add_generation_prompt: Whether to append assistant header for generation.
                               Only used when is_preformatted=False.
        return_formatted_text: If True, include 'formatted_text' in the output.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - packed_loss_mask: List of packed loss mask strings (if use_packed_loss_mask=True).
            - loss_mask: List of loss mask tensors (if use_packed_loss_mask=False).
            - attention_mask: List of attention masks (only if include_attention_mask=True).
            - formatted_text: List of formatted text strings (only if return_formatted_text=True).
    """

    if use_packed_loss_mask:
        results = {"input_ids": [], "packed_loss_mask": []}
    else:
        results = {"input_ids": [], "loss_mask": []}
    if include_attention_mask:
        results["attention_mask"] = []
    if return_formatted_text:
        results["formatted_text"] = []

    parser = create_parser(tokenizer, chat_template)

    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for source, kwargs_item in zip(conversations, kwargs_list):
        if not source:
            continue
        source = _normalize_conversation(source)

        if is_preformatted:
            formatted = source
        else:
            formatted = parser.format(
                source, add_generation_prompt=add_generation_prompt, **kwargs_item
            )

        # Drop (rather than truncate) samples longer than max_length. Truncation
        # is lossy and indistinguishable after the fact, so probe the untruncated
        # length here and skip the whole sample.
        # add_special_tokens=False matches the parse() tokenization below.
        if drop_overlength:
            n_tokens = len(tokenizer(formatted, add_special_tokens=False).input_ids)
            if n_tokens > max_length:
                continue

        input_ids, loss_mask = parser.parse(
            formatted,
            max_length,
            preformatted=True,
            last_turn_only=last_turn_loss_only,
        )

        # Skip sequences with insufficient supervised tokens. For DFlash, anchor
        # sampling needs enough valid positions; sequences with near-zero loss_mask
        # produce zero gradients and waste training compute.
        if loss_mask.sum() < max(1, min_loss_tokens):
            continue

        results["input_ids"].append(input_ids[None, :])
        if use_packed_loss_mask:
            packed = pack_loss_mask(loss_mask)
            results["packed_loss_mask"].append(serialize_packed_loss_mask(packed))
        else:
            results["loss_mask"].append(loss_mask[None, :])
        if include_attention_mask:
            results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        if return_formatted_text:
            results["formatted_text"].append(formatted)
    return results


def _count_token_frequencies(prompts: list) -> Counter:
    """Count token frequencies from tokenized prompts using packed_loss_mask."""

    @numba.njit(cache=True)
    def _histogram(ids, mask, counts):
        for i in range(len(ids)):
            if mask[i] == 1:
                counts[ids[i]] += 1

    all_ids = []
    all_masks = []
    for item in tqdm(prompts, desc="Preparing token data"):
        input_ids = item["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            ids_np = input_ids.squeeze().numpy().astype(np.int64)
        elif isinstance(input_ids, np.ndarray):
            ids_np = input_ids.astype(np.int64)
        else:
            ids_np = np.array(input_ids, dtype=np.int64)

        packed = item.get("packed_loss_mask")
        if packed is not None:
            mask_np = unpack_loss_mask(packed).numpy().astype(np.int64)
        else:
            mask_np = np.ones_like(ids_np)

        all_ids.append(ids_np)
        all_masks.append(mask_np)

    flat_ids = np.concatenate(all_ids)
    flat_masks = np.concatenate(all_masks)

    counts = np.zeros(int(flat_ids.max()) + 1, dtype=np.int64)
    _histogram(flat_ids, flat_masks, counts)

    nonzero = np.nonzero(counts)[0]
    return Counter(dict(zip(nonzero.tolist(), counts[nonzero].tolist())))


def generate_vocab_mapping(
    prompts: list,
    target_vocab_size: int,
    draft_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate vocab mapping (d2t, t2d) from tokenized prompts.

    Args:
        prompts: List of dicts with input_ids and packed_loss_mask
                 (as returned by load_conversation_dataset).
        target_vocab_size: The target vocabulary size.
        draft_vocab_size: The draft vocabulary size.

    Returns:
        (d2t, t2d) tensor tuple for vocab pruning.
    """
    token_dict = _count_token_frequencies(prompts)
    print(f"Found {len(token_dict)} unique tokens from {sum(token_dict.values())} total tokens")

    return process_token_dict_to_mappings(
        token_dict,
        draft_vocab_size,
        target_vocab_size,
    )


def process_token_dict_to_mappings(
    token_dict: Counter,
    draft_vocab_size: int,
    target_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
        print(
            f"Added {draft_vocab_size - len(existing_tokens)} missing tokens to reach draft vocab size: {draft_vocab_size}"
        )
    else:
        warnings.warn(
            f"Unique tokens ({len(token_dict)}) exceed draft vocab size ({draft_vocab_size}). "
            f"{len(token_dict) - draft_vocab_size} tokens will be dropped from the vocab mapping.",
            stacklevel=2,
        )
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        print("Warning: Total token frequency is zero. All tokens will have zero ratio.")
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = sorted(key for key, freq in top_N if key < target_vocab_size)
    used_arr = torch.tensor(used_tokens, dtype=torch.long)

    d2t = used_arr - torch.arange(len(used_arr))
    t2d = torch.zeros(target_vocab_size, dtype=torch.bool)
    t2d[used_arr] = True

    return d2t, t2d
