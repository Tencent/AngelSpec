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

"""
Decode-mode mixin for SglEngine.

Provides speculative-decoding generation and draft model weight sync.
Mixed into SglEngine when train_with_decode is enabled.
"""

from __future__ import annotations

from typing import Any

import ray
import torch

from angelspec.data.utils import serialize_packed_loss_mask
from angelspec.utils.logging import logger


class SglDecodeEngineMixin:
    """Mixin that adds decode-mode generation and weight sync to SglEngine.

    Expects the host class to provide: ``self.args``, ``self.rank``,
    ``self._engine``, ``self._get_tensor_shapes()``, ``self._get_tensor_dtypes()``,
    ``self._extract_image_data()``.
    """

    # -- Engine kwargs --------------------------------------------------------

    # Mapping from args attr -> engine kwarg for optional decode params.
    _DECODE_OPTIONAL_KWARGS = [
        ("decode_cuda_graph_max_bs", "cuda_graph_max_bs"),
        ("decode_max_running_requests", "max_running_requests"),
    ]
    _DECODE_SPEC_KWARGS = [
        ("decode_speculative_num_steps", "speculative_num_steps"),
        ("decode_speculative_eagle_topk", "speculative_eagle_topk"),
        ("decode_speculative_num_draft_tokens", "speculative_num_draft_tokens"),
        ("sglang_speculative_draft_attention_backend", "speculative_draft_attention_backend"),
    ]

    def _build_decode_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Populate *engine_kwargs* with decode-mode speculative decoding params."""
        for attr, kwarg in self._DECODE_OPTIONAL_KWARGS:
            if (val := getattr(self.args, attr, None)) is not None:
                engine_kwargs[kwarg] = val

        spec_algorithm = getattr(self.args, "decode_speculative_algorithm", None)
        if not spec_algorithm:
            return

        spec_draft_path = getattr(self.args, "decode_speculative_draft_model_path", None)
        if spec_draft_path is None:
            raise ValueError(
                "decode.speculative_draft_model_path is null. It is normally auto-created "
                "by _maybe_create_scratch_draft when train_with_decode=true. Set it "
                "explicitly or check that speculative_algorithm is configured correctly."
            )
        engine_kwargs["speculative_algorithm"] = spec_algorithm
        engine_kwargs["speculative_draft_model_path"] = spec_draft_path
        engine_kwargs["speculative_draft_model_quantization"] = "unquant"

        for attr, kwarg in self._DECODE_SPEC_KWARGS:
            if (val := getattr(self.args, attr, None)) is not None:
                engine_kwargs[kwarg] = val

        logger.info(
            f"SglEngine rank {self.rank}: decode mode enabled with algorithm={spec_algorithm}, draft={spec_draft_path}"
        )

    # -- Generation -----------------------------------------------------------

    def _decode_prompt_token_lengths(
        self,
        *,
        use_prompts: bool,
        formatted_prompts: list[str] | None,
        input_ids_list: list[torch.Tensor] | None,
    ) -> list[int | None]:
        if not use_prompts:
            lengths: list[int | None] = []
            assert input_ids_list is not None
            for ids in input_ids_list:
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                lengths.append(int(ids.numel()))
            return lengths

        tokenizer_manager = getattr(self._engine, "tokenizer_manager", None)
        tokenizer = getattr(tokenizer_manager, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return [None for _ in formatted_prompts or []]

        lengths = []
        for prompt in formatted_prompts or []:
            try:
                try:
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                except TypeError:
                    token_ids = tokenizer.encode(prompt)
                lengths.append(len(token_ids))
            except Exception as exc:
                logger.warning(
                    f"SglEngine rank {self.rank}: failed to estimate prompt token "
                    f"length before decode ({exc!r}); letting SGLang tokenize it."
                )
                lengths.append(None)
        return lengths

    def generate_with_decode(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any] | None]:
        """Generate training data with decoding (spec training with actual token generation).

        Unlike generate() which does prefill-only, this method generates new tokens
        and captures hidden states for the full prompt+completion sequence.

        Accepts either pre-tokenized input_ids or formatted prompt strings.
        Exactly one of input_ids_ref or formatted_prompts must be set.

        Args:
            data_id: Data ID(s) for the batch.
            input_ids_ref: Ray ObjectRef or list of input_ids tensors.
            packed_loss_mask_list: List of packed loss_mask strings.
            formatted_prompts: List of already chat-template-formatted prompt strings.
            return_last_hidden_states: Ignored, always stored in mooncake.
            return_logits: Ignored, always stored in mooncake.
            multimodal_inputs: List of multimodal input dicts.

        Returns:
            List of dicts with mooncake_key and tensor metadata.
            Returns None entries for samples where mooncake storage failed.
        """
        if self._engine is None:
            raise RuntimeError("SglEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None

        if use_prompts:
            batch_size = len(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            batch_size = len(input_ids_list)

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = data_id
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        # Build sampling params for decode mode
        max_new_tokens = getattr(self.args, "decode_max_new_tokens", 512)
        min_new_tokens = getattr(self.args, "decode_min_new_tokens", 2)
        if min_new_tokens < 0:
            raise ValueError(f"decode.min_new_tokens must be >= 0, got {min_new_tokens}")
        if min_new_tokens > max_new_tokens:
            raise ValueError(
                f"decode.min_new_tokens ({min_new_tokens}) cannot exceed decode.max_new_tokens ({max_new_tokens})"
            )
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
        }
        stop_token_ids = getattr(self.args, "decode_stop_token_ids", None)
        if stop_token_ids:
            sampling_params["stop_token_ids"] = list(stop_token_ids)
        stop_token_id_set = set(stop_token_ids or [])
        temperature = getattr(self.args, "decode_temperature", 1.0)
        if temperature != 1.0:
            sampling_params["temperature"] = temperature
        top_p = getattr(self.args, "decode_top_p", 1.0)
        if top_p < 1.0:
            sampling_params["top_p"] = top_p
        top_k = getattr(self.args, "decode_top_k", -1)
        if top_k > 0:
            sampling_params["top_k"] = top_k
        logger.debug(f"SglEngine rank {self.rank}: decode sampling_params={sampling_params}")

        outputs: list[dict[str, Any] | None] = [None for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        max_seq_length = getattr(self.args, "max_seq_length", None)
        if max_seq_length:
            prompt_lengths = self._decode_prompt_token_lengths(
                use_prompts=use_prompts,
                formatted_prompts=formatted_prompts,
                input_ids_list=None if use_prompts else input_ids_list,
            )
            active_indices = []
            for i, prompt_len in enumerate(prompt_lengths):
                if prompt_len is not None and prompt_len + min_new_tokens > max_seq_length:
                    logger.warning(
                        f"SglEngine rank {self.rank}: skipping data_id={data_ids[i]} "
                        f"because prompt_tokens={prompt_len} leaves less than "
                        f"min_new_tokens={min_new_tokens} within max_seq_length={max_seq_length}"
                    )
                    continue
                active_indices.append(i)

        if not active_indices:
            return outputs

        active_data_ids = [data_ids[i] for i in active_indices]
        active_multimodal_inputs = (
            [multimodal_inputs[i] for i in active_indices]
            if multimodal_inputs is not None
            else None
        )

        if use_prompts:
            active_formatted_prompts = [formatted_prompts[i] for i in active_indices]
            logger.debug(
                f"SglEngine rank {self.rank}: decode prompt mode processing data_ids={active_data_ids}, "
                f"num_prompts={len(active_formatted_prompts)}"
            )
            engine_kwargs = {
                "prompt": active_formatted_prompts,
                "spec_training_data_id": active_data_ids,
                "sampling_params": sampling_params,
                "return_hidden_states": True,
            }
        else:
            input_ids_list_of_lists = []
            for i in active_indices:
                ids = input_ids_list[i]
                if ids.dim() == 2 and ids.shape[0] == 1:
                    ids = ids.squeeze(0)
                elif ids.dim() > 2:
                    raise ValueError(f"Unexpected input_ids shape: {ids.shape}")
                input_ids_list_of_lists.append(ids.tolist())

            active_packed_loss_mask_list = (
                [packed_loss_mask_list[i] for i in active_indices]
                if packed_loss_mask_list is not None
                else None
            )
            logger.debug(
                f"SglEngine rank {self.rank}: decode mode processing data_ids={active_data_ids}, "
                f"shapes: {[len(ids) for ids in input_ids_list_of_lists]}"
            )
            engine_kwargs = {
                "input_ids": input_ids_list_of_lists,
                "spec_training_data_id": active_data_ids,
                "packed_loss_mask": active_packed_loss_mask_list,
                "sampling_params": sampling_params,
                "return_hidden_states": True,
            }

        image_data = self._extract_image_data(active_multimodal_inputs)
        if image_data is not None:
            engine_kwargs["image_data"] = image_data

        results = self._engine.generate(**engine_kwargs)
        if len(results) != len(active_indices):
            raise RuntimeError(
                f"SglEngine rank {self.rank}: decode expected {len(active_indices)} "
                f"results from SGLang, got {len(results)}"
            )

        # IMPORTANT: Must produce exactly one output per input result to match
        # the zip(entries, outputs, strict=True) in the inference manager.
        for result_index, result in enumerate(results):
            i = active_indices[result_index]
            store_keys = result["meta_info"].get("spec_training_mooncake_store_keys", [])
            if not store_keys:
                logger.warning(
                    f"SglEngine rank {self.rank}: No mooncake keys returned for "
                    f"data_id={active_data_ids[result_index]}, skipping this sample."
                )
                continue

            meta_info = result["meta_info"]
            if "e2e_latency" in meta_info:
                metrics_msg = f"SglEngine rank {self.rank}: DECODE e2e_latency={meta_info['e2e_latency']:.4f}s"
                if "spec_accept_rate" in meta_info:
                    metrics_msg += f", spec_accept_rate={meta_info['spec_accept_rate']:.2%}"
                if "spec_accept_length" in meta_info:
                    metrics_msg += f", spec_accept_length={meta_info['spec_accept_length']:.2f}"
                logger.debug(metrics_msg)

            key = store_keys[0]
            prompt_tokens = meta_info.get("prompt_tokens", 0)
            completion_tokens = meta_info.get("completion_tokens", 0)
            output_ids = result.get("output_ids", [])
            if completion_tokens <= 1:
                text = result.get("text", "")
                logger.warning(
                    f"SglEngine rank {self.rank}: completion_tokens={completion_tokens} for "
                    f"data_id={active_data_ids[result_index]}, finish_reason={meta_info.get('finish_reason')}, "
                    f"output_ids={output_ids}, text={text!r}, dropping sample to avoid "
                    f"a zero-loss mask"
                )
                continue
            if output_ids and output_ids[0] in stop_token_id_set:
                logger.warning(
                    f"SglEngine rank {self.rank}: completion starts with stop token "
                    f"{output_ids[0]} for data_id={active_data_ids[result_index]}, "
                    f"finish_reason={meta_info.get('finish_reason')}, dropping sample."
                )
                continue

            seq_len = prompt_tokens + completion_tokens - 1
            logger.debug(
                f"SglEngine rank {self.rank}: decode mode - "
                f"prompt={prompt_tokens}, completion={completion_tokens}, "
                f"seq_len={seq_len}"
            )

            tensor_shapes = self._get_tensor_shapes(seq_len)
            logger.debug(
                f"SglEngine rank {self.rank}: mooncake_key={key}, seq_len={seq_len}, tensor_shapes={tensor_shapes}"
            )

            # Build loss mask: skip prompt tokens, compute loss on completion tokens.
            # The -1 accounts for next-token prediction (last token has no target).
            if completion_tokens > 0:
                loss_mask = serialize_packed_loss_mask([prompt_tokens, completion_tokens - 1])
            else:
                loss_mask = serialize_packed_loss_mask([prompt_tokens])

            output_dict = {
                "mooncake_key": key,
                "tensor_shapes": tensor_shapes,
                "tensor_dtypes": self._get_tensor_dtypes(),
                "packed_loss_mask": loss_mask,
            }
            if getattr(self.args, "attention_backend", None) == "usp":
                output_dict["metadata"] = {"usp_sharded": True}

            # Add performance metrics if available (for wandb logging)
            _METRIC_KEYS = (
                "e2e_latency",
                "spec_accept_rate",
                "spec_accept_length",
                "prompt_tokens",
                "completion_tokens",
            )
            output_dict.update({k: meta_info[k] for k in _METRIC_KEYS if k in meta_info})

            outputs[i] = output_dict

        accepted_count = sum(output is not None for output in outputs)
        logger.debug(
            f"SglEngine rank {self.rank}: decode generated {accepted_count} mooncake keys "
            f"for {len(data_ids)} data_ids={data_ids}"
        )
        return outputs

    # -- Weight sync ----------------------------------------------------------

    def update_weights_from_disk(self, model_path: str) -> dict:
        """Update draft model weights from disk without restarting the engine.

        Args:
            model_path: Path to the saved draft model weights.

        Returns:
            Dict with 'success' and 'message' keys.
        """
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

        if self._engine is None:
            return {"success": False, "message": "Engine not initialized"}

        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            update_draft_model=True,
            flush_cache=True,
        )

        # sgl.Engine.update_weights_from_disk() does not expose update_draft_model=True.
        # Access the internal event loop and tokenizer_manager directly to pass this flag.
        result = self._engine.loop.run_until_complete(
            self._engine.tokenizer_manager.update_weights_from_disk(obj, None)
        )

        success = getattr(result, "success", False)
        message = getattr(result, "message", str(result))
        logger.info(
            f"SglEngine rank {self.rank}: update_weights_from_disk -> success={success}, message={message}"
        )
        return {"success": success, "message": message}
