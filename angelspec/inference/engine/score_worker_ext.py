"""Packed tree-forward scoring — vLLM worker extension.

Runs a cache-less encoder-only prefill over a packed ``[trunk | branch_0 | ...]``
row with a custom FlexAttention tree mask, and returns the target model's
pre-lm_head (post-final-norm) hidden at the requested predecessor slots.

This is mixed into the vLLM worker via ``worker_extension_cls`` and invoked with
``collective_rpc("score_packed", ...)`` (single-threaded on the worker, serialized
with execute_model). It requires the scoring engine to be built **encoder-only**
(Qwen3 ``is_causal=False`` → every layer is ``EncoderOnlyAttention``, no KV cache,
and ``FlexAttentionImpl.forward`` takes the ENCODER_ONLY branch that consumes the
passed q/k/v directly). A normally-served decoder engine's layers are
``attn_type=DECODER`` and would read the paged KV cache instead — so the entry
guard hard-fails if any attention layer is not ENCODER_ONLY (a wrong teacher
distribution would silently poison training, never crash).
"""

import torch


class ScorePackedWorkerExt:
    """Mixin providing ``score_packed`` on the vLLM worker."""

    def _score_attention_layers(self):
        """All attention modules in the model, keyed by their ``layer_name``.

        Fail-closed: raises if any is not encoder-only. ``is_causal=False`` must
        have taken effect at construction; a DECODER layer here means the forward
        would read an (empty) paged cache and produce a wrong teacher hidden.
        """
        from vllm.v1.attention.backend import AttentionType

        model = self.model_runner.model
        layers = {}
        for name, mod in model.named_modules():
            if hasattr(mod, "attn_type") and hasattr(mod, "layer_name"):
                if mod.attn_type != AttentionType.ENCODER_ONLY:
                    raise RuntimeError(
                        f"score_packed requires an encoder-only scoring engine "
                        f"(build with is_causal=False); layer {name!r} has "
                        f"attn_type={mod.attn_type!r}. Refusing to run — a decoder "
                        f"layer reads the paged KV cache and yields a wrong teacher "
                        f"distribution that silently poisons training."
                    )
                layers[mod.layer_name] = mod
        if not layers:
            raise RuntimeError("score_packed: no attention layers found on the model.")
        return layers

    def _build_flex_metadata(self, num_tokens: int, block_mask, device):
        """Hand-build a cache-less FlexAttentionMetadata carrying our tree mask.

        The ENCODER_ONLY forward branch (flex_attention.py) reads only
        ``num_actual_tokens`` + ``block_mask`` (+ sliding_window/logical_mask_mod
        rebuild triggers, which stay inert here). The paged fields are never read,
        so they are set to EMPTY tensors — any accidental paged access blows up
        loudly rather than reading plausible-looking garbage.
        """
        from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

        long = dict(dtype=torch.long, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        empty2d = torch.empty((0, 0), **long)
        block_m, block_n = block_mask.BLOCK_SIZE
        return FlexAttentionMetadata(
            causal=False,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            query_start_loc=torch.tensor([0, num_tokens], **long),
            query_start_loc_cpu=torch.tensor([0, num_tokens], dtype=torch.long),
            max_seq_len=num_tokens,
            seq_lens=torch.tensor([num_tokens], **long),
            block_table=empty2d,
            slot_mapping=torch.empty((0,), **long),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            total_cache_tokens=0,
            block_size=block_n,
            max_possible_sequence_length=num_tokens,
            num_reqs=1,
            physical_to_logical=empty2d,
            decode_offset=torch.zeros((1,), **long),
            num_blocks_per_seq=torch.zeros((1,), **long),
            persistent_kv_indices=torch.empty((0, 0), **i32),
            persistent_kv_num_blocks=torch.empty((0, 0), **i32),
            # __post_init__ writes per-token request ids here → must be sized >= N.
            persistent_doc_ids=torch.zeros((num_tokens,), **i32),
            num_input_tokens=num_tokens,
            block_mask=block_mask,
            uses_paged_kv=False,
            direct_build=False,
            q_block_size=block_m,
            kv_block_size=block_n,
        )

    @torch.inference_mode()
    def score_packed(
        self,
        packed_ids,
        positions,
        doc_ids,
        anchor_of,
        score_index,
        trunk_doc_of=None,
    ):
        """Encoder-only tree-forward; returns selected pre-lm_head hidden.

        Args (CPU tensors ok — moved to device here):
            packed_ids: (N,) long — [trunk | branch_0 | ...].
            positions:  (N,) long — absolute RoPE positions.
            doc_ids:    (N,) long — 0=trunk, k+1=branch_k, -1=pad (tree mask input).
            anchor_of:  (N,) long — branch token's trunk anchor; -1 for trunk/pad.
            score_index:(M,) long — predecessor slots to return.

        Returns:
            (M, H) float tensor (cpu) — the selected hidden states.
        """
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        from angelspec.models.ops.flex_attention import (
            compile_friendly_create_block_mask,
        )
        from angelspec.models.ops.tree_mask import create_tree_mask_mod

        layers = self._score_attention_layers()  # fail-closed guard

        # Under runner=pooling the top module is a wrapper (e.g. Qwen3ForEmbedding,
        # a Qwen3ForCausalLM subclass). Its ``.model`` is the base Qwen3Model whose
        # forward returns post-final-norm (== pre-lm_head) hidden. Embeddings are
        # reached via ``embed_input_ids`` (this fork) or ``get_input_embeddings``.
        wrapper = self.model_runner.model
        model = getattr(wrapper, "model", wrapper)
        if hasattr(model, "embed_input_ids"):
            embed_fn = model.embed_input_ids
        elif hasattr(model, "get_input_embeddings"):
            embed_fn = model.get_input_embeddings()
        else:
            raise RuntimeError(f"score_packed: no embedding accessor on {type(model).__name__}.")
        device = next(model.parameters()).device
        # collective_rpc serializes tensors to plain lists over the mp channel;
        # rebuild them here (as_tensor is a no-op if a tensor is passed directly).
        packed_ids = torch.as_tensor(packed_ids, dtype=torch.long, device=device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=device)
        doc_ids = torch.as_tensor(doc_ids, dtype=torch.long, device=device)
        anchor_of = torch.as_tensor(anchor_of, dtype=torch.long, device=device)
        score_index = torch.as_tensor(score_index, dtype=torch.long, device=device)
        trunk_doc = (
            torch.as_tensor(trunk_doc_of, dtype=torch.long, device=device).unsqueeze(0)
            if trunk_doc_of is not None
            else None
        )
        n = int(packed_ids.shape[0])

        # RoPE bound (positions are only a cache index; over-bound → silent NaN).
        max_pos = getattr(self.vllm_config.model_config.hf_config, "max_position_embeddings", None)
        if max_pos is not None and n and int(positions.max()) >= max_pos:
            raise ValueError(f"positions exceed RoPE bound: {int(positions.max())} >= {max_pos}")

        # Tree block mask (whole row = one request; our mask owns all visibility).
        mask_mod = create_tree_mask_mod(doc_ids.unsqueeze(0), anchor_of.unsqueeze(0), trunk_doc)
        block_mask = compile_friendly_create_block_mask(
            mask_mod, 1, 1, n, n, device=device, _compile=True
        )
        md = self._build_flex_metadata(n, block_mask, device)
        md_dict = {name: md for name in layers}

        embeds = embed_fn(packed_ids)
        with set_forward_context(
            md_dict,
            self.vllm_config,
            num_tokens=n,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping={},
        ):
            hidden = model(input_ids=None, positions=positions, inputs_embeds=embeds)

        sel = hidden.index_select(0, score_index).to(torch.bfloat16).contiguous()
        # collective_rpc mangles tensor return values, so hand back raw bytes: the
        # (M, H) bf16 hidden reinterpreted as a contiguous uint8 blob. The caller
        # rebuilds it with torch.frombuffer(dtype=bfloat16).view(M, H). bf16 matches
        # the trainer's target dtype exactly (it previously cast fp32 -> bf16), and a
        # single memcpy-level blob avoids pickling ~M*H PyFloat objects per call
        # (the dominant OPD RPC overhead). (Production will instead write _tgt to the
        # eagle store and return nothing.)
        return sel.cpu().view(torch.uint8).numpy().tobytes()
