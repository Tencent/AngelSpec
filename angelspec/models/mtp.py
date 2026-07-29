"""TTT training wrapper for the single-head MTP draft.

Autoregressive test-time-training loop: embed the predicted token, run the MTP
backbone with a growing per-step KV cache, take the loss vs the target, shift
inputs left. Loss is CE + optional KL against the target head; the draft consumes
a single last-layer hidden state with a tied, frozen lm_head.

forward returns (ce_losses, kl_losses, vlosses, acces, acc_counts, acces_gt,
e2e_loss), one list entry per TTT step (e2e_loss is a scalar or None). vlosses is
the per-step CE for logging/eval. acces = draft argmax vs target argmax
(self-distill); acces_gt = draft argmax vs ground-truth token (the serve-acceptance
proxy). With single_step_form set the kl_losses slot carries the tv/kl/lk per-step
distill term; with e2e_weighting the per-step accept rates are coupled into e2e_loss.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from angelspec.models.ops.loss import mtp_loss_from_hs
from angelspec.utils.distributed import get_draft_sp_group
from angelspec.utils.tensor import padding, same_doc_next_mask


class MTPModel(nn.Module):
    def __init__(
        self,
        draft_model,
        length: int = 7,
        attention_backend: str = "sdpa",
        gradient_checkpointing: bool = False,
        use_kl: bool = False,
        kl_topk: Optional[int] = None,
        kl_variant: str = "a",
        ce_use_ground_truth: bool = False,
        on_policy: bool = False,
        single_step_form: Optional[str] = None,
        single_step_eta: float = 3.0,
        e2e_weighting: bool = False,
        e2e_direct: bool = False,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing
        self.use_kl = use_kl
        self.kl_topk = kl_topk
        self.kl_variant = kl_variant
        self.ce_use_ground_truth = ce_use_ground_truth
        # Acceptance-rate-aware distillation. single_step_form ("tv"/"kl"/"lk")
        # replaces the per-step KL term; e2e_weighting couples the per-step accept
        # rates into one loss (e2e_direct = differentiable objective, else W_m
        # surrogate). e2e requires single_step_form.
        self.single_step_form = single_step_form
        self.single_step_eta = float(single_step_eta)
        self.e2e_weighting = bool(e2e_weighting) and single_step_form is not None
        self.e2e_direct = bool(e2e_direct)
        # On-policy TTT: at step>=1 the draft embeds its OWN previous-step argmax
        # prediction (stop-grad) instead of the left-shifted ground-truth token;
        # step 0 stays teacher-forced. Off (default) keeps the teacher-forced rollout.
        self.on_policy = on_policy
        self.vocab_pruning = False  # MTP never prunes the vocab
        # Pure-Ulysses SP: each rank gets the FULL sample, slices its own seq shard,
        # runs the whole TTT loop locally (attention all-to-alls internally), and
        # per-step loss sums are all-reduced across the SP group. Ground-truth shifts,
        # masks, and teacher-hidden windows are prepared on the full sequence before
        # slicing so TTT targets remain exact across SP shard boundaries.
        self._usp_group = get_draft_sp_group() if attention_backend == "usp" else None
        self._usp_size = dist.get_world_size(self._usp_group) if self._usp_group is not None else 1
        self._usp_rank = dist.get_rank(self._usp_group) if self._usp_group is not None else 0

    def warmup_usp_flex(
        self,
        seq_length: int,
        hidden_size: int,
        target_lm_head_weight: torch.Tensor,
        packing: bool = False,
    ) -> None:
        """Compile the flex block-0 kernel identically on every SP rank BEFORE the
        training loop, so no rank triggers a mid-step inductor compile that would
        desync the Ulysses all-to-all collectives (→ NCCL hang).

        Runs one forward+backward on dummy data shaped like a real USP micro-batch
        so the shape-keyed flex_attention compile is cached. `seq_length` is the
        LOCAL shard length; the attention all-to-all expands it to the full seq.
        No-op unless attention_backend == "usp".

        The block-0 ``mask_mod`` has a Python-level ``if doc_ids is not None`` branch,
        so packing (doc_ids present) traces a STRUCTURALLY DIFFERENT graph than the
        plain path. When ``packing`` is on, warm up with a representative two-doc
        layout so the doc-gated kernel compiles here rather than mid-step; the doc-id
        VALUES don't key the kernel (shape-keyed), only the branch presence does.
        """
        if self.attention_backend != "usp":
            return
        import torch.distributed as dist

        device = next(self.draft_model.parameters()).device
        dtype = next(self.draft_model.parameters()).dtype
        full_seq = seq_length * self._usp_size  # MTPModel.forward slices its own shard
        input_ids = torch.zeros(1, full_seq, dtype=torch.long, device=device)
        attn = torch.ones(1, full_seq, dtype=dtype, device=device)
        hidden = torch.zeros(1, full_seq, hidden_size, dtype=dtype, device=device)
        loss_mask = torch.ones(1, full_seq, device=device)
        ctx_doc_ids = None
        base_position_ids = None
        if packing:
            half = full_seq // 2
            ctx_doc_ids = torch.zeros(1, full_seq, dtype=torch.long, device=device)
            ctx_doc_ids[:, half:] = 1
            base_position_ids = torch.cat(
                [
                    torch.arange(half, dtype=torch.long, device=device),
                    torch.arange(full_seq - half, dtype=torch.long, device=device),
                ]
            ).unsqueeze(0)
        was_training = self.training
        self.train()
        try:
            ce, kl, *_ = self.forward(
                input_ids=input_ids,
                attention_mask=attn,
                target_hidden_states=hidden,
                target_lm_head_weight=target_lm_head_weight,
                loss_mask=loss_mask,
                hidden_states=hidden,
                position_ids=None,
                ctx_doc_ids=ctx_doc_ids,
                base_position_ids=base_position_ids,
            )
            loss = sum(c for c in ce) + sum(k for k in kl)
            loss.backward()
        finally:
            self.zero_grad(set_to_none=True)
            if not was_training:
                self.eval()
        if dist.is_initialized():
            dist.barrier()

    def _step_loss(
        self,
        post_norm_hidden: torch.Tensor,
        target_hidden_states_padded: torch.Tensor,
        mask: torch.Tensor,
        idx: int,
        seq_length: int,
        lm_head_weight: torch.Tensor,
        target_lm_head_weight: torch.Tensor,
        gt_labels: torch.Tensor = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        empty = valid_idx.numel() == 0
        if empty:
            # No supervised tokens on this shard/step. A structurally different shortcut
            # (params-only / hidden-sum) would give this rank a different backward graph
            # than peers → Ulysses all-to-all backward runs in a different order → NCCL
            # desync deadlocking the NEXT step. So run the IDENTICAL loss path on one
            # dummy position (index 0) and zero the result below.
            valid_idx = torch.zeros(1, dtype=torch.long, device=mask.device)

        torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
        hs_flat = post_norm_hidden.reshape(-1, post_norm_hidden.shape[-1])
        ths = target_hidden_states_padded[:, idx : idx + seq_length, :]
        ths_flat = ths.reshape(-1, ths.shape[-1])
        gt_flat = gt_labels.reshape(-1) if gt_labels is not None else None

        return_per_pos = self.e2e_weighting
        out = mtp_loss_from_hs(
            hs_flat,
            ths_flat,
            valid_idx,
            lm_head_weight,
            target_lm_head_weight,
            use_kl=self.use_kl,
            kl_topk=self.kl_topk,
            kl_variant=self.kl_variant,
            gt_labels_flat=gt_flat,
            ce_use_ground_truth=self.ce_use_ground_truth,
            single_step_form=self.single_step_form,
            single_step_eta=self.single_step_eta,
            return_per_pos=return_per_pos,
        )
        if return_per_pos:
            ce_sum, kl_sum, correct, count, correct_gt, ell_full, alpha_full = out
        else:
            ce_sum, kl_sum, correct, count, correct_gt = out
            ell_full = alpha_full = None
        if empty:
            ce_sum = ce_sum * 0.0
            kl_sum = kl_sum * 0.0
            correct = correct * 0
            count = count * 0
            correct_gt = correct_gt * 0
        return ce_sum, kl_sum, correct, count, correct_gt, ell_full, alpha_full

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_lm_head_weight: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[torch.Tensor],
        List[torch.Tensor],
    ]:
        batch_size, seq_length, _ = hidden_states.shape

        usp = self._usp_size > 1
        usp_global_offset = 0
        usp_gt_ids_steps = None
        usp_gt_label_steps = None
        usp_mask_steps = None
        target_hidden_states_padded = None
        if usp:
            # Local slicing: each SP rank keeps its own contiguous seq shard; the
            # attention module all-to-alls internally so it still attends the full
            # sequence, only the loss tokens are partitioned here. Right-pad the full
            # sequence to a multiple of sp_size (pad tokens get loss/attention mask 0)
            # so every rank gets an equal shard.
            rem = seq_length % self._usp_size
            if rem != 0:
                pad = self._usp_size - rem
                input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=0)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)
                loss_mask = torch.nn.functional.pad(loss_mask, (0, pad), value=0)
                target_hidden_states = torch.nn.functional.pad(
                    target_hidden_states, (0, 0, 0, pad), value=0.0
                )
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad), value=0.0)
                if position_ids is not None:
                    position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
                if ctx_doc_ids is not None:
                    ctx_doc_ids = torch.nn.functional.pad(ctx_doc_ids, (0, pad), value=-1)
                if base_position_ids is not None:
                    base_position_ids = torch.nn.functional.pad(
                        base_position_ids, (0, pad), value=0
                    )
                seq_length = seq_length + pad

            # Build every TTT shift on the full sequence before taking this rank's
            # shard. Shifting after slicing would replace the last token of each shard
            # with zero instead of reading the first token from the next shard.
            full_same_doc_next = (
                same_doc_next_mask(ctx_doc_ids).to(loss_mask.dtype)
                if ctx_doc_ids is not None
                else None
            )
            full_gt_ids = input_ids
            full_mask = loss_mask
            full_gt_ids_steps = []
            full_gt_label_steps = []
            full_mask_steps = []
            for idx in range(self.length):
                full_gt_ids_steps.append(full_gt_ids)
                full_gt_label_steps.append(padding(full_gt_ids, left=False))
                full_mask_steps.append(full_mask)
                if idx != self.length - 1:
                    full_gt_ids = padding(full_gt_ids, left=False)
                    full_mask = padding(full_mask, left=False)
                    if full_same_doc_next is not None:
                        full_mask = full_mask * full_same_doc_next

            shard = seq_length // self._usp_size
            usp_global_offset = self._usp_rank * shard
            sl = slice(usp_global_offset, usp_global_offset + shard)
            usp_gt_ids_steps = [x[:, sl] for x in full_gt_ids_steps]
            usp_gt_label_steps = [x[:, sl] for x in full_gt_label_steps]
            usp_mask_steps = [x[:, sl] for x in full_mask_steps]

            # Keep a right halo for the teacher hidden windows used by later TTT
            # steps. The halo is local metadata only; attention still receives the
            # normal equal-size shard.
            halo_end = min(usp_global_offset + shard + self.length, seq_length)
            target_hidden_states_padded = target_hidden_states[:, usp_global_offset:halo_end, :]
            halo_pad = shard + self.length - target_hidden_states_padded.shape[1]
            if halo_pad > 0:
                target_hidden_states_padded = torch.nn.functional.pad(
                    target_hidden_states_padded, (0, 0, 0, halo_pad), value=0.0
                )

            input_ids = input_ids[:, sl]
            attention_mask = attention_mask[:, sl]
            target_hidden_states = target_hidden_states[:, sl, :]
            loss_mask = loss_mask[:, sl]
            hidden_states = hidden_states[:, sl, :]
            if position_ids is not None:
                position_ids = position_ids[:, sl]
            if ctx_doc_ids is not None:
                ctx_doc_ids = ctx_doc_ids[:, sl]
            if base_position_ids is not None:
                base_position_ids = base_position_ids[:, sl]
            seq_length = shard

        lm_head_weight = self.draft_model.lm_head_weight
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        if base_position_ids is not None:
            # Packing: doc-local RoPE positions (each doc restarts at 0).
            position_ids = base_position_ids.view(-1, seq_length).long()
        elif position_ids is None:
            device = hidden_states.device
            # Global positions so RoPE is consistent across shards.
            position_ids = torch.arange(
                usp_global_offset,
                usp_global_offset + seq_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Pad target hidden states by `length` along seq so each step can read a
        # left-shifted slice [idx : idx+seq_length]. USP constructed this tensor
        # from the full sequence above so the slice can cross a shard boundary.
        if target_hidden_states_padded is None:
            target_hidden_states_padded = torch.nn.functional.pad(
                target_hidden_states, (0, 0, 0, self.length), value=0.0
            )

        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=0,
            )
            if ctx_doc_ids is not None:
                # Packing doc-gate: forbid attending across doc boundaries.
                same_doc = ctx_doc_ids[:, :, None] == ctx_doc_ids[:, None, :]
                neg_inf = torch.finfo(attention_mask.dtype).min
                attention_mask = attention_mask.masked_fill(~same_doc.unsqueeze(1), neg_inf)

        # Packing: re-gate loss at doc seams after each left-shift (see loop below).
        same_doc_next = (
            same_doc_next_mask(ctx_doc_ids).to(loss_mask.dtype)
            if ctx_doc_ids is not None
            else None
        )

        mask = usp_mask_steps[0] if usp_mask_steps is not None else loss_mask
        ce_losses, kl_losses, vlosses, acces, acc_counts = [], [], [], [], []
        vklosses = []
        acces_gt = []
        # Per-step e2e collectors (flat [M] on the local grid, same index across
        # steps). Only filled when e2e_weighting is on.
        e2e_ell_list, e2e_alpha_list, e2e_mask_list = [], [], []
        cache_keys = None
        cache_values = None

        input_ids = input_ids.clamp(min=0, max=self.draft_model.target_vocab_size - 1)
        # Two decoupled chains:
        #   gt_ids   — ground-truth chain, left-shifted each step, the SOLE source of
        #              the CE/KL target label. Never polluted by on-policy predictions.
        #   embed_ids — what gets embedded. step 0 == gt_ids (teacher-forced); step>=1
        #              is the previous step's own argmax when on_policy, else gt_ids.
        gt_ids = usp_gt_ids_steps[0] if usp_gt_ids_steps is not None else input_ids
        embed_ids = gt_ids

        for idx in range(self.length):
            is_last = idx == self.length - 1

            inputs_embeds = self.draft_model.embed_input_ids(embed_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                hidden_states_out, cache_keys, cache_values = torch_checkpoint(
                    self.draft_model.backbone,
                    inputs_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_keys,
                    cache_values,
                    True,
                    ctx_doc_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states_out, cache_keys, cache_values = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                    ctx_doc_ids=ctx_doc_ids,
                )

            # Ground-truth next-token label: at position j the draft predicts the
            # token AFTER gt_ids[j], i.e. gt_ids shifted left by one. Derived from
            # gt_ids (NOT embed_ids) so on-policy argmax can never pollute the target.
            gt_labels = (
                usp_gt_label_steps[idx]
                if usp_gt_label_steps is not None
                else padding(gt_ids, left=False)
            )

            ce_sum, kl_sum, correct, count, correct_gt, ell_full, alpha_full = self._step_loss(
                post_norm_hidden=hidden_states_out,
                target_hidden_states_padded=target_hidden_states_padded,
                mask=mask,
                idx=idx,
                seq_length=seq_length,
                lm_head_weight=lm_head_weight,
                target_lm_head_weight=target_lm_head_weight,
                gt_labels=gt_labels,
            )
            if self.e2e_weighting:
                e2e_ell_list.append(ell_full)
                e2e_alpha_list.append(alpha_full)
                e2e_mask_list.append(mask.reshape(-1).float())

            # The MTP head feeds its own post-norm hidden back as the next step's
            # hidden source.
            hidden_states = hidden_states_out

            metric_denom = count.clamp_min(1.0)
            if usp:
                # Backward loss normalizes by the GLOBAL (all-shard) supervised-token
                # count so the post-backward grad SUM over the SP group reconstructs the
                # true full-sequence mean gradient. Dividing by the LOCAL count and then
                # summing per-shard grads would inflate the gradient by ~sp_size (each
                # token's effective weight becomes 1/c_local instead of 1/C_global).
                # This is an UNCONDITIONAL scalar all-reduce — every rank always runs it
                # (symmetric → deadlock-free), unlike the zero-loss-shard *backward*
                # asymmetry the connect-graph trick below guards against.
                global_count = count.detach().clone()
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=self._usp_group)
                loss_denom = global_count.clamp_min(1.0)
                # CRITICAL: the zero-valued term must flow through hidden_states_out
                # (which came via the Ulysses all-to-all), NOT raw params — else a
                # zero-loss shard skips the all-to-all backward while peers run it →
                # NCCL deadlock. Added AFTER the division so an empty shard (real loss 0)
                # still carries the connect graph into backward on every rank.
                connect = hidden_states_out.sum() * 0.0
                ce = ce_sum / loss_denom + connect
                kl = kl_sum / loss_denom + connect
            else:
                ce = ce_sum / metric_denom
                kl = kl_sum / metric_denom

            # Keep local means plus the local count. The trainer converts them back
            # to sums and performs one global SUM before dividing by the global count,
            # so displayed CE/KL/accuracy are exact even for unbalanced/empty shards.
            metric_ce = (ce_sum / metric_denom).detach()
            metric_kl = (kl_sum / metric_denom).detach()
            has_valid = float(count.detach().float().cpu()) > 0.0
            metric_acc = (
                (correct / metric_denom).detach() if has_valid else correct.detach().float() * 0.0
            )
            metric_acc_gt = (
                (correct_gt / metric_denom).detach()
                if has_valid
                else correct_gt.detach().float() * 0.0
            )

            ce_losses.append(ce)
            kl_losses.append(kl)
            vlosses.append(metric_ce)
            vklosses.append(metric_kl)
            acces.append(metric_acc)
            acces_gt.append(metric_acc_gt)
            acc_counts.append(count.detach().float().to(device=ce.device))

            if not is_last:
                # Advance the ground-truth chain (drives the next step's CE label).
                if usp_gt_ids_steps is not None:
                    gt_ids = usp_gt_ids_steps[idx + 1]
                    mask = usp_mask_steps[idx + 1]
                else:
                    gt_ids = padding(gt_ids, left=False)
                    mask = padding(mask, left=False)
                    # Packing: drop predictions that crossed into the next doc. gt_ids is
                    # not cleaned — its dirty token is contained by the doc-gated attention.
                    if same_doc_next is not None:
                        mask = mask * same_doc_next
                # Next step's embedded token: on-policy → this step's own argmax
                # (stop-grad, trains on the serve distribution); off-policy → the
                # left-shifted ground-truth token (teacher forcing).
                if self.on_policy:
                    with torch.no_grad():
                        pred = self.draft_model.compute_logits(hidden_states_out).argmax(-1)
                    embed_ids = pred.detach().clamp(
                        min=0, max=self.draft_model.target_vocab_size - 1
                    )
                else:
                    embed_ids = gt_ids

        e2e_loss = self._e2e_loss(e2e_ell_list, e2e_alpha_list, e2e_mask_list)
        return ce_losses, kl_losses, vlosses, acces, acc_counts, acces_gt, e2e_loss, vklosses

    def _e2e_loss(self, ell_list, alpha_list, mask_list):
        """Couple the per-step accept rates α_i = 1 − TV_i into one loss over the TTT
        chain. A chain at index j counts only where every step's mask is valid
        (``valid_all = Π_i mask_i``). Under USP, each rank contributes its local
        numerator but divides by the SP-global valid-chain count, so the later
        gradient SUM reconstructs the full-sequence mean.

          * e2e_direct: E[L] = Σ_j Π_{i≤j} α_i,  loss = 1 − E[L]/γ.
          * else (W_m): detached weights W_m = (1/γ)·pre[m]·S[m] on ell_m.
        """
        if not self.e2e_weighting or len(ell_list) == 0:
            return None
        gamma = len(ell_list)
        ell_stack = torch.stack(ell_list, dim=0)
        alpha_stack = torch.stack(alpha_list, dim=0)
        mask_stack = torch.stack(mask_list, dim=0)
        valid_all = mask_stack.prod(dim=0)
        local_count = valid_all.sum()
        denom = local_count.detach().clone()
        if self._usp_size > 1:
            dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=self._usp_group)
        denom = denom.clamp_min(1.0)

        if self.e2e_direct:
            prefix = torch.cumprod(alpha_stack, dim=0)
            expected_len = prefix.sum(dim=0)
            loss_per_pos = 1.0 - expected_len / float(gamma)
            local_sum = (loss_per_pos * valid_all).sum()
            # Explicitly preserve the Ulysses backward graph on empty shards.
            connect = ell_stack.sum() * 0.0
            return local_sum / denom + connect

        alpha_det = alpha_stack.detach()
        pre = [torch.ones_like(alpha_det[0])]
        for m in range(1, gamma):
            pre.append(pre[m - 1] * alpha_det[m - 1])
        S = [None] * gamma
        S[gamma - 1] = torch.ones_like(alpha_det[0])
        for m in range(gamma - 2, -1, -1):
            S[m] = 1.0 + alpha_det[m + 1] * S[m + 1]
        W = torch.stack([(pre[m] * S[m]) / float(gamma) for m in range(gamma)], dim=0)
        weighted = (W * ell_stack).sum(dim=0)
        local_sum = (weighted * valid_all).sum()
        connect = ell_stack.sum() * 0.0
        return local_sum / denom + connect
