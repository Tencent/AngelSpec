"""Branch-request teacher scoring shared by DFlash-family trainers."""

import time
from typing import Optional

import torch
import torch.distributed as dist

from angelspec.utils.logging import logger


class DFlashOPDMixin:
    def _opd_loss_name(self) -> str:
        return getattr(self.args, "dflash_opd_loss", "two_stream_k3")

    def _opd_scored_slots(self) -> int:
        """Slots scored per branch: every drafted token, so ``block_size - 1``."""
        return int(getattr(self.args, "dflash_block_size", 16)) - 1

    def _opd_accept_sampling(self) -> dict:
        """The target's serving sampling parameters, as tempered_target_probs takes them."""
        return {
            "temperature": float(getattr(self.args, "dflash_opd_accept_temperature", 0.9)),
            "top_k": int(getattr(self.args, "dflash_opd_accept_top_k", 0)),
            "top_p": float(getattr(self.args, "dflash_opd_accept_top_p", 1.0)),
        }

    def _opd_metric_keys(self) -> tuple:
        """Key set the configured OPD loss emits (must match across DP ranks)."""
        from angelspec.models.ops.dfly_opd_loss import (
            opd_greedy_metric_keys,
            opd_tempered_metric_keys,
        )
        from angelspec.models.ops.loss import _OPD_METRIC_KEYS

        if self._opd_loss_name() == "greedy_accept":
            return opd_greedy_metric_keys(int(getattr(self.args, "dflash_block_size", 16)) - 1)
        if self._opd_loss_name() == "tempered_accept":
            return opd_tempered_metric_keys(self._opd_scored_slots())
        return _OPD_METRIC_KEYS

    def _branch_opd_on(self) -> bool:
        return (
            bool(self.score_engine)
            and getattr(self.args, "dflash_opd_enabled", False)
            and getattr(self.args, "dflash_opd_backend", "packed_tree") == "branch_requests"
        )

    def set_score_engine(self, score_engine):
        """Bind this DP rank's verify engine (branch backend) or keep the pool."""
        super().set_score_engine(score_engine)
        self._verify_engine = None
        if not self._branch_opd_on():
            return
        if len(score_engine) != self.dp_size:
            raise ValueError(
                f"Branch-request OPD needs exactly one verify engine per DP rank "
                f"({self.dp_size}); got {len(score_engine)}. The trunk lives in ONE "
                f"engine's prefix cache, so a rank's branches cannot be spread around. "
                f"Set opd_score_engine_num_gpus = dp_size * inference_num_gpus_per_engine."
            )
        self._verify_engine = score_engine[self.dp_rank]

    def _prime_branch_opd(self, batch: dict, input_ids: torch.Tensor) -> None:
        """Prefill this micro-batch's trunk into the verify engine's prefix cache."""
        self._branch_opd_state = None
        if not self._branch_opd_on():
            return
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                f"Branch-request OPD scores one trunk per micro-batch; got "
                f"micro_batch_size={input_ids.shape[0]}. Raise "
                f"draft_accumulation_steps instead."
            )
        mask = batch.get("attention_mask")
        trunk_len = int(mask[0].sum()) if mask is not None else int(input_ids.shape[1])
        trunk_ids = input_ids[0, :trunk_len].tolist()
        media = (batch.get("opd_media") or [None])[0]
        self._branch_opd_state = {
            "trunk_ids": trunk_ids,
            "media": media,
            "warm": self._verify_engine.opd_warm.remote(trunk_ids, media),
        }

    def _branch_opd_scored_blocks(
        self, keep: torch.Tensor, anchors: torch.Tensor, trunk_len: int, width: int
    ) -> torch.Tensor:
        """Valid block indices this step scores — same selection as the tree layout."""
        keep = keep.bool() & (anchors + width < trunk_len)
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        max_anchors = getattr(self.args, "dflash_opd_max_anchors", None)
        if max_anchors is not None and selected.numel() > max_anchors > 0:
            sel = torch.linspace(0, selected.numel() - 1, steps=max_anchors)
            selected = selected[sel.round().long().unique()]
        return selected

    def _compute_opd_loss_branch(
        self,
        input_ids: torch.Tensor,
        opd: dict,
        last_hidden_states: Optional[torch.Tensor],
        target_norm,
    ):
        """OPD teacher via branch prompts on a decoder engine (multimodal-capable)."""
        import ray

        from angelspec.models.ops.dfly_opd_loss import (
            opd_greedy_accept_loss_from_hs,
            opd_tempered_accept_loss_from_hs,
        )
        from angelspec.models.ops.loss import opd_two_stream_kl_from_hs

        state = self._branch_opd_state
        draft_hidden = opd["draft_hidden"]  # [1, n_blocks*bs, D] (grad)
        device = draft_hidden.device
        bs = int(getattr(self.args, "dflash_block_size", 16))
        width = self._opd_scored_slots()  # every drafted token, B-1 of them
        zero = draft_hidden.new_zeros(())
        empty = {k: zero.detach().clone() for k in self._opd_metric_keys()}
        if last_hidden_states is None:
            raise ValueError(
                "Branch-request OPD needs the prefill engine's last_hidden_states for "
                "the corpus-matching teacher slots; enable store_last_hidden_states."
            )

        trunk_len = len(state["trunk_ids"])
        blocks = self._branch_opd_scored_blocks(
            opd["block_keep_mask"][0], opd["anchor_positions"][0], trunk_len, width
        )
        if blocks.numel() == 0:
            ray.get(state["warm"])
            return draft_hidden.sum() * 0, empty

        anchors = opd["anchor_positions"][0].index_select(0, blocks)  # [K]
        proposals = opd["proposals"][0].index_select(0, blocks)  # [K, bs]
        branch_tokens = proposals[:, 1 : width + 1]  # [K, width] == p_1..p_width

        # Teacher rows at predecessor slots a+0 .. a+width-1, seeded from the corpus
        # prefill and normalized the way the draft logits' teacher is. The target
        # head is frozen, so this carries no gradient.
        slot_offsets = torch.arange(width, device=device)
        slot_positions = anchors.unsqueeze(1) + slot_offsets.unsqueeze(0)  # [K, width]
        with torch.no_grad():
            teacher = last_hidden_states[0].index_select(0, slot_positions.reshape(-1))
            teacher = teacher.view(anchors.numel(), width, -1)
            if target_norm is not None:
                teacher = target_norm(teacher)
            teacher = teacher.to(draft_hidden.dtype)
        # Preserve the independent prefill reference BEFORE any verifier overwrite.
        prefill_teacher = teacher.detach().clone()

        # Leading proposals that still match the corpus; only p_1..p_{width-1} can
        # ever appear inside a branch prompt, so the match run is capped there.
        gt = (
            input_ids[0]
            .index_select(0, slot_positions[:, 1:].reshape(-1))
            .view(anchors.numel(), width - 1)
        )
        matched = (branch_tokens[:, : width - 1] == gt).long().cumprod(dim=1).sum(dim=1)  # [K]
        # First slot the verify engine must supply. With the GT shortcut, slots
        # 0..matched are already correct in ``teacher`` (their conditioning prefix is
        # still the corpus prefix). Without it, every slot — including slot 0, whose
        # prefix is pure trunk — comes from the verify engine, so one branch never
        # mixes two engines' hidden states.
        shortcut = bool(getattr(self.args, "dflash_opd_gt_shortcut", True))
        first_slot = matched + 1 if shortcut else torch.zeros_like(matched)

        need = torch.nonzero(first_slot <= (width - 1), as_tuple=False).flatten()
        verify_branches = int(need.numel())
        if verify_branches:
            anchors_cpu = anchors.cpu().tolist()
            tokens_cpu = branch_tokens.cpu().tolist()
            first_cpu = first_slot.cpu().tolist()
            need_cpu = need.cpu().tolist()
            trunk_ids = state["trunk_ids"]
            prefix_lens, tails = [], []
            for i in need_cpu:
                lo, a = first_cpu[i], anchors_cpu[i]
                # Slot j sits at absolute position a+j and holds the anchor token at
                # j == 0, else proposal p_j == branch_tokens[j-1].
                prefix_lens.append(a + lo)
                tails.append(
                    [trunk_ids[a] if j == 0 else tokens_cpu[i][j - 1] for j in range(lo, width)]
                )
            _t0 = time.perf_counter()
            result = ray.get(
                self._verify_engine.opd_verify.remote(
                    state["trunk_ids"],
                    prefix_lens,
                    tails,
                    state["warm"],
                    require_complete=True,
                    media=state["media"],
                )
            )
            self._opd_last_verify = {
                "rpc_seconds": time.perf_counter() - _t0,
                "warm_seconds": (
                    float(ray.get(state["warm"])) if state["warm"] is not None else 0.0
                ),
                **{
                    k: float(result.get(k, 0))
                    for k in (
                        "prepare_time",
                        "generate_time",
                        "capture_time",
                        "cached_tokens",
                        "computed_tokens",
                    )
                },
                "branches": verify_branches,
            }
            rows = result["rows"].to(device=device, dtype=draft_hidden.dtype)
            branch_of = result["branch"].to(device=device, dtype=torch.long)
            pos = result["pos"].to(device=device, dtype=torch.long)
            if rows.shape[0] != branch_of.numel() or pos.numel() != branch_of.numel():
                raise RuntimeError("Misaligned OPD teacher rows from the verify engine")
            block_of = need.index_select(0, branch_of)
            slot_of = pos - anchors.index_select(0, block_of)
            if bool(((slot_of < 0) | (slot_of >= width)).any()):
                raise RuntimeError(
                    f"Verify returned out-of-range OPD slots: "
                    f"{slot_of.min().item()}..{slot_of.max().item()} outside [0, {width})"
                )
            teacher = teacher.index_put((block_of, slot_of), rows)
        elif state["warm"] is not None:
            ray.get(state["warm"])

        self._check_teacher_parity(state, anchors, prefill_teacher)

        student_slots = (
            blocks.unsqueeze(1) * bs + torch.arange(1, width + 1, device=device).unsqueeze(0)
        ).reshape(-1)
        student = draft_hidden[0].index_select(0, student_slots)
        teacher_flat = teacher.reshape(-1, teacher.shape[-1])
        if self._opd_loss_name() == "tempered_accept":
            return opd_tempered_accept_loss_from_hs(
                student,
                teacher_flat,
                branch_tokens.reshape(-1),
                self.target_lm_head_weight,
                width,
                **self._opd_accept_sampling(),
                relax_temperature=float(getattr(self.args, "dflash_opd_relax_temperature", 1.0)),
                vocab_chunk=int(getattr(self.args, "dflash_opd_vocab_chunk", 64)),
            )
        if self._opd_loss_name() == "greedy_accept":
            return opd_greedy_accept_loss_from_hs(
                student,
                teacher_flat,
                self.target_lm_head_weight,
                bs,
                vocab_chunk=int(getattr(self.args, "dflash_opd_vocab_chunk", 64)),
                logprob_min_clamp=getattr(self.args, "dflash_opd_logprob_min_clamp", -10.0),
            )
        return opd_two_stream_kl_from_hs(
            student,
            teacher_flat,
            branch_tokens.reshape(-1),
            self.target_lm_head_weight,
            bs,
            response_stream_weight=float(
                getattr(self.args, "dflash_opd_response_stream_weight", 1.0)
            ),
            rejected_stream_weight=float(
                getattr(self.args, "dflash_opd_rejected_stream_weight", 1.0)
            ),
            position_decay=float(getattr(self.args, "dflash_opd_position_decay", 0.8)),
            position_decay_enabled=bool(
                getattr(self.args, "dflash_opd_position_decay_enabled", True)
            ),
            loss_max_clamp=getattr(self.args, "dflash_opd_loss_max_clamp", 10.0),
            logprob_min_clamp=getattr(self.args, "dflash_opd_logprob_min_clamp", -10.0),
        )

    def _check_teacher_parity(self, state, anchors, teacher):
        import ray

        from angelspec.inference.engine.opd_parity import (
            assert_teacher_parity,
            teacher_parity_metrics,
        )

        steps = self.args.dflash_opd_teacher_parity_steps
        if steps <= 0 or (self.global_step >= steps and self.global_step % 50 != 0):
            return
        count = min(anchors.numel(), self.args.dflash_opd_teacher_parity_branches)
        probe_anchors = anchors[:count].tolist()
        # Recompute the anchor row even when no proposal matches the corpus.
        # Its teacher always belongs to the independently captured prefill prefix.
        result = ray.get(
            self._verify_engine.opd_verify.remote(
                state["trunk_ids"],
                probe_anchors,
                [[state["trunk_ids"][a]] for a in probe_anchors],
                state["warm"],
                media=state["media"],
            )
        )
        actual = result["rows"].to(device=teacher.device, dtype=teacher.dtype)
        with torch.no_grad():
            comparison = teacher_parity_metrics(
                teacher[:count, 0],
                actual,
                self.target_lm_head_weight,
                **self._opd_accept_sampling(),
            )
        logger.info("OPD independent teacher parity step=%s: %s", self.global_step, comparison)
        assert_teacher_parity(comparison, self.args.dflash_opd_teacher_parity_tol)

    def _format_opd_metrics(self, m: dict) -> str:
        """One log line for the reduced OPD metrics of whichever loss is configured."""
        if self._opd_loss_name() == "two_stream_k3":
            return (
                f"resp_kl={m['opd/resp_kl']:.4f} rej_kl={m['opd/rej_kl']:.4f} "
                f"accept_rate={m['opd/accept_rate']:.3f} "
                f"accepted={int(m['opd/accepted_cnt'])} rejected={int(m['opd/rejected_cnt'])} "
                f"clamp_frac={m['opd/clamp_fraction']:.3f}"
            )
        slot = "opd/accept_{}" if self._opd_loss_name() == "tempered_accept" else "opd/alpha_{}"
        per_slot, i = [], 0
        while slot.format(i) in m:
            per_slot.append(f"{m[slot.format(i)]:.3f}")
            i += 1
        lengths = " ".join(
            f"{key.split('/', 1)[1]}={m[key]:.3f}"
            for key in ("opd/accept_len", "opd/greedy_accept_len", "opd/accept_len_surrogate")
            if key in m
        )
        return f"{lengths} branches={int(m['opd/branches'])} per_slot=[{' '.join(per_slot)}]"

    def _reduce_opd_metrics(self, all_step_metrics: list[dict]) -> dict:
        """DP-reduce the two-stream OPD sum/count scalars into logged means."""
        opd_dicts = [m["_opd"] for m in all_step_metrics if m.get("_opd")]
        if not opd_dicts:
            return {}
        keys = self._opd_metric_keys()

        totals = {k: torch.stack([d[k] for d in opd_dicts]).sum() for k in keys}
        flat = torch.stack([totals[k] for k in keys]).float()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        t = {k: flat[i] for i, k in enumerate(keys)}

        if self._opd_loss_name() == "tempered_accept":
            branches = t["opd/branches"].clamp(min=1.0)
            out = {"opd/branches": t["opd/branches"].item()}

            def mean(name: str, i: int) -> float:
                return (t[f"opd/{name}_sum_{i}"] / branches).item()

            accept_len = greedy_len = 1.0
            for i in range(self._opd_scored_slots()):
                out[f"opd/accept_{i}"] = mean("accept", i)
                out[f"opd/accept_soft_{i}"] = mean("soft", i)
                out[f"opd/accept_best_{i}"] = mean("best", i)
                out[f"opd/greedy_hit_{i}"] = mean("hit", i)
                accept_len += mean("prefix", i)
                greedy_len += mean("hit_prefix", i)
            # Serving convention (+1 for the target's own token). ``accept_len`` is the
            # expected accept length at the serving temperature along the drafter's
            # own proposals; ``greedy_accept_len`` scores the same branches under
            # greedy verification, the twin of eval/*/simulated_acc_len_tgt.
            out["opd/accept_len"] = accept_len
            out["opd/greedy_accept_len"] = greedy_len
            return out

        if self._opd_loss_name() == "greedy_accept":
            width = int(getattr(self.args, "dflash_block_size", 16)) - 1
            branches = t["opd/branches"].clamp(min=1.0)
            out: dict = {"opd/branches": t["opd/branches"].item()}
            surrogate = hard = 0.0
            for i in range(width):
                alpha = (t[f"opd/alpha_sum_{i}"] / branches).item()
                prefix = (t[f"opd/prefix_sum_{i}"] / branches).item()
                hit = (t[f"opd/hit_sum_{i}"] / branches).item()
                hit_prefix = (t[f"opd/hit_prefix_sum_{i}"] / branches).item()
                out[f"opd/alpha_{i}"] = alpha
                out[f"opd/greedy_hit_{i}"] = hit
                surrogate += prefix
                hard += hit_prefix
            # Serving convention (+1 for the target's own token). ``accept_len`` is
            # the hard, directly comparable twin of eval/*/simulated_acc_len_tgt;
            # ``accept_len_surrogate`` is what the loss actually descends.
            out["opd/accept_len"] = 1.0 + hard
            out["opd/accept_len_surrogate"] = 1.0 + surrogate
            return out

        accepted = t["opd/accepted_cnt"]
        rejected = t["opd/rejected_cnt"]
        scored = t["opd/scored_cnt"].clamp(min=1.0)
        return {
            "opd/resp_kl": (t["opd/resp_kl_sum"] / accepted.clamp(min=1.0)).item(),
            "opd/rej_kl": (t["opd/rej_kl_wsum"] / t["opd/rej_eff_w_sum"].clamp(min=1.0)).item(),
            "opd/accept_rate": (accepted / (accepted + rejected).clamp(min=1.0)).item(),
            "opd/accepted_cnt": accepted.item(),
            "opd/rejected_cnt": rejected.item(),
            "opd/clamp_fraction": (t["opd/clamp_frac_sum"] / scored).item(),
        }


def validate_branch_opd(args, config):
    if not getattr(args, "dflash_opd_enabled", False):
        if getattr(args, "dflash_opd_on_policy_prev", False):
            raise ValueError("dflash_opd_on_policy_prev requires dflash_opd_enabled")
        return
    if args.dflash_opd_backend not in ("packed_tree", "branch_requests"):
        raise ValueError("dflash_opd_backend must be packed_tree or branch_requests")
    if args.dflash_opd_loss not in ("two_stream_k3", "greedy_accept", "tempered_accept"):
        raise ValueError("Unknown dflash_opd_loss")
    if args.dflash_opd_backend == "packed_tree":
        if args.dflash_opd_loss != "two_stream_k3":
            raise ValueError("Acceptance OPD objectives require branch_requests")
        return
    if args.inference_engine_type != "vllm" or not args.enable_opd_score_engine:
        raise ValueError("Branch OPD requires vLLM and enable_opd_score_engine=true")
    if args.micro_batch_size != 1 or args.attention_backend == "usp" or args.sp_ulysses_size != 1:
        raise ValueError("Branch OPD requires micro_batch_size=1 and SP=1 without USP")
    if args.dflash_packing or args.colocate or args.placement_strategy == "custom":
        raise ValueError(
            "Branch OPD requires separate roles, unpacked data and non-custom placement"
        )
    if args.vllm_pp_size != 1 or args.vllm_nnodes != 1 or args.vllm_data_parallel_size != 1:
        raise ValueError("Branch OPD engines require single-node TP with PP=1 and engine DP=1")
    if args.inference_num_gpus_per_engine != args.vllm_tp_size:
        raise ValueError("Branch OPD engine GPU count must equal vllm.tp_size")
    if args.opd_score_engine_num_gpus != args.dp_size * args.vllm_tp_size:
        raise ValueError("Branch OPD requires exactly one TP verify engine per DP rank")
    per_node = args.inference_num_gpus_per_node
    for gpus in (
        args.training_num_nodes * args.training_num_gpus_per_node,
        args.inference_num_gpus,
        args.opd_score_engine_num_gpus,
    ):
        if gpus <= 0 or gpus % per_node:
            raise ValueError("Branch OPD roles must each occupy whole nodes")
    if not args.store_last_hidden_states or not args.last_hidden_states_prenorm:
        raise ValueError("Branch OPD requires stored prenorm final teacher hidden states")
    if args.dflash_block_size < 3 or args.dflash_opd_vocab_chunk <= 0:
        raise ValueError("Branch OPD requires block_size>=3 and a positive vocabulary chunk")
    if args.dflash_loss_checkpoint_chunks:
        raise ValueError("Branch OPD requires dflash_loss_checkpoint_chunks=false")
    if args.dflash_opd_accept_temperature <= 0 or args.dflash_opd_relax_temperature <= 0:
        raise ValueError("OPD temperatures must be positive")
    if not 0 < args.dflash_opd_accept_top_p <= 1:
        raise ValueError("dflash_opd_accept_top_p must be in (0, 1]")
    if args.dflash_opd_teacher_parity_branches < 1 or args.dflash_opd_teacher_parity_tol <= 0:
        raise ValueError("Teacher parity requires positive probe count and tolerance")
    for name in (
        "enable_hidden_correction_topk",
        "attentive_correction",
        "enable_path_selector",
        "markov_rank",
    ):
        if getattr(config, name, False):
            raise ValueError(f"Branch OPD cannot represent the logits of {name}")
    if args.mtp_draft_opd:
        raise ValueError("MTP and DFlash OPD cannot both be enabled")
