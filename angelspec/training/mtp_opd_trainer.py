"""Two-pass Hy4 flash MTP draft-OPD with complete on-policy teacher scoring."""

import json
import os
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.api import CheckpointException

from angelspec.models.mtp_opd import mtp_opd_loss
from angelspec.training.mtp_trainer import MTPTrainer, _remap_legacy_fused_expert_key
from angelspec.utils.logging import logger


def opd_contract(args):
    return {
        "algorithm": "mtp_draft_opd",
        "version": 2,
        "optimizer": "adamw",
        "loss": getattr(args, "mtp_opd_loss", "lk_alpha"),
        "temperature": float(getattr(args, "mtp_opd_temperature", 1.0)),
        "fp8_indexer": bool(getattr(args, "mtp_opd_fp8_indexer", True)),
        "steps": int(args.mtp_opd_block_size),
        "target": str(Path(args.target_model_path).resolve()),
        "dataset": str(Path(args.train_data_path).resolve()),
    }


def validate_opd_resume(args):
    if not args.load_path:
        return
    root = Path(args.load_path).expanduser()
    if not (root / "model").is_dir():
        step = getattr(args, "ckpt_step", None)
        if step is None:
            step = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
        root = root / f"iter_{step:07d}"
    saved = json.loads((root / "opd_contract.json").read_text())
    if saved != opd_contract(args):
        raise ValueError(
            "Resume must use a complete matching draft-OPD checkpoint; initialize old MTP weights with mtp_opd_init_model"
        )
    for part in ("model", "optimizer", "lr_scheduler"):
        if not (root / part / ".metadata").is_file():
            raise ValueError(f"Incomplete OPD resume checkpoint: {root / part}")
    if not (root / "rng.pt").is_file() or not (root / "meta.json").is_file():
        raise ValueError(f"Incomplete OPD resume checkpoint: {root}")
    metadata = json.loads((root / "meta.json").read_text())
    if "global_step" not in metadata:
        raise ValueError(f"OPD resume checkpoint has no global_step: {root}")


def initialize_opd_weights(draft_model, model_path):
    """Read only model tensors, before creating fresh optimizer master weights."""
    if not model_path or not (Path(model_path) / ".metadata").is_file():
        raise ValueError("mtp_opd_init_model must point at a DCP model directory")
    from angelspec.utils.distributed import get_gloo_group

    # Loading happens only on rank 0; relay errors before peers enter FSDP broadcasts.
    error = [None]
    if dist.get_rank() == 0:
        try:
            from tools.convert_mtp_to_hf import _load_dcp_state_dict, _strip_to_draft

            state = _strip_to_draft(_load_dcp_state_dict(model_path))
            state = {
                _remap_legacy_fused_expert_key(k): v
                for k, v in state.items()
                if k not in ("embed_tokens.weight", "lm_head_weight")
            }
            expected = set(draft_model.state_dict()) - {"embed_tokens.weight"}
            if set(state) != expected:
                raise ValueError(
                    f"OPD initialization key mismatch: missing={expected - set(state)}, extra={set(state) - expected}"
                )
            for name, param in draft_model.frozen_fp32_parameters().items():
                if state[name].dtype != torch.float32:
                    raise ValueError(f"Frozen {name} must retain checkpoint FP32 precision")
                param.data = param.data.float()
            draft_model.load_state_dict(state, strict=False)
            logger.info(
                "OPD initialized %d model tensors from %s; optimizer/schedule/step are fresh",
                len(state),
                model_path,
            )
        except (Exception, CheckpointException) as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0, group=get_gloo_group())
    if error[0] is not None:
        raise RuntimeError(f"OPD model initialization failed: {error[0]}")


class MTPOPDTrainer(MTPTrainer):
    def init_model(self, *args, **kwargs):
        validate_opd_resume(self.args)
        # Hy4 flash's 16-expert BF16 scatter and concurrent GEMMs otherwise change
        # the sampled policy between no-grad rollout and gradient replay.
        # Configure before the first model GEMM / FSDP forward.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        result = super().init_model(*args, **kwargs)
        # Reuse the exact FP32 head for sampling/replay instead of converting
        # the 120832 x 3072 frozen matrix for every chunk.
        self.target_lm_head_weight = self.target_lm_head_weight.float()
        self.draft_model.set_lm_head_weight(self.target_lm_head_weight)
        return result

    def set_score_engine(self, score_engine):
        super().set_score_engine(score_engine)
        if len(score_engine or []) != self.dp_size:
            raise ValueError(f"OPD needs one TP verify engine per DP group ({self.dp_size})")
        self.verify_engine = score_engine[self.dp_rank]

    @property
    def _sp_size(self):
        return dist.get_world_size(self.sp_group) if self.sp_group is not None else 1

    @property
    def _sp_rank(self):
        return dist.get_rank(self.sp_group) if self.sp_group is not None else 0

    @property
    def _metric_group(self):
        return self.grad_sync_mesh.get_group() if self.grad_sync_mesh is not None else None

    def _gather(self, tensor):
        if self._sp_size == 1:
            return tensor
        out = [torch.empty_like(tensor) for _ in range(self._sp_size)]
        dist.all_gather(out, tensor.contiguous(), group=self.sp_group)
        return torch.cat(out)

    def _broadcast(self, tensor):
        if self._sp_size > 1:
            dist.broadcast(
                tensor, src=dist.get_process_group_ranks(self.sp_group)[0], group=self.sp_group
            )

    def _prepare(self, batch, step, micro):
        unavailable = int(batch.pop("_mooncake_unavailable", 0))
        if self.args.fail_on_unavailable_mooncake:
            self._check_mooncake_batch(unavailable, step, micro)
        length = int(batch["attention_mask"][0].sum())
        device = torch.device("cuda", torch.cuda.current_device())
        ids = batch["input_ids"][0, :length].to(device)
        hidden = batch["last_hidden_states"][0, :length].detach().to(device)
        positions = batch.get("position_ids")
        if positions is not None:
            positions = positions[..., :length].to(device)
        cand = (batch["loss_mask"][0, 2:length] > 0).nonzero().flatten().to(device) + 1
        # Media-adjacent context tokens cannot be used as an MTP text anchor.
        if cand.numel():
            cand = cand[~((ids[cand] >= 120118) & (ids[cand] <= 120124))]
        invalid = torch.tensor(int(length < 3 or not cand.numel()), device=ids.device)
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self._metric_group)
        if invalid.item():
            raise RuntimeError(
                "OPD received an invalid/placeholder conversation or no usable supervised anchor on a rank"
            )
        seed = int(self.args.seed) + step * 1000003 + micro * 1009 + self.dp_rank * 9176
        rng = torch.Generator().manual_seed(seed)
        take = min(int(self.args.mtp_opd_max_anchors), cand.numel())
        if cand.numel() > take:
            cand = (
                cand[torch.randperm(cand.numel(), generator=rng)[:take].to(cand.device)]
                .sort()
                .values
            )
        width = (take + self._sp_size - 1) // self._sp_size
        padded = cand.new_full((width * self._sp_size,), int(cand[-1]))
        padded[:take] = cand
        sl = slice(self._sp_rank * width, (self._sp_rank + 1) * width)
        anchors = padded[sl]
        valid = (
            torch.arange(self._sp_rank * width, (self._sp_rank + 1) * width, device=ids.device)
            < take
        )
        media = batch.get("opd_media", [None])[0]
        warm = (
            self.verify_engine.opd_warm.remote(ids.tolist(), media) if self._sp_rank == 0 else None
        )
        rng_cuda = torch.Generator(device=ids.device).manual_seed(seed + self._sp_rank * 4711)
        with torch.no_grad():
            drafts = self.model(
                input_ids=ids,
                target_hidden=hidden,
                anchors=anchors,
                position_ids=positions,
                num_steps=self.args.mtp_opd_block_size,
                generator=rng_cuda,
            ).drafts
        all_drafts = self._gather(drafts)
        steps = self.args.mtp_opd_block_size
        gt_pos = padded[:, None] + torch.arange(1, steps, device=ids.device)
        match = (all_drafts[:, : steps - 1] == ids[gt_pos.clamp(max=length - 1)]) & (
            gt_pos < length
        )
        m = match.int().cumprod(1).sum(1)
        need = (
            ((m < steps - 1) & (torch.arange(len(padded), device=ids.device) < take))
            .nonzero()
            .flatten()
        )
        verify = None
        if self._sp_rank == 0 and need.numel():
            prefixes = (padded[need] + m[need] + 1).tolist()
            tails = [
                d[mm : steps - 1] for d, mm in zip(all_drafts[need].tolist(), m[need].tolist())
            ]
            verify = self.verify_engine.opd_verify.remote(
                ids.tolist(), prefixes, tails, warm, require_complete=True, media=media
            )
        return dict(
            ids=ids,
            hidden=hidden,
            positions=positions,
            anchors=anchors,
            drafts=drafts,
            valid=valid,
            padded=padded,
            take=take,
            m=m,
            need=need,
            verify=verify,
            warm=warm,
            sl=sl,
            delete_keys=batch.get("_mooncake_delete_keys", []),
        )

    def _teacher(self, ro):
        s, h = self.args.mtp_opd_block_size, ro["hidden"]
        anchors, m = ro["padded"], ro["m"]
        steps = torch.arange(s, device=h.device)
        teacher = h[(anchors[:, None] + steps).clamp(max=len(h) - 1)].clone()
        # A verifier failure must stop every DP/SP rank before gradient collectives.
        error = None
        if self._sp_rank == 0:
            try:
                covered = steps[None] <= m[:, None]
                if ro["verify"] is not None:
                    result = ray.get(ro["verify"])
                    ro["verify_metrics"] = (
                        float(result["time"]),
                        float(result.get("repair_time", 0)),
                        int(result.get("repair_branches", 0)),
                        len(ro["need"]),
                    )
                    branch = result["branch"].to(h.device).long()
                    pos = result["pos"].to(h.device).long()
                    rows = result["rows"].to(h.device)
                    if branch.numel() != pos.numel() or rows.shape != (
                        branch.numel(),
                        h.shape[-1],
                    ):
                        raise RuntimeError("Misaligned OPD teacher rows")
                    if bool(((branch < 0) | (branch >= len(ro["need"]))).any()):
                        raise RuntimeError("Invalid OPD branch index")
                    a = ro["need"][branch]
                    st = pos - anchors[a]
                    if (
                        bool(((st <= m[a]) | (st >= s)).any())
                        or (a * s + st).unique().numel() != st.numel()
                    ):
                        raise RuntimeError("Duplicate/out-of-range OPD teacher positions")
                    teacher[a, st] = rows.to(h.dtype)
                    covered[a, st] = True
                elif ro["warm"] is not None:
                    ray.get(ro["warm"])
                if not bool(covered[: ro["take"]].all()):
                    raise RuntimeError("Incomplete teacher after prefix-cache repair")
            except Exception as exc:
                error = exc
        failed = torch.tensor(int(error is not None), device=h.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self._metric_group)
        if failed.item():
            raise RuntimeError(
                f"OPD teacher failed collectively: {error or 'another rank failed'}"
            ) from error
        self._broadcast(teacher)
        return teacher[ro["sl"]]

    def _train_core_from_queue(self, step, num_batches):
        self.model.train()
        rollouts = [
            self._prepare(batch, step, i)
            for i, batch in enumerate(self._iter_batches_from_queue(num_batches))
        ]
        device = rollouts[0]["hidden"].device
        s = self.args.mtp_opd_block_size
        denom = torch.tensor(
            sum(int(ro["valid"].sum()) * s for ro in rollouts), device=device, dtype=torch.float32
        )
        dist.all_reduce(denom, group=self._metric_group)
        # Native HSDP averages DP and SP; manual finalization turns SP into SUM.
        scale = self.dp_size / denom
        totals = torch.zeros(2 + 2 * s, device=device, dtype=torch.float64)
        verify_totals = torch.zeros(4, device=device, dtype=torch.float64)
        seq_tokens = torch.tensor(
            sum(ro["ids"].numel() for ro in rollouts) / self._sp_size,
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(seq_tokens, group=self._metric_group)
        delete_keys = []
        for i, ro in enumerate(rollouts):
            teacher = self._teacher(ro)
            if self._sp_rank == 0 and "verify_metrics" in ro:
                verify_totals += verify_totals.new_tensor(ro["verify_metrics"])
            self.model.set_requires_all_reduce(i == num_batches - 1)
            student = self.model(
                input_ids=ro["ids"],
                target_hidden=ro["hidden"],
                anchors=ro["anchors"],
                drafts=ro["drafts"],
                position_ids=ro["positions"],
                num_steps=s,
            ).hidden
            losses, alpha = mtp_opd_loss(
                student,
                teacher,
                self.target_lm_head_weight,
                loss_type=self.args.mtp_opd_loss,
            )
            # e2e_tv returns one prefix loss per step. The existing global
            # anchors*steps denominator implements 1 - mean(prefix products).
            loss = (losses * ro["valid"][:, None]).sum() * scale
            loss.backward()
            totals[0] += (losses.detach() * ro["valid"][:, None]).sum()
            totals[1] += ro["valid"].sum()
            totals[2 : 2 + s] += (alpha * ro["valid"][:, None]).sum(0)
            totals[2 + s :] += (alpha.cumprod(dim=-1) * ro["valid"][:, None]).sum(0)
            delete_keys.extend(ro["delete_keys"])
            rollouts[i] = None
        self._usp_manual_grad_allreduce()
        grad_norm = self.optimizer.step()
        self.global_step += 1
        dist.barrier(group=self._metric_group)
        if delete_keys and self._sp_rank == 0:
            self._flush_mooncake_deletes(delete_keys)
        dist.all_reduce(totals, group=self._metric_group)
        dist.all_reduce(verify_totals, group=self._metric_group)
        metrics = {
            "train/loss": float(totals[0] / denom),
            "train/anchors": float(totals[1]),
            "train/grad_norm": float(grad_norm),
            "train/lr": float(self.optimizer.get_learning_rate()),
            "data/seq_tokens": float(seq_tokens),
        }
        metrics.update(
            {f"train/opd_alpha_{i + 1}": float(totals[i + 2] / totals[1]) for i in range(s)}
        )
        prefix_means = totals[2 + s :] / totals[1]
        metrics.update(
            {
                f"train/opd_prefix_alpha_product_{i + 1}": float(value)
                for i, value in enumerate(prefix_means)
            }
        )
        # Includes the +1 verifier token, as in eval's length convention, but
        # remains an alpha-product proxy rather than actual serving counters.
        metrics["train/opd_e2e_accept_length_proxy"] = float(1 + prefix_means.sum())
        metrics["train/opd_e2e_tv_loss"] = float(1 - prefix_means.mean())
        metrics.update(
            {
                f"opd/{name}": float(value)
                for name, value in zip(
                    (
                        "verify_seconds_sum",
                        "repair_seconds_sum",
                        "repair_branches",
                        "verify_branches",
                    ),
                    verify_totals,
                )
            }
        )
        logger.info("OPD step=%s metrics=%s", self.global_step, metrics)
        return metrics

    def save_model(self, step, force_sync=False):
        if not self.args.checkpoint_dir:
            return
        super().save_model(step, force_sync)
        error = None
        if dist.get_rank() == 0:
            path = Path(self.args.checkpoint_dir) / f"iter_{step + 1:07d}"
            try:
                marker = path / "opd_contract.json.tmp"
                marker.write_text(json.dumps(opd_contract(self.args), indent=2) + "\n")
                marker.replace(path / "opd_contract.json")
            except OSError as exc:
                error = exc
        device = next(self.draft_model.parameters()).device
        failed = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self._metric_group)
        if failed.item():
            raise RuntimeError(f"OPD checkpoint contract save failed: {error}") from error
