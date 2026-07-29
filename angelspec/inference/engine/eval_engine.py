"""Persistent eval engine — a long-lived Ray actor that runs real speculative
decoding (target + current draft) and reports engine-native acceptance metrics
offline (no HTTP server).

Pinned to a dedicated node/GPU via the eval role placement group. The target loads
once at init and stays resident; reload_draft rebuilds the LLM in-actor (vLLM can't
hot-swap the drafter) but keeps the node warm. Metrics come from LLM.get_metrics().
"""

from __future__ import annotations

import os

from angelspec.ray.ray_actor import RayActor
from angelspec.utils.logging import logger, setup_file_logging


class EvalEngine(RayActor):
    """Long-lived spec-decode eval engine (one target + swappable draft)."""

    def __init__(
        self,
        args,
        rank: int = 0,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 1,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._llm = None
        self._draft_dir = None
        self.local_gpu_id = None
        setup_file_logging("eval", self.rank, group=engine_group)

    # -- lifecycle ------------------------------------------------------------

    def init(self) -> None:
        """Pin GPUs and register any vendored HF configs. Does NOT build the LLM
        yet — the LLM is (re)built on the first ``reload_draft`` once we have a
        draft to serve."""
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            gpu_ids = [str(self.base_gpu_id + i) for i in range(self.num_gpus_per_engine)]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            logger.info(
                "EvalEngine rank %d: base_gpu_id=%s CUDA_VISIBLE_DEVICES=%s",
                self.rank,
                self.base_gpu_id,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        logger.info(
            "EvalEngine rank %d: init complete (LLM built lazily on reload_draft)", self.rank
        )

    def is_ready(self) -> bool:
        return True

    # -- draft management -----------------------------------------------------

    def reload_draft(self, draft_model_path: str) -> bool:
        """(Re)build the spec-decode LLM with the given draft dir as the drafter.

        vLLM's speculative_config is fixed at LLM construction, so a draft swap
        rebuilds the LLM (target re-read too). Runs at eval cadence, not per-request.
        Returns True on success.
        """
        self._teardown_llm()
        spec_method = getattr(self.args, "online_eval_speculative_method", "mtp")
        num_spec = int(getattr(self.args, "online_eval_num_spec_tokens", 3))
        max_len = int(getattr(self.args, "max_seq_length", 8192)) or 8192

        # Match the training-side extract engine's precision path (HF mode) so the
        # draft sees the same hidden distribution it was trained under.
        os.environ.setdefault("VLLM_PRECISION_MODE", "HF")
        os.environ.setdefault("VLLM_ENABLE_HPC_ROUTER", "0")
        os.environ.setdefault("VLLM_ENABLE_HPC_ROPE_NORM", "0")

        from vllm import LLM

        spec_cfg = self._build_spec_config(spec_method, num_spec, draft_model_path)
        # For MTP the served dir IS the model (target + spliced MTP layer, vLLM
        # auto-detects hy_v3_mtp); for dflash/eagle3 the model is the plain target
        # and the draft is a separate path inside spec_cfg.
        model_path = draft_model_path if spec_method == "mtp" else self.args.target_model_path
        logger.info(
            "EvalEngine: building spec-decode LLM (method=%s, num_spec=%d, model=%s, draft=%s)",
            spec_method,
            num_spec,
            model_path,
            draft_model_path,
        )
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=self.num_gpus_per_engine,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
            gpu_memory_utilization=float(getattr(self.args, "online_eval_gpu_util", 0.85)),
            max_model_len=max_len,
            enforce_eager=True,
            # get_metrics() reads Prometheus counters; offline LLM disables stat
            # logging by default (llm.py), so enable it or get_metrics() asserts.
            disable_log_stats=False,
            speculative_config=spec_cfg,
            **self._extra_llm_kwargs(),
        )
        self._draft_dir = draft_model_path
        return True

    def _build_spec_config(self, method: str, num_spec: int, draft_path: str) -> dict:
        """Per-method speculative_config. (Only mtp is wired for now.)"""
        if method == "mtp":
            # For hy_v3 MTP, the served dir IS the target dir with the MTP
            # layer spliced in; vLLM auto-detects hy_v3_mtp. The draft_path here is
            # that served dir, so it doubles as the model.
            return {"num_speculative_tokens": num_spec, "method": "mtp"}
        # dflash / eagle3: draft is a separate model dir
        return {
            "num_speculative_tokens": num_spec,
            "method": method,
            "model": draft_path,
        }

    def _extra_llm_kwargs(self) -> dict:
        out = {}
        moe = os.environ.get("MOE_BACKEND")
        if moe:
            out["moe_backend"] = moe
        rp = getattr(self.args, "online_eval_reasoning_parser", None)
        if rp:
            out["reasoning_parser"] = rp
        return out

    # -- eval -----------------------------------------------------------------

    def run_eval(self, prompts: list[str], max_new_tokens: int = 256) -> dict:
        """Run offline spec-decode generation over prompts, return acceptance
        metrics (counter delta over this call). Never raises — {} on error."""
        if self._llm is None:
            logger.warning("EvalEngine.run_eval called before reload_draft; skipping")
            return {}
        try:
            before = self._snapshot_counters()
            self._chat_generate(prompts, max_new_tokens)
            metrics = self._metrics_from_delta(before, self._snapshot_counters(), "online_eval/")
            if metrics:
                logger.info(
                    "EvalEngine: real accept_len=%.3f", metrics["online_eval/real_accept_len"]
                )
            return metrics
        except Exception:
            logger.exception("EvalEngine.run_eval failed")
            return {}

    def run_eval_datasets(
        self,
        named_prompts: list[tuple[str, list[str]]],
        max_new_tokens: int = 256,
        reasoning_effort: str | None = None,
    ) -> dict:
        """Run one eval per named dataset (sequential, counter-diffed per set) and
        report per-dataset acceptance + a cross-dataset mean under the legacy
        ``online_eval/real_accept_len`` key. Never raises — {} on error."""
        if self._llm is None:
            logger.warning("EvalEngine.run_eval_datasets called before reload_draft; skipping")
            return {}
        out: dict = {}
        accept_lens: list[float] = []
        for name, prompts in named_prompts:
            if not prompts:
                logger.warning("EvalEngine: dataset %s has no prompts, skipping", name)
                continue
            try:
                before = self._snapshot_counters()
                self._chat_generate(prompts, max_new_tokens, reasoning_effort)
                m = self._metrics_from_delta(
                    before, self._snapshot_counters(), f"online_eval/{name}/"
                )
            except Exception:
                logger.exception("EvalEngine: dataset %s eval failed", name)
                continue
            if not m:
                continue
            out.update(m)
            al = m[f"online_eval/{name}/real_accept_len"]
            accept_lens.append(al)
            logger.info("EvalEngine: [%s] real accept_len=%.3f (n=%d)", name, al, len(prompts))
        if accept_lens:
            out["online_eval/real_accept_len"] = sum(accept_lens) / len(accept_lens)
            out["online_eval/real_num_datasets"] = float(len(accept_lens))
            logger.info(
                "EvalEngine: mean accept_len=%.3f over %d datasets",
                out["online_eval/real_accept_len"],
                len(accept_lens),
            )
        return out

    def _chat_generate(
        self, prompts: list[str], max_new_tokens: int, reasoning_effort: str | None = None
    ) -> None:
        """Chat-templated offline generation (chat completions + reasoning_effort)."""
        from vllm import SamplingParams

        from angelspec.controller.online_eval import resolve_sampling

        if reasoning_effort is None:
            reasoning_effort = getattr(self.args, "online_eval_reasoning_effort", None)

        s = resolve_sampling(self.args)
        logger.info(
            "EvalEngine chat sampling=%s reasoning_effort=%s n_prompts=%d",
            s,
            reasoning_effort,
            len(prompts),
        )
        sp = SamplingParams(
            temperature=s["temperature"],
            top_p=s["top_p"],
            top_k=s["top_k"],
            seed=s["seed"],
            max_tokens=int(max_new_tokens),
        )
        messages = [[{"role": "user", "content": p}] for p in prompts]
        chat_kwargs = {}
        if reasoning_effort:
            chat_kwargs["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        self._llm.chat(messages, sp, use_tqdm=False, **chat_kwargs)

    def _snapshot_counters(self) -> dict:
        """Raw cumulative spec-decode counters from vLLM Prometheus metrics."""
        by_name = {m.name: m for m in self._llm.get_metrics()}

        def val(name):
            m = by_name.get(name)
            v = getattr(m, "value", None) if m is not None else None
            return float(v) if v is not None else 0.0

        per_pos_m = by_name.get("vllm:spec_decode_num_accepted_tokens_per_pos")
        per_pos = [float(v) for v in getattr(per_pos_m, "values", []) or []] if per_pos_m else []
        return {
            "n_drafts": val("vllm:spec_decode_num_drafts"),
            "n_draft_tokens": val("vllm:spec_decode_num_draft_tokens"),
            "n_accepted": val("vllm:spec_decode_num_accepted_tokens"),
            "per_pos": per_pos,
        }

    def _metrics_from_delta(self, before: dict, after: dict, prefix: str) -> dict:
        """Acceptance stats from the counter delta between two snapshots.

        acceptance_len = 1 + Δaccepted / Δdrafts
        per_pos_rate[i] = Δaccepted_at_pos[i] / Δdrafts
        overall_rate    = Δaccepted / Δdraft_tokens
        """
        d_drafts = after["n_drafts"] - before["n_drafts"]
        d_draft_tokens = after["n_draft_tokens"] - before["n_draft_tokens"]
        d_accepted = after["n_accepted"] - before["n_accepted"]
        if d_drafts <= 0:
            logger.warning("EvalEngine: no spec_decode drafts in delta (prefix=%s)", prefix)
            return {}

        out = {
            f"{prefix}real_accept_len": 1.0 + (d_accepted / d_drafts),
            f"{prefix}real_num_drafts": float(d_drafts),
        }
        if d_draft_tokens > 0:
            out[f"{prefix}real_overall_accept_rate"] = d_accepted / d_draft_tokens
        # before-snapshot per_pos may be empty (no metrics yet before the first
        # generate); pad missing entries with 0 and key off the after length.
        before_pp, after_pp = before["per_pos"], after["per_pos"]
        for i in range(len(after_pp)):
            b = before_pp[i] if i < len(before_pp) else 0.0
            out[f"{prefix}real_pos{i}"] = (after_pp[i] - b) / d_drafts
        return out

    def _teardown_llm(self) -> None:
        if self._llm is not None:
            try:
                del self._llm
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._llm = None

    def shutdown(self) -> None:
        self._teardown_llm()
