"""Online real spec-decode eval — architecture-agnostic controller + per-arch adapters.

Complements the forward-only simulated eval in controller/eval.py (draft on cached
teacher-forcing hidden states). This exports the current draft, loads it into a real
vLLM spec-decode engine, drives a fixed prompt set, and reads the engine-native
acceptance length / per-position rate.

Arch-specific logic lives in an ArchAdapter picked by draft model_arch: how to turn
a training checkpoint into a vLLM-servable dir, plus the --speculative-config +
serve env. Stage 1 serves on a throwaway subprocess; stage 2 uses a persistent
EvalEngine actor. No vLLM fork changes, no in-place drafter hot-swap.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from angelspec.utils.logging import logger

# vLLM prints, per spec-decode metrics flush:
#   SpecDecoding metrics: Mean acceptance length: 2.05, ... Per-position acceptance rate: 0.566, 0.309, 0.177, ...
_ACC_LEN_RE = re.compile(r"Mean acceptance length:\s*([0-9.]+)")
_PER_POS_RE = re.compile(r"Per-position acceptance rate:\s*([0-9.,\s]+?)(?:,\s*Avg|$)")
_SERVER_READY_RE = re.compile(r"Application startup complete|Starting vLLM server on")


# ── Per-arch adapters ────────────────────────────────────────────────────────


@dataclass
class ServedBuild:
    """Result of turning a training ckpt into a servable dir."""

    model_path: str  # dir passed to vllm serve --model
    spec_config: dict  # --speculative-config JSON
    serve_env: dict = field(default_factory=dict)  # extra env for the serve process
    extra_serve_args: list[str] = field(default_factory=list)


class ArchAdapter:
    """Turns a training checkpoint into a vLLM-servable spec-decode dir."""

    #: draft ``model_arch`` / ``model_type`` values this adapter handles
    handles: tuple[str, ...] = ()

    def build_served_dir(self, ckpt_dir: str, work_dir: str, args) -> ServedBuild:
        raise NotImplementedError


class MTPArchAdapter(ArchAdapter):
    """Hy3/v4 single-head MTP.

    Invokes the verified CLI chain as subprocesses (convert_mtp_to_hf.py →
    build_small_mtp_served.py) so the already-validated code path runs unchanged.
    The scaffold (a served dir with a dedicated layer-N shard) comes from
    ``online_eval_mtp_scaffold``.
    """

    handles = ("mtp",)

    def build_served_dir(self, ckpt_dir: str, work_dir: str, args) -> ServedBuild:
        # dense-Qwen3 MTP (mlp_type="dense") uses a different servable layout than
        # the hy_v3 path: mtp.* keys + runtime-injected Qwen3MTPForCausalLM
        # (registered in every vLLM process by vllm_runtime_patch's general_plugins
        # entry point), NOT model.layers.<N>.* for native hy_v3_mtp.
        draft_cfg = getattr(args, "draft_model_config", None)
        mlp_type = "moe"
        if draft_cfg and os.path.isfile(draft_cfg):
            try:
                mlp_type = json.load(open(draft_cfg)).get("mlp_type", "moe")
            except Exception:
                pass
        if mlp_type == "dense":
            return self._build_qwen3_dense(ckpt_dir, work_dir, args)

        repo = _repo_root()
        py = _python_exe()
        scaffold = getattr(args, "online_eval_mtp_scaffold", None)
        if not scaffold or not os.path.isdir(scaffold):
            raise ValueError(
                "MTP online eval requires online_eval_mtp_scaffold=<served dir with a "
                f"dedicated spec-layer shard>; got {scaffold!r}"
            )
        spec_layer = int(getattr(args, "online_eval_mtp_spec_layer", 0)) or _infer_spec_layer(
            scaffold
        )

        head_dir = os.path.join(work_dir, "head")
        served_dir = os.path.join(work_dir, "served")

        convert_cmd = [
            py,
            os.path.join(repo, "tools", "convert_mtp_to_hf.py"),
            "--input-dir",
            ckpt_dir,
            "--output-dir",
            head_dir,
            "--spec-layer",
            str(spec_layer),
            "-f",
        ]
        _run(convert_cmd, "convert_mtp_to_hf")

        build_cmd = [
            py,
            os.path.join(repo, "tools", "build_small_mtp_served.py"),
            "--scaffold",
            scaffold,
            "--head",
            os.path.join(head_dir, "model.safetensors"),
            "--out",
            served_dir,
            "--spec-layer",
            str(spec_layer),
        ]
        _run(build_cmd, "build_small_mtp_served")

        num_spec = int(getattr(args, "online_eval_num_spec_tokens", 3))
        return ServedBuild(
            model_path=served_dir,
            spec_config={"num_speculative_tokens": num_spec, "method": "mtp"},
            # match the training-side extract engine's precision (HF mode); keep
            # HPC accel paths off (external `hpc` kernel module not installed).
            serve_env={
                "VLLM_PRECISION_MODE": "HF",
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                "VLLM_ENABLE_HPC_ROUTER": "0",
                "VLLM_ENABLE_HPC_ROPE_NORM": "0",
            },
            extra_serve_args=["--reasoning-parser", "hy_v3"],
        )

    def _build_qwen3_dense(self, ckpt_dir: str, work_dir: str, args) -> ServedBuild:
        """Dense-Qwen3 MTP: append the trained mtp.* head onto the Qwen3 target dir
        (tools/convert_mtp_dense_to_qwen3_mtp.py) and serve it with method=mtp. The
        target is the hidden-state source model (args.target_model_path); vLLM picks
        our runtime-registered Qwen3MTPForCausalLM via the num_nextn_predict_layers
        promotion in vllm_runtime_patch.apply_qwen3_mtp."""
        repo = _repo_root()
        py = _python_exe()
        base = getattr(args, "target_model_path", None)
        if not base or not os.path.isdir(base):
            raise ValueError(
                f"dense-Qwen3 MTP online eval requires target_model_path=<Qwen3 dir>; got {base!r}"
            )
        served_dir = os.path.join(work_dir, "served")
        _run(
            [
                py,
                os.path.join(repo, "tools", "convert_mtp_dense_to_qwen3_mtp.py"),
                "--input-dir",
                ckpt_dir,
                "--base",
                base,
                "--out",
                served_dir,
                "-f",
            ],
            "convert_mtp_dense_to_qwen3_mtp",
        )
        num_spec = int(getattr(args, "online_eval_num_spec_tokens", 3))
        return ServedBuild(
            model_path=served_dir,
            spec_config={"num_speculative_tokens": num_spec, "method": "mtp"},
            # Pure Qwen3 target: no hy_v3 HPC/reasoning-parser paths.
            serve_env={
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                "VLLM_ENABLE_HPC_ROUTER": "0",
                "VLLM_ENABLE_HPC_ROPE_NORM": "0",
            },
            extra_serve_args=[],
        )


_ARCH_ADAPTERS: dict[str, ArchAdapter] = {}


def _register(adapter: ArchAdapter) -> None:
    for name in adapter.handles:
        _ARCH_ADAPTERS[name] = adapter


_register(MTPArchAdapter())


def get_arch_adapter(model_arch: str) -> Optional[ArchAdapter]:
    return _ARCH_ADAPTERS.get(model_arch)


def resolve_sampling(args) -> dict:
    """Effective acceptance sampling params from the online_eval config.

    argmax → greedy (temperature 0.0); rs → the configured temperature/top_p/
    top_k/seed. Shared by the actor and subprocess paths so both agree.
    """
    mode = str(getattr(args, "online_eval_sampling_mode", "argmax")).lower()
    if mode == "rs":
        return {
            "temperature": float(getattr(args, "online_eval_temperature", 1.0)),
            "top_p": float(getattr(args, "online_eval_top_p", 1.0)),
            "top_k": int(getattr(args, "online_eval_top_k", -1)),
            "seed": getattr(args, "online_eval_seed", None),
        }
    return {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": None}


# ── Controller ───────────────────────────────────────────────────────────────


@dataclass
class OnlineEvalState:
    enabled: bool
    interval: int
    dataset: Optional[str]
    prompt_key: str
    limit: int
    max_new_tokens: int
    gpus: str  # CUDA_VISIBLE_DEVICES for the eval serve (subprocess fallback)
    port: int
    adapter: Optional[ArchAdapter]
    model_arch: str
    sampling_mode: str = "argmax"
    sampling: dict = field(
        default_factory=lambda: {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": None}
    )
    # Multi-dataset mode (vendored eval_prompts). When ``datasets`` is set, the
    # actor path runs each set separately and reports per-dataset + mean metrics.
    datasets: Optional[list[str]] = None
    per_dataset_limit: Optional[int] = None
    dataset_sample_size: int = 1000
    dataset_seed: int = 42
    eval_prompts_dir: Optional[str] = None
    reasoning_effort: str = "no_think"
    eval_engine: object = None  # persistent EvalEngine actor (stage 2), else None


def setup_online_eval(args, model_arch: str, eval_engine=None) -> OnlineEvalState:
    """Validate config and pick the arch adapter. Returns a disabled state if off.

    ``eval_engine``: the persistent EvalEngine Ray actor (stage 2). When present,
    eval runs on that dedicated actor (machine-isolated, target loaded once);
    when None, falls back to the stage-1 throwaway subprocess serve.
    """
    disabled = OnlineEvalState(False, 0, None, "conversations", 0, 0, "", 0, None, model_arch)
    enabled = bool(getattr(args, "online_eval_enabled", False))
    if not enabled:
        return disabled

    adapter = get_arch_adapter(model_arch)
    if adapter is None:
        logger.warning(
            "online_eval_enabled but no adapter for model_arch=%r (have %s); disabling online eval.",
            model_arch,
            sorted(_ARCH_ADAPTERS),
        )
        return disabled

    dataset = getattr(args, "online_eval_dataset", None)
    datasets = getattr(args, "online_eval_datasets", None) or None
    if not dataset and not datasets:
        logger.warning(
            "online_eval_enabled but neither online_eval_dataset nor online_eval_datasets is set; disabling."
        )
        return disabled

    gpus = _resolve_eval_gpus(args)
    port = int(getattr(args, "online_eval_port", 8130))
    mode = "dedicated-actor" if eval_engine is not None else "subprocess"
    sampling_mode = str(getattr(args, "online_eval_sampling_mode", "argmax")).lower()
    sampling = resolve_sampling(args)
    logger.info(
        "Online real eval ENABLED: arch=%s interval=%d dataset=%s datasets=%s mode=%s "
        "accept=%s sampling=%s gpus=%s port=%d",
        model_arch,
        int(getattr(args, "online_eval_interval", 500)),
        dataset,
        datasets,
        mode,
        sampling_mode,
        sampling,
        gpus,
        port,
    )
    return OnlineEvalState(
        enabled=True,
        interval=int(getattr(args, "online_eval_interval", 500)),
        dataset=dataset,
        prompt_key=getattr(args, "online_eval_prompt_key", "conversations"),
        limit=int(getattr(args, "online_eval_limit", 64)),
        max_new_tokens=int(getattr(args, "online_eval_max_new_tokens", 256)),
        gpus=gpus,
        port=port,
        adapter=adapter,
        model_arch=model_arch,
        sampling_mode=sampling_mode,
        sampling=sampling,
        datasets=list(datasets) if datasets else None,
        per_dataset_limit=getattr(args, "online_eval_per_dataset_limit", None),
        dataset_sample_size=int(getattr(args, "online_eval_dataset_sample_size", 1000)),
        dataset_seed=int(getattr(args, "online_eval_dataset_seed", 42)),
        eval_prompts_dir=getattr(args, "online_eval_eval_prompts_dir", None),
        reasoning_effort=str(getattr(args, "online_eval_reasoning_effort", "no_think")),
        eval_engine=eval_engine,
    )


def maybe_run_online_eval(state: OnlineEvalState, step: int, train_group, args) -> dict:
    """Non-blocking online eval (stage-2 actor path).

    The draft export (a brief FSDP gather across the training ranks) is done
    SYNCHRONOUSLY, then serve+eval (the ~1-4min spec-decode on the separate
    EvalEngine actor) runs in a BACKGROUND thread so training does not pause for it.
    Returns the metrics of a PREVIOUSLY-launched async eval that just finished
    (tagged with its launch step), or {}. Never raises into the loop. The subprocess
    fallback (no persistent eval_engine) stays blocking.
    """
    # 1) collect a previously-launched async eval that has finished (non-blocking).
    done = _collect_async_eval(state)

    due = state.enabled and state.interval > 0 and step > 0 and step % state.interval == 0
    if not due:
        return done

    # 2a) subprocess fallback (no persistent actor): still blocking.
    if state.eval_engine is None:
        try:
            return _run_online_eval_subprocess(state, step, train_group, args) or done
        except Exception:
            logger.exception("Online real eval failed at step %d (continuing training)", step)
            return done

    # 2b) actor path: don't stack evals (the single EvalEngine runs them serially).
    prev = getattr(state, "_async_thread", None)
    if prev is not None and prev.is_alive():
        logger.warning(
            "[online-eval step %d] previous async eval still running; skipping this interval", step
        )
        return done
    try:
        _launch_async_online_eval(state, step, train_group, args)
    except Exception:
        logger.exception("Online real eval launch failed at step %d (continuing training)", step)
    return done


def _make_eval_work_dir(args, step: int) -> str:
    """Per-eval work dir on a SHARED filesystem (NOT node-local /tmp).

    save_draft_model_for_serving writes pytorch_model.bin from trainer rank 0 (a
    training node) while the servable conversion / serve runs on the driver/eval
    node. Under tempfile.mkdtemp() (/tmp) those are different physical dirs on a
    multi-node cluster, so convert can't see the export ("metadata is None").
    Anchor under output_dir (cephfs) so every node sees the same files.
    """
    base = os.path.join(
        os.path.abspath(getattr(args, "output_dir", ".") or "."), "online_eval_work"
    )
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"step{step}_", dir=base)


def _launch_async_online_eval(state: OnlineEvalState, step: int, train_group, args) -> None:
    """Export the draft (SYNC — brief FSDP gather on the training ranks) then run
    serve+eval on the persistent EvalEngine actor in a background daemon thread, so the
    training loop resumes immediately after the export."""
    import threading

    work_dir = _make_eval_work_dir(args, step)
    ckpt_dir = os.path.join(work_dir, "draft_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.monotonic()
    train_group.save_draft_model_for_serving(ckpt_dir)  # collective across train ranks
    logger.info(
        "[online-eval step %d] exported draft in %.1fs; serve+eval dispatched to background",
        step,
        time.monotonic() - t0,
    )

    def _worker():
        m = {}
        try:
            m = _actor_serve_and_eval(state, step, work_dir, ckpt_dir, args)
        except Exception:
            logger.exception("[online-eval step %d] async eval failed (training unaffected)", step)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            # set result LAST; the main thread only reads it after the thread ends
            # (is_alive()==False), so this needs no lock.
            state._async_result = m or {}

    state._async_result = {}
    state._async_step = step
    t = threading.Thread(target=_worker, name=f"online-eval-{step}", daemon=True)
    state._async_thread = t
    t.start()


def _actor_serve_and_eval(
    state: OnlineEvalState, step: int, work_dir: str, ckpt_dir: str, args
) -> dict:
    """(Runs in the background thread.) Build the served dir, (re)load it into the
    EvalEngine actor, drive the eval prompt set(s), return metrics tagged with step.
    Only touches the exported disk snapshot + the separate EvalEngine actor (GPU-
    isolated) — never the training weights/collectives."""
    import ray

    build = state.adapter.build_served_dir(ckpt_dir, work_dir, args)

    ok = ray.get(state.eval_engine.reload_draft.remote(build.model_path))
    if not ok:
        raise RuntimeError("eval engine reload_draft returned False")

    if state.datasets:
        named_prompts = _load_named_eval_prompts(state)
        if not named_prompts:
            logger.warning("[online-eval step %d] no eval prompts loaded", step)
            return {}
        metrics = ray.get(
            state.eval_engine.run_eval_datasets.remote(
                named_prompts, state.max_new_tokens, state.reasoning_effort
            )
        )
    else:
        prompts = _load_eval_prompts(state)
        if not prompts:
            logger.warning("[online-eval step %d] no eval prompts loaded", step)
            return {}
        metrics = ray.get(state.eval_engine.run_eval.remote(prompts, state.max_new_tokens))
    if metrics:
        logger.info(
            "[online-eval step %d] REAL accept_len=%.3f pos0=%s (async)",
            step,
            metrics.get("online_eval/real_accept_len", 0.0),
            metrics.get("online_eval/real_pos0"),
        )
        metrics["online_eval/step"] = step
    return metrics


def _collect_async_eval(state: OnlineEvalState, block: bool = False) -> dict:
    """Return the metrics of the in-flight async eval if it has finished (or, with
    block=True, wait for it). {} if none / not done. Clears the slot when collected."""
    t = getattr(state, "_async_thread", None)
    if t is None:
        return {}
    if block:
        t.join()
    elif t.is_alive():
        return {}
    metrics = getattr(state, "_async_result", {}) or {}
    state._async_thread = None
    state._async_result = {}
    return metrics


def drain_online_eval(state: OnlineEvalState) -> dict:
    """Block until any in-flight async eval finishes and return its metrics. Call once
    at the end of training so the final interval's eval isn't lost when the process exits."""
    return _collect_async_eval(state, block=True)


def _load_eval_prompts(state: OnlineEvalState) -> list[str]:
    """Load up to `limit` chat-formatted prompt strings from the eval jsonl.

    Reuses the same jsonl the stage-1 path fed eval_accept_rate.py. Each line is
    a conversation under `prompt_key`; we render just the prompt turns (drop the
    trailing assistant answer) so the engine generates fresh under spec-decode.
    """
    import json as _json

    path = state.dataset
    key = state.prompt_key
    prompts: list[str] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                convo = obj.get(key) or obj.get("conversations") or obj.get("messages")
                if not convo:
                    continue
                # keep turns up to (not including) the last assistant turn
                turns = [
                    t for t in convo if isinstance(t, dict) and t.get("role") != "assistant"
                ] or convo[:-1]
                text = "\n".join(
                    str(t.get("content", "")) for t in turns if isinstance(t, dict)
                ).strip()
                if text:
                    prompts.append(text)
                if len(prompts) >= state.limit:
                    break
    except Exception:
        logger.exception("[online-eval] failed loading prompts from %s", path)
    return prompts


def _load_named_eval_prompts(state: OnlineEvalState) -> list[tuple[str, list[str]]]:
    """Load vendored prompts for each dataset in ``state.datasets``.

    Returns ``[(name, [prompt, ...]), ...]``; datasets that fail to load are
    logged and skipped rather than aborting the whole eval.
    """
    from angelspec.controller import eval_datasets

    limit = state.per_dataset_limit if state.per_dataset_limit is not None else state.limit
    out: list[tuple[str, list[str]]] = []
    for name in state.datasets:
        try:
            prompts = eval_datasets.load_dataset_prompts(
                name,
                sample_size=state.dataset_sample_size,
                seed=state.dataset_seed,
                limit=limit,
                prompts_dir=state.eval_prompts_dir,
            )
        except Exception:
            logger.exception("[online-eval] failed loading dataset %s", name)
            continue
        if prompts:
            out.append((name, prompts))
        else:
            logger.warning("[online-eval] dataset %s produced no prompts", name)
    return out


def _run_online_eval_subprocess(state: OnlineEvalState, step: int, train_group, args) -> dict:
    if state.datasets and not state.dataset:
        logger.warning(
            "[online-eval] multi-dataset (online_eval_datasets) is only supported on "
            "the stage-2 dedicated eval engine (set online_eval.num_gpus > 0). "
            "No single online_eval_dataset jsonl to fall back to; skipping."
        )
        return {}
    work_dir = _make_eval_work_dir(args, step)
    ckpt_dir = os.path.join(work_dir, "draft_ckpt")
    server = None
    try:
        # 1. export current draft weights (arch-agnostic; native trainer format)
        os.makedirs(ckpt_dir, exist_ok=True)
        t0 = time.monotonic()
        train_group.save_draft_model_for_serving(ckpt_dir)
        logger.info("[online-eval step %d] exported draft in %.1fs", step, time.monotonic() - t0)

        # 2. arch adapter → servable dir + spec config
        build = state.adapter.build_served_dir(ckpt_dir, work_dir, args)

        # 3. serve it (throwaway server on the reserved GPU(s))
        server, log_path = _launch_serve(state, build, args)
        if not _wait_server_ready(server, log_path, timeout=600):
            raise RuntimeError("eval server did not become ready")

        # 4. drive eval prompts (server accumulates spec-decode metrics)
        _drive_eval_prompts(state, args)

        # 5. parse engine-native acceptance metrics from the server log
        metrics = _parse_spec_metrics(log_path)
        if metrics:
            logger.info(
                "[online-eval step %d] REAL accept_len=%.3f per_pos=%s",
                step,
                metrics.get("online_eval/real_accept_len", 0.0),
                metrics.get("online_eval/real_per_pos", []),
            )
            metrics["online_eval/step"] = step
        return metrics
    finally:
        if server is not None:
            _terminate(server)
        shutil.rmtree(work_dir, ignore_errors=True)


# ── serve subprocess management ──────────────────────────────────────────────


def _launch_serve(state: OnlineEvalState, build: ServedBuild, args):
    py = _python_exe()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = state.gpus
    env.setdefault("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = _env_lib() + ":" + env["LD_LIBRARY_PATH"]
    # proxies break intra-host vLLM worker setup
    for p in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(p, None)
    env.update(build.serve_env)

    max_len = int(getattr(args, "max_seq_length", 8192)) or 8192
    log_path = os.path.join(tempfile.gettempdir(), f"online_eval_serve_{state.port}.log")
    cmd = [
        py,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        build.model_path,
        "--speculative-config",
        json.dumps(build.spec_config),
        "--tensor-parallel-size",
        "1",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(getattr(args, "online_eval_gpu_util", 0.85)),
        "--max-model-len",
        str(max_len),
        "--hf-overrides",
        json.dumps({"max_position_embeddings": max_len}),
        "--enforce-eager",
        "--moe-backend",
        os.environ.get("MOE_BACKEND", "triton"),
        "--port",
        str(state.port),
        "--served-model-name",
        "online-eval",
        *build.extra_serve_args,
    ]
    logf = open(log_path, "w")
    logger.info("[online-eval] launching serve on GPU(s)=%s port=%d", state.gpus, state.port)
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    return proc, log_path


def _wait_server_ready(proc, log_path: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.error("[online-eval] serve process exited early (code %s)", proc.returncode)
            return False
        try:
            with open(log_path) as f:
                if _SERVER_READY_RE.search(f.read()):
                    return True
        except FileNotFoundError:
            pass
        time.sleep(5)
    logger.error("[online-eval] serve did not become ready within %ds", timeout)
    return False


def _drive_eval_prompts(state: OnlineEvalState, args) -> None:
    """Reuse tools/eval_accept_rate.py to fire concurrent requests at the server."""
    py = _python_exe()
    repo = _repo_root()
    cmd = [
        py,
        os.path.join(repo, "tools", "eval_accept_rate.py"),
        "--port",
        str(state.port),
        "--jsonl",
        state.dataset,
        "--prompt-key",
        state.prompt_key,
        "--limit",
        str(state.limit),
        "--temperature",
        str(state.sampling["temperature"]),
        "--top-p",
        str(state.sampling["top_p"]),
        "--top-k",
        str(state.sampling["top_k"]),
    ]
    if state.sampling.get("seed") is not None:
        cmd += ["--seed", str(state.sampling["seed"])]
    env = dict(os.environ)
    for p in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(p, None)
    _run(cmd, "eval_accept_rate", env=env, timeout=1800)


def _parse_spec_metrics(log_path: str) -> dict:
    """Take the last SpecDecoding metrics line the server flushed."""
    try:
        with open(log_path) as f:
            text = f.read()
    except FileNotFoundError:
        return {}
    acc_lens = _ACC_LEN_RE.findall(text)
    per_pos = _PER_POS_RE.findall(text)
    if not acc_lens:
        logger.warning("[online-eval] no SpecDecoding metrics found in server log")
        return {}
    out = {"online_eval/real_accept_len": float(acc_lens[-1])}
    if per_pos:
        rates = [float(x) for x in per_pos[-1].replace(" ", "").strip(",").split(",") if x]
        out["online_eval/real_per_pos"] = rates
        if rates:
            out["online_eval/real_pos0"] = rates[0]
    return out


def _terminate(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ── helpers ──────────────────────────────────────────────────────────────────


def _run(cmd: list[str], name: str, env: dict | None = None, timeout: int = 900) -> None:
    logger.info("[online-eval] $ %s", " ".join(cmd))
    res = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(
            "[online-eval] %s failed (code %d)\nstdout:\n%s\nstderr:\n%s",
            name,
            res.returncode,
            res.stdout[-2000:],
            res.stderr[-2000:],
        )
        raise RuntimeError(f"{name} failed (code {res.returncode})")


def _repo_root() -> str:
    # controller/online_eval.py -> angelspec/ -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _python_exe() -> str:
    import sys

    return sys.executable


def _env_lib() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(_python_exe())), "lib")


def _resolve_eval_gpus(args) -> str:
    """Which GPU(s) the throwaway eval server binds.

    Stage 1 shared mode: reuse the last inference-segment GPU (training holds the
    front cards; inference the back). We bind a single card by its *local* index
    within the process's CUDA_VISIBLE_DEVICES view.
    """
    explicit = getattr(args, "online_eval_gpus", None)
    if explicit:
        return str(explicit)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return visible.split(",")[-1].strip()
    return "0"


def _infer_spec_layer(scaffold: str) -> int:
    """Read the spec layer index from the scaffold config (num_hidden_layers)."""
    cfg = os.path.join(scaffold, "config.json")
    try:
        with open(cfg) as f:
            d = json.load(f)
        return int(d.get("num_hidden_layers"))
    except Exception as e:
        raise ValueError(
            f"could not infer spec layer from {cfg}; set online_eval_mtp_spec_layer"
        ) from e


def infer_model_arch(args) -> str:
    """Best-effort read of the draft ``model_type`` from the draft config JSON.

    Returns "" when it can't be determined (online eval then no-ops with a warning).
    """
    path = getattr(args, "draft_model_config", None)
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path) as f:
            return str(json.load(f).get("model_type", ""))
    except Exception:
        return ""
