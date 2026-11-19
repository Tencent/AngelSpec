"""Single-node parity check for Hy4 flash hidden-state capture.

Run three steps on one 8-GPU node (``native`` and ``capture`` each load the model):

    python scripts/hy_v4_flash_capture_parity.py native  --model $TARGET --out $DIR
    python scripts/hy_v4_flash_capture_parity.py capture --model $TARGET --out $DIR
    python scripts/hy_v4_flash_capture_parity.py compare --model $TARGET --out $DIR

``native`` runs stock vLLM with AngelSpec's plugin disabled and records what the
attention iHC pre-gate of ``--probe-layer`` makes of the residual stream. ``capture`` runs
with the plugin and ``extract_hidden_states``, reading the cache through AngelSpec's
Mooncake connector with Mooncake swapped for a file sink. ``compare`` checks that (1) the
captured pre-norm final hidden state (aux id ``num_hidden_layers``) reproduces the capture
run's own prompt logprobs through the target's final norm and lm_head, and (2) the probed
layer's capture and the prompt logprobs match the native run where prefill attends densely
(the first chunk, at most ``index_topk`` tokens). The probed layer must not carry Engram,
whose residual the capture excludes.

With ``--image`` / ``--video`` (repeatable, local files) the prompt asks about that media.
``native`` hands vLLM one placeholder per item to expand itself; ``capture`` sends the
prompt AngelSpec's data pipeline expands, so ``compare`` also checks that both expansions
are the same tokens.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from angelspec.inference.engine.mooncake_hidden_states_connector import (
        MooncakeHiddenStatesConnector,
    )
except ImportError:  # pragma: no cover - angelspec missing
    MooncakeHiddenStatesConnector = object

PROMPT_TOKENS = 3000
FILE_SINK_ENV = "HY4_FLASH_PARITY_SINK"
MM_PROCESSOR_KWARGS = {"min_pixels": 4096, "max_pixels": 4194304}


class _HostBuffer:
    def __init__(self, size):
        import torch

        self.size = size
        self._tensor = torch.empty(size, dtype=torch.uint8)
        self.ptr = self._tensor.data_ptr()

    def copy_from_tensor(self, tensor, offset=0):
        import torch

        data = tensor.contiguous().view(-1).view(torch.uint8).cpu()
        self._tensor[offset : offset + data.numel()].copy_(data)
        return data.numel()


class _FileStore:
    """Stands in for DraftMooncakeStore: stages exactly like it, then saves to disk."""

    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def put_kv_cache_chunks(
        self,
        key,
        kv_cache,
        slot_mapping,
        input_ids,
        hidden_size,
        num_training_layers,
        chunk_tokens,
    ):
        import torch
        from safetensors.torch import save_file

        from angelspec.transfer.mooncake.draft_store import (
            HIDDEN_STATES_STORAGE_DTYPE,
            _stage_kv_cache_chunks_into_buffer,
        )

        n = input_ids.numel()
        width = (num_training_layers + 1) * hidden_size
        buf = _HostBuffer(n * 8 + n * width * 2)
        ptrs, sizes = _stage_kv_cache_chunks_into_buffer(
            buf, kv_cache, slot_mapping, input_ids, hidden_size, num_training_layers, chunk_tokens
        )
        torch.cuda.synchronize()

        def read(i, dtype, shape):
            off = ptrs[i] - buf.ptr
            return buf._tensor[off : off + sizes[i]].view(dtype).reshape(shape).clone()

        out = {"input_ids": read(0, torch.int64, (n,))}
        if num_training_layers:
            out["hidden_states"] = read(
                1, HIDDEN_STATES_STORAGE_DTYPE, (n, num_training_layers * hidden_size)
            )
        out["last_hidden_states"] = read(
            len(ptrs) - 1, HIDDEN_STATES_STORAGE_DTYPE, (n, hidden_size)
        )
        save_file(out, str(self.path / f"{key}.safetensors"))

    def flush(self):
        pass

    def pop_failed_put_keys(self):
        return set()

    def mark_put_failed(self, keys):
        raise RuntimeError(f"hidden-state capture failed for {sorted(keys)}")


class AngelSpecFileConnector(MooncakeHiddenStatesConnector):
    def _ensure_mooncake_store(self) -> bool:
        if self._get_tp_rank() != 0:
            return False
        if self._mooncake_store is None:
            self._mooncake_store = _FileStore(Path(os.environ[FILE_SINK_ENV]))
        return True


def _install_probe(worker, layer):
    model = worker.model_runner.model
    module = getattr(model, "language_model", model).model.layers[layer].hc_attn_layer.hc_pre
    rows = []
    module.register_forward_hook(lambda _m, _i, out: rows.append(out[0].detach().float().cpu()))
    worker._angelspec_probe_rows = rows


def _dump_probe(worker, path):
    import torch

    if worker.rank == 0:
        torch.save(torch.cat(worker._angelspec_probe_rows), path)


def _text_prompt(tokenizer) -> str:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    ids = tokenizer(readme.read_text())["input_ids"][:PROMPT_TOKENS]
    body = tokenizer.decode(ids)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{body}\n\nSummarize the text above."}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _media_prompt(args, tokenizer, capture: bool):
    import re

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    from angelspec.data.dataset import _hy4_image_newlines_to_placeholder
    from angelspec.data.parse import create_parser
    from angelspec.data.template import TEMPLATE_REGISTRY
    from angelspec.data.utils import (
        HY4_IMAGE_END,
        HY4_IMAGE_PAD,
        HY4_IMAGE_START,
        HY4_VIDEO_END,
        HY4_VIDEO_PAD,
        HY4_VIDEO_START,
        Hy4VideoLayout,
        render_hy4_media,
    )
    from angelspec.inference.engine.vllm_engine import VllmEngine

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("hy4-think"))
    content = [{"type": "image", "image": path} for path in args.image]
    content += [{"type": "video", "video": path} for path in args.video]
    content.append({"type": "text", "text": "Describe each image and video in detail."})
    text, media = render_hy4_media(
        parser,
        [{"role": "user", "content": content}],
        processor.image_processor,
        MM_PROCESSOR_KWARGS,
        Hy4VideoLayout(args.model, MM_PROCESSOR_KWARGS) if args.video else None,
        add_generation_prompt=True,
    )
    if capture:
        ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"])
        ids = _hy4_image_newlines_to_placeholder(ids, tokenizer)[0].tolist()
    else:
        # Videos first: their frames are image spans too.
        for start, pad, end in (
            (HY4_VIDEO_START, HY4_VIDEO_PAD, HY4_VIDEO_END),
            (HY4_IMAGE_START, HY4_IMAGE_PAD, HY4_IMAGE_END),
        ):
            span = re.escape(start) + ".*?" + re.escape(end)
            text = re.sub(span, lambda _, s=start + pad + end: s, text, flags=re.S)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    data = {}
    if media["images"]:
        data["image"] = [Image.open(path).convert("RGB") for path in media["images"]]
    if media["videos"]:
        data["video"] = [VllmEngine._load_video_file(item) for item in media["videos"]]
    return {"prompt_token_ids": ids, "multi_modal_data": data}


def _run_engine(args, capture: bool):
    if not capture:
        os.environ["VLLM_PLUGINS"] = ""
        # collective_rpc pickles the probe functions to the workers.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    text = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="mp",
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": len(args.image), "video": len(args.video)},
        mm_processor_kwargs=MM_PROCESSOR_KWARGS if args.image or args.video else None,
        engram_config={"cpu_offload": args.engram_cpu_offload},
    )
    if capture:
        storage = out / "hidden_states"
        storage.mkdir(parents=True, exist_ok=True)
        os.environ[FILE_SINK_ENV] = str(storage)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parent), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
        kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [
                        args.probe_layer,
                        text["num_hidden_layers"],
                    ]
                }
            },
        }
        kwargs["kv_transfer_config"] = {
            "kv_connector": "AngelSpecFileConnector",
            "kv_connector_module_path": "hy_v4_flash_capture_parity",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cuda",
        }

    llm = LLM(**kwargs)
    if not capture:
        llm.collective_rpc(_install_probe, args=(args.probe_layer,))
    media = args.image or args.video
    prompt = _media_prompt(args, tokenizer, capture) if media else _text_prompt(tokenizer)
    if capture and media:
        from angelspec.inference.engine.vllm_runtime_patch import PRETOKENIZED_MM_ENV

        # Only this request is pre-expanded; engine start-up profiles with dummy prompts.
        os.environ[PRETOKENIZED_MM_ENV] = "1"
    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1),
        use_tqdm=False,
    )
    if not capture:
        llm.collective_rpc(_dump_probe, args=(str(out / "native_probe.pt"),))
    records = []
    for result in outputs:
        ids = list(result.prompt_token_ids)
        actual = [None] + [result.prompt_logprobs[i][ids[i]].logprob for i in range(1, len(ids))]
        record = {"prompt_token_ids": ids, "actual_logprobs": actual}
        if capture:
            key = (result.kv_transfer_params or {})["mooncake_key"]
            record["hidden_states_path"] = str(out / "hidden_states" / f"{key}.safetensors")
        records.append(record)
    name = "capture.json" if capture else "native.json"
    (out / name).write_text(json.dumps(records))
    print(f"WROTE {name}: {[len(r['prompt_token_ids']) for r in records]} tokens", flush=True)


def _target_tensor(model_dir: Path, key: str):
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    with safe_open(str(model_dir / index["weight_map"][key]), framework="pt") as f:
        return f.get_tensor(key)


def _compare(args):
    import torch
    from safetensors.torch import load_file
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4RMSNorm

    out = Path(args.out)
    (ref,) = json.loads((out / "native.json").read_text())
    (cap,) = json.loads((out / "capture.json").read_text())
    probe = torch.load(out / "native_probe.pt")
    model_dir = Path(args.model)
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    if args.probe_layer + 1 in (text.get("engram_layer_ids") or ()):
        raise ValueError(f"layer {args.probe_layer} carries Engram; probe another layer")
    norm = HYV4RMSNorm(text["hidden_size"], eps=text["rms_norm_eps"])
    norm.weight.data.copy_(_target_tensor(model_dir, "model.norm.weight"))
    lm_head = _target_tensor(model_dir, "lm_head.weight").float()

    assert ref["prompt_token_ids"] == cap["prompt_token_ids"], "prompts differ"
    ids = torch.tensor(ref["prompt_token_ids"])
    payload = load_file(cap["hidden_states_path"])
    assert payload["input_ids"].tolist() == ref["prompt_token_ids"], "captured token ids differ"
    probed, final = payload["hidden_states"].float(), payload["last_hidden_states"]
    assert torch.isfinite(probed).all() and torch.isfinite(final).all(), "non-finite capture"
    assert probe.shape == probed.shape, (tuple(probe.shape), tuple(probed.shape))

    logprobs = torch.log_softmax(norm(final.to(torch.bfloat16)).float() @ lm_head.T, dim=-1)
    captured_lp = logprobs[:-1].gather(1, ids[1:, None]).squeeze(1)
    ref_lp = torch.tensor(ref["actual_logprobs"][1:])
    cap_lp = torch.tensor(cap["actual_logprobs"][1:])
    self_diff = (captured_lp - cap_lp).abs()
    dense = min(text["index_topk"], args.max_num_batched_tokens) - 1
    forward_diff = (ref_lp - cap_lp).abs()[:dense]
    probe_diff = (probe[:dense] - probed[:dense]).abs()
    probe_rel = (probe_diff.norm() / probe[:dense].norm()).item()

    print(
        f"tokens={len(ids)} probe layer {args.probe_layer} hidden={tuple(probed.shape)}\n"
        f"  captured final vs capture-run logprob |diff| mean={self_diff.mean():.4f} "
        f"p99={torch.quantile(self_diff, 0.99):.4f} max={self_diff.max():.4f}\n"
        f"  capture vs native logprob, dense prefix ({dense} tokens) |diff| "
        f"mean={forward_diff.mean():.4f} max={forward_diff.max():.4f}\n"
        f"  probed capture vs native pre-gate, dense prefix: max |diff|={probe_diff.max():.4g} "
        f"rel={probe_rel:.3g}",
        flush=True,
    )
    ok = self_diff.mean() < args.tol and forward_diff.mean() < args.tol and probe_rel < 1e-2
    print("PARITY_OK" if ok else "PARITY_FAILED", flush=True)
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["native", "capture", "compare"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--tol", type=float, default=0.05, help="max mean |logprob diff|")
    parser.add_argument("--probe-layer", type=int, default=25, help="a layer without Engram")
    parser.add_argument("--image", action="append", default=[], help="an image to ask about")
    parser.add_argument("--video", action="append", default=[], help="a video to ask about")
    parser.add_argument(
        "--engram-cpu-offload",
        action="store_true",
        help="keep the Engram tables in pinned host memory instead of on the GPUs",
    )
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.mode == "compare":
        _compare(args)
    else:
        _run_engine(args, capture=args.mode == "capture")


if __name__ == "__main__":
    main()
