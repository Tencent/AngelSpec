#!/usr/bin/env python3
"""Drive benchmark datasets against a running vLLM (spec-decode) server.

This does NOT score correctness. It only fires concurrent chat-completion
requests so the *server* prints its speculative-decoding acceptance-rate
statistics. Point it at whichever port your `vllm serve` is listening on.

Data loading
------------
This machine cannot reach huggingface.co directly, and the new huggingface_hub
(1.17) `snapshot_download` chokes on the hf-mirror.com -> huggingface.co -> xet
redirect chain. So instead of `datasets.load_dataset(name)`, each dataset here
records its HF repo + the exact file glob + reader, and we read those files
through the mirror via fsspec (`hf://...` with endpoint=hf-mirror), which DOES
follow the redirects. Run with the OA proxy + mirror exported:

    export http_proxy=http://your-proxy:port
    export https_proxy=http://your-proxy:port
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN=<token>            # optional for public datasets

Usage:
    python tools/eval_accept_rate.py --dataset gsm8k --port 8021
    python tools/eval_accept_rate.py --dataset all   --port 10086
    python tools/eval_accept_rate.py --dataset math500 --limit 10 --concurrency 8

Available datasets:
    arc_challenge gsm8k hellaswag humaneval math500 mbpp mmlu mmlu_pro triviaqa
"""

import argparse
import asyncio
import os
import string
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# Mirror must be set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
SEED = 42


# ---------------------------------------------------------------------------
# Per-dataset prompt builders (input: one row as a dict)
# ---------------------------------------------------------------------------
def _arc_challenge(ex: dict) -> str:
    choices = "\n".join(
        f"{lb}. {t}" for lb, t in zip(ex["choices"]["label"], ex["choices"]["text"])
    )
    return (
        f"Question: {ex['question']}\n\nChoices:\n{choices}\n\n"
        "Please select the correct answer and explain your reasoning."
    )


def _gsm8k(ex: dict) -> str:
    return ex["question"]


def _hellaswag(ex: dict) -> str:
    endings = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(ex["endings"]))
    return (
        f"Context: {ex['ctx']}\n\nWhich ending is most plausible?\n{endings}\n\n"
        "Select the number of the best ending and explain why."
    )


def _humaneval(ex: dict) -> str:
    return (
        "Complete the following Python function. "
        "Return the full function implementation in a single ```python ... ``` code block. "
        "Do not include any explanation outside the code block.\n\n"
        f"```python\n{ex['prompt']}```"
    )


def _math500(ex: dict) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer inside \\boxed{}.\n\n"
        f"Problem: {ex['problem'].strip()}"
    )


def _mbpp(ex: dict) -> str:
    # nested/array columns come back as numpy arrays from pandas, so avoid
    # `or` (ambiguous truth value) and index/convert explicitly.
    description = str(ex.get("prompt") or ex.get("text") or "").strip()
    tests = ex.get("test_list")
    tests = list(tests) if tests is not None else []
    tests_block = "\n".join(str(t) for t in tests)
    return (
        "Write a Python function that satisfies the description below. "
        "Your function must pass the provided assert tests. "
        "Return the full function implementation in a single ```python ... ``` code block. "
        "Do not include any explanation outside the code block.\n\n"
        f"Description:\n{description}\n\n"
        f"Tests:\n```python\n{tests_block}\n```"
    )


_MMLU_LABELS = ["A", "B", "C", "D"]


def _mmlu(ex: dict) -> str:
    subject = ex.get("subject", "")
    lines = [
        (
            f"The following is a multiple choice question about {subject.replace('_', ' ')}."
            if subject
            else "The following is a multiple choice question."
        ),
        "",
        f"Question: {ex['question'].strip()}",
    ]
    lines += [f"{lb}. {c}" for lb, c in zip(_MMLU_LABELS, ex["choices"])]
    lines += [
        "",
        "Please reason step by step, and put your final answer (a single letter A, B, C, or D) "
        "on the last line in the form: Answer: <letter>.",
    ]
    return "\n".join(lines)


_MMLU_PRO_LABELS = list(string.ascii_uppercase)  # up to 10 options A-J


def _mmlu_pro(ex: dict) -> str:
    category = ex.get("category", "")
    options = ex["options"]
    labels = _MMLU_PRO_LABELS[: len(options)]
    lines = [
        (
            f"The following is a multiple choice question about {category}."
            if category
            else "The following is a multiple choice question."
        ),
        "",
        f"Question: {ex['question'].strip()}",
    ]
    lines += [f"{lb}. {o}" for lb, o in zip(labels, options)]
    lines += [
        "",
        "Please reason step by step, and put your final answer "
        f"(a single letter from {labels[0]} to {labels[-1]}) "
        "on the last line in the form: Answer: <letter>.",
    ]
    return "\n".join(lines)


def _triviaqa(ex: dict) -> str:
    return f"Answer the following question concisely:\n\n{ex['question']}"


# ---------------------------------------------------------------------------
# Dataset registry — repo + exact file path on the Hub + reader
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    repo: str  # HF dataset repo id
    path: str  # file path within the repo (relative), e.g. "main/test-00000-of-00001.parquet"
    fmt: str  # "parquet" | "jsonl" | "csv"
    build_prompt: Callable[[dict], str]
    max_tokens: int = 1024
    sample_size: Optional[int] = 1000  # cap; None = whole split
    shuffle: bool = False  # shuffle(seed=SEED) before truncating
    read_kwargs: dict = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "arc_challenge": DatasetSpec(
        "allenai/ai2_arc",
        "ARC-Challenge/test-00000-of-00001.parquet",
        "parquet",
        _arc_challenge,
        max_tokens=512,
    ),
    # gpqa removed: Idavidrein/gpqa is a gated repo (token lacks access); the
    # open mirrors have a different schema. Re-add with a working repo if needed.
    "gsm8k": DatasetSpec(
        "madrylab/gsm8k-platinum",
        "main/test-00000-of-00001.parquet",
        "parquet",
        _gsm8k,
    ),
    "hellaswag": DatasetSpec(
        "Rowan/hellaswag",
        "data/validation-00000-of-00001.parquet",
        "parquet",
        _hellaswag,
        max_tokens=256,
        shuffle=True,
    ),
    "humaneval": DatasetSpec(
        "openai/openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet",
        "parquet",
        _humaneval,
        shuffle=True,
    ),
    "math500": DatasetSpec(
        "HuggingFaceH4/MATH-500",
        "test.jsonl",
        "jsonl",
        _math500,
        max_tokens=2048,
        shuffle=True,
    ),
    "mbpp": DatasetSpec(
        "google-research-datasets/mbpp",
        "sanitized/test-00000-of-00001.parquet",
        "parquet",
        _mbpp,
        shuffle=True,
    ),
    "mmlu": DatasetSpec(
        "cais/mmlu",
        "all/test-00000-of-00001.parquet",
        "parquet",
        _mmlu,
        shuffle=True,
    ),
    "mmlu_pro": DatasetSpec(
        "TIGER-Lab/MMLU-Pro",
        "data/test-00000-of-00001.parquet",
        "parquet",
        _mmlu_pro,
        shuffle=True,
    ),
    "triviaqa": DatasetSpec(
        "mandarjoshi/trivia_qa",
        "rc.nocontext/validation-00000-of-00001.parquet",
        "parquet",
        _triviaqa,
        max_tokens=256,
        shuffle=True,
    ),
}


# ---------------------------------------------------------------------------
# Data loading via the mirror (fsspec hf:// follows the redirect chain)
# ---------------------------------------------------------------------------
def _read_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    # hf_hub_download for a SINGLE file goes through the same LFS path that
    # `snapshot_download` does and hits the xet redirect bug; so read straight
    # off the mirror's resolve URL with fsspec/pandas, which follows redirects.
    storage = {"endpoint": HF_ENDPOINT, "token": HF_TOKEN}
    uri = f"hf://datasets/{spec.repo}/{spec.path}"
    if spec.fmt == "parquet":
        return pd.read_parquet(uri, storage_options=storage, **spec.read_kwargs)
    if spec.fmt == "jsonl":
        return pd.read_json(uri, lines=True, storage_options=storage, **spec.read_kwargs)
    if spec.fmt == "csv":
        return pd.read_csv(uri, storage_options=storage, **spec.read_kwargs)
    raise ValueError(f"unknown fmt {spec.fmt}")


def _build_prompts(spec: DatasetSpec, limit: Optional[int]) -> list[str]:
    print(f"Loading hf://datasets/{spec.repo}/{spec.path} via {HF_ENDPOINT} ...")
    df = _read_dataframe(spec)
    print(f"Raw rows: {len(df)}")

    n = spec.sample_size
    if n is not None:
        if spec.shuffle:
            df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        df = df.iloc[: min(n, len(df))]
        print(f"After sampling (size={n}, shuffle={spec.shuffle}): {len(df)}")

    # to_dict("records") gives plain dicts; nested cols (choices/endings/options)
    # come back as numpy arrays/lists, which the builders handle.
    rows = df.to_dict("records")
    prompts = [spec.build_prompt(r) for r in rows]
    if limit is not None:
        prompts = prompts[:limit]
        print(f"After --limit: {len(prompts)}")
    return prompts


# ---------------------------------------------------------------------------
# Request driver
# ---------------------------------------------------------------------------
async def _get_model_name(client: AsyncOpenAI) -> str:
    models = await client.models.list()
    return models.data[0].id


async def _send_one(
    client, model, idx, prompt, sem, max_tokens, temperature, top_p=1.0, top_k=-1, seed=None
):
    async with sem:
        try:
            extra_body = {} if top_k in (-1, None) else {"top_k": top_k}
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                extra_body=extra_body,
            )
            content = resp.choices[0].message.content or ""
            print(f"[{idx}] OK  ({len(content)} chars)")
            return idx, content
        except Exception as e:  # noqa: BLE001 — best-effort load driver
            print(f"[{idx}] ERR {type(e).__name__}: {e}")
            return idx, None


def _build_prompts_from_jsonl(path: str, prompt_key: str, limit: Optional[int]) -> list[str]:
    """Extract the first user-turn content from each conversation in a jsonl.

    Used to drive the acceptance-rate benchmark on the *training* eval split
    (same distribution as training) so the result is comparable to the trainer's
    eval acc_0 — isolating framework correctness from train/serve domain shift.
    """
    import json as _json

    prompts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            convs = row.get(prompt_key, [])
            user_msg = next((m.get("content", "") for m in convs if m.get("role") == "user"), None)
            if user_msg:
                prompts.append(user_msg)
    print(f"Loaded {len(prompts)} prompts from {path} (key={prompt_key})")
    if limit is not None:
        prompts = prompts[:limit]
        print(f"After --limit: {len(prompts)}")
    return prompts


async def _run_prompts(client, model, name, prompts, args, max_tokens=1024):
    print(f"\n========== {name} ==========")
    print(f"Total: {len(prompts)} samples")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    tasks = [
        _send_one(
            client,
            model,
            i,
            p,
            sem,
            max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.seed,
        )
        for i, p in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    dt = time.time() - t0
    ok = sum(1 for _, c in results if c is not None)
    print(
        f"\n[{name}] Done: {ok}/{len(results)}  Time: {dt:.1f}s  Throughput: {len(results) / dt:.2f} req/s"
    )


async def _run_dataset(client, model, name, spec, args):
    print(f"\n========== {name} ==========")
    prompts = _build_prompts(spec, args.limit)
    print(f"Total: {len(prompts)} samples")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    tasks = [
        _send_one(
            client,
            model,
            i,
            p,
            sem,
            spec.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.seed,
        )
        for i, p in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    dt = time.time() - t0

    ok = sum(1 for _, c in results if c is not None)
    print(
        f"\n[{name}] Done: {ok}/{len(results)}  Time: {dt:.1f}s  Throughput: {len(results) / dt:.2f} req/s"
    )


async def main_async(args):
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key)
    model = args.model or await _get_model_name(client)
    print(f"Server: {base_url}  Model: {model}")

    if args.jsonl:
        prompts = _build_prompts_from_jsonl(args.jsonl, args.prompt_key, args.limit)
        await _run_prompts(client, model, os.path.basename(args.jsonl), prompts, args)
        return

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        await _run_dataset(client, model, name, DATASETS[name], args)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        default=None,
        choices=[*DATASETS.keys(), "all"],
        help="Dataset to run, or 'all' to run every dataset in sequence.",
    )
    p.add_argument(
        "--jsonl",
        default=None,
        help="Path to a conversations jsonl (e.g. the training eval split). "
        "Drives the benchmark on training-distribution prompts instead of a "
        "built-in dataset; mutually exclusive with --dataset.",
    )
    p.add_argument(
        "--prompt-key",
        default="conversations",
        help="Top-level key holding the message list in --jsonl rows (default: conversations).",
    )
    p.add_argument("--port", type=int, default=8021, help="vLLM server port (default 8021).")
    p.add_argument("--base-url", default=None, help="Full base URL; overrides --port.")
    p.add_argument("--api-key", default="EMPTY", help="vLLM ignores this by default.")
    p.add_argument("--model", default=None, help="Model id; default = first model on server.")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1, help="vLLM top_k (passed via extra_body).")
    p.add_argument(
        "--seed", type=int, default=None, help="Sampling seed for reproducible rs runs."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate to N prompts after sampling (for quick debug runs).",
    )
    args = p.parse_args()
    if not args.jsonl and not args.dataset:
        p.error("one of --dataset or --jsonl is required")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
