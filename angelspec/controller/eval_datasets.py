"""Vendored multi-dataset eval prompts for online spec-decode acceptance eval.

Two roles:
  * ``build_<name>_prompt`` — pure per-dataset prompt formatters (no network).
  * ``load_dataset_prompts`` — train-time reader over the vendored jsonl under
    ``angelspec/data/eval_prompts/<name>.jsonl``. No HF / network / ``datasets``
    dependency, so the whole repo migrates with the eval sets baked in.

mtbench is multi-turn upstream; we keep only the first user turn as a single-turn
acceptance-rate proxy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATASETS = ("gsm8k", "humaneval", "mbpp", "math500", "mtbench", "livecodebench")

# HF identifiers — used ONLY by the offline dump tool, never at train time.
DATASET_SPECS = {
    "gsm8k": {"repo": "madrylab/gsm8k-platinum", "split": "test"},
    "humaneval": {"repo": "openai/openai_humaneval", "split": "test"},
    "mbpp": {"repo": "google-research-datasets/mbpp", "config": "sanitized", "split": "test"},
    "math500": {"repo": "HuggingFaceH4/MATH-500", "split": "test"},
    "mtbench": {"repo": "HuggingFaceH4/mt_bench_prompts", "split": "train"},
    "livecodebench": {"repo": "livecodebench/code_generation_lite", "split": "test"},
}

# angelspec/controller/eval_datasets.py -> angelspec/data/eval_prompts
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_prompts"


# ── per-dataset formatters (pure; fed one HF example dict) ────────────────────


def build_gsm8k_prompt(example: dict) -> str:
    return str(example["question"]).strip()


def build_math500_prompt(example: dict) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer inside \\boxed{}.\n\n"
        f"Problem: {str(example['problem']).strip()}"
    )


def build_humaneval_prompt(example: dict) -> str:
    return (
        "Complete the following Python function. "
        "Return the full function implementation in a single ```python ... ``` code block. "
        "Do not include any explanation outside the code block.\n\n"
        f"```python\n{example['prompt']}```"
    )


def build_mbpp_prompt(example: dict) -> str:
    description = (example.get("prompt") or example.get("text") or "").strip()
    tests = "\n".join(example.get("test_list") or [])
    return (
        "Write a Python function that satisfies the description below. "
        "Your function must pass the provided assert tests. Return the "
        "full function implementation in a single ```python ... ``` "
        "code block. Do not include any explanation outside the code "
        f"block.\n\nDescription:\n{description}\n\nTests:\n```python\n{tests}\n```"
    )


def build_mtbench_prompt(example: dict) -> str:
    """mt_bench_prompts stores a list of turns; keep only the first user turn."""
    turns = example.get("prompt") or example.get("turns") or []
    if isinstance(turns, str):
        turns = [turns]
    turns = [str(t).strip() for t in turns if str(t).strip()]
    return turns[0] if turns else ""


def _stringify_value(value) -> str:
    """list/dict/json-string field -> text suitable for a prompt (livecodebench)."""
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ""
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except Exception:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _first_non_empty(example: dict, keys: list[str]) -> str:
    for key in keys:
        if key in example:
            text = _stringify_value(example[key])
            if text:
                return text
    return ""


def build_livecodebench_prompt(example: dict) -> str:
    title = _first_non_empty(example, ["question_title", "title", "name"])
    question = _first_non_empty(
        example,
        ["question_content", "content", "prompt", "description", "problem_statement"],
    )
    starter_code = _first_non_empty(example, ["starter_code", "starter", "code_stub"])
    public_tests = _first_non_empty(example, ["public_test_cases", "examples", "sample_tests"])
    metadata = _first_non_empty(example, ["metadata"])
    difficulty = _first_non_empty(example, ["difficulty"])
    platform = _first_non_empty(example, ["platform"])

    parts = [
        "Solve the following competitive programming problem in Python.",
        "Return the full solution in a single ```python ... ``` code block.",
        "Do not include any explanation outside the code block.",
    ]
    context_lines = []
    if title:
        context_lines.append(f"Title: {title}")
    if platform:
        context_lines.append(f"Platform: {platform}")
    if difficulty:
        context_lines.append(f"Difficulty: {difficulty}")
    if context_lines:
        parts.append("\n".join(context_lines))
    if question:
        parts.append(f"Problem:\n{question}")
    if starter_code:
        parts.append(f"Starter code:\n```python\n{starter_code}\n```")
    if public_tests:
        parts.append(f"Public test cases / examples:\n```text\n{public_tests}\n```")
    if metadata:
        parts.append(f"Metadata:\n```json\n{metadata}\n```")
    return "\n\n".join(parts)


PROMPT_BUILDERS = {
    "gsm8k": build_gsm8k_prompt,
    "math500": build_math500_prompt,
    "humaneval": build_humaneval_prompt,
    "mbpp": build_mbpp_prompt,
    "mtbench": build_mtbench_prompt,
    "livecodebench": build_livecodebench_prompt,
}


# ── train-time reader (local jsonl only) ──────────────────────────────────────


def load_dataset_prompts(
    name: str,
    *,
    sample_size: int = 1000,
    seed: int = 42,
    limit: int | None = None,
    prompts_dir: str | None = None,
) -> list[str]:
    """Read vendored ``<name>.jsonl`` (each line ``{"prompt": ...}``), shuffle with
    ``seed``, cap at ``sample_size``, then truncate to ``limit``. No network."""
    base = Path(prompts_dir) if prompts_dir else _DEFAULT_PROMPTS_DIR
    path = base / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"eval prompts for {name!r} not found at {path}; run "
            f"tools/dump_eval_prompts.py to generate the vendored jsonl."
        )
    prompts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompt = json.loads(line).get("prompt")
            if prompt:
                prompts.append(prompt)

    rng = random.Random(seed)
    rng.shuffle(prompts)
    if sample_size:
        prompts = prompts[: min(sample_size, len(prompts))]
    if limit is not None:
        prompts = prompts[: min(limit, len(prompts))]
    return prompts
