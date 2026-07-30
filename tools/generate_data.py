"""
Re-generate a dataset from a target model's output distribution,
which better aligns the draft model with the target model.

Usage:
1. Set up one or more SGLang servers for the target model:

python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 128 \
    --tp 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

2. Regenerate the dataset:

python tools/generate_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

CONNECTION_ERROR_KEYWORDS = [
    "ConnectionError",
    "Timeout",
    "timed out",
    "ECONNREFUSED",
    "Connection refused",
    "RemoteDisconnected",
    "SSLError",
    "ReadTimeout",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)",
    )
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Concurrent requests per server (default: 64)",
    )
    parser.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address(es) for sglang model server",
    )
    parser.add_argument(
        "--is-reasoning-model",
        action="store_true",
        help="Whether the model is a reasoning model",
    )
    parser.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of successful samples to generate (default: all)",
    )
    parser.add_argument(
        "--sample-max-retries",
        type=int,
        default=5,
        help="Maximum retries per sample (default: 5)",
    )
    return parser.parse_args()


def get_random_reasoning_effort() -> str:
    """Weighted random: LOW(4), MEDIUM(4), HIGH(2)."""
    return random.choices(["low", "medium", "high"], weights=[4, 4, 2], k=1)[0]


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens
    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def is_connection_error(error_msg: str) -> bool:
    if not error_msg:
        return False
    return any(kw in error_msg for kw in CONNECTION_ERROR_KEYWORDS)


def call_sglang(
    args, server_address: str, data: Dict[str, Any], max_tokens=None
) -> Dict[str, Any]:
    """Regenerate assistant turns for a single conversation via chat completions."""
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    messages = data["conversations"]
    regenerated_messages = []
    total_output_tokens = 0

    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)
            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)
            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data

            total_output_tokens += resp.usage.completion_tokens
            resp_msg = {"role": "assistant", "content": resp.choices[0].message.content}
            if args.is_reasoning_model:
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or (
                    resp.choices[0].message.model_extra or {}
                ).get("reasoning")
                resp_msg["thinking"] = reasoning
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data

    data["output_tokens"] = total_output_tokens
    data["input_tokens"] = resp.usage.prompt_tokens
    data["context_length"] = data["input_tokens"] + total_output_tokens
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


def health_check_server(args, server_address: str) -> bool:
    dummy = {"conversations": [{"role": "user", "content": "Hello, how are you?"}]}
    try:
        result = call_sglang(args, server_address, dummy, max_tokens=1)
    except Exception:
        return False
    if result is None:
        return False
    if result.get("status") == "error":
        return not is_connection_error(str(result.get("error", "")))
    return True


def wait_for_healthy_servers(args) -> List[str]:
    while True:
        valid = []
        for addr in args.server_address:
            if health_check_server(args, addr):
                valid.append(addr)
            else:
                print(f"Server {addr} is not available")
        if valid:
            print(f"Using {len(valid)} server(s): {valid}")
            return valid
        print("No valid server available, waiting...")
        time.sleep(5)


def load_checkpoint(output_file_path: str):
    """Load previously processed IDs and metrics from an existing output file."""
    processed_ids = set()
    success_count = 0
    token_sum, token_min, token_max = 0, None, 0

    if not os.path.exists(output_file_path):
        return processed_ids, success_count, token_sum, token_min, token_max

    print(f"Resuming from existing output: {output_file_path}")
    with open(output_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_id = record.get("data_id")
            if isinstance(data_id, int):
                processed_ids.add(data_id)
            ctx_len = record.get("context_length")
            if ctx_len is not None and record.get("conversations") is not None:
                token_sum += ctx_len
                token_min = ctx_len if token_min is None else min(token_min, ctx_len)
                token_max = max(token_max, ctx_len)
                success_count += 1

    print(f"Found {len(processed_ids)} previously processed samples.")
    return processed_ids, success_count, token_sum, token_min, token_max


def main():
    args = parse_arguments()

    if not (0.0 <= args.temperature <= 2.0):
        raise ValueError("Temperature must be between 0.0 and 2.0")
    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Servers: {args.server_address}")
    print(f"  Input: {args.input_file_path}")
    print(f"  Output: {args.output_file_path}")

    valid_servers = wait_for_healthy_servers(args)
    processed_ids, success_count, token_sum, token_min, token_max = load_checkpoint(
        args.output_file_path
    )

    if args.num_samples is not None and success_count >= args.num_samples:
        print(f"Already have {success_count}/{args.num_samples} samples. Nothing to do.")
        return

    remaining_data = []
    with open(args.input_file_path, "r") as f:
        for data_id, line in enumerate(f):
            if data_id in processed_ids:
                continue
            if (
                args.num_samples is not None
                and (success_count + len(remaining_data)) >= args.num_samples
            ):
                break
            data = json.loads(line.strip())
            data["data_id"] = data_id
            remaining_data.append(data)

    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    error_count = 0
    retry_counts: Dict[int, int] = {}
    output_mode = "a" if os.path.exists(args.output_file_path) else "w"

    def handle_result(future, server_addr, output_fh, error_fh):
        nonlocal valid_servers, success_count, error_count, token_sum, token_min, token_max

        regen_data = future.result()
        if regen_data.get("status") == "error":
            if is_connection_error(str(regen_data.get("error", ""))):
                if server_addr in valid_servers:
                    print(f"Removing unhealthy server: {server_addr}")
                    valid_servers.remove(server_addr)

            data_id = regen_data.get("data_id")
            if isinstance(data_id, int):
                retry_counts[data_id] = retry_counts.get(data_id, 0) + 1
                if retry_counts[data_id] < args.sample_max_retries:
                    return regen_data

            error_fh.write(json.dumps(regen_data, ensure_ascii=False) + "\n")
            error_fh.flush()
            error_count += 1
            return None

        ctx_len = regen_data.get("context_length", 0)
        token_sum += ctx_len
        token_min = ctx_len if token_min is None else min(token_min, ctx_len)
        token_max = max(token_max, ctx_len)
        output_fh.write(json.dumps(regen_data, ensure_ascii=False) + "\n")
        output_fh.flush()
        success_count += 1
        return None

    with open(error_file_path, "w") as error_fh:
        while remaining_data:
            retry_records = []
            server_count = max(1, len(valid_servers))
            pending: Dict = {}
            per_server: Dict[str, int] = {addr: 0 for addr in valid_servers}

            with open(args.output_file_path, output_mode) as output_fh:
                with tqdm(total=len(remaining_data), desc="Processing") as pbar:
                    with ThreadPoolExecutor(
                        max_workers=args.concurrency * server_count
                    ) as executor:
                        for i, data in enumerate(remaining_data):
                            if not valid_servers:
                                valid_servers = wait_for_healthy_servers(args)
                                per_server = {addr: 0 for addr in valid_servers}

                            server_addr = valid_servers[i % len(valid_servers)]

                            while per_server.get(server_addr, 0) >= args.concurrency:
                                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                                for f in done:
                                    addr = pending.pop(f)
                                    per_server[addr] -= 1
                                    retry = handle_result(f, addr, output_fh, error_fh)
                                    if retry is not None:
                                        retry_records.append(retry)
                                    pbar.update(1)

                            future = executor.submit(
                                lambda a, d: call_sglang(args, a, d), server_addr, data
                            )
                            pending[future] = server_addr
                            per_server[server_addr] = per_server.get(server_addr, 0) + 1

                        for f in as_completed(pending):
                            addr = pending[f]
                            retry = handle_result(f, addr, output_fh, error_fh)
                            if retry is not None:
                                retry_records.append(retry)
                            pbar.update(1)

            if not retry_records:
                break
            remaining_data = []
            for record in retry_records:
                record.pop("status", None)
                record.pop("error", None)
                remaining_data.append(record)
            output_mode = "a"

    print("\nProcessing completed!")
    if success_count > 0:
        avg = token_sum / success_count
        print(
            f"Context length stats ({success_count} samples): min={token_min}, max={token_max}, avg={avg:.2f}"
        )
    else:
        print("No successful samples.")
    print(f"Total: {success_count} succeeded, {error_count} failed.")


if __name__ == "__main__":
    main()
