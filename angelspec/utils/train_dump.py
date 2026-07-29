# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Debug utilities for saving and comparing training data.

Used to verify correctness between train_entry.py and train/train_eagle3.py.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from angelspec.utils.logging import logger


def extract_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract gradients from model parameters.

    Args:
        model: The model to extract gradients from.

    Returns:
        Dict mapping parameter names to gradient tensors (cloned and detached).
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone().cpu()
    return gradients


def extract_model_weights(model: nn.Module, filter_prefix: str = None) -> Dict[str, torch.Tensor]:
    """Extract model weights from parameters.

    Args:
        model: The model to extract weights from.
        filter_prefix: Optional prefix to filter parameter names (e.g., 'draft_model').

    Returns:
        Dict mapping parameter names to weight tensors (cloned and detached).
    """
    weights = {}
    for name, param in model.named_parameters():
        if filter_prefix is not None and not name.startswith(filter_prefix):
            continue
        weights[name] = param.detach().clone().cpu()
    return weights


def should_dump_step(step: int, max_dump_steps: int = 5) -> bool:
    """Check if current step should be dumped.

    Args:
        step: Current training step (1-indexed).
        max_dump_steps: Maximum number of steps to dump.

    Returns:
        True if step should be dumped, False otherwise.
    """
    return step <= max_dump_steps


def dump_eagle3_batch(
    args,
    *,
    step: int,
    batch_idx: int,
    batch: Dict[str, torch.Tensor],
    plosses: Optional[List[torch.Tensor]] = None,
    acces: Optional[List[torch.Tensor]] = None,
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    total_loss: Optional[torch.Tensor] = None,
    model_weights: Optional[Dict[str, torch.Tensor]] = None,
    hidden_states: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    last_hidden_states: Optional[torch.Tensor] = None,
):
    """Save Eagle3 training batch and metrics for debugging.

    Args:
        args: Arguments containing save_debug_train_data path template.
        step: Current optimizer step.
        batch_idx: Batch index within the step.
        batch: Dict of input tensors (input_ids, attention_mask, etc.).
        plosses: Optional list of per-position losses.
        acces: Optional list of per-position accuracies.
        gradients: Optional dict mapping param names to gradient tensors.
        total_loss: Optional total weighted loss scalar.
        model_weights: Optional dict mapping param names to weight tensors.
        hidden_states: Optional aux hidden states tensor (intermediate layers).
        target: Optional target logits tensor.
        last_hidden_states: Optional last hidden states tensor (final layer before lm_head).
    """
    path_template = getattr(args, "save_debug_train_data", None)
    if path_template is None:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    path = Path(path_template.format(step=step, rank=rank, batch_idx=batch_idx))
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "step": step,
        "batch_idx": batch_idx,
        "rank": rank,
        "batch": {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
    }

    if plosses is not None:
        data["plosses"] = [pl.detach().cpu() for pl in plosses]
    if acces is not None:
        data["acces"] = [ac.detach().cpu() for ac in acces]
    if gradients is not None:
        data["gradients"] = {k: v.cpu() for k, v in gradients.items()}
    if total_loss is not None:
        data["total_loss"] = total_loss.detach().cpu()
    if model_weights is not None:
        data["model_weights"] = {k: v.cpu() for k, v in model_weights.items()}
    if hidden_states is not None:
        data["hidden_states"] = hidden_states.detach().cpu()
    if target is not None:
        data["target"] = target.detach().cpu()
    if last_hidden_states is not None:
        data["last_hidden_states"] = last_hidden_states.detach().cpu()

    logger.info(f"Save Eagle3 debug batch to {path}")
    torch.save(data, path)


def compare_eagle3_outputs(
    path1: str,
    path2: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = False,
) -> Dict:
    """Compare two Eagle3 training dumps for correctness verification.

    Args:
        path1: Path to first dump file.
        path2: Path to second dump file.
        rtol: Relative tolerance for torch.allclose.
        atol: Absolute tolerance for torch.allclose.
        verbose: If True, print detailed comparison info.

    Returns:
        Dict with comparison results including match status and diffs.
    """
    data1 = torch.load(path1, weights_only=False)
    data2 = torch.load(path2, weights_only=False)

    results = {
        "match": True,
        "diffs": {},
        "summary": {},
    }

    def _compare_tensors(t1, t2, name: str):
        """Compare two tensors and record differences."""
        if t1.shape != t2.shape:
            results["match"] = False
            results["diffs"][name] = {"error": f"Shape mismatch: {t1.shape} vs {t2.shape}"}
            return

        if t1.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            match = torch.equal(t1, t2)
            if not match:
                diff_count = (t1 != t2).sum().item()
                results["match"] = False
                results["diffs"][name] = {"diff_count": diff_count, "total": t1.numel()}
        else:
            match = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
            if not match:
                diff = (t1.float() - t2.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                results["match"] = False
                results["diffs"][name] = {"max_diff": max_diff, "mean_diff": mean_diff}

        if verbose:
            status = "✓" if name not in results["diffs"] else "✗"
            print(f"  {status} {name}: shape={tuple(t1.shape)}, dtype={t1.dtype}")

    for key in ["input_ids", "attention_mask", "loss_mask", "target", "hidden_states"]:
        if key not in data1.get("batch", {}) or key not in data2.get("batch", {}):
            continue
        t1 = data1["batch"][key]
        t2 = data2["batch"][key]
        _compare_tensors(t1, t2, f"batch/{key}")

    for metric_key in ["plosses", "acces"]:
        if metric_key in data1 and metric_key in data2:
            if len(data1[metric_key]) != len(data2[metric_key]):
                results["match"] = False
                results["diffs"][metric_key] = {
                    "error": f"Length mismatch: {len(data1[metric_key])} vs {len(data2[metric_key])}"
                }
                continue
            for i, (m1, m2) in enumerate(zip(data1[metric_key], data2[metric_key])):
                _compare_tensors(m1, m2, f"{metric_key}[{i}]")

    if "total_loss" in data1 and "total_loss" in data2:
        _compare_tensors(data1["total_loss"], data2["total_loss"], "total_loss")

    if "gradients" in data1 and "gradients" in data2:
        grads1, grads2 = data1["gradients"], data2["gradients"]
        common_keys = set(grads1.keys()) & set(grads2.keys())
        only_in_1 = set(grads1.keys()) - set(grads2.keys())
        only_in_2 = set(grads2.keys()) - set(grads1.keys())

        results["summary"]["gradient_keys"] = {
            "common": len(common_keys),
            "only_in_file1": len(only_in_1),
            "only_in_file2": len(only_in_2),
        }

        if only_in_1 or only_in_2:
            results["match"] = False
            if only_in_1:
                results["diffs"]["gradients_only_in_file1"] = list(only_in_1)[:10]
            if only_in_2:
                results["diffs"]["gradients_only_in_file2"] = list(only_in_2)[:10]

        for key in sorted(common_keys):
            _compare_tensors(grads1[key], grads2[key], f"gradients/{key}")

    if "model_weights" in data1 and "model_weights" in data2:
        weights1, weights2 = data1["model_weights"], data2["model_weights"]
        common_keys = set(weights1.keys()) & set(weights2.keys())
        only_in_1 = set(weights1.keys()) - set(weights2.keys())
        only_in_2 = set(weights2.keys()) - set(weights1.keys())

        results["summary"]["weight_keys"] = {
            "common": len(common_keys),
            "only_in_file1": len(only_in_1),
            "only_in_file2": len(only_in_2),
        }

        if only_in_1 or only_in_2:
            results["match"] = False
            if only_in_1:
                results["diffs"]["weights_only_in_file1"] = list(only_in_1)[:10]
            if only_in_2:
                results["diffs"]["weights_only_in_file2"] = list(only_in_2)[:10]

        for key in sorted(common_keys):
            _compare_tensors(weights1[key], weights2[key], f"model_weights/{key}")

    return results


def compare_eagle3_batches(
    dir1: str,
    dir2: str,
    pattern: str = "*.pt",
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Dict:
    """Compare all matching Eagle3 dumps in two directories.

    Args:
        dir1: First directory path.
        dir2: Second directory path.
        pattern: Glob pattern for finding dump files.
        rtol: Relative tolerance for torch.allclose.
        atol: Absolute tolerance for torch.allclose.
        verbose: If True, print comparison progress.

    Returns:
        Dict with overall match status and per-file results.
    """
    dir1, dir2 = Path(dir1), Path(dir2)
    files1 = {f.name: f for f in dir1.glob(pattern)}
    files2 = {f.name: f for f in dir2.glob(pattern)}

    common_files = set(files1.keys()) & set(files2.keys())

    overall_results = {
        "match": True,
        "files_compared": len(common_files),
        "files_only_in_dir1": list(set(files1.keys()) - common_files),
        "files_only_in_dir2": list(set(files2.keys()) - common_files),
        "per_file_results": {},
    }

    for filename in sorted(common_files):
        if verbose:
            print(f"\nComparing {filename}...")
        result = compare_eagle3_outputs(
            str(files1[filename]),
            str(files2[filename]),
            rtol=rtol,
            atol=atol,
            verbose=verbose,
        )
        overall_results["per_file_results"][filename] = result
        if not result["match"]:
            overall_results["match"] = False

    if verbose:
        status = "✓ All match" if overall_results["match"] else "✗ Differences found"
        print(f"\nOverall: {status}")
        print(f"  Files compared: {len(common_files)}")
        if overall_results["files_only_in_dir1"]:
            print(f"  Files only in dir1: {overall_results['files_only_in_dir1']}")
        if overall_results["files_only_in_dir2"]:
            print(f"  Files only in dir2: {overall_results['files_only_in_dir2']}")

    return overall_results


def load_and_print_eagle3_dump(path: str, show_gradients: bool = False):
    """Load and print summary of an Eagle3 training dump.

    Args:
        path: Path to the dump file.
        show_gradients: If True, also print gradient summary.
    """
    data = torch.load(path, weights_only=False)

    print(f"=== Eagle3 Training Dump: {path} ===")
    print(f"Step: {data.get('step', data.get('inference_id', 'N/A'))}")
    print(f"Batch IDX: {data.get('batch_idx', 'N/A')}")
    print(f"Rank: {data.get('rank', 'N/A')}")

    if "batch" in data:
        print("\nBatch tensors:")
        for key, tensor in data["batch"].items():
            if torch.is_tensor(tensor):
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    if "total_loss" in data:
        print(f"\nTotal loss: {data['total_loss'].item():.6f}")

    if "plosses" in data:
        print("\nPosition losses:")
        for i, pl in enumerate(data["plosses"]):
            print(f"  pos_{i}: {pl.item():.6f}")

    if "acces" in data:
        print("\nPosition accuracies:")
        for i, ac in enumerate(data["acces"]):
            print(f"  pos_{i}: {ac.item():.4f}")

    if show_gradients and "gradients" in data:
        print(f"\nGradients ({len(data['gradients'])} parameters):")
        for name, grad in sorted(data["gradients"].items())[:20]:
            norm = grad.float().norm().item()
            print(f"  {name}: shape={tuple(grad.shape)}, norm={norm:.6f}")
        if len(data["gradients"]) > 20:
            print(f"  ... and {len(data['gradients']) - 20} more parameters")

    if "model_weights" in data:
        print(f"\nModel weights ({len(data['model_weights'])} parameters):")
        for name, weight in sorted(data["model_weights"].items())[:20]:
            norm = weight.float().norm().item()
            print(f"  {name}: shape={tuple(weight.shape)}, norm={norm:.6f}")
        if len(data["model_weights"]) > 20:
            print(f"  ... and {len(data['model_weights']) - 20} more parameters")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compare Eagle3 training dumps")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    compare_parser = subparsers.add_parser("compare", help="Compare two dump files")
    compare_parser.add_argument("path1", help="Path to first dump file or directory")
    compare_parser.add_argument("path2", help="Path to second dump file or directory")
    compare_parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    compare_parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    compare_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    show_parser = subparsers.add_parser("show", help="Show contents of a dump file")
    show_parser.add_argument("path", help="Path to dump file")
    show_parser.add_argument("--gradients", action="store_true", help="Show gradient info")

    args = parser.parse_args()

    if args.command == "compare":
        p1, p2 = Path(args.path1), Path(args.path2)
        if p1.is_dir() and p2.is_dir():
            results = compare_eagle3_batches(
                args.path1, args.path2, rtol=args.rtol, atol=args.atol, verbose=args.verbose
            )
        else:
            results = compare_eagle3_outputs(
                args.path1, args.path2, rtol=args.rtol, atol=args.atol, verbose=args.verbose
            )
        if not results["match"]:
            print("\nDifferences found:")
            for key, diff in results.get("diffs", {}).items():
                print(f"  {key}: {diff}")
            sys.exit(1)
        else:
            print("All tensors match!")
            sys.exit(0)

    elif args.command == "show":
        load_and_print_eagle3_dump(args.path, show_gradients=args.gradients)

    else:
        parser.print_help()
        sys.exit(1)
