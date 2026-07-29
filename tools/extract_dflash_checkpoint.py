#!/usr/bin/env python3
"""Extract DFlash draft model weights from FSDP checkpoint to a simple state_dict .pt file.

Uses the same loading approach as tools/convert_to_hf.py — no distributed setup needed.
"""

import argparse
import os

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.filesystem import FileSystemReader
from typing_extensions import override


class _WrappedStorageReader(FileSystemReader):
    """FileSystemReader that auto-locates the 'model/' subdirectory."""

    def __init__(self, path: str):
        model_dir = os.path.join(path, "model")
        if os.path.isdir(model_dir):
            path = model_dir
        super().__init__(path)

    def read_metadata(self):
        metadata = super().read_metadata()
        return metadata


class _EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir", required=True, help="Path to FSDP checkpoint directory (iter_XXXXXXX)"
    )
    parser.add_argument("--output", default="dflash_draft.pt", help="Output .pt file path")
    args = parser.parse_args()

    # Try loading as a simple checkpoint first
    if os.path.isfile(args.checkpoint_dir):
        state = torch.load(args.checkpoint_dir, map_location="cpu")
        draft_keys = {
            k: v
            for k, v in state.items()
            if "draft_model" in k or "context_proj" in k or "context_norm" in k
        }
        if draft_keys:
            torch.save(draft_keys, args.output)
            print(f"Saved {len(draft_keys)} keys to {args.output}")
            return

    # Load from FSDP distributed checkpoint using no_dist mode
    ckpt_dir = args.checkpoint_dir
    print(f"Loading FSDP checkpoint from {ckpt_dir}...")

    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=_WrappedStorageReader(ckpt_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    print(f"Loaded {len(state_dict)} keys from checkpoint")

    # Extract draft model keys
    draft_state = {}
    skipped = []
    for k, v in state_dict.items():
        if "draft_model." not in k:
            skipped.append(k)
            continue
        # Extract the part after "draft_model."
        new_key = k.split("draft_model.")[-1]
        if new_key in ("t2d", "d2t"):  # Vocab pruning tables
            continue
        draft_state[new_key] = v

    if not draft_state:
        print("Warning: no draft_model keys found. All keys:")
        for k in sorted(state_dict.keys()):
            print(f"  {k}")
        print("\nSaving all keys as-is...")
        draft_state = state_dict

    torch.save(draft_state, args.output)
    print(f"\nSaved {len(draft_state)} draft model keys to {args.output}")
    for k in sorted(draft_state.keys())[:15]:
        shape = draft_state[k].shape if hasattr(draft_state[k], "shape") else "n/a"
        print(f"  {k}: {shape}")
    if len(draft_state) > 15:
        print(f"  ... ({len(draft_state) - 15} more)")

    if skipped:
        print(f"\nSkipped {len(skipped)} non-draft keys")


if __name__ == "__main__":
    main()
