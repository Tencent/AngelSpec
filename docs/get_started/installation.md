# Installation

## Requirements

- Linux, CUDA 12.4+
- Python 3.11+
- One or more NVIDIA GPUs. The [quickstart](quickstart.md) needs 8 GPUs
  (4 inference + 4 training); multi-node examples need RDMA-capable interconnect for Mooncake.

> **CUDA build matching.** PyPI's default `torch` / `vllm` wheels track the latest CUDA
> (currently CUDA 13) and will fail to load on an older driver (e.g. `libcudart.so.13: cannot
> open ...`). On a CUDA 12.x host, install CUDA-matched wheels — see
> [CUDA-matched wheels](#cuda-matched-wheels) below.

## Quick setup

AngelSpec provides a helper that creates a `angelspec` conda environment and installs the
package with the inference backend of your choice:

```bash
# Install with vLLM
./tools/build_conda.sh 1 vllm
micromamba activate angelspec

# Or install with SGLang
./tools/build_conda.sh
micromamba activate angelspec
```

To install into your **current** environment instead of a fresh one:

```bash
./tools/build_conda.sh current sglang   # or 'vllm' or 'both'
```

## Backend extras

If you manage the environment yourself, install AngelSpec with the extra for your backend:

```bash
pip install -e ".[vllm]"     # vLLM backend
pip install -e ".[sglang]"   # SGLang backend
```

Optional Flash Attention support:

```bash
pip install -e ".[fa]"
```

> Flash Attention is optional. Draft models use FlexAttention and the inference backends ship
> their own fused kernels, so a missing `flash_attn` is not a blocker.

## CUDA-matched wheels

`pip install torch` / `pip install vllm` install wheels built for the **latest** CUDA
(currently CUDA 13). On a **CUDA 12.x** host these fail to load (`libcudart.so.13: cannot
open shared object file` for vLLM; "the NVIDIA driver on your system is too old" for torch).
Install wheels that match your CUDA minor version (`cu126` / `cu128` / `cu129`):

```bash
# torch (pick the tag matching your CUDA — cu129 shown)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# vLLM (the version is part of the wheel-index path; pick the tag matching your CUDA)
pip install "vllm==0.22.1" \
    --extra-index-url https://wheels.vllm.ai/0.22.1/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129
```

Then install AngelSpec and Mooncake as below. On a **CUDA 13** host the plain
`./tools/build_conda.sh` / `pip install -e ".[vllm]"` path works with no extra index.

Verified combo on CUDA 12.9: `torch==2.11.0+cu129`, `vllm==0.22.1+cu129`.

> **libstdc++ (self-managed envs).** vLLM's FlashInfer JIT kernels need `libstdc++` with
> `GLIBCXX_3.4.32+`. A conda / micromamba `conda-forge` Python (what `build_conda.sh` creates)
> already ships a new-enough one. A bare `venv` built on an old base Python may not — install a
> newer `libstdcxx-ng` (conda) or put a newer `libstdc++.so.6` first on `LD_LIBRARY_PATH`.

## Mooncake (hidden-state transfer)

Hidden states are streamed from inference to training through the
[Mooncake](https://github.com/kvcache-ai/Mooncake) store. Multi-node runs use RDMA; single-node
runs can use TCP. See [Multi-Node Training](../advanced_features/multi_node.md) for cluster setup.

Install the prebuilt wheel (CUDA-independent; not pulled in by the AngelSpec extras):

```bash
pip install mooncake-transfer-engine
```

**Known issue — Mooncake SEGFAULT on TCP-only hosts.** The current Mooncake release can
SEGFAULT on TCP-only hosts. Until the [upstream issue](https://github.com/kvcache-ai/Mooncake/issues/1986)
is fixed, set:

```bash
export MC_STORE_MEMCPY=0
```

## Verify the install

Run the smallest example end to end (see [Quickstart](quickstart.md)):

```bash
./examples/qwen3-8b-dfly/run.sh training.num_train_steps=20
```
