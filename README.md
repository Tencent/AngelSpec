<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo_banner_white.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/logo_banner.png">
  <img src="docs/_static/logo_banner.png" alt="AngelSpec" width="420">
</picture>

**A unified training framework for MTP and block-parallel speculative decoding**

[![arXiv](https://img.shields.io/badge/arXiv-2607.25852-b31b1b.svg)](https://arxiv.org/abs/2607.25852)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://angelspec.readthedocs.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/AngelSlim/angelspec)
[![License](https://img.shields.io/badge/License-AngelSpec%20License-green.svg)](LICENSE)

</div>

AngelSpec is developed by the Tencent Hunyuan AI Infra team, which is a torch-native framework for training speculative-decoding draft models, covering both autoregressive MTP drafting and the block-parallel DFlash family. It is the training framework behind the [technical report](https://arxiv.org/abs/2607.25852): all drafters in the report — the TTT-trained MTP drafter and the DFly family — are trained and released with it.

## Latest News

- **[2026/11/21]** We add training for [**DeepSeek-V4.1-Flash**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [**GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash), [**Qwen3.5**](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) and [**Qwen3.8**](https://huggingface.co/Qwen/Qwen3.8-27B). 🔥🔥🔥
- **[2026/11/20]** We add training for **DFly2** on [**Hy4**](https://huggingface.co/tencent/Hy4). 🔥🔥🔥
- **[2026/11/18]** We release **AngelSpec v0.2.0**, adding training for **DFly2**, the speculative-decoding drafter that ships with [**Hy4 flash**](https://huggingface.co/tencent/Hy4-flash). DFly2 reuses the target's **sparse KV cache and top-k indices**, keeping drafting cost **constant at any context length**. Training for more open-source models, on-policy distillation, and the training and inference images are coming soon. 🔥🔥🔥
- **[2026/07/29]** We release AngelSpec v0.1.0, supporting MTP and block-parallel speculative decoding training. Check out our [technical report](https://arxiv.org/abs/2607.25852) and [released models](https://huggingface.co/collections/AngelSlim/angelspec).

## Key Features

- **7 draft architectures** — DFly2, DFly, DFlash, DFlare, Eagle3, DSpark, MTP — behind one training pipeline; switching is a config change
- **MTP training with TTT** — on-policy multi-depth rollout with memory close to a single causal pass; long-context training up to 256k via Ulysses sequence parallelism
- **Acceptance-aligned objectives** — CE, top-k KL, LK losses, D-PACE weighting, and end-to-end TV, composable through configuration
- **MLA / DSA drafters** — drafters that read the target's MLA latents and DSA candidates
- **Document-aware sequence packing** — Megatron-style fixed-length packing with strict cross-document isolation, on both the DFlash and MTP paths
- **Online evaluation** — genuine speculative decoding against the latest checkpoint during training, reporting mean accepted length and per-position acceptance as measured by the serving engine

## Architecture

<p align="center">
  <img src="docs/_static/framework.png" alt="AngelSpec Framework" width="800">
</p>

Inference and training run as separate GPU worker groups connected by a [Mooncake](https://github.com/kvcache-ai/Mooncake) tensor store, so hidden-state generation and optimization scale independently. This disaggregated foundation comes from [TorchSpec](https://github.com/lightseekorg/TorchSpec); AngelSpec extends it with the architectures, objectives, and training features above.

## Draft Architectures

| Architecture | Method | Key Idea |
|-------------|--------|----------|
| **DFly2** | Block-parallel | Target's own DSA picks and MLA latents as draft context + MTP fusion (Hy4 flash's drafter) |
| **DFly** | Block-parallel | Hybrid target conditioning + hidden-correction AR head |
| **DFlash** | Block-parallel | Anchor sampling + parallel block generation |
| **DFlare** | Block-parallel | DFlash + learnable per-layer target fusion |
| **Eagle3** | Autoregressive TTT | Test-time training with input fusion |
| **DSpark** | Hybrid | DFlash backbone + EAGLE-style autoregressive head |
| **MTP** | Single-head TTT | Full MoE decoder layer as draft (Hy3-native) |

See the [draft-model family](https://angelspec.readthedocs.io/en/latest/concepts/draft_model_family.html) docs for details and trade-offs.

### DFly2

DFly2 is [Hy4 flash](https://huggingface.co/tencent/Hy4-flash)'s speculative-decoding drafter, trained with AngelSpec and built directly on the target model's Top-k Sparse KV: each draft layer shares the KV cache (KV Share) and the compressed indexer's selection of up to 2,048 positions (Index Share) with its corresponding sparse attention layer. This yields three advantages:

- Constant cost: each draft token attends to at most ~2K keys, regardless of context length.
- Zero extra KV memory: no separate KV cache is maintained for the drafter.
- PD-disaggregation ready: not involved in prefill; the prefill-to-decode transfer adds only one last-layer hidden state.

<p align="center">
  <img src="docs/_static/dfly2.png" alt="DFly2" width="800">
</p>

## Quick Start

```bash
# Install AngelSpec + the vLLM backend
pip install -e ".[vllm]"
pip install mooncake-transfer-engine

# Single-node quickstart (8 GPUs: 4 inference + 4 training)
./examples/qwen3-8b-dfly/run.sh

# Override config values from CLI
./examples/qwen3-8b-dfly/run.sh training.learning_rate=5e-5 training.num_train_steps=500
```

Or set up a conda environment (installs mooncake too):

```bash
./tools/build_conda.sh 1 vllm     # or: sglang
micromamba activate angelspec
```

> **CUDA 12.x hosts:** PyPI's default `torch` / `vllm` wheels target CUDA 13 and won't load on a CUDA-12 driver. Install CUDA-matched wheels first — see [Installation](https://angelspec.readthedocs.io/en/latest/get_started/installation.html).

## Examples

<details open>
<summary><b>Multi-node (Hy3 target)</b></summary>

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy3-dfly](examples/hy3-dfly/) | DFly | From scratch | [AngelSlim/Hy3-DFly-Block8](https://huggingface.co/AngelSlim/Hy3-DFly-Block8) |
| [hy3-mtp](examples/hy3-mtp/) | MTP | From scratch | [AngelSlim/Hy3-MTP-TTT3](https://huggingface.co/AngelSlim/Hy3-MTP-TTT3) |

</details>

<details open>
<summary><b>Multi-node (Hy4 and Hy4 flash targets)</b></summary>

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy-v4-dfly2](examples/hy-v4-dfly2/) | DFly2 | From scratch | [tencent/Hy4](https://huggingface.co/tencent/Hy4) |
| [hy-v4-flash-dfly2](examples/hy-v4-flash-dfly2/) | DFly2 | From scratch | [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) |

</details>

<details>
<summary><b>Single-node (Qwen3-8B target)</b></summary>

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen3-8b-dspark](examples/qwen3-8b-dspark/) | DSpark | From scratch | — |
| [qwen3-8b-dfly](examples/qwen3-8b-dfly/) | DFly | From scratch | [AngelSlim/Qwen3-8B-DFly-Block8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-Block8) |
| [qwen3-8b-mtp](examples/qwen3-8b-mtp/) | MTP | From scratch | [AngelSlim/Qwen3-8B-MTP-TTT3](https://huggingface.co/AngelSlim/Qwen3-8B-MTP-TTT3) |
| [qwen3-8b-dfly-cpt](examples/qwen3-8b-dfly-cpt/) | DFly | CPT (from ckpt) | Continue from AngelSlim/Qwen3-8B-DFly-Block8 |

</details>

<details>
<summary><b>Single-node (Qwen3.5 targets)</b></summary>

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen35-9b-dflash](examples/qwen35-9b-dflash/) | DFlash | From scratch | — |
| [qwen35-35b-a3b-mtp](examples/qwen35-35b-a3b-mtp/) | MTP | From scratch | — |

</details>

<details>
<summary><b>Available training configs</b></summary>

| Config | Architecture | Features |
|--------|-------------|----------|
| `configs/vllm_qwen3_8b_dfly.yaml` | DFly | Block-parallel + hidden correction |
| `configs/vllm_qwen3_8b_dflare.yaml` | DFlare | Per-layer target fusion |
| `configs/sglang_qwen3_8b_dflash.yaml` | DFlash | SGLang backend |
| `configs/sglang_qwen3_8b_dspark.yaml` | DSpark | Markov head + confidence |
| `configs/vllm_qwen3_8b_mtp_pack_usp_40k.yaml` | MTP | Packing + USP 40k seq |
| `configs/vllm_dsv41_flash_dfly2.yaml` | DFly2 | DeepSeek-V4.1-Flash, 32k |
| `configs/vllm_dsv41_flash_dspark.yaml` | DSpark | DeepSeek-V4.1-Flash |
| `configs/vllm_glm53_flash_dfly2.yaml` | DFly2 | GLM-5.3-Flash, 32k |

</details>

## Released Models

Draft models trained with AngelSpec:

| Model | Architecture | Target | Mode |
|-------|-------------|--------|------|
| [AngelSlim/Hy3-DFly-Block8](https://huggingface.co/AngelSlim/Hy3-DFly-Block8) | DFly | Hy3 | No-think |
| [AngelSlim/Hy3-DFly-Block8-Think-High](https://huggingface.co/AngelSlim/Hy3-DFly-Block8-Think-High) | DFly | Hy3 | High-think |
| [AngelSlim/Hy3-MTP-TTT3](https://huggingface.co/AngelSlim/Hy3-MTP-TTT3) | MTP | Hy3 | No-think |
| [AngelSlim/Qwen3-8B-DFly-Block8](https://huggingface.co/AngelSlim/Qwen3-8B-DFly-Block8) | DFly | Qwen3-8B | No-think |
| [AngelSlim/Qwen3-8B-MTP-TTT3](https://huggingface.co/AngelSlim/Qwen3-8B-MTP-TTT3) | MTP | Qwen3-8B | No-think |

## Benchmark

### Offline Throughput (HY3-295B-A21B, TP=8)

<p align="center">
  <img src="docs/_static/main_result.png" alt="Main throughput results" width="800">
</p>

Output-token throughput (Tok/s) and speedup relative to AR at temperature 1 across concurrency levels. Each cell uses 3 × 120 s windows; Avg. is the arithmetic mean across six datasets.

## License

This project is released under [LICENSE](LICENSE). AngelSpec is built upon [TorchSpec](https://github.com/lightseekorg/TorchSpec) by LightSeek Foundation and uses [Mooncake](https://github.com/kvcache-ai/Mooncake) for disaggregated hidden-state transfer.

## Citation

If you find AngelSpec useful, please cite:

```bibtex
@article{angelspec2026,
  title   = {AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding},
  author  = {Liu, Hong and Cen, Rui and Shi, Junhan and Qin, Guangshuo and Zhang, Jiebin and Liu, Tianyu and Fan, Runzhi and Zhao, Guoliang and Xie, Ruobing and Zhang, Kai and Liu, Song and Yu, Guanghua and Zhu, Jianchen},
  journal = {arXiv preprint arXiv:2607.25852},
  year    = {2026}
}
```

## Projects in teams

- [AngelSlim](https://github.com/Tencent/AngelSlim): A more accessible, comprehensive, and efficient toolkit for large model compression.
