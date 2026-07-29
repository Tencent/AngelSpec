# About AngelSpec

AngelSpec is a torch-native framework for training speculative-decoding draft models. It is
built upon [TorchSpec](https://github.com/lightseekorg/TorchSpec) and extends it with additional
draft architectures, target-model support, and training capabilities for production-scale runs.

## Background

Speculative decoding speeds up LLM inference by letting a small **draft model** propose the next
few tokens and a large **target model** verify them in parallel, without changing the target
model's output distribution. Modern draft models (Eagle3 and its successors) train in *feature
space*: the draft model learns from the target model's intermediate **hidden states**, so
training needs those hidden states for every token of data.

## Disaggregated design

```{image} ../_static/framework.png
:alt: AngelSpec framework overview
:width: 700px
```

AngelSpec separates inference from training:

- **Inference engines** run the frozen target model and extract multi-layer hidden states.
- The **[Mooncake](https://github.com/kvcache-ai/Mooncake) store** moves those tensors from
  inference to training over RDMA, without staging them to disk.
- **Training workers** consume the streamed hidden states to train the draft model under FSDP2.
- A **controller** orchestrates batching, backpressure, and evaluation.

Because the two sides only share a tensor store, they run on separate GPU pools and scale
independently — add inference engines for more hidden-state throughput, or add trainers for more
optimization throughput. See [Disaggregated Architecture](../concepts/disaggregated_architecture.md)
for the full pipeline.

## Draft architectures

AngelSpec supports several draft architectures, selected by config:

| Architecture | Method | Loss | Key idea |
|--------------|--------|------|----------|
| **Eagle3** | Autoregressive TTT | Forward KL | Test-time training with input fusion |
| **DFlash** | Block-parallel | CE + exponential decay | Anchor sampling + parallel block generation |
| **DFlare** | Block-parallel | CE + exponential decay | DFlash with learnable per-layer target fusion |
| **DSpark** | Hybrid | CE + L1 + confidence | DFlash backbone with an EAGLE-style autoregressive head |
| **MTP** | Single-head TTT | CE + KL | A full MoE decoder layer as the draft model |

See [The Draft-Model Family](../concepts/draft_model_family.md) for the details and trade-offs.

## Capabilities

- **Multi-backend inference** — vLLM (first-class), SGLang, and HuggingFace
- **Long-sequence training** — Ulysses sequence parallelism (USP) for 128k+ contexts
- **Sequence packing** — doc-aware packing with cross-document attention isolation
- **Online evaluation** — spec-decode acceptance rate measured during training
- **Multi-node** — large MoE target models across nodes, connected by Mooncake RDMA
- **Vocabulary pruning** — reduce the draft `lm_head` to a smaller token set at train or convert time

## Inference backends

| Backend | Support tier |
|---------|--------------|
| [vLLM](https://github.com/vllm-project/vllm) | First-class |
| [SGLang](https://github.com/sgl-project/sglang) | Community |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Community |

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md) — train a draft model on a single node
- [Disaggregated Architecture](../concepts/disaggregated_architecture.md) — how the pipeline fits together
