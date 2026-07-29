# Disaggregated Architecture

AngelSpec separates the two halves of feature-space draft training — generating the target
model's hidden states, and optimizing the draft model — so they run as independent groups of
GPU workers connected by a shared tensor store.

```{image} ../_static/framework.png
:alt: AngelSpec framework: inference engines stream hidden states through the Mooncake store to training workers, with realtime evaluation server.
:width: 750px
```

## Why disaggregate

Feature-space draft models learn from the target model's hidden states, so every training step
needs those states for its batch. Producing them and optimizing the draft model have very
different resource profiles: the target model is large and inference-shaped, while the draft
model is small and training-shaped. Coupling them in one process forces the two to share GPUs
and scale together.

AngelSpec instead runs them as separate worker groups that communicate only through a tensor
store. Each side scales on its own: add inference engines for more hidden-state throughput, or
add trainers for more optimization throughput.

## Components

All GPU workloads run as [Ray](https://www.ray.io/) actors, coordinated by a central controller.

- **Inference engines** run the frozen target model and extract hidden states from selected
  layers during prefill. AngelSpec supports [vLLM, SGLang, and HuggingFace](../basic_usage/inference_backends.md)
  backends behind a common engine interface.
- **Mooncake store** ([Mooncake](https://github.com/kvcache-ai/Mooncake)) holds the hidden-state
  tensors in shared memory and moves them over RDMA, without materializing them to disk.
- **Training workers** consume the streamed tensors and train the draft model under
  [FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html). Each worker
  is a Ray actor owning one GPU and a slot in the data-parallel group.
- **Controller** orchestrates the pipeline: it loads and batches data, dispatches prompts to
  inference, applies backpressure so inference does not outrun training, and drives evaluation.

## Data flow

1. The controller loads the dataset and feeds prompts to the inference engines.
2. An engine prefills the target model and captures hidden states from the configured layers.
3. Those tensors are written to the Mooncake store; only lightweight metadata (keys and shapes)
   travels through the controller's queues.
4. Each training worker pulls the tensors for its shard of the batch directly from Mooncake and
   runs the draft model's forward/backward.
5. The optimizer steps, metrics are aggregated, and checkpoints are written periodically.

Because the store decouples producer from consumer, inference and training proceed
asynchronously, with backpressure keeping the in-flight buffer bounded.

## Related pages

- [The Draft-Model Family](draft_model_family.md) — what the training side produces
- [Multi-Node Training](../advanced_features/multi_node.md) — running across nodes
- [Performance Metrics](../advanced_features/performance_metrics.md) — finding pipeline bottlenecks
