# Inference Backends

AngelSpec runs the frozen target model on inference engines that extract hidden states and stream
them to training. The backend is selected by `model.target_model_backend`, and all backends
produce the same batch format for the trainer, so switching backends does not change the training
side.

| Backend | `target_model_backend` | Support tier | Notes |
|---------|------------------------|--------------|-------|
| vLLM | `vllm` | First-class | Extracts hidden states via a worker extension; multi-node TP |
| SGLang | `sglang` | Community | Async serving; multi-node TP |
| HuggingFace | `hf` | Community | Simplest; good for small setups and the quickstart |

## vLLM

vLLM is a first-class backend. AngelSpec hooks into the model forward pass through vLLM's
**worker extension** mechanism to capture hidden states directly inside the worker process,
avoiding RPC serialization overhead. Hidden-state extraction is enabled through vLLM's
speculative-config interface (`method="extract_hidden_states"`), and the captured tensors are
written to the Mooncake store by a KV connector.

```bash
./examples/qwen3-8b-dfly/run.sh
```

Models that vLLM does not ship with are registered at runtime, so no manual patching of the vLLM
installation is required.

## SGLang

SGLang serves the target model with async inference. AngelSpec applies a patch to enable
hidden-state extraction:

```bash
./examples/qwen3-8b-dspark/run.sh
```

## HuggingFace

The HuggingFace backend runs the target model with Transformers directly. It has the fewest
dependencies and is useful for debugging or small setups:

```bash
python3 -m angelspec.train_entry --config configs/hf_qwen3_8b.yaml
```

## Choosing a backend

- For the smallest setup or a first run, use **HuggingFace**.
- For throughput on a single node or a few nodes, use **SGLang** or **vLLM**.
- For very large target models that must be sharded across nodes, use a backend with multi-node
  tensor parallelism (**vLLM** or **SGLang**) — see [Multi-Node Training](../advanced_features/multi_node.md).
