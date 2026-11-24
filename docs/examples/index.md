# Examples

End-to-end training examples organized by target model and scale. Each corresponds to a
directory under `examples/` with a `run.sh` script.

## Multi-Node (Hy3)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy3-dfly](../../examples/hy3-dfly/) | DFly | From scratch | AngelSlim/Hy3-DFly-B8 |
| [hy3-mtp](../../examples/hy3-mtp/) | MTP | From scratch | AngelSlim/Hy3-MTP-TTT3 |

These require a Ray cluster and Mooncake RDMA across nodes. The Hy3 target model is not
publicly available — use these as reference configurations for your own large models.

## Multi-Node (Hy4 and Hy4 flash)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy-v4-dfly2](../../examples/hy-v4-dfly2/) | DFly2 | From scratch | Ships with Hy4 |
| [hy-v4-flash-dfly2](../../examples/hy-v4-flash-dfly2/) | DFly2 | From scratch | Ships with Hy4 flash |

## Single-Node (Qwen3-8B)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen3-8b-dspark](../../examples/qwen3-8b-dspark/) | DSpark | From scratch | — |
| [qwen3-8b-dfly](../../examples/qwen3-8b-dfly/) | DFly | From scratch | AngelSlim/Qwen3-8B-DFly-B8 |
| [qwen3-8b-mtp](../../examples/qwen3-8b-mtp/) | MTP | From scratch | AngelSlim/Qwen3-8B-MTP-TTT3 |
| [qwen3-8b-dfly-cpt](../../examples/qwen3-8b-dfly-cpt/) | DFly | CPT | Continue from released ckpt |

**Start here:** `qwen3-8b-dfly` is the recommended first example (8 GPUs, vLLM backend).

## Single-Node (Qwen3.5)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen35-9b-dflash](../../examples/qwen35-9b-dflash/) | DFlash | From scratch | — |
| [qwen35-35b-a3b-mtp](../../examples/qwen35-35b-a3b-mtp/) | MTP | From scratch | — |

## Data

All examples read conversational JSONL data. Sample data ships under `examples/data/`.
See [Data Preparation](../basic_usage/data_preparation.md) for format details.
