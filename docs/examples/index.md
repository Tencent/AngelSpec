# Examples

End-to-end training examples organized by target model and scale. Each corresponds to a
directory under [`examples/`](https://github.com/Tencent/AngelSpec/tree/main/examples) with a
`run.sh` script.

## Multi-Node (Hy4 and Hy4 flash)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy-v4-dfly2](https://github.com/Tencent/AngelSpec/tree/main/examples/hy-v4-dfly2) | DFly2 | From scratch | Ships with Hy4 |
| [hy-v4-flash-dfly2](https://github.com/Tencent/AngelSpec/tree/main/examples/hy-v4-flash-dfly2) | DFly2 | From scratch | Ships with Hy4 flash |

## Single-Node (Qwen3-8B)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen3-8b-dspark](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3-8b-dspark) | DSpark | From scratch | — |
| [qwen3-8b-dfly](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3-8b-dfly) | DFly | From scratch | AngelSlim/Qwen3-8B-DFly-Block8 |
| [qwen3-8b-mtp](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3-8b-mtp) | MTP | From scratch | AngelSlim/Qwen3-8B-MTP-TTT3 |
| [qwen3-8b-dfly-cpt](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3-8b-dfly-cpt) | DFly | CPT | Continue from released ckpt |

**Start here:** `qwen3-8b-dfly` is the recommended first example (8 GPUs, vLLM backend).

## Single-Node (Qwen3.5)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen35-9b-dflash](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen35-9b-dflash) | DFlash | From scratch | — |
| [qwen35-35b-a3b-mtp](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen35-35b-a3b-mtp) | MTP | From scratch | — |

## Single-Node (Qwen3-VL)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen3vl-8b-dfly](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3vl-8b-dfly) | DFly | From scratch | — |
| [qwen3vl-8b-mtp](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3vl-8b-mtp) | MTP | From scratch | — |
| [qwen3vl-30b-dfly](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3vl-30b-dfly) | DFly | From scratch | — |
| [qwen3vl-30b-mtp](https://github.com/Tencent/AngelSpec/tree/main/examples/qwen3vl-30b-mtp) | MTP | From scratch | — |

These train on text-only data by default.

## Reference Recipes (Hy3)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy3-dfly](https://github.com/Tencent/AngelSpec/tree/main/examples/hy3-dfly) | DFly | From scratch | AngelSlim/Hy3-DFly-Block8 |
| [hy3-mtp](https://github.com/Tencent/AngelSpec/tree/main/examples/hy3-mtp) | MTP | From scratch | AngelSlim/Hy3-MTP-TTT3 |

These require a Ray cluster and Mooncake RDMA across nodes.

## Data

All examples read conversational JSONL data. Sample data ships under
[`examples/data/`](https://github.com/Tencent/AngelSpec/tree/main/examples/data).
See [Data Preparation](../basic_usage/data_preparation.md) for format details.
