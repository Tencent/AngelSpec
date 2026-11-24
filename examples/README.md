# Examples

Training examples organized by target model and scale.

## Single-Node (Qwen3-8B)

These examples use [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as the
target model. All run on a single 8-GPU node.

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen3-8b-dspark](qwen3-8b-dspark/) | DSpark | From scratch | — |
| [qwen3-8b-dfly](qwen3-8b-dfly/) | DFly | From scratch | AngelSlim/Qwen3-8B-DFly-B8 |
| [qwen3-8b-mtp](qwen3-8b-mtp/) | MTP | From scratch | AngelSlim/Qwen3-8B-MTP-TTT3 |
| [qwen3-8b-dfly-cpt](qwen3-8b-dfly-cpt/) | DFly | Continual training | — |

**Start here:** `qwen3-8b-dfly` is the recommended first example if you have 8 GPUs.

## Single-Node (Qwen3.5)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [qwen35-9b-dflash](qwen35-9b-dflash/) | DFlash | From scratch | — |
| [qwen35-35b-a3b-mtp](qwen35-35b-a3b-mtp/) | MTP | From scratch | — |

## Multi-Node (Hy3)

These examples target Hy3 (large MoE) and require 2+ nodes with RDMA.
The target model is not publicly available — use these as reference configurations
for your own large target models.

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy3-dfly](hy3-dfly/) | DFly | From scratch | AngelSlim/Hy3-DFly-B8 |
| [hy3-mtp](hy3-mtp/) | MTP | From scratch | AngelSlim/Hy3-MTP-TTT3 |

## Multi-Node (Hy4 and Hy4 flash)

| Example | Architecture | Mode | Released Model |
|---------|-------------|------|----------------|
| [hy-v4-dfly2](hy-v4-dfly2/) | DFly2 | From scratch | Ships with Hy4 |
| [hy-v4-flash-dfly2](hy-v4-flash-dfly2/) | DFly2 | From scratch | Ships with Hy4 flash |

## Sample Data

`data/sample_conversations.jsonl` contains a small dataset for testing.
`data/eval_conversations.jsonl` contains evaluation prompts for online eval.

For real training, prepare your data as described in
[Data Preparation](../docs/basic_usage/data_preparation.md).
