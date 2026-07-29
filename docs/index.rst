AngelSpec Documentation
=======================

AngelSpec is a torch-native framework for training speculative-decoding draft models,
covering both autoregressive MTP drafting and the block-parallel DFlash family. Inference
and training are disaggregated: inference engines generate target-model hidden states and
stream them — via the `Mooncake <https://github.com/kvcache-ai/Mooncake>`_ store over
RDMA — to distributed training workers, so each side scales independently.

AngelSpec supports 6 draft architectures (Eagle3, DFlash, DFlare, DFly, DSpark, MTP) across
multiple inference backends (vLLM, SGLang, HuggingFace).

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/about.md
   get_started/installation.md
   get_started/quickstart.md

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/speculative_decoding.md
   concepts/disaggregated_architecture.md
   concepts/draft_model_family.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   basic_usage/data_preparation.md
   basic_usage/training.md
   basic_usage/inference_backends.md
   basic_usage/checkpoint_conversion.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced_features/multi_node.md
   advanced_features/long_sequence_usp.md
   advanced_features/sequence_packing.md
   advanced_features/performance_metrics.md
   advanced_features/customization.md

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index.md
