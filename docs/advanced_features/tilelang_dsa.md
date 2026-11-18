# TileLang DSA kernels

`dsa_backend: tilelang` runs the sparse (DSA) attention of the MLA drafters and the MLA MTP
head on two kernels: a lightning indexer that picks each query's top-k keys, and sparse MLA
forward and backward over those keys. Install them with:

```bash
pip install "angelspec[tilelang]"
```

The kernels compile on first use and are cached under `~/.tilelang/cache`.
`dsa_backend: torch_absorbed` runs the same computation in plain PyTorch.

The kernels need bf16 inputs, a 512-wide latent with 64 RoPE channels (or none), index lists
padded with `-1` to a multiple of 64, and at most 128 indexer heads.

`dsa_fp8_indexer: true` rounds the indexer's q and k to FP8 before scoring, so training selects
the keys an FP8 inference indexer would.

`ANGELSPEC_DSA_INDEXER_QUERY_CHUNK` bounds the indexer's logits memory: `auto` (default) scores
at most 2048 query rows and about 128 MiB of logits per call, a number fixes the row count, and
`0` scores all rows at once.

The backward accumulates the KV gradient with atomics, so runs are not bitwise reproducible.

The kernels in `angelspec/models/ops/tilelang_dsa_kernels.py` are adapted from TileLang's
`examples/deepseek_v32` (MIT License).
