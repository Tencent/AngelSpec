# Qwen3.5 MTP assembly

Both tools write a new Hugging Face model directory and leave their inputs untouched.

```bash
python tools/replace_qwen35_mtp.py --base-model /path/to/qwen35-with-mtp \
  --input /path/to/native-qwen35-mtp --output /path/to/qwen35-replaced

python tools/add_qwen35_mtp.py --base-model /path/to/qwen35-without-mtp \
  --input /path/to/native-qwen35-mtp --output /path/to/qwen35-added
```

`replace` needs a base that has an MTP module and replaces all of it; `add` needs a base
without one. The input is a directory or file of native `mtp.*` tensors (`mtp.fc`,
`mtp.pre_fc_norm_*`, `mtp.norm`, `mtp.layers.*`), or a complete Qwen3.5 model whose MTP
tensors are taken. Unquantized dense and MoE bases are supported; FP8/FP4 weights, dedicated
MTP embeddings and attention-bias variants are not. Key sets, shapes and dtypes are checked,
but the MTP must have been trained against the base's architecture, tokenizer, embedding and
output head. `--dry-run` validates without writing.
