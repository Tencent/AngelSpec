import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from angelspec.models.draft.mtp import MTPConfig, MTPDraftModel
from angelspec.training.qwen35_mtp import draft_to_native, validate_qwen35_target


def export(draft_path, target_path, output):
    import torch

    draft_path, target_path, output = map(Path, (draft_path, target_path, output))
    metadata = json.loads((draft_path / "config.json").read_text())
    if metadata.get("model_type") != "mtp":
        raise ValueError("Expected an AngelSpec MTP export")
    config = MTPConfig(**metadata)
    target_config = json.loads((target_path / "config.json").read_text())
    validate_qwen35_target(config, target_config)
    with torch.device("meta"):
        draft = MTPDraftModel(config, attention_backend="flex_attention")
    from angelspec.training.qwen35_mtp import read_weights

    state = read_weights(draft_path)
    # The router trains in fp32; the checkpoint stores every MTP tensor in bf16.
    native = {k: v.to(torch.bfloat16) for k, v in draft_to_native(state).items()}
    from angelspec.training.qwen35_mtp import native_to_draft

    remapped = native_to_draft(native)
    expected = {
        key: value for key, value in draft.state_dict().items() if key != "embed_tokens.weight"
    }
    if set(expected) != set(remapped):
        raise ValueError("Exported draft weights are incomplete")
    for key, value in remapped.items():
        if value.shape != expected[key].shape or not value.is_floating_point():
            raise ValueError(f"Invalid draft tensor: {key}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_file(native, str(output / "model.safetensors"))
    text = target_config.get("text_config", target_config)
    text["mtp_num_hidden_layers"] = 1
    text["mtp_use_dedicated_embeddings"] = False
    target_config["angelspec_mtp"] = {
        "source_draft": str(draft_path.resolve()),
        "format": "native_qwen3_5",
        "text_only": True,
        "gpu_validated": False,
    }
    (output / "config.json").write_text(json.dumps(target_config, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.draft, args.target, args.output))


if __name__ == "__main__":
    main()
