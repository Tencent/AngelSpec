import argparse
import json

from angelspec.data.offline import HiddenDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="AngelSpec offline feature assets and inspection")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", required=True, help="Directory containing manifest.json")
        sub.add_argument("--format", choices=("auto", "native", "dfly"), default="auto")
        sub.add_argument("--tokenized-path")
        sub.add_argument("--hidden-size", type=int)
        sub.add_argument("--context-layers", type=int)
        sub.add_argument("--vocab-size", type=int)
    status = commands.add_parser("status")
    status.add_argument("--manifest", required=True, help="Online dump directory")
    status.add_argument("--verify", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", required=True, help="Incomplete online dump directory")
    finalize.add_argument(
        "--verify",
        choices=("full", "header"),
        default="full",
        help="full re-reads every file; header checks sizes and file metadata only",
    )
    commands.add_parser("export-assets").add_argument("--config", "-c", required=True)
    args, overrides = parser.parse_known_args(argv)
    if args.command == "export-assets":
        if any("=" not in value or value.startswith("-") for value in overrides):
            parser.error("Configuration overrides must be section.key=value")
        from angelspec import AutoDraftModelConfig
        from angelspec.config.train_config import config_to_flat_args, load_config
        from angelspec.data.offline_assets import (
            export_target_assets,
            mtp_init_prefix,
            target_weight_prefixes,
        )

        settings = config_to_flat_args(load_config(args.config, cli_args=overrides))
        if not settings.target_model_path or not settings.offline_frozen_target_path:
            parser.error(
                "export-assets requires model.target_model_path and offline.frozen_target_path"
            )
        draft = (
            AutoDraftModelConfig.from_file(settings.draft_model_config)
            if settings.draft_model_config
            else None
        )
        identity = export_target_assets(
            settings.target_model_path,
            settings.offline_frozen_target_path,
            init_prefix=mtp_init_prefix(settings, draft),
            weight_prefixes=target_weight_prefixes(settings.target_model_path, draft),
        )
        result = {"frozen_target_path": settings.offline_frozen_target_path, "identity": identity}
        print(json.dumps(result, indent=2))
        return result
    if args.command in ("status", "finalize"):
        if overrides:
            parser.error(f"Unrecognized arguments: {overrides}")
        from angelspec.inference.online_dump import (
            finalize_online_dump,
            online_dump_status,
        )

        result = (
            finalize_online_dump(args.manifest, verify=args.verify)
            if args.command == "finalize"
            else online_dump_status(args.manifest, verify=args.verify)
        )
        print(json.dumps(result, indent=2))
        return result
    if overrides:
        parser.error(f"Unrecognized arguments: {overrides}")
    dataset = HiddenDataset(
        args.manifest,
        format=args.format,
        tokenized_path=args.tokenized_path,
        verify_hashes=True,
    )
    result = {
        "format": dataset.format,
        "manifest_sha256": dataset.identity,
        "samples": len(dataset.records),
        "train": sum(r["split"] == "train" for r in dataset.records),
        "validation": sum(r["split"] == "validation" for r in dataset.records),
        "max_seq_len": max(r["seq_len"] for r in dataset.records),
    }
    if not dataset.legacy:
        result.update(
            {
                "layer_ids": dataset.layer_ids,
                "layer_semantics": dataset.recipe["layer_semantics"],
                "final_layer_id": dataset.final_layer_id,
                "hidden_size": dataset.hidden_size,
            }
        )
    if args.command == "verify" and not dataset.legacy:
        result["verified"] = dataset.verify()
    elif args.command == "verify":
        hidden_size = args.hidden_size or dataset.recipe.get("hidden_size")
        layers = args.context_layers
        if layers is None and "hidden_layer_ids" in dataset.recipe:
            layers = len(dataset.recipe["hidden_layer_ids"])
        if not hidden_size or layers is None or layers < 0:
            parser.error("verify requires --hidden-size and --context-layers for DFly datasets")
        result["verified"] = dataset.verify(
            hidden_size, layers, args.vocab_size or dataset.recipe.get("vocab_size")
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
