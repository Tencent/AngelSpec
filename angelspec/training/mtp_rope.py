from angelspec.training.position_config import configure_draft_positions


def configure_mtp_rope(config, target_model_path):
    if getattr(config, "model_type", None) != "mtp":
        return config
    return configure_draft_positions(config, target_model_path)
