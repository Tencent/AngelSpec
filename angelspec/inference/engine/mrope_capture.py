import torch

from angelspec.data.positions import build_position_ids


def compare_target_positions(input_ids, target_position_ids, recipe, *, multimodal_metadata=None):
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if target_position_ids.ndim == 2:
        target_position_ids = target_position_ids.unsqueeze(0)
    rebuilt = build_position_ids(input_ids, recipe, multimodal_metadata=multimodal_metadata)
    expected = target_position_ids.to(device=rebuilt.device, dtype=torch.long)
    if rebuilt.shape != expected.shape:
        raise ValueError(f"Position shapes differ: {rebuilt.shape} != {expected.shape}")
    error = int((rebuilt - expected).abs().max()) if rebuilt.numel() else 0
    if error:
        raise ValueError(f"Reconstructed positions differ from target; max error={error}")
    return {"position_recipe": recipe.identity, "tokens": input_ids.shape[-1], "max_error": error}
