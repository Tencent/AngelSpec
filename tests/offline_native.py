import torch

from angelspec.data.offline import layer_recipe, layer_tensor_name


def native_recipe(
    aux_layers, final_layer, hidden_size, vocab_size, *, storage_dtype="float32", **extra
):
    """A native v2 capture recipe for a target whose last layer is ``final_layer``."""
    recipe = {
        **layer_recipe(
            [*aux_layers, final_layer],
            target_num_hidden_layers=final_layer + 1,
            hidden_size=hidden_size,
        ),
        "capture_aux_layer_order": list(aux_layers),
        "target_metadata_sha256": {"config.json": "test"},
        "vocab_size": vocab_size,
        "storage_dtype": storage_dtype,
    }
    recipe.update(extra)
    return recipe


def to_native(tensors, aux_layers, final_layer):
    """Split a trainer-layout sample (concatenated aux + last hidden) into per-layer tensors."""
    length = tensors["input_ids"].shape[-1]
    width = tensors["last_hidden_states"].shape[-1]
    native = {
        "input_ids": tensors["input_ids"].reshape(length),
        "loss_mask": tensors["loss_mask"].reshape(length).to(torch.uint8),
        layer_tensor_name(final_layer): tensors["last_hidden_states"].reshape(length, width),
    }
    if aux_layers:
        hidden = tensors["hidden_states"].reshape(length, len(aux_layers), width)
        for position, layer in enumerate(aux_layers):
            native[layer_tensor_name(layer)] = hidden[:, position].contiguous()
    return native
