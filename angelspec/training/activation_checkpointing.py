from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)


def enable_dflash_activation_checkpointing(draft_model):
    for index, layer in enumerate(draft_model.layers):
        kernel = getattr(getattr(layer, "self_attn", None), "dsa_kernel", None)
        if kernel is not None and not kernel.recompute_safe:
            raise ValueError(
                f"dsa_backend={kernel.name!r} with these options does not reproduce its forward "
                "on recompute, so it cannot run under activation checkpointing"
            )
        if not isinstance(layer, CheckpointWrapper):
            draft_model.layers[index] = checkpoint_wrapper(
                layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
