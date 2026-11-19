from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)


def enable_dflash_activation_checkpointing(draft_model):
    for index, layer in enumerate(draft_model.layers):
        if not isinstance(layer, CheckpointWrapper):
            draft_model.layers[index] = checkpoint_wrapper(
                layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
