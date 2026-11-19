"""DFly2: the DSpark drafter under the name of the generation Hy4 flash ships.

Same model, trainer, checkpoints and ``dfly_*`` target export as DSpark. Besides
``model_type`` and ``architectures``, only the defaults differ: no Markov head, as in
the drafter Hy4 flash ships.
"""

from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel


class DFly2Config(DSparkConfig):
    model_type = "dfly2"

    # transformers makes a config subclass without its own __init__ a dataclass whose
    # generated __init__ never runs DSparkConfig's.
    def __init__(self, markov_rank: int = 0, confidence_head_with_markov: bool = False, **kwargs):
        super().__init__(
            markov_rank=markov_rank,
            confidence_head_with_markov=confidence_head_with_markov,
            **kwargs,
        )


class DFly2DraftModel(DSparkDraftModel):
    config_class = DFly2Config
