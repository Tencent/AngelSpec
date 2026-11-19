"""DFly training wrapper — DFlash backbone + TreeFlash hidden-states correction.

The correction itself is a DFlash-family feature: the drafter builds it from
``enable_hidden_correction`` and ``DFlashModel._compute_draft_logits`` applies it
before the LM head. DFly is the path that turns it on by default, and this
subclass is the wrapper identity ``DFlyTrainer`` builds (kept as its own name so
DFly checkpoints and logs stay distinguishable from plain DFlash ones).
"""

from angelspec.models.dflash import DFlashModel


class DFlyModel(DFlashModel):
    """DFly training wrapper (DFlash backbone + hidden-states correction)."""
