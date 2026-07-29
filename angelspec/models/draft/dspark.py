"""
DSpark draft model: DFlash backbone + EAGLE-style Markov and confidence heads.

DSpark shares DFlash's block-diffusion drafter (dual-source KV injection, anchor
sampling, MASK-token noise stream) and adds two heads on top:

  - Markov head: a low-rank learned bigram bias added to the draft logits,
    conditioned on the (teacher-forced) previous token. Improves the per-token
    distribution without touching the backbone.
  - Confidence head (AcceptRatePredictor): predicts a per-draft-position
    acceptance probability, trained against the empirical draft-vs-target
    accept rate (used at inference time for adaptive block length).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel, DFlashRMSNorm


class DSparkConfig(DFlashConfig):
    """
    Configuration for the DSpark draft model. Extends :class:`DFlashConfig`.
    """

    model_type = "dspark"

    def __init__(
        self,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        markov_pos_adaptive: bool = False,
        markov_alpha_max: float = 1.0,
        markov_alpha_start: float = 0.1,
        markov_alpha_end: float = 1.0,
        markov_smooth_lambda: float = 0.0,
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        enable_hidden_correction: bool = True,
        hidden_correction_intermediate_size: Optional[int] = None,
        hidden_correction_pos_adaptive: bool = False,
        hidden_correction_alpha_max: float = 0.8,
        hidden_correction_alpha_start: float = 0.1,
        hidden_correction_alpha_end: float = 0.5,
        hidden_correction_smooth_lambda: float = 0.0,
        block_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.markov_rank = markov_rank
        self.markov_head_type = markov_head_type
        # Position-adaptive transition strength for the Markov logit bias
        # (``logits_i += alpha_i * bias_i``), sharing the same design as the
        # hidden-states correction below: ``alpha_i = alpha_max * sigmoid(w_i)``
        # is a per-in-block-position learnable scalar, initialized to a
        # monotonically increasing ramp (weak prefix, strong suffix). Adds only
        # ``block_size`` scalar parameters. NOTE: the idea doc recommends the
        # hidden-states-injection variant over this logits-bias variant; this is
        # provided so both heads can carry an independent alpha curve.
        self.markov_pos_adaptive = markov_pos_adaptive
        self.markov_alpha_max = markov_alpha_max
        self.markov_alpha_start = markov_alpha_start
        self.markov_alpha_end = markov_alpha_end
        self.markov_smooth_lambda = markov_smooth_lambda
        self.enable_confidence_head = enable_confidence_head
        self.confidence_head_with_markov = confidence_head_with_markov
        # TreeFlash hidden-states correction (formula (1)): a lightweight SwiGLU
        # applied to the drafter's output hidden state, conditioned on the
        # previous token's embedding, added back in residual form.
        self.enable_hidden_correction = enable_hidden_correction
        # Intermediate width of the correction SwiGLU. ``None`` -> lightweight
        # default of ``hidden_size`` (kept small since this rides on top of the
        # full backbone MLP).
        self.hidden_correction_intermediate_size = hidden_correction_intermediate_size
        # Position-adaptive transition strength for the hidden-states correction
        # (``h'_i = h_i + alpha_i * delta_i``). ``alpha_i = alpha_max * sigmoid(w_i)``
        # is a per-in-block-position learnable scalar, initialized to a
        # monotonically increasing ramp (weak prefix correction, strong suffix
        # correction) to match the per-position drift-accumulation prior. Adds
        # only ``block_size`` scalar parameters, no extra compute at inference.
        self.hidden_correction_pos_adaptive = hidden_correction_pos_adaptive
        self.hidden_correction_alpha_max = hidden_correction_alpha_max
        self.hidden_correction_alpha_start = hidden_correction_alpha_start
        self.hidden_correction_alpha_end = hidden_correction_alpha_end
        # Optional smoothness regularizer weight lambda for the alpha curve
        # (``lambda * sum_i (alpha_i - alpha_{i-1})^2``); 0 disables it.
        self.hidden_correction_smooth_lambda = hidden_correction_smooth_lambda
        # Block length K (number of in-block positions). Required to size the
        # position-adaptive alpha vector; injected by the trainer from its
        # ``block_size`` knob when not set explicitly in the config JSON.
        self.block_size = block_size


class PositionAdaptiveAlpha(nn.Module):
    """Per-in-block-position transition strength ``alpha_i`` (length ``block_size``).

    Implements ``alpha_i = alpha_max * sigmoid(w_i)`` with a learnable logit
    ``w_i`` initialized so ``alpha_i`` follows a monotonically increasing ramp
    from ``alpha_start`` to ``alpha_end`` (weak prefix correction, strong suffix
    correction — matching the per-position drift-accumulation prior). Adds only
    ``block_size`` scalar parameters and no inference-time compute beyond a
    broadcasted multiply.

    Optionally exposes a smoothness regularizer
    ``smooth_lambda * sum_i (alpha_i - alpha_{i-1})^2`` via :meth:`smooth_loss`.
    """

    def __init__(
        self,
        *,
        block_size: int,
        alpha_max: float = 1.0,
        alpha_start: float = 0.1,
        alpha_end: float = 1.0,
        smooth_lambda: float = 0.0,
    ):
        super().__init__()
        if block_size is None or int(block_size) <= 0:
            raise ValueError(
                "PositionAdaptiveAlpha requires a positive block_size (the block "
                f"length K); got {block_size!r}. It is injected by the trainer "
                "from ``dflash_block_size``; set it in the config JSON when "
                "building the model outside the trainer."
            )
        self.block_size = int(block_size)
        self.alpha_max = float(alpha_max)
        self.smooth_lambda = float(smooth_lambda)

        # Initialize the ramp in alpha-space, then invert the sigmoid to obtain
        # the logit initialization ``w_i``.
        if self.block_size == 1:
            ramp = torch.full((1,), float(alpha_end))
        else:
            ramp = torch.linspace(float(alpha_start), float(alpha_end), self.block_size)
        frac = (ramp / self.alpha_max).clamp(1e-4, 1.0 - 1e-4)
        w_init = torch.log(frac / (1.0 - frac))
        self.alpha_logit = nn.Parameter(w_init)

    def alpha(self) -> torch.Tensor:
        """Return the current ``alpha`` vector of shape ``[block_size]``."""
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def smooth_loss(self) -> Optional[torch.Tensor]:
        """Smoothness penalty on the alpha curve, or ``None`` when disabled."""
        if self.smooth_lambda <= 0.0 or self.block_size < 2:
            return None
        a = self.alpha()
        diff = a[1:] - a[:-1]
        return self.smooth_lambda * (diff * diff).sum()


class VanillaMarkov(nn.Module):
    """Low-rank learned bigram (Markov) logit bias on the previous token."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        pos_adaptive: bool = False,
        block_size: Optional[int] = None,
        alpha_max: float = 1.0,
        alpha_start: float = 0.1,
        alpha_end: float = 1.0,
        smooth_lambda: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        # TODO: markow_w2 out_features should match "draft_vocab_size" if pruning is used.
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

        # Position-adaptive per-in-block-position strength on the logit bias.
        self.pos_alpha: Optional[PositionAdaptiveAlpha] = None
        if pos_adaptive:
            self.pos_alpha = PositionAdaptiveAlpha(
                block_size=block_size,
                alpha_max=alpha_max,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                smooth_lambda=smooth_lambda,
            )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # ``base_logits`` is ``[B, n_blocks, block_size, V]``; the bias is
        # broadcast per in-block position when position-adaptive alpha is on.
        if base_logits.size(2) == 0:
            return base_logits
        bias = self.compute_step_bias(token_ids)
        if self.pos_alpha is not None:
            alpha = self.pos_alpha.alpha().to(bias.dtype)  # [block_size]
            bias = bias * alpha.view(1, 1, -1, 1)
        return base_logits + bias


class AcceptRatePredictor(nn.Module):
    """Per-position acceptance-probability head (linear projection to a logit)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class HiddenStatesCorrection(nn.Module):
    """TreeFlash hidden-states correction (formula (1)).

    Refines the drafter's output hidden state by conditioning on the previous
    token, without breaking the ``O(1)`` complexity of the backbone::

        h'_{t+i} = h_{t+i} + SwiGLU( norm(h_{t+i}) :: norm(e_{t+i-1}) )

    where

      * ``h_{t+i}``  : the drafter's original output hidden state at position
        ``t + i`` (fed to the LM head);
      * ``norm(h_{t+i})`` (``h̃_{t+i}`` in the paper) : the RMS-normalized hidden
        state;
      * ``e_{t+i-1}`` : the RMS-normalized input embedding of the previous token
        ``x_{t+i-1}`` (teacher-forced);
      * ``::``        : concatenation along the feature dimension;
      * ``SwiGLU``    : a Swish-gated linear unit producing a correction with the
        same dimension as ``h_{t+i}``.

    The output projection is zero-initialized so the correction starts at 0 and
    the model degenerates back to DFlash at initialization (residual form).
    """

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        pos_adaptive: bool = False,
        block_size: Optional[int] = None,
        alpha_max: float = 0.8,
        alpha_start: float = 0.1,
        alpha_end: float = 0.5,
        smooth_lambda: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embed_size = int(embed_size)
        self.intermediate_size = int(intermediate_size)

        # Position-adaptive per-in-block-position strength on the residual delta.
        self.pos_alpha: Optional[PositionAdaptiveAlpha] = None
        if pos_adaptive:
            self.pos_alpha = PositionAdaptiveAlpha(
                block_size=block_size,
                alpha_max=alpha_max,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                smooth_lambda=smooth_lambda,
            )

        # Normalize each input stream independently before concatenation, so the
        # hidden state and the (differently-scaled) token embedding contribute on
        # a comparable footing.
        self.hidden_norm = DFlashRMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.embed_norm = DFlashRMSNorm(self.embed_size, eps=rms_norm_eps)

        in_features = self.hidden_size + self.embed_size
        self.gate_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Residual zero-init: correction = 0 at start -> exactly recovers DFlash.
        nn.init.zeros_(self.down_proj.weight)

    def forward(
        self, hidden_states: torch.Tensor, prev_token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Apply the residual correction.

        Args:
            hidden_states: ``[..., hidden_size]`` drafter output hidden states.
            prev_token_embeds: ``[..., embed_size]`` input embeddings of the
                previous token (same leading shape as ``hidden_states``).

        Returns:
            Corrected hidden states, same shape/dtype as ``hidden_states``.
        """
        h_norm = self.hidden_norm(hidden_states)
        e_norm = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        gate_in = torch.cat([h_norm, e_norm], dim=-1)
        delta = self.down_proj(F.silu(self.gate_proj(gate_in)) * self.up_proj(gate_in))
        delta = delta.to(hidden_states.dtype)

        # Scale the correction per in-block position: ``h'_i = h_i + alpha_i * delta_i``.
        # ``hidden_states`` is ``[B, n_blocks * block_size, hidden]``; fold out the
        # block-position axis to broadcast the ``[block_size]`` alpha vector.
        if self.pos_alpha is not None:
            K = self.pos_alpha.block_size
            bsz, n_pos, hid = delta.shape
            if n_pos % K != 0:
                raise ValueError(
                    f"HiddenStatesCorrection position count {n_pos} is not a "
                    f"multiple of block_size {K}; cannot apply position-adaptive alpha."
                )
            alpha = self.pos_alpha.alpha().to(delta.dtype)  # [block_size]
            delta = (delta.view(bsz, -1, K, hid) * alpha.view(1, 1, K, 1)).reshape(bsz, n_pos, hid)

        return hidden_states + delta


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            pos_adaptive=bool(getattr(config, "markov_pos_adaptive", False)),
            block_size=getattr(config, "block_size", None),
            alpha_max=float(getattr(config, "markov_alpha_max", 1.0)),
            alpha_start=float(getattr(config, "markov_alpha_start", 0.1)),
            alpha_end=float(getattr(config, "markov_alpha_end", 1.0)),
            smooth_lambda=float(getattr(config, "markov_smooth_lambda", 0.0)),
        )
    raise NotImplementedError(
        f"markov_head_type={markov_head_type!r} is not supported yet; only 'vanilla' "
        "is implemented in AngelSpec as it is recommended by the authors."
    )


def build_hidden_correction(config) -> Optional[nn.Module]:
    """Build the TreeFlash hidden-states correction module, or ``None``.

    The previous-token embedding dimension equals the draft ``hidden_size``
    (the drafter reuses the target model's token embedding). The SwiGLU
    intermediate width defaults to ``hidden_size`` (lightweight) unless
    ``hidden_correction_intermediate_size`` is set on the config.
    """
    if not bool(getattr(config, "enable_hidden_correction", False)):
        return None

    hidden_size = int(config.hidden_size)
    intermediate = getattr(config, "hidden_correction_intermediate_size", None)
    intermediate = int(intermediate) if intermediate else hidden_size
    return HiddenStatesCorrection(
        hidden_size=hidden_size,
        embed_size=hidden_size,
        intermediate_size=intermediate,
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        pos_adaptive=bool(getattr(config, "hidden_correction_pos_adaptive", False)),
        block_size=getattr(config, "block_size", None),
        alpha_max=float(getattr(config, "hidden_correction_alpha_max", 0.8)),
        alpha_start=float(getattr(config, "hidden_correction_alpha_start", 0.1)),
        alpha_end=float(getattr(config, "hidden_correction_alpha_end", 0.5)),
        smooth_lambda=float(getattr(config, "hidden_correction_smooth_lambda", 0.0)),
    )


class DSparkDraftModel(DFlashDraftModel):
    config_class = DSparkConfig

    def __init__(self, config: DSparkConfig):
        super().__init__(config)

        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        self.markov_head = build_markov_head(config)

        # TreeFlash hidden-states correction (formula (1)); ``None`` when disabled.
        self.hidden_correction = build_hidden_correction(config)

        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_input_dim = self.hidden_size
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov=True requires a Markov head (markov_rank > 0)."
                    )
                conf_input_dim += self.markov_rank
            self.confidence_head = AcceptRatePredictor(conf_input_dim)
