# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch

from angelspec.training.lr_scheduler import LRSchedulerWithWarmup
from angelspec.training.muon_utils import (
    adjust_lr_for_muon,
    zeropower_via_newtonschulz5,
)
from angelspec.utils.logging import print_on_rank0


def _is_muon_matrix(name: str, param: torch.Tensor) -> bool:
    """Whether a parameter is optimized by Muon (>=2-D weight matrix).

    Follows the standard Muon convention: matrix parameters (ndim >= 2) EXCEPT
    two kinds that stay on AdamW alongside all 1-D params (norms, biases):

    * Lookup tables and classification heads, whose rows are per-token or
      per-class embeddings. Orthogonalizing them forces an artificial
      decorrelation between rows that carries no meaning: ``embed_tokens``,
      ``lm_head``, DSpark's ``confidence_head`` ([1, hidden] scalar readout) and
      ``markov_head`` (a per-token embedding and its vocabulary-sized readout),
      and the path selector's ``codebook`` tables (indexed by token
      id, so a row is a per-token embedding exactly like ``embed_tokens``; only a
      handful of rows are even touched per step, and Newton-Schulz would
      additionally all-gather the full [vocab, rank] matrix every step).
    * Small gating tables that mix a handful of channels rather than acting as
      linear maps: ``layer_fusion_weights`` (DFlare's [num_layers, T] mixing
      logits).

    The MoE ``router`` gate is deliberately NOT excluded: it stays on Muon, as it
    has been for every MTP run on this branch.
    """
    if param.ndim < 2:
        return False
    lowered = name.lower()
    for excluded in (
        "embed_tokens",
        "lm_head",
        "confidence_head",
        "markov_head",
        "codebook",
        "layer_fusion_weights",
    ):
        if excluded in lowered:
            return False
    return True


def moe_lr_scales(draft_model, scale: float, router_scale: Optional[float] = None) -> dict:
    """LR multipliers for a draft's MoE FFNs, keyed by param name.

    Covers each Hy3MoE's routed and shared experts and router, plus the layer's
    post-attention norm feeding it: what DFlash's target_mlp_init_layers copies from
    the target, and the MoE half of a warm-started MTP layer. The router takes
    ``router_scale`` (default ``scale``).
    """
    from angelspec.models.draft.mtp import Hy3MoE

    router_scale = scale if router_scale is None else router_scale
    scales = {}
    for name, module in draft_model.named_modules():
        if not isinstance(module, Hy3MoE):
            continue
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                is_router = param_name.startswith("router.")
                scales[f"{name}.{param_name}"] = router_scale if is_router else scale
        layer_name = name.rpartition(".")[0]
        norm = getattr(draft_model.get_submodule(layer_name), "post_attention_layernorm", None)
        if norm is not None:
            for param_name, param in norm.named_parameters():
                if param.requires_grad:
                    scales[f"{layer_name}.post_attention_layernorm.{param_name}"] = scale
    return scales


def muon_head_rows(model) -> dict:
    """Rows each attention head owns, keyed by full param name, for the matrices their
    modules ask Muon to orthogonalize head by head (a ``muon_head_rows()`` method naming
    the module's own params)."""
    rows = {}
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    for prefix, module in model.named_modules():
        declare = getattr(module, "muon_head_rows", None)
        if callable(declare):
            for name, head_rows in declare().items():
                full = f"{prefix}.{name}" if prefix else name
                # Skip what is frozen, as ``moe_lr_scales`` does. A module declares how its
                # matrices want to be orthogonalized whether or not this run trains them,
                # and the optimizer rejects a name it cannot find among trainable params.
                if full in trainable:
                    rows[full] = int(head_rows)
    return rows


class BF16Optimizer:
    """AdamW (default) or Muon over an fp32 master copy of bf16 model params.

    The fp32 master-weight machinery (fp32 params + fp32 grads, gradient clip and
    optimizer step in fp32, copy back to bf16) is shared by both optimizer types.
    With ``optimizer_type="muon"`` the >=2-D weight matrices are updated by
    Momentum-Orthogonalized-by-Newton-Schulz on their fp32 master copy, while all
    remaining params (norms, embeddings, LM head, fusion logits) stay on AdamW.
    Under FSDP2 the fp32 master params are sharded DTensors; Newton-Schulz gathers
    each matrix, orthogonalizes it, and the result is resharded back to the
    param's own placement before the update.

    ``lr_scales`` maps param names to LR multipliers. A scaled param follows the
    shared warmup/decay curve times its scale (its own AdamW group, or a per-matrix
    factor under Muon); unscaled params, and every param when it is empty, keep
    the single-group behaviour.

    A >2-D Muon param (fused MoE experts ``[E, in, out]``) is by default orthogonalized
    one ``[in, out]`` matrix at a time; FSDP2 shards such a stack along dim 0, so every
    rank updates the whole experts it holds without gathering anything.
    ``muon_per_expert=False`` flattens it to one ``[E, in*out]`` matrix instead.

    ``muon_head_rows`` maps a 2-D Muon param's name to the rows each attention head
    owns (see ``muon_head_rows()``). Each head's block of rows is then orthogonalized
    on its own (head-wise Muon), and its LR follows one head's shape.
    """

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        decay_style="cosine",
        min_lr=0.0,
        wsd_decay_steps=None,
        wsd_decay_style=None,
        betas=(0.9, 0.999),
        optimizer_type="adamw",
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_matched_adamw_rms=0.2,
        muon_weight_decay=None,
        lr_scales=None,
        muon_per_expert=True,
        muon_head_rows=None,
    ):
        self.model = model
        self.optimizer_type = optimizer_type.lower()
        self.muon_per_expert = bool(muon_per_expert)
        if self.optimizer_type not in ("adamw", "muon"):
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Use 'adamw' or 'muon'.")

        # named_parameters() order stays aligned with fp32 master list (checkpoint fp32_params[i]).
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.param_names = [n for n, _ in named]
        self.param_name_to_index = {name: i for i, name in enumerate(self.param_names)}
        self.model_params = [p for _, p in named]
        self.max_grad_norm = max_grad_norm
        # Two decay rates because the groups hold different kinds of tensor. 0.1 on
        # weight matrices is the Muon/Moonlight convention, but the AdamW group is
        # norms, gates and embeddings, which are conventionally decayed far less or
        # not at all. Defaults to a single shared value so existing configs are
        # unchanged.
        self.weight_decay = weight_decay
        self.muon_weight_decay = weight_decay if muon_weight_decay is None else muon_weight_decay

        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        for mp in self.fp32_params:
            mp.requires_grad = True

        lr_scales = dict(lr_scales or {})
        unknown = sorted(set(lr_scales) - set(self.param_names))
        if unknown:
            raise ValueError(f"lr_scales name no trainable parameter: {unknown[:4]}")
        if not all(scale >= 0 for scale in lr_scales.values()):
            raise ValueError(
                f"lr_scales must not be negative, got {sorted(set(lr_scales.values()))}"
            )
        self.lr_scales = [float(lr_scales.get(name, 1.0)) for name in self.param_names]

        muon_head_rows = dict(muon_head_rows or {})
        unknown = sorted(set(muon_head_rows) - set(self.param_names))
        if unknown:
            raise ValueError(f"muon_head_rows name no trainable parameter: {unknown[:4]}")
        self.muon_head_rows = [muon_head_rows.get(name, 0) for name in self.param_names]
        for i, rows in enumerate(self.muon_head_rows):
            mp = self.fp32_params[i]
            if rows and (mp.ndim != 2 or rows <= 0 or mp.size(0) % rows):
                raise ValueError(
                    f"{self.param_names[i]} {tuple(mp.shape)} does not split into heads of {rows} rows"
                )

        if self.optimizer_type == "muon":
            self.muon_momentum = muon_momentum
            self.muon_nesterov = muon_nesterov
            self.muon_ns_steps = muon_ns_steps
            self.muon_matched_adamw_rms = muon_matched_adamw_rms

            self.muon_indices = [
                i
                for i, (n, mp) in enumerate(zip(self.param_names, self.fp32_params))
                if _is_muon_matrix(n, mp)
            ]
            muon_index_set = set(self.muon_indices)
            adamw_indices = [i for i in range(len(self.fp32_params)) if i not in muon_index_set]
            if not adamw_indices:
                raise ValueError(
                    "Muon requires at least one non-matrix parameter (norm/bias) for "
                    "the internal AdamW group; none were found."
                )
            # Pre-allocate momentum buffers so the checkpoint state-dict is stable from step 0.
            self.muon_bufs = {i: torch.zeros_like(self.fp32_params[i]) for i in self.muon_indices}

            self.optimizer = torch.optim.AdamW(
                self._adamw_param_groups(adamw_indices),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
            stacked = sum(1 for i in self.muon_indices if self.fp32_params[i].ndim > 2)
            stacked_mode = "per expert, on local shards" if self.muon_per_expert else "flattened"
            head_wise = sum(1 for i in self.muon_indices if self.muon_head_rows[i])
            print_on_rank0(
                f"[Muon] {len(self.muon_indices)} matrix params on Newton-Schulz "
                f"(ns_steps={muon_ns_steps}, momentum={muon_momentum}, nesterov={muon_nesterov}); "
                f"{len(adamw_indices)} params on AdamW. "
                f"weight_decay: muon={self.muon_weight_decay}, adamw={self.weight_decay}."
                + (f" {stacked} >2-D params {stacked_mode}." if stacked else "")
                + (f" {head_wise} params per attention head." if head_wise else "")
            )
        else:
            self.muon_indices = []
            self.muon_bufs = {}
            self.optimizer = torch.optim.AdamW(
                self._adamw_param_groups(range(len(self.fp32_params))),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        # Restoring a checkpoint brings back its groups' lr_scale; load_state_dict
        # puts these configured ones back.
        self._group_lr_scales = [g.get("lr_scale", 1.0) for g in self.optimizer.param_groups]
        if any(scale != 1.0 for scale in self.lr_scales):
            summary = ", ".join(
                f"x{scale:g}: {self.lr_scales.count(scale)} tensors / "
                f"{sum(p.numel() for p, s in zip(self.model_params, self.lr_scales) if s == scale)} params"
                for scale in sorted(set(self.lr_scales), reverse=True)
            )
            print_on_rank0(f"[LR scales] {summary}")

        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
            decay_style=decay_style,
            min_lr=min_lr,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )

    def _adamw_param_groups(self, indices):
        """The AdamW params among ``indices``, one group per LR scale.

        With no scaled param this is the plain param list (one group, as before).
        Otherwise the scale-1.0 group comes first, so ``param_groups[0]`` keeps
        the base LR that ``get_learning_rate`` reports and Muon builds on.

        Sets ``adamw_indices`` to the fp32 indices in the order AdamW holds them:
        its state is keyed by that position, which checkpoint loading relies on.
        """
        by_scale = {}
        for i in indices:
            by_scale.setdefault(self.lr_scales[i], []).append(i)
        if set(by_scale) <= {1.0}:
            self.adamw_indices = by_scale.get(1.0, [])
            return [self.fp32_params[i] for i in self.adamw_indices]
        groups = [(1.0, by_scale.pop(1.0, []))] + sorted(by_scale.items(), reverse=True)
        self.adamw_indices = [i for _, group in groups for i in group]
        return [
            {"params": [self.fp32_params[i] for i in group], "lr_scale": scale}
            for scale, group in groups
        ]

    def _muon_update(self):
        """Update the Muon (matrix) params in-place on their fp32 master copy.

        The base LR is the current scheduled LR of the scale-1.0 AdamW group (single
        scheduler drives both); each matrix takes its own LR scale, then is rescaled
        by ``adjust_lr_for_muon``.
        """
        from torch.distributed.tensor import DTensor, distribute_tensor

        base_lr = self.optimizer.param_groups[0]["lr"]
        wd = self.muon_weight_decay
        momentum = self.muon_momentum
        for i in self.muon_indices:
            mp = self.fp32_params[i]
            g = mp.grad
            if g is None:
                continue
            if self.muon_per_expert and mp.ndim > 2:
                self._muon_update_stacked(i, base_lr * self.lr_scales[i])
                continue
            if g.ndim > 2:
                g = g.view(g.size(0), -1)
            buf = self.muon_bufs[i]
            if buf.ndim > 2:
                # Match the flattened grad (e.g. MoE fused experts [E,in,out] ->
                # [E,in*out]). The view shares storage with the pre-allocated 3D
                # buffer, so in-place mul_/add_ still land in it (checkpoint-stable),
                # and u.view(mp.shape) below reshapes the update back. Without this,
                # buf stays 3D while g is 2D -> broadcast error at add_ (the >2D path
                # is exercised by MTP's fused experts, not by DFlash's 2D Linears).
                buf = buf.view(buf.size(0), -1)
            buf.mul_(momentum).add_(g)
            g = g.add(buf, alpha=momentum) if self.muon_nesterov else buf

            head_rows = self.muon_head_rows[i]
            if head_rows:
                full = g.full_tensor() if isinstance(g, DTensor) else g
                heads = full.view(-1, head_rows, full.size(-1))
                u_full = zeropower_via_newtonschulz5(heads, steps=self.muon_ns_steps).view(
                    full.shape
                )
                matrix_shape = heads.shape[1:]
            else:
                u_full = zeropower_via_newtonschulz5(g, steps=self.muon_ns_steps)
                matrix_shape = g.shape
            if isinstance(mp, DTensor):
                u = distribute_tensor(u_full, mp.device_mesh, mp.placements)
            else:
                u = u_full

            lr = base_lr * self.lr_scales[i]
            # Match the LR rule to the matrix Newton-Schulz actually orthogonalized:
            # for fused experts g is the flattened [E, in*out], not the 3-D [E, in, out],
            # and a head-wise matrix is one head's rows.
            adjusted_lr = adjust_lr_for_muon(lr, matrix_shape, self.muon_matched_adamw_rms)
            mp.data.mul_(1 - lr * wd)
            mp.data.add_(u.view(mp.shape).to(mp.dtype), alpha=-adjusted_lr)

    @torch.no_grad()
    def _muon_update_stacked(self, i, lr):
        """Muon on a stack of matrices (fused experts ``[E, in, out]``), one per expert.

        Newton-Schulz batches over the leading dims, and the LR rule uses one
        expert's ``[in, out]``. Under FSDP2 the stack is sharded along dim 0 (plus
        replication under HSDP), so the local shard holds whole experts and the
        update needs no communication.
        """
        from torch.distributed.tensor import DTensor, Replicate, Shard

        mp, buf = self.fp32_params[i], self.muon_bufs[i]
        g = mp.grad
        if isinstance(mp, DTensor):
            if any(p != Shard(0) and p != Replicate() for p in mp.placements):
                raise ValueError(
                    f"per-expert Muon needs whole experts on each rank (Shard(0) / Replicate), "
                    f"got {mp.placements} for {self.param_names[i]}"
                )
            mp, g, buf = mp.to_local(), g.to_local(), buf.to_local()
        buf.mul_(self.muon_momentum).add_(g)
        update = g.add(buf, alpha=self.muon_momentum) if self.muon_nesterov else buf
        u = zeropower_via_newtonschulz5(update, steps=self.muon_ns_steps)
        adjusted_lr = adjust_lr_for_muon(lr, mp.shape[-2:], self.muon_matched_adamw_rms)
        mp.mul_(1 - lr * self.muon_weight_decay)
        mp.add_(u.to(mp.dtype), alpha=-adjusted_lr)

    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        # fp32 gradients live only inside step(); a persistent buffer would add four
        # bytes per parameter to the forward/backward peak.
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = torch.empty_like(mp).copy_(p.grad) if p.grad is not None else None

        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        if grad_norm > 0.0:
            if self.optimizer_type == "muon":
                self._muon_update()
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
                mp.grad = None

        return grad_norm

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def load_state_dict(self, state_dict):
        # AdamW-default checkpoints store the inner AdamW state_dict directly (pre-Muon runs
        # resume unchanged); Muon wraps it and adds the Newton-Schulz momentum buffers.
        if self.optimizer_type != "muon":
            self.optimizer.load_state_dict(state_dict)
            self._restore_group_lr_scales()
            print_on_rank0("Successfully loaded optimizer state_dict.")
            return
        self.optimizer.load_state_dict(state_dict["adamw"])
        self._restore_group_lr_scales()
        for i_str, buf in state_dict.get("muon_bufs", {}).items():
            self.muon_bufs[int(i_str)].data.copy_(buf)
        print_on_rank0("Successfully loaded Muon optimizer state_dict.")

    def _restore_group_lr_scales(self):
        """Put the configured LR scales back over the restored ones, so changing a
        scale takes effect on resume (from the next scheduler step)."""
        for group, scale in zip(self.optimizer.param_groups, self._group_lr_scales):
            if scale != 1.0 or "lr_scale" in group:
                group["lr_scale"] = scale

    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            for mp, p in zip(self.fp32_params, self.model_params):
                mp.data.copy_(p.data.to(torch.float32))

    @torch.no_grad()
    def add_to_parameter(self, name: str, delta: torch.Tensor) -> None:
        """Add a full fp32 controller update to a model parameter and its master copy.

        Used by non-gradient controllers such as aux-free MoE expert bias.
        Under FSDP2, ``mp`` is a sharded DTensor; every rank holds the same
        globally reduced ``delta``, which is distributed to the master's own
        placements before the update.
        """
        if name not in self.param_name_to_index:
            raise KeyError(f"optimizer has no trainable parameter {name!r}")
        index = self.param_name_to_index[name]
        param, master = self.model_params[index], self.fp32_params[index]

        from torch.distributed.tensor import DTensor, distribute_tensor

        update = delta.to(device=master.device, dtype=master.dtype)
        if isinstance(master, DTensor):
            update = distribute_tensor(update, master.device_mesh, master.placements)
        master.data.add_(update)
        param.data.copy_(master.data.to(param.dtype))

    def state_dict(self):
        if self.optimizer_type != "muon":
            return self.optimizer.state_dict()
        return {
            "adamw": self.optimizer.state_dict(),
            "muon_bufs": {str(i): buf for i, buf in self.muon_bufs.items()},
        }

    def get_learning_rate(self, lr_scale=1.0):
        """Current LR of params with ``lr_scale`` (the base LR by default)."""
        for group in self.optimizer.param_groups:
            if group.get("lr_scale", 1.0) == lr_scale:
                return group["lr"]
        # Only Muon matrices carry this scale; their LR is the base LR times it.
        return self.optimizer.param_groups[0]["lr"] * lr_scale

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def lr_scheduler(self):
        return self.scheduler
