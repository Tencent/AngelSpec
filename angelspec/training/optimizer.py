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

import torch

from angelspec.training.lr_scheduler import LRSchedulerWithWarmup
from angelspec.training.muon_utils import adjust_lr_for_muon, zeropower_via_newtonschulz5
from angelspec.utils.logging import print_on_rank0


def _is_muon_matrix(name: str, param: torch.Tensor) -> bool:
    """Whether a parameter is optimized by Muon (>=2-D weight matrix).

    Follows the standard Muon convention: matrix parameters (ndim >= 2) EXCEPT
    the token embedding and LM head, which — like all 1-D params (norms, biases)
    and small fusion logits — stay on AdamW. ``layer_fusion_weights`` (DFlare's
    [num_layers, T] mixing logits) is 2-D but semantically a small routing table,
    not a linear map, so it is explicitly kept on AdamW.
    """
    if param.ndim < 2:
        return False
    lowered = name.lower()
    for excluded in ("embed_tokens", "lm_head", "layer_fusion_weights"):
        if excluded in lowered:
            return False
    return True


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
    ):
        self.model = model
        self.optimizer_type = optimizer_type.lower()
        if self.optimizer_type not in ("adamw", "muon"):
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Use 'adamw' or 'muon'.")

        # named_parameters() order stays aligned with fp32 master list (checkpoint fp32_params[i]).
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.param_names = [n for n, _ in named]
        self.model_params = [p for _, p in named]
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay

        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        self.fp32_grads = [torch.zeros_like(mp) for mp in self.fp32_params]
        for mp in self.fp32_params:
            mp.requires_grad = True

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
            adamw_fp32 = [mp for i, mp in enumerate(self.fp32_params) if i not in muon_index_set]
            if not adamw_fp32:
                raise ValueError(
                    "Muon requires at least one non-matrix parameter (norm/bias) for "
                    "the internal AdamW group; none were found."
                )
            # Pre-allocate momentum buffers so the checkpoint state-dict is stable from step 0.
            self.muon_bufs = {i: torch.zeros_like(self.fp32_params[i]) for i in self.muon_indices}

            self.optimizer = torch.optim.AdamW(
                adamw_fp32, lr=lr, weight_decay=weight_decay, betas=betas
            )
            print_on_rank0(
                f"[Muon] {len(self.muon_indices)} matrix params on Newton-Schulz "
                f"(ns_steps={muon_ns_steps}, momentum={muon_momentum}, nesterov={muon_nesterov}); "
                f"{len(adamw_fp32)} params on AdamW."
            )
        else:
            self.muon_indices = []
            self.muon_bufs = {}
            self.optimizer = torch.optim.AdamW(
                self.fp32_params, lr=lr, weight_decay=weight_decay, betas=betas
            )

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

    def _muon_update(self):
        """Update the Muon (matrix) params in-place on their fp32 master copy.

        The base LR is the current scheduled AdamW LR (single scheduler drives
        both groups); each matrix is then rescaled by ``adjust_lr_for_muon``.
        """
        from torch.distributed.tensor import DTensor, distribute_tensor

        lr = self.optimizer.param_groups[0]["lr"]
        wd = self.weight_decay
        momentum = self.muon_momentum
        for i in self.muon_indices:
            mp = self.fp32_params[i]
            g = mp.grad
            if g is None:
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

            u_full = zeropower_via_newtonschulz5(g, steps=self.muon_ns_steps)
            if isinstance(mp, DTensor):
                u = distribute_tensor(u_full, mp.device_mesh, mp.placements)
            else:
                u = u_full

            adjusted_lr = adjust_lr_for_muon(lr, mp.shape, self.muon_matched_adamw_rms)
            mp.data.mul_(1 - lr * wd)
            mp.data.add_(u.view(mp.shape).to(mp.dtype), alpha=-adjusted_lr)

    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        with torch.no_grad():
            for p, mp, g in zip(self.model_params, self.fp32_params, self.fp32_grads):
                if p.grad is not None:
                    g.copy_(p.grad)
                    mp.grad = g
                else:
                    mp.grad = None

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
            print_on_rank0("Successfully loaded optimizer state_dict.")
            return
        self.optimizer.load_state_dict(state_dict["adamw"])
        for i_str, buf in state_dict.get("muon_bufs", {}).items():
            self.muon_bufs[int(i_str)].data.copy_(buf)
        print_on_rank0("Successfully loaded Muon optimizer state_dict.")

    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            for mp, p in zip(self.fp32_params, self.model_params):
                mp.data.copy_(p.data.to(torch.float32))

    def state_dict(self):
        if self.optimizer_type != "muon":
            return self.optimizer.state_dict()
        return {
            "adamw": self.optimizer.state_dict(),
            "muon_bufs": {str(i): buf for i, buf in self.muon_bufs.items()},
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def lr_scheduler(self):
        return self.scheduler
