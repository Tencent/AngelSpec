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

from angelspec.models.dflash import DFlashModel
from angelspec.models.dfly import DFlyModel
from angelspec.models.dspark import DSparkModel
from angelspec.models.eagle3 import (
    Eagle3Model,
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from angelspec.models.mtp import MTPModel
from angelspec.models.ops.loss import compiled_forward_kl_loss
from angelspec.models.ops.loss_mask import compute_assistant_loss_mask

__all__ = [
    "Eagle3Model",
    "DFlashModel",
    "DFlyModel",
    "DSparkModel",
    "MTPModel",
    "compute_lazy_target_padded",
    "compute_target_p_padded",
    "compiled_forward_kl_loss",
    "compute_assistant_loss_mask",
]
