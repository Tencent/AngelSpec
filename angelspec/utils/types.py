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

from dataclasses import dataclass, field

import torch


@dataclass
class InferenceInput:
    """Input entry waiting to be sent to inference.

    For Eagle3 distillation training, input_ids and packed_loss_mask are provided
    from batch preprocessing. For other training modes, prompt is used.
    """

    data_id: str
    prompt: str | list[dict[str, str]] | None = None
    input_ids: torch.Tensor | None = None
    packed_loss_mask: str | None = None
    formatted_prompt: str | None = None
    metadata: dict = field(default_factory=dict)
    multimodal_inputs: dict = None


@dataclass
class InferenceOutput:
    """Output from inference, contains mooncake key and tensor metadata."""

    data_id: str
    mooncake_key: str
    tensor_shapes: dict[str, tuple[int, ...]]
    tensor_dtypes: dict[str, torch.dtype] | None = None
    packed_loss_mask: str | None = None
    metadata: dict = field(default_factory=dict)
