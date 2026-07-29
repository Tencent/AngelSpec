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

from angelspec.utils.logging import logger


def _format_bytes(size: int) -> str:
    """Format bytes in a human-readable form."""
    if size < 0:
        return f"{size}B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024


def calculate_eagle3_buffer_size(
    max_seq_len: int,
    batch_size: int,
    hidden_dim: int,
    num_aux_layers: int = 3,
    include_last_hidden_states: bool = True,
    safety_margin: float = 1.1,
) -> int:
    """
    Calculate the required buffer size for receiving Eagle3 output tensors.

    Tensors actually transferred via Mooncake (without batch dim per sample):
    - hidden_states: (seq, hidden_dim * num_aux_layers), bfloat16
    - input_ids: (seq,), int64
    - last_hidden_states: (seq, hidden_dim), bfloat16 (optional)

    Note: target/logits are computed locally, not transferred.
    """
    bfloat16_size = 2
    int64_size = 8

    hidden_states_size = batch_size * max_seq_len * hidden_dim * num_aux_layers * bfloat16_size
    input_ids_size = batch_size * max_seq_len * int64_size

    total = hidden_states_size + input_ids_size

    if include_last_hidden_states:
        last_hidden_states_size = batch_size * max_seq_len * hidden_dim * bfloat16_size
        total += last_hidden_states_size

    total_with_margin = int(total * safety_margin)

    alignment = 256
    aligned_size = ((total_with_margin + alignment - 1) // alignment) * alignment

    logger.debug(
        "Calculated Eagle3 buffer size: %.1fMB (seq=%s, batch=%s, hidden=%s)",
        aligned_size / (1024**2),
        max_seq_len,
        batch_size,
        hidden_dim,
    )

    return aligned_size
