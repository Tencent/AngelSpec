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

import numba
import numpy as np
import torch


@numba.njit(cache=True)
def _numba_loss_mask(ids, header, header_len, end, end_len, out, skip_after):
    n = len(ids)
    i = 0
    while i <= n - header_len:
        match = True
        for k in range(header_len):
            if ids[i + k] != header[k]:
                match = False
                break
        if not match:
            i += 1
            continue
        j = i + header_len + skip_after
        found_end = False
        while j <= n - end_len:
            end_match = True
            for k in range(end_len):
                if ids[j + k] != end[k]:
                    end_match = False
                    break
            if end_match:
                found_end = True
                break
            out[j] = 1
            j += 1
        if not found_end:
            for k in range(j, n):
                out[k] = 1
        i = j + end_len
    return out


def _to_contiguous_int64_numpy(t: torch.Tensor) -> np.ndarray:
    if t.is_cuda:
        t = t.cpu()
    if t.dtype != torch.int64:
        t = t.to(torch.int64)
    t = t.contiguous()
    return t.numpy()


def compute_assistant_loss_mask(
    input_ids: torch.Tensor,
    assistant_header_ids: list[int],
    end_token_ids: list[int],
    last_turn_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor:
    """Compute loss mask where 1s mark assistant content tokens only.

    Uses a JIT-compiled single-pass O(N) scan to find assistant header
    sequences and pair them with end token sequences.
    Computes on CPU and returns result on the same device as input_ids.

    Args:
        input_ids: 1-D tensor of token IDs (CPU or CUDA).
        assistant_header_ids: Token ID sequence marking the start of assistant content.
        end_token_ids: Token ID sequence marking the end of assistant content.
        last_turn_only: If True, only the last assistant turn is marked.
        skip_after_header: Number of tokens to skip after header match before
            marking content. Use 1 when the header excludes a trailing newline
            that BPE may merge with subsequent content.

    Returns:
        1-D long tensor on the same device as input_ids, with 1s for assistant
        content tokens and 0s elsewhere.
    """
    device = input_ids.device if isinstance(input_ids, torch.Tensor) else torch.device("cpu")
    ids_np = (
        _to_contiguous_int64_numpy(input_ids)
        if isinstance(input_ids, torch.Tensor)
        else np.asarray(input_ids, dtype=np.int64)
    )
    header_np = np.array(assistant_header_ids, dtype=np.int64)
    end_np = np.array(end_token_ids, dtype=np.int64)
    out = np.zeros(len(ids_np), dtype=np.int64)
    _numba_loss_mask(
        ids_np, header_np, len(header_np), end_np, len(end_np), out, skip_after_header
    )

    if last_turn_only and out.any():
        last_one = len(out) - 1 - np.argmax(out[::-1])
        first_of_last = last_one
        while first_of_last > 0 and out[first_of_last - 1] == 1:
            first_of_last -= 1
        out[:first_of_last] = 0

    result = torch.from_numpy(out)
    return result.to(device) if device.type != "cpu" else result
