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

from transformers import AutoTokenizer

from angelspec.utils.logging import logger


def load_tokenizer(name_or_path: str, **kwargs):
    kwargs.setdefault("use_fast", True)
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def get_assistant_token_ids(
    args,
) -> tuple[list[int] | None, list[int] | None, int]:
    """Derive assistant_header_ids, end_token_ids, and skip_after_header.

    Returns:
        (header_ids, end_ids, skip_after_header) where skip_after_header is the
        number of trailing-newline tokens stripped from the header to avoid BPE
        merge issues. Pass this to ``compute_assistant_loss_mask`` so it skips
        the formatting newline after the role name.
    """
    from angelspec.data.template import TEMPLATE_REGISTRY

    chat_template_name = getattr(args, "chat_template", None)
    if not chat_template_name:
        return None, None, 0

    template = TEMPLATE_REGISTRY.get(chat_template_name)
    if not template.assistant_header or not template.end_of_turn_token:
        return None, None, 0

    tokenizer = load_tokenizer(args.target_model_path, trust_remote_code=True)

    full_ids = tokenizer.encode(template.assistant_header, add_special_tokens=False)
    header_text = template.assistant_header.rstrip("\n")
    stripped_ids = tokenizer.encode(header_text, add_special_tokens=False)
    # BPE tokenizers may merge the header's trailing \n with content-leading
    # newlines into a single multi-char token, breaking subsequence matching.
    # We tokenize without the trailing \n and tell the mask function to skip
    # the newline token(s) that follow.
    skip_after_header = len(full_ids) - len(stripped_ids)

    end_token_ids = tokenizer.encode(template.end_of_turn_token, add_special_tokens=False)
    logger.info(
        f"Assistant loss mask token IDs: header={stripped_ids}, end={end_token_ids}, "
        f"skip_after_header={skip_after_header}"
    )
    return stripped_ids, end_token_ids, skip_after_header
