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

import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from angelspec.data.template import ChatTemplate

if TYPE_CHECKING:
    from typing import Any

    Conversation = List[Dict[str, Any]]

__all__ = [
    "GeneralParser",
    "Hy3ThinkParser",
    "HarmonyParser",
    "KimiK25Parser",
    "MiniMaxParser",
    "create_parser",
    "has_thinking_content",
]

_HAS_THINKING_RE = re.compile(r"<think>(?!\s*</think>)")


def has_thinking_content(conversation: list) -> bool:
    """Detect whether any assistant message contains real thinking content.

    Checks for non-empty <think> blocks in message content and for
    separate thinking/thinking_content/reasoning_content/reasoning fields
    on the message dict (covers preserved_thinking outputs from engines).
    Must be called on the raw conversation BEFORE formatting, since
    formatters (e.g. KimiK25Parser) inject empty <think></think> tags.
    """
    for msg in conversation:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and _HAS_THINKING_RE.search(content):
            return True
        for field in ("thinking", "thinking_content", "reasoning_content", "reasoning"):
            if msg.get(field):
                return True
    return False


class Parser(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @abstractmethod
    def format(self, conversation: "Conversation", **kwargs) -> str:
        """Apply chat template to conversation messages, return formatted text.

        This is the first phase: formatting only, no tokenization.
        Use this when sending prompts to an inference engine that
        handles tokenization internally.
        """
        pass

    @abstractmethod
    def parse(
        self, conversation: "Conversation", max_length: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format, tokenize, and compute loss mask.

        This is the full two-phase pipeline for offline dataset building.
        Returns (input_ids, loss_mask) tensors.
        """
        pass

    def _prepare_text(self, conversation: "Conversation", preformatted: bool, **kwargs) -> str:
        if preformatted:
            return conversation
        return self.format(conversation, **kwargs)

    def _ensure_pad_token(self):
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def _tokenize_with_loss_mask(
        self,
        text: str,
        max_length: int,
        assistant_pattern: str,
        last_turn_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and compute loss mask via a single offset-mapped pass.

        The text is tokenized exactly once with return_offsets_mapping; each
        assistant content span (char range from assistant_pattern) is mapped to
        a token range by binary-searching the per-token offsets. O(text_len).

        Args:
            last_turn_only: If True, only compute loss for the last assistant turn.
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        matches = list(re.finditer(assistant_pattern, text))
        if last_turn_only and matches:
            matches = matches[-1:]
        if not matches:
            return input_ids, loss_mask

        # Map each char span to a token range via the per-token offsets.
        # start: a token belongs to the content only if it starts at/after
        #   content_start; a token straddling the boundary stays with the prefix,
        #   matching len(encode(text[:content_start])) — re-tokenizing the prefix
        #   alone would split that straddling token off.
        # end: count tokens whose end offset is <= content_end.
        # Both offset arrays are non-decreasing over the truncated sequence.
        start_offsets = encoding.offset_mapping[0][:, 0].numpy()
        end_offsets = encoding.offset_mapping[0][:, 1].numpy()
        n_tokens = len(input_ids)

        for match in matches:
            start_token_idx = int(np.searchsorted(start_offsets, match.start(1), side="left"))
            end_token_idx = int(np.searchsorted(end_offsets, match.end(1), side="right"))

            actual_start = min(start_token_idx, n_tokens)
            actual_end = min(end_token_idx, n_tokens)

            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        return input_ids, loss_mask


class GeneralParser(Parser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = f"{chat_template.end_of_turn_token}"
        self.assistant_message_separator = f"{chat_template.assistant_header}"

    def _apply_chat_template(self, messages, **kwargs) -> str:
        kwargs.setdefault("add_generation_prompt", False)
        conversation = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        return conversation

    def format(self, conversation: "Conversation", **kwargs) -> str:
        messages = []

        if conversation[0]["role"] == "system":
            warnings.warn(
                "The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
            )
            messages.append({"role": "system", "content": conversation[0]["content"]})
            conversation = conversation[1:]
        else:
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

        for j, sentence in enumerate(conversation):
            role = sentence["role"]
            if j == 0:
                if role != "user":
                    warnings.warn(
                        f"Conversation must start with a 'user' role, but found '{role}'. Conversation truncated."
                    )
                    break
            else:
                prev_role = conversation[j - 1]["role"]
                if role == "tool" and prev_role not in ["assistant", "tool"]:
                    warnings.warn(
                        f"A 'tool' message must follow an 'assistant' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                    )
                    break
                if role == "assistant" and prev_role not in ["user", "tool"]:
                    warnings.warn(
                        f"An 'assistant' message must follow a 'user' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                    )
                    break
            messages.append(sentence)

        # Non-thinking training: drop assistant reasoning_content so Qwen3's jinja
        # renders an empty <think></think> (plain `qwen`).
        if getattr(self.chat_template, "strip_reasoning", False):
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]

        try:
            return self._apply_chat_template(messages, **kwargs)
        except (ValueError, TypeError):
            warnings.warn("Tokenizer does not have a chat_template, using fallback rendering.")
            add_generation_prompt = kwargs.get("add_generation_prompt", False)
            parts = []
            bos_token = getattr(self.tokenizer, "bos_token", None)
            user_header = self.chat_template.user_header or ""
            assistant_header = self.chat_template.assistant_header or ""
            end_of_turn = self.chat_template.end_of_turn_token or ""

            if bos_token:
                parts.append(bos_token)

            for msg in messages:
                if msg["role"] == "system":
                    parts.append(msg["content"])
                elif msg["role"] == "user":
                    parts.append(f"{user_header}{msg['content']}")
                elif msg["role"] == "assistant":
                    parts.append(f"{assistant_header}{msg['content']}{end_of_turn}")

            if add_generation_prompt:
                parts.append(assistant_header)

            return "".join(parts)

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"([\s\S]*?(?:"
            + re.escape(self.chat_template.end_of_turn_token)
            + "|$))"
        )
        return self._tokenize_with_loss_mask(
            text,
            max_length,
            assistant_pattern,
            last_turn_only=last_turn_only,
        )


class Hy3ThinkParser(GeneralParser):
    """CoT-aware parser for Hy3 (template: hy3-think).

    Differs from GeneralParser in two ways:
    1. format(): forces is_training=True when applying the tokenizer's jinja
       chat template, so assistant `reasoning_content` is rendered into
       <think>...</think>. Also passes through `tools` / `reasoning_effort`
       kwargs so the agent tool definitions enter the training text. When a
       turn has no reasoning_content, the jinja template auto-emits an empty
       <think></think> (verified), so no manual think-tag handling is needed.
    2. parse(): inherits GeneralParser.parse, whose loss-mask anchor is the
       bare assistant header "<｜hy_Assistant｜>". Supervision therefore covers
       the full think + answer (+ tool_calls) span up to the eos token.
    """

    def _apply_chat_template(self, messages, **kwargs) -> str:
        kwargs.setdefault("is_training", True)
        return super()._apply_chat_template(messages, **kwargs)


class HarmonyParser(Parser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.reasoning_levels = ["low", "medium", "high"]
        self.default_reasoning_level = "low"

    def build_single_turn_prompt(
        self,
        prompt_text: str,
        role: str,
        content: str,
    ) -> str:
        if role == "system":
            prompt_text = f"<|start|>system<|message|>{content}<|end|>"
        elif role == "assistant_reasoning_effort":
            prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {content.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        elif role == "user":
            prompt_text += f"<|start|>user<|message|>{content}<|end|>"
        elif role == "assistant_analysis":
            prompt_text += f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
        elif role == "assistant_commentary":
            prompt_text += f"<|start|>assistant<|channel|>commentary<|message|>{content}<|end|>"
        elif role == "assistant_final":
            prompt_text += f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
        else:
            raise ValueError(f"Unknown role: {role}")
        return prompt_text

    def format(self, conversation: "Conversation", **kwargs) -> str:
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        prompt_text = ""
        for j, message in enumerate(conversation):
            if (
                j == 0
                and message["role"] != "system"
                and message["role"] != "assistant_reasoning_effort"
            ):
                prompt_text = self.build_single_turn_prompt(
                    prompt_text,
                    "assistant_reasoning_effort",
                    self.default_reasoning_level,
                )
            prompt_text = self.build_single_turn_prompt(
                prompt_text, message["role"], message["content"]
            )
        if add_generation_prompt:
            prompt_text += "<|start|>assistant<|channel|>analysis<|message|>"
        return prompt_text

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> List[torch.Tensor]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        pattern = re.compile(r"<\|start\|>assistant([\s\S]*?)(?=<\|start\|>user<\|message\|>|$)")

        matches = list(pattern.finditer(text))
        if last_turn_only and matches:
            matches = matches[-1:]

        for match in matches:
            start_char = match.start(1)
            end_char = match.end(1)

            for idx, (ts, te) in enumerate(offsets):
                if ts >= start_char and te <= end_char:
                    loss_mask[idx] = 1

        return input_ids, loss_mask


class ThinkingParser(GeneralParser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)

    def _apply_chat_template(self, messages, **kwargs) -> str:
        # ThinkingParser always generates prompt before appending last assistant content
        kwargs.pop("add_generation_prompt", None)
        if messages[-1]["role"] == "assistant":
            conversation_history = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                **kwargs,
            )
            conversation = (
                conversation_history
                + messages[-1]["content"]
                + self.chat_template.end_of_turn_token
            )
            return conversation
        else:
            raise Exception(f"The last message is not assistant but {messages[-1]['role']}")

    def format(self, conversation: "Conversation", **kwargs) -> str:
        if self.chat_template.enable_thinking:
            kwargs["enable_thinking"] = True
        return super().format(conversation, **kwargs)


class KimiK25Parser(Parser):
    """
    Parser for Kimi-K2.5 model with manual string formatting.
    The default system prompt might cause confusion to users and unexpected behaviours, so we remove it.
    The token <|media_start|> is incorrect; it has been replaced with <|media_begin|> in the chat template. We also remove the system prompt from the chat template.

    Handles:
    - Converting <|image|> placeholders to Kimi media token structure
    - Preserving existing <think>...</think> blocks in assistant responses
    - Generating loss mask for assistant content (including thinking)
    """

    MEDIA_TOKEN = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    USER_HEADER = "<|im_user|>user<|im_middle|>"
    ASSISTANT_HEADER = "<|im_assistant|>assistant<|im_middle|>"
    SYSTEM_HEADER = "<|im_system|>system<|im_middle|>"
    TOOL_HEADER = "<|im_system|>tool<|im_middle|>"
    END_TOKEN = "<|im_end|>"
    IMAGE_PLACEHOLDER = "<|image|>"
    TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
    TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
    TOOL_CALL_BEGIN = "<|tool_call_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)

    def _format_content(self, content: str, role: str) -> str:
        """Format message content, replacing image placeholders for user messages."""
        if role == "user":
            return content.replace(self.IMAGE_PLACEHOLDER, self.MEDIA_TOKEN + "\n")
        return content

    THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>")

    def _strip_thinking(self, content: str) -> str:
        return self.THINK_PATTERN.sub("", content)

    def _format_tool_calls(self, tool_calls: list) -> str:
        """Format structured tool_calls into Kimi-native inline tokens."""
        tc_parts = []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            tc_args = tc.get("function", {}).get("arguments", "")
            tc_parts.append(
                f"{self.TOOL_CALL_BEGIN}{tc_id}{self.TOOL_CALL_ARGUMENT_BEGIN}{tc_args}{self.TOOL_CALL_END}"
            )
        return self.TOOL_CALLS_SECTION_BEGIN + "".join(tc_parts) + self.TOOL_CALLS_SECTION_END

    def format(self, conversation: "Conversation", **kwargs) -> str:
        """Build conversation string with Kimi-K2.5 tokens.

        Strips <think>...</think> from all assistant turns except the last one,
        aligned with Kimi's native multi-turn behavior.

        When expand_media_tokens=False, image placeholders are kept as-is so
        that sglang's multimodal processor can match them against image_data.
        """
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        expand_media_tokens = kwargs.pop("expand_media_tokens", True)
        parts = []

        last_assistant_idx = max(
            (i for i, msg in enumerate(conversation) if msg["role"] == "assistant"),
            default=-1,
        )

        for idx, msg in enumerate(conversation):
            role = msg["role"]
            content = msg.get("content", "")

            if not isinstance(content, str):
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") in ("image", "image_url"):
                            if expand_media_tokens:
                                text_parts.append(self.MEDIA_TOKEN + "\n")
                            else:
                                text_parts.append(self.IMAGE_PLACEHOLDER)
                    content = "".join(text_parts)
                else:
                    content = str(content)
            else:
                if expand_media_tokens:
                    content = self._format_content(content, role)

            if role == "assistant" and idx != last_assistant_idx:
                content = self._strip_thinking(content)

            if role == "system":
                parts.append(f"{self.SYSTEM_HEADER}{content}{self.END_TOKEN}")
            elif role == "user":
                parts.append(f"{self.USER_HEADER}{content}{self.END_TOKEN}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    content += self._format_tool_calls(tool_calls)
                if not content.startswith("<think>"):
                    content = "<think></think>" + content
                parts.append(f"{self.ASSISTANT_HEADER}{content}{self.END_TOKEN}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id:
                    content = f"## Return of {tool_call_id}\n{content}"
                parts.append(f"{self.TOOL_HEADER}{content}{self.END_TOKEN}")

        if add_generation_prompt:
            parts.append(f"{self.ASSISTANT_HEADER}<think>")

        return "".join(parts)

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        assistant_pattern = (
            re.escape(self.ASSISTANT_HEADER) + r"([\s\S]*?)" + re.escape(self.END_TOKEN)
        )
        return self._tokenize_with_loss_mask(
            text,
            max_length,
            assistant_pattern,
            last_turn_only=last_turn_only,
        )


class MiniMaxParser(Parser):
    """Parser for MiniMax-M2 model with manual string formatting.

    Handles:
    - Interleaved thinking: preserves <think>...</think> only in the last
      assistant turn (after the last user message), strips it from earlier turns.
    - Tool calls with <minimax:tool_call> XML format.
    - Consecutive tool responses grouped under a single ]~b]tool header.
    - Multimodal content with vision token placeholders.
    """

    BOS = "]~!b["
    ROLE_PREFIX = "]~b]"
    END_TOKEN = "[e~["
    SYSTEM_HEADER = "]~!b[]~b]system\n"
    USER_HEADER = "]~b]user\n"
    ASSISTANT_HEADER = "]~b]ai\n"
    TOOL_HEADER = "]~b]tool"
    TOOL_CALL_BEGIN = "<minimax:tool_call>"
    TOOL_CALL_END = "</minimax:tool_call>"
    MEDIA_TOKEN = "]<]start of image[>[" + "]<]vision pad[>[" + "]<]end of image[>["
    IMAGE_PLACEHOLDER = "<image>"

    THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>")

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        if chat_template.image_placeholder:
            self.IMAGE_PLACEHOLDER = chat_template.image_placeholder

    def _format_content(self, content: str, role: str) -> str:
        if role == "user":
            return content.replace(self.IMAGE_PLACEHOLDER, self.MEDIA_TOKEN + "\n")
        return content

    def _strip_thinking(self, content: str) -> str:
        return self.THINK_PATTERN.sub("", content).lstrip("\n")

    def _extract_thinking(self, content: str) -> Tuple[str, str]:
        """Split content into (reasoning, remaining) following MiniMax jinja logic."""
        if "</think>" in content:
            before, after = content.split("</think>", 1)
            reasoning = before.split("<think>")[-1].strip("\n")
            remaining = after.strip("\n")
            return reasoning, remaining
        return "", content

    def _format_tool_calls(self, tool_calls: list) -> str:
        """Format structured tool_calls into MiniMax XML tags."""
        tc_parts = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {"raw": args}
            invoke_lines = [f'<invoke name="{name}">']
            if isinstance(args, dict):
                for k, v in args.items():
                    val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    invoke_lines.append(f'<parameter name="{k}">{val}</parameter>')
            invoke_lines.append("</invoke>")
            tc_parts.append("\n".join(invoke_lines))
        return "\n" + self.TOOL_CALL_BEGIN + "\n" + "\n".join(tc_parts) + "\n" + self.TOOL_CALL_END

    def format(self, conversation: "Conversation", **kwargs) -> str:
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        expand_media_tokens = kwargs.pop("expand_media_tokens", True)
        parts = []

        messages = list(conversation)
        system_content = None
        if messages and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            messages = messages[1:]

        if system_content is None:
            system_content = self.chat_template.system_prompt or "You are a helpful assistant."

        parts.append(f"{self.SYSTEM_HEADER}{system_content}{self.END_TOKEN}\n")

        last_user_idx = max(
            (i for i, msg in enumerate(messages) if msg["role"] == "user"),
            default=-1,
        )

        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg.get("content", "") or ""

            if not isinstance(content, str):
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") in ("image", "image_url"):
                            if expand_media_tokens:
                                text_parts.append(self.MEDIA_TOKEN + "\n")
                            else:
                                text_parts.append(self.IMAGE_PLACEHOLDER)
                    content = "".join(text_parts)
                else:
                    content = str(content)
            else:
                if expand_media_tokens:
                    content = self._format_content(content, role)

            if role == "user":
                parts.append(f"{self.USER_HEADER}{content}{self.END_TOKEN}\n")
            elif role == "assistant":
                reasoning = msg.get("reasoning_content", "")
                if not reasoning:
                    reasoning, content = self._extract_thinking(content)

                assistant_text = ""
                if reasoning and idx > last_user_idx:
                    assistant_text += f"<think>\n{reasoning}\n</think>\n\n"

                if content:
                    assistant_text += content

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    assistant_text += self._format_tool_calls(tool_calls)

                parts.append(f"{self.ASSISTANT_HEADER}{assistant_text}{self.END_TOKEN}\n")
            elif role == "tool":
                if idx == 0 or messages[idx - 1]["role"] != "tool":
                    parts.append(self.TOOL_HEADER)
                parts.append(f"\n<response>{content}</response>")
                if idx == len(messages) - 1 or messages[idx + 1]["role"] != "tool":
                    parts.append(f"{self.END_TOKEN}\n")

        if add_generation_prompt:
            parts.append(f"{self.ASSISTANT_HEADER}<think>\n")

        return "".join(parts)

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        last_turn_only: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._prepare_text(conversation, preformatted, **kwargs)
        self._ensure_pad_token()

        assistant_pattern = (
            re.escape(self.ASSISTANT_HEADER) + r"([\s\S]*?)" + re.escape(self.END_TOKEN)
        )
        return self._tokenize_with_loss_mask(
            text,
            max_length,
            assistant_pattern,
            last_turn_only=last_turn_only,
        )


def create_parser(tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate) -> Parser:
    """Create the appropriate parser based on chat_template.parser_type."""
    if chat_template.parser_type == "general":
        return GeneralParser(tokenizer, chat_template)
    elif chat_template.parser_type == "hy3-think":
        return Hy3ThinkParser(tokenizer, chat_template)
    elif chat_template.parser_type == "thinking":
        return ThinkingParser(tokenizer, chat_template)
    elif chat_template.parser_type == "openai-harmony":
        return HarmonyParser(tokenizer, chat_template)
    elif chat_template.parser_type == "kimi-k25":
        return KimiK25Parser(tokenizer, chat_template)
    elif chat_template.parser_type == "minimax-m2":
        return MiniMaxParser(tokenizer, chat_template)
    else:
        raise ValueError(f"Invalid parser type: {chat_template.parser_type}")
