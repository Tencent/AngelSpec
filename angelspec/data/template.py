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

from typing import List

from pydantic import BaseModel


class ChatTemplate(BaseModel):
    assistant_header: str | None
    user_header: str | None
    system_prompt: str | None
    end_of_turn_token: str | None
    parser_type: str = "general"
    enable_thinking: bool = False
    strip_reasoning: bool = (
        False  # drop assistant reasoning_content before rendering (non-thinking training)
    )
    image_placeholder: str = "<image>"


class TemplateRegistry:
    def __init__(self):
        self.templates = {}

    def register(self, name: str, template: ChatTemplate, override: bool = False):
        assert override or name not in self.templates, (
            f"Chat template for the model type {name} has already been registered"
        )
        self.templates[name] = template

    def get(self, name: str) -> ChatTemplate:
        return self.templates[name]

    def get_all_template_names(self) -> List[str]:
        return list(self.templates.keys())


TEMPLATE_REGISTRY = TemplateRegistry()

TEMPLATE_REGISTRY.register(
    name="llama3",
    template=ChatTemplate(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_header="<|start_header_id|>user<|end_header_id|>",
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        end_of_turn_token="<|eot_id|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="llama4",
    template=ChatTemplate(
        assistant_header="<|header_start|>assistant<|header_end|>\n\n",
        user_header="<|header_start|>user<|header_end|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|eot|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

# Like "qwen" but strips assistant reasoning_content so <think> renders empty
# (plain `qwen`), for aligning non-thinking runs.
TEMPLATE_REGISTRY.register(
    name="qwen-nothink",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        strip_reasoning=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen2-vl",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi3",
    template=ChatTemplate(
        assistant_header="<|assistant|>\n",
        user_header="<|user|>\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi4",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant<|im_sep|>",
        user_header="<|im_start|>user<|im_sep|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="phi4-mini",
    template=ChatTemplate(
        assistant_header="<|assistant|>",
        user_header="<|user|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="gpt-oss-naive",
    template=ChatTemplate(
        assistant_header="<|start|>assistant<|channel|>analysis<|message|>",
        user_header="<|start|>user<|message|>",
        system_prompt=None,
        end_of_turn_token="<|end|>",
    ),
)


TEMPLATE_REGISTRY.register(
    name="gpt-oss",
    template=ChatTemplate(
        assistant_header=None,
        user_header=None,
        system_prompt=None,
        end_of_turn_token=None,
        parser_type="openai-harmony",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-r1-distill",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        end_of_turn_token=None,
        system_prompt=None,
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen3-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)


TEMPLATE_REGISTRY.register(
    name="qwen3-instruct",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>\n\n</think>\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen3-next-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n<think>\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="kimi-k2-thinking",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="kimi-k2-instruct",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="kimi-k25-vlm",
    template=ChatTemplate(
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        user_header="<|im_user|>user<|im_middle|>",
        system_prompt=None,
        end_of_turn_token="<|im_end|>",
        parser_type="kimi-k25",
        image_placeholder="<|image|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-v3",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<｜end▁of▁sentence｜>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="ling-flash-2.0",
    template=ChatTemplate(
        assistant_header="<role>ASSISTANT</role>",
        user_header="<role>HUMAN</role>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|role_end|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="deepseek-v32",
    template=ChatTemplate(
        assistant_header="<｜Assistant｜>",
        user_header="<｜User｜>",
        system_prompt="",
        end_of_turn_token="<｜end▁of▁sentence｜>",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

TEMPLATE_REGISTRY.register(
    name="minimax-m2",
    template=ChatTemplate(
        assistant_header="]~b]ai\n",
        user_header="]~b]user\n",
        system_prompt="You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
        end_of_turn_token="[e~[",
        parser_type="minimax-m2",
        image_placeholder="<image>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="hy3",
    template=ChatTemplate(
        assistant_header="<｜hy_Assistant:opensource｜><think:opensource></think:opensource>",
        user_header="<｜hy_User:opensource｜>",
        system_prompt="",
        end_of_turn_token="<｜hy_eos:opensource｜>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="hy3-think",
    template=ChatTemplate(
        assistant_header="<｜hy_Assistant:opensource｜>",
        user_header="<｜hy_User:opensource｜>",
        system_prompt="",
        end_of_turn_token="<｜hy_eos:opensource｜>",
        parser_type="hy3-think",
    ),
)
