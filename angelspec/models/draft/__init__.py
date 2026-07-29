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

from angelspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from angelspec.models.draft.base import Eagle3DraftModel
from angelspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM
from angelspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from angelspec.models.draft.dfly import DFlyDraftModel
from angelspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from angelspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from angelspec.models.draft.treeflash_dspark_dflare import TreeflashDSparkDFlareDraftModel

__all__ = [
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "Eagle3DeepseekV2ForCausalLM",
    "Eagle3DraftModel",
    "DFlashConfig",
    "DFlashDraftModel",
    "DFlyDraftModel",
    "DSparkConfig",
    "DSparkDraftModel",
    "LlamaForCausalLMEagle3",
    "TreeflashDSparkDFlareDraftModel",
]
