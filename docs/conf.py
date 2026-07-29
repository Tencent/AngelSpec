import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("angelspec")
except Exception:
    # Fallback when the package is not installed (e.g. RTD build).
    import re

    _pyproject = Path(__file__).parent.parent / "pyproject.toml"
    _match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), re.MULTILINE)
    __version__ = _match.group(1) if _match else "0.0.0"

sys.path.insert(0, os.path.abspath("../.."))

DOCS_PATH = Path(__file__).parent
ROOT_PATH = DOCS_PATH.parent

project = "AngelSpec"
copyright = f"{datetime.now().year}, AngelSpec"
author = "AngelSpec Team"

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "substitution",
]

myst_heading_anchors = 5
myst_ref_domains = ["std", "py"]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Meta files, not site pages.
    "README.md",
]
pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_copy_source = True
html_last_updated_fmt = ""

_REPO_URL = "https://github.com/Tencent/angelspec"

html_theme_options = {
    "repository_url": _REPO_URL,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "logo": {"text": "AngelSpec"},
    "show_navbar_depth": 2,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "show_toc_level": 2,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

htmlhelp_basename = "angelspecdoc"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

autodoc_preserve_defaults = True
navigation_with_keys = False

autodoc_mock_imports = [
    "torch",
    "transformers",
    "triton",
    "ray",
    "vllm",
    "sglang",
]
