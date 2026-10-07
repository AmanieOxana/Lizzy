"""Small Sphinx site built from the maintained Markdown documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "Lizzy"
author = "Lizzy contributors"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3
autodoc_typehints = "none"
autodoc_member_order = "bysource"
html_theme = "alabaster"
html_title = "Lizzy: structure-aware synthesis"
html_baseurl = "https://amanieoxana.github.io/Lizzy/"
html_theme_options = {"description": "Hamiltonian evolution from algebraic structure"}
