# Configuration file for Sphinx documentation

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "pyflowreg"
author = "Philipp Flotho"
copyright = "2025, Philipp Flotho"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/FlowRegSuite/pyflowreg",
    "logo": {"text": "PyFlowReg"},
    "navbar_end": ["navbar-icon-links"],
    "show_nav_level": 2,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
    "linkify",
    "dollarmath",
]
myst_heading_anchors = 3
myst_url_schemes = ["http", "https"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = False
napoleon_use_rtype = False

autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

extlinks = {
    "opencv": ("https://docs.opencv.org/4.x/%s", ""),
}

bibtex_bibfiles = ["references.bib"]

nitpicky = True
