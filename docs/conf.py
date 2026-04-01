"""Sphinx configuration for ORMAS-CI documentation."""

import os
import sys

# Add project root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information --
project = "ORMAS-CI"
copyright = "2026, Bryce M. Westheimer"
author = "Bryce M. Westheimer"
release = "0.3.0"

# -- General configuration --
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# MyST-Parser configuration
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
myst_heading_anchors = 3

# Source file configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pyscf": ("https://pyscf.org/", None),
}

# -- HTML output --
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

html_title = "ORMAS-CI"
html_short_title = "ORMAS-CI"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Root document (renamed from master_doc in Sphinx 4.0)
root_doc = "index"
