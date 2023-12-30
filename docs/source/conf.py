# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# add parallelproj to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'python')))

import json

# get version string from file
with open(os.path.join('..', '..', 'package.json')) as f:
    version = json.load(f)['version']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'parallelproj'
copyright = '2023, Georg Schramm'
author = 'Georg Schramm'
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax', 
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_design',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'

# -- napoleon options --------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- autodoc options ----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "private-members": False,
    "special-members": "__init__,__call__",
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}

autodoc_member_order = 'bysource'

# -- coverage builder options -------------------------------------------------
# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True
coverage_statistics_to_stdout = True

# -- nbsphinx options ---------------------------------------------------------
nbsphinx_custom_formats = {
    '.pct.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}
