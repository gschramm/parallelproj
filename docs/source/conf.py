# Configuration file for the Sphinx documentation builder.
import json

# -- Project information

project = 'parallelproj'
copyright = 'Georg Schramm'
author = 'Georg Schramm'

# get version string from file
with open('../../package.json') as f:
    version = json.load(f)['version']
release = version

# -- General configuration

extensions = [
    'sphinx.ext.duration', 'sphinx.ext.doctest', 'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax', 'sphinx.ext.viewcode'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# napoleon options
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autosummary_generate = True
autosummary_imported_members = True

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

## Don't show class signature with the class' name.
#autodoc_class_signature = "separated"

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "special-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__"
}