# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'parallelproj'
copyright = 'Georg Schramm'
author = 'Georg Schramm'

release = '1.3'
version = '1.3.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode'
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