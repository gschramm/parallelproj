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
html_logo = 'parllelproj-logo2.svg'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    #'style_nav_header_background': '#1e8449',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
  'display_github': True,
  'github_user': 'gschramm',
  'github_repo': 'parallelproj',
  'github_version': 'master/docs/source/',
}



# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'

# -- napoleon options --------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- autodoc options ----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "private-members": False,
    "special-members": "__call__",
    "show-inheritance": True,
}

autoclass_content = 'both'

# -- coverage builder options -------------------------------------------------
# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True
coverage_statistics_to_stdout = True

# -- nbsphinx options ---------------------------------------------------------
nbsphinx_custom_formats = {
    '.pct.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}

# -- print warnings for undocumented things ------------------------------------
# https://stackoverflow.com/questions/14141170/how-can-i-just-list-undocumented-members-with-sphinx-autodoc
# set up the types of member to check that are documented
members_to_watch = ['function','class','method','attribute','property']

def warn_undocumented_members(app, what, name, obj, options, lines):
    if(what in members_to_watch and len(lines)==0):
        # warn to terminal during build
        print("WARNING: ", what, "is undocumented: ", name, "(%d)"% len(lines))
        # or modify the docstring so the rendered output is highlights the omission
        lines.append(".. Warning:: %s '%s' undocumented" % (what, name))

def setup(app):
    app.connect('autodoc-process-docstring', warn_undocumented_members)