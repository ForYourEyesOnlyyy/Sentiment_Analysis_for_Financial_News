# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Sentiment Analysis for Financial News'
copyright = '2024, Maxim Martyshov, Elisey Smirnov, Roman Makeev'
author = 'Maxim Martyshov, Elisey Smirnov, Roman Makeev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',            # For Google-style and NumPy-style docstrings
    'sphinx.ext.autosummary',         # For automatic summary tables
    'sphinx_autodoc_typehints',       # For type hints in function signatures
    'myst_parser',                    # For Markdown files
    'nbsphinx',                       # For Jupyter notebooks
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'             # Adds links to source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}
nbsphinx_allow_errors = True
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

source_suffix = ['.rst', '.md', '.ipynb']

autosummary_generate = True