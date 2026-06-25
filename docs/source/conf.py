# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fue'
copyright = '2026, Sebastian Linden'
author = 'Sebastian Linden'
release = '12.08.2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Extracts docstrings from source code
    'sphinx.ext.napoleon', # Parses Google/NumPy style docstrings
    'sphinx.ext.viewcode', # Adds links to the highlighted source code
    'sphinxcontrib.bibtex',# Added for BibTeX support
]

# Configure the path to your master .bib file(s)
bibtex_bibfiles = ['references.bib']

# Optional: Set the default bibliography style (e.g., 'alpha', 'plain', 'unsrt')
bibtex_default_style = 'alpha'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))