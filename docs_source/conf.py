# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
import os
#sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = 'cbeam'
copyright = '2024, Jon Lin'
author = 'Jon Lin'
release = '0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx.ext.doctest','matplotlib.sphinxext.plot_directive','sphinx.ext.napoleon','sphinx.ext.githubpages']

templates_path = ['_templates']
exclude_patterns = []
plot_include_source = True

doctest_global_setup = '''
skip_tests = False
'''
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/jw-lin/cbeam",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs_source/"
}