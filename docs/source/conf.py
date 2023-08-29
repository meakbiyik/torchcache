# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# Source code dir relative to this file
sys.path.insert(0, os.path.abspath(".." + os.sep + ".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "torchcache"
copyright = "2023, Eren Akbiyik"
author = "Eren Akbiyik"
release = "v0.1.0"

html_title = "torchcache"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
]
if os.environ.get("READTHEDOCS") == "True":
    autosummary_generate = False  # Turn it off on readthedocs, otherwise it will fail
else:
    autosummary_generate = True
add_module_names = False
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
