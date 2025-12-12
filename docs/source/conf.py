# conf.py
import os
import sys
import sphinx_rtd_theme
from pathlib import Path

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))  # point to your package root

version_file = Path(__file__).resolve().parents[2] / "mdvcontainment" / "_version.py"
version_dict = {}
with open(version_file, "r") as f:
    exec(f.read(), version_dict)


# -- Project information -----------------------------------------------------
project = 'mdvcontainment'
author = 'BMH Bruininks'
release = version_dict["__version__"]
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # Core automatic documentation from docstrings
    'sphinx.ext.napoleon',   # For Google/Numpy style docstrings
    'sphinx.ext.viewcode',   # Adds links to highlighted source code
    'sphinx.ext.autosummary',# Generates summary tables for modules/classes
    'myst_parser',           # For Markdown support
]

autosummary_generate = True   # Generate stub files automatically
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'show-inheritance': True,
    'inherited-members': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sphinx = True
html_show_sourcelink = True
