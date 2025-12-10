# conf.py
import os
import sys
import subprocess
import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))  # point to your package root

# -- Project information -----------------------------------------------------
project = 'mdvcontainment'
author = 'BMH Bruininks'
release = 'v2.0.0'
version = 'v2.0.0'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # Core automatic documentation from docstrings
    'sphinx.ext.napoleon',   # For Google/Numpy style docstrings
    'sphinx.ext.viewcode',   # For regular .py files
    'sphinx.ext.linkcode',   # For .pyx files
    'sphinx.ext.autosummary',# Generates summary tables for modules/classes
    'myst_parser',           # For Markdown support
]

# Get current git branch
def get_git_branch():
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=os.path.abspath('..'),
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return branch
    except:
        return 'main'  # fallback to main if git command fails

# GitHub repository info
github_user = 'BartBruininks'  
github_repo = 'mdvcontainment'  
github_branch = get_git_branch()

# linkcode configuration for .pyx files
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
    module = info['module']
    filename = module.replace('.', '/')
    
    # Check if this is a Cython module (.pyx file)
    base_path = os.path.abspath('..')
    pyx_path = os.path.join(base_path, f'{filename}.pyx')
    
    if os.path.exists(pyx_path):
        # Link to GitHub for .pyx files
        return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{filename}.pyx"
    
    # Return None for .py files - let viewcode handle them
    return None

autosummary_generate = True
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