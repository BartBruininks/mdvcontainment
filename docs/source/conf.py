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

# linkcode configuration for .pyx and .py files with inheritance and property support
def linkcode_resolve(domain, info):
    print(f"\n=== linkcode_resolve called ===")
    print(f"Domain: {domain}")
    print(f"Info: {info}")
    
    if domain != 'py':
        print("Not Python domain, returning None")
        return None
    if not info['module']:
        print("No module in info, returning None")
        return None
    
    # Import the module to inspect it
    try:
        import importlib
        mod = importlib.import_module(info['module'])
        print(f"Successfully imported module: {info['module']}")
    except ImportError as e:
        print(f"Failed to import module: {e}")
        return None
    
    # Get the actual object (class or function)
    obj = mod
    parts = info['fullname'].split('.')
    for i, part in enumerate(parts):
        try:
            obj = getattr(obj, part)
            print(f"Got attribute: {part}")
        except AttributeError as e:
            print(f"Failed to get attribute {part}: {e}")
            return None
    
    print(f"Final object: {obj}")
    print(f"Object type: {type(obj)}")
    
    # Check if this is a property
    is_property = isinstance(obj, property)
    print(f"Is property: {is_property}")
    
    # If it's a property, get the fget function
    if is_property:
        obj = obj.fget
        if obj is None:
            print("Property has no getter (fget is None)")
            return None
        print(f"Property getter function: {obj}")
    
    # For methods, try to find where they're actually defined
    try:
        import inspect
        
        # Unwrap to get to the actual function (handles decorators, etc.)
        obj = inspect.unwrap(obj)
        print(f"Unwrapped object: {obj}")
        
        # Get the module where this object is actually defined
        try:
            actual_module = inspect.getmodule(obj)
            if actual_module is None:
                print("inspect.getmodule returned None")
                return None
            module_name = actual_module.__name__
            print(f"Actual module where defined: {module_name}")
        except Exception as e:
            print(f"Failed to get module: {e}")
            module_name = info['module']
        
        # Get the file where it's defined
        try:
            source_file = inspect.getsourcefile(obj)
            if source_file is None:
                print("inspect.getsourcefile returned None")
                return None
            print(f"Source file: {source_file}")
        except Exception as e:
            print(f"Failed to get source file: {e}")
            return None
        
        # Convert absolute path to relative path from repo root
        # Go up two levels from docs/source to get to repo root
        base_path = os.path.abspath('../..')
        print(f"Base path (repo root): {base_path}")
        
        # Check if source is in the repo checkout
        if source_file.startswith(base_path):
            rel_path = os.path.relpath(source_file, base_path)
            print(f"Relative path from repo: {rel_path}")
        # If not, check if it's in site-packages (non-editable install)
        elif 'site-packages' in source_file:
            print("File is in site-packages, attempting to map to repo")
            # Extract the path after site-packages/
            parts = source_file.split('site-packages/')
            if len(parts) > 1:
                # Get the package-relative path (e.g., mdvcontainment/mda_containment.py)
                rel_path = parts[1]
                print(f"Mapped to relative path: {rel_path}")
            else:
                print("Could not extract path from site-packages")
                return None
        else:
            print(f"Source file not in repo or site-packages")
            return None
        
        # Try to get line number
        try:
            source, lineno = inspect.getsourcelines(obj)
            linespec = f"#L{lineno}"
            print(f"Line number: {lineno}")
        except Exception as e:
            print(f"Failed to get line number: {e}")
            linespec = ""
        
        # Build GitHub URL
        url = f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{rel_path}{linespec}"
        print(f"Generated URL: {url}")
        return url
        
    except Exception as e:
        print(f"!!! Exception in main try block: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to original behavior
        print("=== Falling back to original behavior ===")
        module = info['module']
        filename = module.replace('.', '/')
        base_path = os.path.abspath('../..')
        pyx_path = os.path.join(base_path, f'{filename}.pyx')
        
        print(f"Checking for .pyx file: {pyx_path}")
        if os.path.exists(pyx_path):
            url = f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{filename}.pyx"
            print(f"Fallback URL: {url}")
            return url
        
        print("No .pyx file found, returning None")
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
html_css_files = ['custom.css']
html_show_sphinx = True
html_show_sourcelink = True