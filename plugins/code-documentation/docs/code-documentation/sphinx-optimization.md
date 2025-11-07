# Sphinx Optimization

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: Sphinx configuration, autodoc setup, and build optimization

## Overview

Comprehensive Sphinx documentation setup with autodoc, Napoleon, intersphinx, and modern themes for professional documentation generation.

## Complete Sphinx Configuration

### conf.py Template

```python
# docs/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'My Project'
copyright = '2024, Author Name'
author = 'Author Name'
version = '1.0.0'
release = '1.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'myst_parser',  # For Markdown support
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Napoleon settings (Google/NumPy docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# Templates and static files
templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'  # or 'furo', 'pydata_sphinx_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'
html_show_sourcelink = True
html_show_sphinx = False

# MyST Markdown parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]
```

## Autodoc Directives

### Module Documentation

```rst
mymodule
========

.. automodule:: mymodule
   :members:
   :undoc-members:
   :show-inheritance:
   :synopsis: Brief module description

Overview
--------
This module provides...

Key Classes
-----------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ClassName1
   ClassName2

Functions
---------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   function1
   function2

Examples
--------
.. code-block:: python

   import mymodule
   result = mymodule.function1()
```

### Class Documentation

```rst
.. autoclass:: mymodule.MyClass
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
   :exclude-members: __weakref__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      method1
      method2
      method3

   .. rubric:: Examples

   .. code-block:: python

      obj = MyClass()
      obj.method1()
```

## Index Structure

### index.rst Template

```rst
Welcome to Project Documentation
=================================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules
   api/classes
   api/functions

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/basic
   examples/advanced
   examples/integrations

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide:

   contributing
   architecture
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

## Build Automation

### Makefile

```makefile
# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/api/generated/

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

coverage:
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
```

## Theme Customization

### Custom CSS

```css
/* docs/_static/custom.css */

:root {
    --primary-color: #2962ff;
    --code-bg: #f5f5f5;
}

.rst-content .highlight {
    background: var(--code-bg);
    border-radius: 4px;
    padding: 1em;
}

.rst-content code {
    font-family: 'Fira Code', monospace;
    font-size: 0.9em;
}

.wy-side-nav-search {
    background-color: var(--primary-color);
}
```

## Build Verification

### Quality Checks Script

```python
# scripts/check_docs.py

import subprocess
import sys

def run_command(cmd, description):
    """Run command and check for errors"""
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ {description} failed:")
        print(result.stderr)
        return False
    else:
        print(f"✓ {description} passed")
        return True

def main():
    checks = [
        ("cd docs && make clean", "Cleaning old build"),
        ("cd docs && make html", "Building HTML documentation"),
        ("cd docs && make linkcheck", "Checking links"),
        ("interrogate src/ --fail-under 80", "Checking docstring coverage"),
    ]

    results = []
    for cmd, desc in checks:
        results.append(run_command(cmd, desc))

    if all(results):
        print("\n✓ All documentation checks passed!")
        sys.exit(0)
    else:
        print("\n✗ Some documentation checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Usage Examples

### Building Documentation

```bash
# Clean build
cd docs
make clean
make html

# Open in browser
open build/html/index.html

# Check for broken links
make linkcheck

# Check coverage
make coverage

# Live reload during development
make livehtml
```

### Auto-generating API Docs

```bash
# Generate module stubs
sphinx-apidoc -o docs/source/api src/ --force --separate

# Build with autosummary
sphinx-build -b html docs/source docs/build
```
