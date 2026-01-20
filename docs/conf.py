import os
import sys

# Add custom extensions to path
sys.path.insert(0, os.path.abspath('_ext'))

project = 'Claude Code Plugin Marketplace'
copyright = '2026, DeepMind & Anthropic'
author = 'DeepMind & Anthropic'
release = '2.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'plugin_directives',  # Our custom extension
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Options for HTML output
html_theme = 'furo'  # Using Furo as per previous setup
html_static_path = ['_static']
html_title = "Claude Code Plugins"

# Custom styling could go here, but we'll stick to defaults for now
