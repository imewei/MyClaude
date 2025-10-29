# Plugin Marketplace Documentation

This directory contains the Sphinx documentation for the Claude Code Plugin Marketplace.

## Quick Start

### Prerequisites

- Python >= 3.12
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Building Documentation

Build the HTML documentation:
```bash
make html
```

The generated documentation will be available in `_build/html/index.html`.

### Live Development Server

For live-reloading during development:
```bash
make livehtml
# or
make autobuild
```

This will start a development server at http://localhost:8000 that automatically rebuilds the documentation when you save changes.

### Other Build Targets

- `make clean` - Remove all built documentation
- `make dirhtml` - Build HTML with separate directories per page
- `make singlehtml` - Build all documentation as a single HTML page

## Documentation Structure

```
docs/
├── conf.py                  # Sphinx configuration
├── index.rst                # Main landing page
├── _static/                 # Static assets (CSS, images, etc.)
├── _templates/              # Custom Sphinx templates
├── categories/              # Category landing pages
├── plugins/                 # Individual plugin documentation pages
└── guides/                  # Quick-start guides and tutorials
```

## Writing Documentation

### RST Format

This documentation uses reStructuredText (RST) format. See the [RST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for syntax reference.

### Code Examples

Use the `.. code-block::` directive for code examples:

```rst
.. code-block:: python

   def hello():
       print("Hello, world!")
```

### Cross-References

Link to other documentation pages using the `:doc:` directive:

```rst
:doc:`/plugins/python-development`
:doc:`/categories/scientific-computing`
```

## Testing

Run the Sphinx infrastructure tests:
```bash
pytest test-corpus/sphinx-infrastructure/test_sphinx_build.py -v
```

## CI/CD

Documentation is automatically built and validated on every commit via GitHub Actions. See `.github/workflows/sphinx-docs.yml` for details.

## Versioning

This documentation uses sphinx-multiversion for versioned documentation support. Configuration is in `conf.py`.

## Theme

We use the Read the Docs theme (`sphinx_rtd_theme`). Theme options are configured in `conf.py`.

## Extensions

The following Sphinx extensions are enabled:

- `sphinx.ext.autodoc` - Auto-generate documentation from docstrings
- `sphinx.ext.napoleon` - Support for Google/NumPy style docstrings
- `sphinx.ext.intersphinx` - Link to other project documentation
- `sphinx.ext.viewcode` - Add links to highlighted source code
- `sphinx_copybutton` - Add copy button to code blocks

## Troubleshooting

### Build Errors

If you encounter build errors, try:
```bash
make clean
make html
```

### Missing Dependencies

If extensions fail to load, ensure all requirements are installed:
```bash
pip install -r requirements.txt
```

## Contributing

When adding new documentation:

1. Write your content in RST format
2. Add it to the appropriate directory (`plugins/`, `categories/`, `guides/`)
3. Update the relevant `toctree` directive to include your new page
4. Build and verify: `make html`
5. Test with live server: `make livehtml`

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [RST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
