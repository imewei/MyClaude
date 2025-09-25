---
description: Generate comprehensive Python package documentation for PyPI and ReadTheDocs
category: documentation-changelogs
allowed-tools: Read, Edit, Write, Glob, Bash
---

# Python Documentation Generator for PyPI & ReadTheDocs

Generate comprehensive, professional documentation for Python packages that meets PyPI and ReadTheDocs standards.

## Process:

### 1. **Project Analysis & Discovery**
   - **Package Structure**: Analyze `pyproject.toml`, `setup.py`, `setup.cfg` for metadata
   - **Module Discovery**: Scan source code for modules, classes, and functions
   - **Dependency Analysis**: Extract requirements and optional dependencies
   - **Configuration Files**: Identify `docs/`, `README.md`, `CHANGELOG.md`, `LICENSE`
   - **API Documentation**: Analyze docstrings and type hints
   - **Test Coverage**: Review test structure and coverage reports

### 2. **PyPI-Compliant Documentation Structure**

#### **A. Core Files Creation/Update:**
   ```
   README.md                    # PyPI landing page
   CHANGELOG.md                 # Version history
   CONTRIBUTING.md              # Contribution guidelines
   LICENSE                      # License file
   MANIFEST.in                  # Package data inclusion
   ```

#### **B. Sphinx Documentation Setup:**
   ```
   docs/
   ├── conf.py                  # Sphinx configuration
   ├── index.rst                # Main documentation index
   ├── installation.rst         # Installation guide
   ├── quickstart.rst          # Quick start tutorial
   ├── api/                     # API documentation
   │   ├── index.rst
   │   └── modules.rst
   ├── tutorials/               # Tutorials and examples
   ├── changelog.rst            # Change log
   ├── contributing.rst         # Contribution guide
   └── _static/                 # Static assets
   ```

### 3. **Documentation Content Generation**

#### **A. README.md Enhancement:**
   - **Project badges**: PyPI version, build status, coverage, downloads
   - **Description**: Clear, compelling project description
   - **Features**: Key features and benefits
   - **Installation**: Multiple installation methods (pip, conda, source)
   - **Quick Start**: Minimal working example
   - **Documentation Links**: Links to full docs
   - **License and Contributing**: Clear legal and contribution info

#### **B. API Documentation:**
   - **Auto-generated docs**: Use Sphinx autodoc for docstring extraction
   - **Type hints**: Document parameter and return types
   - **Usage examples**: Code examples for key functions/classes  
   - **Cross-references**: Link related functions and modules
   - **Inheritance diagrams**: Class hierarchy visualization

#### **C. User Guides:**
   - **Installation Guide**: Detailed installation for different platforms
   - **Tutorial Series**: Step-by-step learning progression
   - **How-To Guides**: Task-oriented documentation
   - **Reference Manual**: Complete API reference
   - **FAQ**: Common questions and troubleshooting

### 4. **ReadTheDocs Integration**

#### **A. Configuration Files:**
   ```yaml
   # .readthedocs.yaml
   version: 2
   build:
     os: ubuntu-22.04
     tools:
       python: "3.12"
   sphinx:
     configuration: docs/conf.py
   python:
     install:
       - requirements: docs/requirements.txt
       - method: pip
         path: .
   ```

#### **B. Sphinx Configuration (docs/conf.py):**
   - **Theme**: Use `sphinx-rtd-theme` or `furo` for modern look
   - **Extensions**: Include `sphinx.ext.autodoc`, `sphinx.ext.viewcode`, etc.
   - **API Documentation**: Auto-generate from docstrings
   - **Cross-references**: Enable intersphinx for external library links
   - **Search**: Configure search functionality

#### **C. Requirements Management:**
   ```
   docs/requirements.txt        # Documentation build dependencies
   ```

### 5. **Documentation Content Standards**

#### **A. Docstring Standards (NumPy/Google Style):**
   ```python
   def example_function(param1: str, param2: int = 0) -> bool:
       """Brief description of the function.
       
       Longer description explaining the function's purpose,
       behavior, and any important implementation details.
       
       Parameters
       ----------
       param1 : str
           Description of param1
       param2 : int, optional
           Description of param2, by default 0
           
       Returns
       -------
       bool
           Description of return value
           
       Raises
       ------
       ValueError
           When invalid input is provided
           
       Examples
       --------
       >>> example_function("hello", 5)
       True
       """
   ```

#### **B. Module Documentation:**
   - **Module docstrings**: Clear purpose and overview
   - **Public API**: Document what users should import/use
   - **Examples**: Working code examples
   - **Version info**: When features were added/changed

### 6. **Advanced Documentation Features**

#### **A. Interactive Examples:**
   - **Jupyter notebooks**: Tutorial notebooks in `docs/notebooks/`
   - **Binder integration**: Live executable examples
   - **Code execution**: Use `sphinx-execute-code` for tested examples
   - **Plots and figures**: Auto-generated visualizations

#### **B. Version Management:**
   - **Version scheme**: Semantic versioning (x.y.z)
   - **Changelog format**: Keep a Changelog standard
   - **API stability**: Document breaking changes
   - **Deprecation warnings**: Clear migration paths

#### **C. Internationalization (i18n):**
   - **Language support**: Configure for multiple languages if needed
   - **Locale directories**: Organize translation files
   - **Build configuration**: Multi-language ReadTheDocs setup

### 7. **Quality Assurance**

#### **A. Documentation Testing:**
   ```bash
   # Test documentation build
   sphinx-build -b html docs docs/_build/html
   
   # Test links
   sphinx-build -b linkcheck docs docs/_build/linkcheck
   
   # Test code examples
   sphinx-build -b doctest docs docs/_build/doctest
   ```

#### **B. Content Review:**
   - **Grammar and spelling**: Use automated checkers
   - **Code examples**: Ensure all examples work
   - **Screenshots**: Update UI screenshots if applicable
   - **Links**: Verify all external links are valid

#### **C. Accessibility:**
   - **Alt text**: For all images and diagrams
   - **Color contrast**: Ensure readable color schemes
   - **Navigation**: Clear, keyboard-accessible navigation
   - **Screen readers**: Semantic HTML structure

### 8. **Automation & CI/CD**

#### **A. GitHub Actions Workflow:**
   ```yaml
   name: Documentation
   on: [push, pull_request]
   jobs:
     docs:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Setup Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.12'
         - name: Install dependencies
           run: |
             pip install -r docs/requirements.txt
         - name: Build documentation
           run: |
             sphinx-build -b html docs docs/_build/html
         - name: Deploy to GitHub Pages
           if: github.ref == 'refs/heads/main'
           uses: peaceiris/actions-gh-pages@v3
   ```

#### **B. Pre-commit Hooks:**
   - **Documentation linting**: Check RST/Markdown syntax
   - **Link checking**: Verify external links
   - **Spelling**: Automated spell checking
   - **Format checking**: Consistent formatting

### 9. **PyPI Package Metadata**

#### **A. pyproject.toml Enhancement:**
   ```toml
   [project]
   name = "your-package"
   description = "Clear, compelling description"
   readme = "README.md"
   license = {text = "MIT"}
   authors = [{name = "Your Name", email = "you@example.com"}]
   keywords = ["keyword1", "keyword2", "keyword3"]
   classifiers = [
       "Development Status :: 4 - Beta",
       "Intended Audience :: Developers",
       "License :: OSI Approved :: MIT License",
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 3.12",
   ]
   
   [project.urls]
   Homepage = "https://github.com/username/repo"
   Documentation = "https://your-package.readthedocs.io/"
   Repository = "https://github.com/username/repo.git"
   Changelog = "https://github.com/username/repo/blob/main/CHANGELOG.md"
   ```

#### **B. Long Description:**
   - **README as description**: Use README.md content
   - **Content-Type**: Specify `text/markdown`
   - **Rich formatting**: Use badges, tables, code blocks
   - **Screenshots**: Include visual examples if applicable

## Documentation Checklist:

### **Essential Files:**
- [ ] README.md with badges and examples
- [ ] CHANGELOG.md following Keep a Changelog
- [ ] LICENSE file
- [ ] CONTRIBUTING.md with development setup
- [ ] docs/conf.py with proper Sphinx configuration
- [ ] .readthedocs.yaml configuration

### **Content Quality:**
- [ ] All public APIs have docstrings
- [ ] Type hints for parameters and returns
- [ ] Working code examples in docstrings
- [ ] Installation instructions tested
- [ ] Tutorial covers main use cases
- [ ] FAQ addresses common issues

### **Technical Standards:**
- [ ] Sphinx documentation builds without errors
- [ ] All links are valid (linkcheck passes)
- [ ] Code examples execute correctly (doctest passes)
- [ ] Mobile-responsive design
- [ ] Search functionality works
- [ ] Version selector functional

### **PyPI Compliance:**
- [ ] Package metadata complete in pyproject.toml
- [ ] README renders correctly on PyPI
- [ ] All required classifiers included
- [ ] Project URLs point to correct resources
- [ ] License clearly specified

This comprehensive approach ensures your Python package documentation meets professional standards and provides an excellent user experience on both PyPI and ReadTheDocs platforms.