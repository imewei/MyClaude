---
description: Comprehensively update and optimize Sphinx docs, README, and related codebase documentation with AST-based content extraction
argument-hint: [--full] [--sphinx] [--readme] [--api] [--format=<type>]
color: blue
agents:
  primary:
    - documentation-architect
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "sphinx|numpy|scipy|pandas|matplotlib|scientific.*computing" OR files "*.ipynb|*.rst|docs/conf.py"
    - agent: fullstack-developer
      trigger: files "package.json|src/components/" OR pattern "react|vue|angular|typescript"
    - agent: systems-architect
      trigger: complexity > 50 OR pattern "architecture|design.*pattern"
    - agent: code-quality
      trigger: pattern "quality|testing|coverage"
    - agent: visualization-interface
      trigger: pattern "diagram|flow.*chart|architecture.*diagram|api.*diagram" OR files "*diagram*|docs/images/|docs/figures/"
  orchestrated: true
---

# Comprehensive Documentation Update & Optimization

## Phase 1: Project Intelligence Gathering

### 1.1 Repository State Analysis
!`git status --porcelain`

### 1.2 Recent Changes Detection
!`git log --since="2 weeks ago" --pretty=format:"%h - %an, %ar : %s" --stat | head -200`

### 1.3 Changed Files Analysis
!`git diff HEAD~10 --name-only | head -50`

### 1.4 Recent Commits Detail
!`git log --oneline -20`

### 1.5 Project Structure Discovery
!`find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" \) ! -path "*/node_modules/*" ! -path "*/.venv/*" ! -path "*/venv/*" ! -path "*/.git/*" ! -path "*/dist/*" ! -path "*/build/*" | head -100`

### 1.6 Documentation Files Inventory
!`find . -type f \( -name "*.md" -o -name "*.rst" -o -name "*.txt" -o -name "*.adoc" \) ! -path "*/node_modules/*" ! -path "*/.venv/*" ! -path "*/.git/*" | head -50`

### 1.7 Configuration Files
!`find . -maxdepth 2 -name "package.json" -o -name "setup.py" -o -name "pyproject.toml" -o -name "Cargo.toml" -o -name "go.mod" 2>/dev/null`

## Phase 2: Existing Documentation State

### 2.1 README Files
@README.md
@README.rst
@readme.md

### 2.2 Sphinx Documentation Discovery
!`find . -path "*/docs/conf.py" -o -path "*/doc/conf.py" -o -path "*/documentation/conf.py" 2>/dev/null`

### 2.3 Sphinx Configuration (if exists)
@docs/conf.py
@doc/conf.py
@documentation/conf.py

### 2.4 Sphinx Index/Main Files
@docs/index.rst
@docs/index.md
@doc/index.rst

### 2.5 API Documentation
@docs/api.rst
@docs/api/index.rst
@API.md

### 2.6 CHANGELOG Analysis
@docs/CHANGELOG.md
@CHANGES.md
@HISTORY.md

### 2.7 Contributing Guidelines
@CONTRIBUTING.md
@CONTRIBUTE.md

### 2.8 Project Metadata
@package.json
@setup.py
@pyproject.toml
@Cargo.toml

## Phase 3: AST-Based Code Analysis

**CRITICAL**: Use AST parsing to extract comprehensive code structure. For each language in the project:

### 3.1 Python AST Extraction
For Python files, extract:
- **Modules**: All `.py` files with their module docstrings
- **Classes**: Class names, docstrings, inheritance, methods
- **Functions**: Function signatures, docstrings, parameters, return types
- **Constants/Variables**: Module-level constants
- **Type Hints**: Full type annotation information
- **Decorators**: Applied decorators and their purposes

**Approach**:
1. Parse all Python files using AST
2. Extract docstrings (Google, NumPy, reStructuredText formats)
3. Build comprehensive module/class/function hierarchy
4. Compare with existing Sphinx autodoc directives
5. Identify undocumented or poorly documented items

### 3.2 JavaScript/TypeScript AST Extraction
For JS/TS files, extract:
- **Modules/Exports**: All exported components, functions, classes
- **Interfaces/Types**: TypeScript type definitions
- **Classes**: Class structures with JSDoc comments
- **Functions**: Function signatures with JSDoc
- **React Components**: Props, state, lifecycle methods
- **API Routes**: Endpoint definitions and handlers

**Approach**:
1. Parse JS/TS files using appropriate AST parser
2. Extract JSDoc comments and TypeScript type info
3. Identify exported APIs and their documentation
4. Cross-reference with API documentation

### 3.3 Go AST Extraction (if applicable)
- **Packages**: Package documentation
- **Types/Structs**: Exported types
- **Functions**: Exported functions with godoc comments
- **Interfaces**: Interface definitions

### 3.4 Rust AST Extraction (if applicable)
- **Crates/Modules**: Module structure
- **Structs/Enums**: Type definitions
- **Traits**: Trait definitions
- **Functions**: Public functions with doc comments

## Phase 4: Documentation Gap Analysis

### 4.1 Cross-Reference Analysis
Compare AST-extracted code structure with documentation:

**For each code element (class/function/module):**
- ✓ Is it documented in Sphinx/API docs?
- ✓ Is the documentation up-to-date with current signature?
- ✓ Are all parameters documented?
- ✓ Are return types documented?
- ✓ Are exceptions/errors documented?
- ✓ Are examples provided?
- ✓ Are there usage guidelines?

**Create Gap Report**:
```
UNDOCUMENTED ITEMS:
- Classes: [list classes without docstrings or Sphinx entries]
- Functions: [list functions without proper documentation]
- Modules: [list modules lacking documentation]
- API Endpoints: [list undocumented endpoints]

OUTDATED DOCUMENTATION:
- Signature Changes: [functions with changed signatures]
- Deprecated Items: [items marked deprecated but still in docs]
- Removed Items: [documented items no longer in code]

INCOMPLETE DOCUMENTATION:
- Missing Parameters: [functions with undocumented params]
- Missing Return Docs: [functions without return documentation]
- Missing Examples: [complex functions without examples]
```

### 4.2 Sphinx-Specific Analysis
If Sphinx documentation exists:
- Check for broken references (`:ref:`, `:doc:`, `:class:`, etc.)
- Verify all autodoc directives resolve correctly
- Identify missing `.. automodule::`, `.. autoclass::` directives
- Check cross-reference consistency
- Verify toctree completeness

### 4.3 README Completeness Check
Verify README contains:
- ✓ Project description and purpose
- ✓ Installation instructions (all platforms)
- ✓ Quick start guide with examples
- ✓ Core features overview
- ✓ API/CLI usage examples
- ✓ Configuration options
- ✓ Dependencies and requirements
- ✓ Contributing guidelines link
- ✓ License information
- ✓ Links to full documentation
- ✓ Badges (CI, coverage, version, etc.)

## Phase 5: Intelligent Documentation Generation

### 5.1 Sphinx Documentation Updates

**If Sphinx exists**, systematically update:

#### 5.1.1 Update API Reference
- Generate/update `.. automodule::` directives for all Python modules
- Add missing `.. autoclass::` for all classes
- Add missing `.. autofunction::` for all functions
- Ensure proper `:members:`, `:undoc-members:`, `:show-inheritance:` options
- Add `:noindex:` where appropriate to avoid duplicates

#### 5.1.2 Update/Create Module Documentation
For each major module/package:
```rst
Module Name
===========

.. automodule:: module_name
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------
[High-level description extracted from module docstring]

Key Classes
-----------
.. autosummary::
   :toctree: generated/

   ClassName1
   ClassName2

Functions
---------
.. autosummary::
   :toctree: generated/

   function1
   function2

Examples
--------
[Add usage examples]
```

#### 5.1.3 Update Index/TOC
- Ensure `docs/index.rst` has complete toctree
- Add missing sections to navigation
- Organize by logical grouping (API, Tutorials, Examples, etc.)

#### 5.1.4 Create Missing Sections
If absent, create:
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start tutorial
- `api/index.rst` - API reference landing page
- `examples/index.rst` - Examples and tutorials
- `changelog.rst` - Link to CHANGELOG
- `contributing.rst` - Contributor guide

#### 5.1.5 Sphinx Configuration Optimization
Update `conf.py` to ensure:
- All necessary extensions are enabled (autodoc, napoleon, viewcode, intersphinx)
- Napoleon configured for Google/NumPy style docstrings
- Intersphinx mapping for external libraries
- Proper theme configuration
- Auto-generated API docs setup

### 5.2 README Optimization

**Generate comprehensive README** with:

```markdown
# Project Name

[Badges: CI/CD, Coverage, Version, License, etc.]

## Overview
[Compelling 2-3 sentence project description]

## Features
- ✨ Feature 1
- 🚀 Feature 2
- 🎯 Feature 3

## Installation

### Prerequisites
- Requirement 1
- Requirement 2

### Using pip/npm/cargo
\`\`\`bash
[installation command]
\`\`\`

### From Source
\`\`\`bash
git clone [repo]
cd [project]
[setup commands]
\`\`\`

## Quick Start

\`\`\`[language]
[Simple, compelling example showing core functionality]
\`\`\`

## Documentation

Full documentation: [link to docs]

- [Installation Guide](docs/installation.rst)
- [API Reference](docs/api/index.rst)
- [Examples & Tutorials](docs/examples/index.rst)

## Usage Examples

### Example 1: [Common Use Case]
\`\`\`[language]
[code example]
\`\`\`

### Example 2: [Another Use Case]
\`\`\`[language]
[code example]
\`\`\`

## Configuration

[Configuration options if applicable]

## Development

### Setup Development Environment
\`\`\`bash
[setup commands]
\`\`\`

### Running Tests
\`\`\`bash
[test commands]
\`\`\`

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## License
[License info]

## Acknowledgments
[Credits, inspirations, etc.]
```

### 5.3 API Documentation Enhancement

If API endpoints exist (REST, GraphQL, etc.):

#### Create/Update API.md
```markdown
# API Reference

## Authentication
[Auth details]

## Base URL
\`https://api.example.com/v1\`

## Endpoints

### [Resource Name]

#### GET /resource
[Description]

**Parameters:**
- \`param1\` (type) - Description
- \`param2\` (type) - Description

**Response:**
\`\`\`json
{
  "example": "response"
}
\`\`\`

**Example:**
\`\`\`bash
curl -X GET https://api.example.com/v1/resource
\`\`\`

[Continue for all endpoints...]
```

### 5.4 CHANGELOG Generation/Update

Based on git history analysis:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- [Recent additions from git log]

### Changed
- [Recent changes from git log]

### Fixed
- [Recent fixes from git log]

## [Version X.Y.Z] - YYYY-MM-DD
[Continue with version history...]
```

### 5.5 Code Examples & Tutorials

Create `docs/examples/` directory with:
- Basic usage examples
- Advanced use cases
- Integration examples
- Troubleshooting guides

Each example should be:
- **Runnable**: Complete, working code
- **Annotated**: Inline comments explaining key parts
- **Realistic**: Actual use cases, not toy examples

## Phase 6: Optimization & Quality Assurance

### 6.1 Documentation Quality Checks

**Verify all updates:**
- ✓ No broken links (internal or external)
- ✓ All code examples are syntactically valid
- ✓ Consistent terminology throughout
- ✓ Proper formatting (RST/Markdown)
- ✓ No orphaned files or sections
- ✓ Proper cross-references
- ✓ Version numbers are current

### 6.2 Sphinx Build Verification

If Sphinx exists, ensure:
```bash
cd docs/
make clean
make html  # Should build without errors or warnings
```

Check for:
- ✗ No build errors
- ✗ No warnings about missing references
- ✗ No autodoc import errors
- ✗ All pages render correctly

### 6.3 Documentation Coverage Report

Generate coverage report:
```
DOCUMENTATION COVERAGE SUMMARY
================================

Python Code Coverage:
- Modules documented: X/Y (Z%)
- Classes documented: X/Y (Z%)
- Functions documented: X/Y (Z%)
- With examples: X/Y (Z%)

JavaScript/TypeScript Coverage:
- Components documented: X/Y (Z%)
- Functions documented: X/Y (Z%)
- Interfaces documented: X/Y (Z%)

Overall Documentation Health: [Score/Grade]

TOP PRIORITIES FOR IMPROVEMENT:
1. [Most critical gap]
2. [Second priority]
3. [Third priority]
```

### 6.4 Accessibility & Readability

Ensure documentation is:
- **Clear**: Written for target audience (developers, users, etc.)
- **Concise**: No unnecessary verbosity
- **Organized**: Logical flow and structure
- **Searchable**: Good headings, keywords, index entries
- **Accessible**: Proper alt text for images, semantic markup

## Phase 7: Final Deliverables

### 7.1 Files to Update/Create

Based on analysis, update or create:
- ✅ `README.md` (or `README.rst`)
- ✅ `docs/index.rst` (Sphinx main page)
- ✅ `docs/api/` (API reference with autodoc)
- ✅ `docs/installation.rst`
- ✅ `docs/quickstart.rst`
- ✅ `docs/examples/` (examples directory)
- ✅ `docs/CHANGELOG.md`
- ✅ `CONTRIBUTING.md` (if missing)
- ✅ `API.md` (if REST/GraphQL API exists)
- ✅ `docs/conf.py` (Sphinx configuration)

### 7.2 Summary Report

Provide comprehensive summary:
```markdown
# Documentation Update Summary

## Changes Made

### New Documentation
- Created: [list new files]
- Added sections: [list new sections]

### Updated Documentation
- README: [summary of changes]
- Sphinx Docs: [summary of changes]
- API Docs: [summary of changes]
- Examples: [summary of changes]

### Documentation Coverage Improvement
- Before: X% documented
- After: Y% documented
- Improvement: +Z%

## Identified Gaps

### Critical (Must Address)
1. [Gap 1]
2. [Gap 2]

### Important (Should Address)
1. [Gap 1]
2. [Gap 2]

### Nice to Have
1. [Gap 1]
2. [Gap 2]

## Next Steps

1. Review generated documentation
2. Add custom examples for complex features
3. Set up documentation CI/CD
4. Consider adding:
   - Interactive documentation (Swagger/OpenAPI)
   - Video tutorials
   - Architecture diagrams
   - Deployment guides

## Build Instructions

To build and view updated Sphinx docs:
\`\`\`bash
cd docs/
pip install -r requirements.txt  # or requirements-docs.txt
make html
open _build/html/index.html  # or start docs/_build/html/index.html on Windows
\`\`\`
```

## Execution Instructions

### Argument Handling: $ARGUMENTS

**Supported flags:**
- `--full`: Complete documentation overhaul (most comprehensive)
- `--sphinx`: Focus on Sphinx documentation only
- `--readme`: Focus on README update only
- `--api`: Focus on API documentation only
- `--format=sphinx|mkdocs|hugo`: Specify documentation format
- `--no-ast`: Skip AST analysis (faster but less comprehensive)
- `--dry-run`: Analyze and report without making changes

### Default Behavior (no arguments)
Comprehensive update of all documentation types with full AST analysis.

### Intelligence Level: ULTRATHINK

This command operates with maximum intelligence:
- **Adaptive**: Automatically detects project type and documentation needs
- **Comprehensive**: Covers all documentation formats and types
- **AST-Driven**: Uses code parsing for accurate, complete coverage
- **Gap-Aware**: Identifies and prioritizes documentation gaps
- **Quality-Focused**: Ensures consistency, accuracy, and completeness
- **Context-Aware**: Understands project evolution via git analysis

### Success Criteria

Documentation update is complete when:
- ✅ All code elements have docstrings/comments
- ✅ Sphinx builds without errors/warnings
- ✅ README is comprehensive and current
- ✅ API documentation covers all endpoints
- ✅ Examples are working and illustrative
- ✅ No broken links or references
- ✅ Documentation coverage > 90%
- ✅ All recent changes are reflected in docs

## Advanced Features

### AST Parsing Implementation Guidance

**For Python:**
```python
import ast
import inspect

# Parse module
with open('module.py') as f:
    tree = ast.parse(f.read())

# Extract classes, functions
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        # Extract class info
        class_name = node.name
        docstring = ast.get_docstring(node)
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    elif isinstance(node, ast.FunctionDef):
        # Extract function info
        func_name = node.name
        docstring = ast.get_docstring(node)
        args = [arg.arg for arg in node.args.args]
```

**For TypeScript/JavaScript:**
Use appropriate parser (e.g., @babel/parser, typescript compiler API) to extract:
- Exported functions/classes
- JSDoc comments
- Type definitions
- React component props

### Sphinx Autodoc Configuration

Ensure `conf.py` includes:
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Auto-generate API docs
autosummary_generate = True
```

### Documentation Testing

After update, validate with:
```bash
# Sphinx linkcheck
cd docs && make linkcheck

# Docstring coverage (Python)
interrogate -v .

# Build verification
cd docs && make html
```

## Final Notes

This command provides **ULTRATHINK-level** documentation intelligence by:
1. **Understanding** code structure through AST parsing
2. **Detecting** gaps via comprehensive cross-referencing
3. **Generating** accurate, complete documentation
4. **Optimizing** for readability and maintainability
5. **Verifying** quality through automated checks

The result is documentation that is **accurate**, **comprehensive**, **maintainable**, and **developer-friendly**.
