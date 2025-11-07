---
version: "1.0.3"
category: "code-documentation"
command: "/update-docs"
description: Comprehensively update and optimize Sphinx docs, README, and related codebase documentation with AST-based content extraction
argument-hint: [--full] [--sphinx] [--readme] [--api] [--format=<type>]
color: blue
execution_modes:
  quick: "15-20 minutes - README and critical doc updates only"
  standard: "30-45 minutes - Comprehensive doc update with AST analysis"
  comprehensive: "60-90 minutes - Full documentation overhaul with CI/CD setup"
agents:
  primary:
    - docs-architect
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "sphinx|numpy|scipy|pandas|matplotlib|scientific.*computing" OR files "*.ipynb|*.rst|docs/conf.py"
    - agent: fullstack-developer
      trigger: files "package.json|src/components/" OR pattern "react|vue|angular|typescript"
    - agent: systems-architect
      trigger: complexity > 50 OR pattern "architecture|design.*pattern"
    - agent: visualization-interface
      trigger: pattern "diagram|flow.*chart|architecture.*diagram" OR files "*diagram*|docs/images/"
  orchestrated: true
---

# Comprehensive Documentation Update & Optimization

Systematically update Sphinx documentation, README, API docs using AST-based code analysis to ensure complete, accurate documentation coverage.

## Execution Modes

| Mode | Time | Scope | Output |
|------|------|-------|--------|
| **quick** | 15-20 min | README + critical updates | Updated README, quick fixes |
| **standard** (default) | 30-45 min | Full AST analysis + docs | README, Sphinx, API docs |
| **comprehensive** | 60-90 min | Everything + CI/CD | Complete doc site + automation |

## Quick Reference Documentation

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| AST Parsing | [ast-parsing-implementation.md](../docs/code-documentation/ast-parsing-implementation.md) | ~400 |
| Sphinx Setup | [sphinx-optimization.md](../docs/code-documentation/sphinx-optimization.md) | ~350 |
| API Templates | [api-documentation-templates.md](../docs/code-documentation/api-documentation-templates.md) | ~550 |
| Automation | [documentation-automation.md](../docs/code-documentation/documentation-automation.md) | ~300 |

**Total External Documentation**: ~1,600 lines of implementation details

## Arguments

$ARGUMENTS

**Supported flags**:
- `--full`: Complete documentation overhaul (same as comprehensive mode)
- `--sphinx`: Focus on Sphinx documentation only
- `--readme`: Focus on README update only
- `--api`: Focus on API documentation only
- `--format=sphinx|mkdocs|hugo`: Specify documentation format
- `--no-ast`: Skip AST analysis (faster but less comprehensive)
- `--dry-run`: Analyze and report without making changes

## Phase 1: Project Intelligence Gathering

### Repository Analysis

**Git status and recent changes**:
!`git status --porcelain`
!`git log --since="2 weeks ago" --pretty=format:"%h - %an, %ar : %s" --stat | head -200`
!`git diff HEAD~10 --name-only | head -50`

**Project structure**:
!`find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" \) ! -path "*/node_modules/*" ! -path "*/.venv/*" ! -path "*/.git/*" | head -100`

**Documentation inventory**:
!`find . -type f \( -name "*.md" -o -name "*.rst" \) ! -path "*/node_modules/*" ! -path "*/.git/*" | head -50`

**Configuration files**:
!`find . -maxdepth 2 -name "package.json" -o -name "setup.py" -o -name "pyproject.toml" 2>/dev/null`

### Existing Documentation State

**README files**: @README.md @README.rst

**Sphinx docs**: @docs/conf.py @docs/index.rst

**API docs**: @docs/api.rst @API.md

**Metadata**: @package.json @setup.py @pyproject.toml

## Phase 2: AST-Based Code Analysis

**Extract complete code structure** using AST parsing:

**For Python projects**:
1. Parse all `.py` files with AST
2. Extract modules, classes, functions with docstrings
3. Identify type hints and parameters
4. Build module/class/function hierarchy
5. Compare with existing Sphinx autodoc directives

**For JavaScript/TypeScript**:
1. Parse `.js`/`.ts` files
2. Extract exports, interfaces, types
3. Collect JSDoc comments
4. Identify React components and props

**Implementation**: See [ast-parsing-implementation.md](../docs/code-documentation/ast-parsing-implementation.md) for:
- `PythonASTExtractor` complete implementation
- `TypeScriptExtractor` with TypeScript compiler API
- `GoAnalyzer` and `RustExtractor` patterns
- Parameter, return type, and docstring extraction

## Phase 3: Documentation Gap Analysis

**Cross-reference code with documentation**:

### Gap Detection

For each code element:
- ✓ Is it documented in Sphinx/API docs?
- ✓ Is signature up-to-date?
- ✓ Are all parameters documented?
- ✓ Are return types documented?
- ✓ Are exceptions documented?
- ✓ Are examples provided?

**Create gap report**:
```
UNDOCUMENTED:
- Classes: [list]
- Functions: [list]
- Modules: [list]

OUTDATED:
- Signature changes: [list]
- Deprecated items: [list]

INCOMPLETE:
- Missing params: [list]
- Missing returns: [list]
- Missing examples: [list]
```

## Phase 4: Sphinx Documentation Updates

**For Python projects with Sphinx**:

### Update API Reference

1. **Generate autodoc directives** for all modules
2. **Add autoclass/autofunction** for missing items
3. **Update index.rst** with complete toctree
4. **Create module pages** with autosummary

**Template**:
```rst
Module Name
===========

.. automodule:: module_name
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: generated/

   ClassName1
   ClassName2
```

### Optimize conf.py

**Key settings**: See [sphinx-optimization.md](../docs/code-documentation/sphinx-optimization.md#complete-sphinx-configuration) for:
- Complete `conf.py` template with autodoc, Napoleon, intersphinx
- Autosummary configuration
- Theme customization
- Extension setup

### Build Verification

```bash
cd docs/
make clean
make html  # Should build without errors
make linkcheck  # Verify all links
```

## Phase 5: README Optimization

**Generate comprehensive README**:

### Sections to Include

1. **Project overview** (2-3 sentences)
2. **Badges** (CI/CD, coverage, version)
3. **Features** (bulleted list)
4. **Installation** (pip, npm, from source)
5. **Quick start** (simple example)
6. **Documentation** (links to full docs)
7. **Configuration** (env vars table)
8. **Development** (setup, testing)
9. **Contributing** (link to guidelines)
10. **License**

**Template**: See [documentation-automation.md](../docs/code-documentation/documentation-automation.md#readme-generation) for complete template code

## Phase 6: API Documentation Enhancement

**For projects with REST APIs**:

1. **Extract endpoints** from route decorators
2. **Generate OpenAPI spec** from code
3. **Create code examples** (Python, JavaScript, cURL)
4. **Document authentication** methods
5. **Add request/response examples**

**OpenAPI generation**: See [api-documentation-templates.md](../docs/code-documentation/api-documentation-templates.md#complete-openapi-30-template)

## Phase 7: Documentation Automation (Comprehensive mode)

**Set up CI/CD for documentation**:

### GitHub Actions Workflow

```yaml
name: Generate Documentation

on:
  push:
    paths:
      - 'src/**'
      - 'docs/**'

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Generate docs
      run: |
        sphinx-build -b html docs/source docs/build
    - name: Deploy
      uses: peaceiris/actions-gh-pages@v3
```

**Complete workflow**: See [documentation-automation.md](../docs/code-documentation/documentation-automation.md#github-actions-workflow)

### Pre-commit Hooks

```yaml
repos:
  - repo: local
    hooks:
      - id: doc-coverage
        entry: interrogate
        args: ['--fail-under', '80']
```

### Documentation Coverage Check

```python
from documentation_automation import DocCoverage

coverage = DocCoverage()
results = coverage.check_coverage('src/')
# Returns: function_coverage, class_coverage, missing_docs
```

**Implementation**: See [documentation-automation.md](../docs/code-documentation/documentation-automation.md#documentation-coverage-validation)

## Mode-Specific Execution

### Quick Mode (15-20 minutes)
**Actions**:
- Update README with recent changes
- Fix broken links
- Update version numbers
**Skip**: AST analysis, Sphinx rebuild, CI/CD

### Standard Mode (30-45 minutes) - DEFAULT
**Actions**:
- Full AST analysis
- Update README
- Update Sphinx docs with new autodoc directives
- Generate API docs
- Run build verification
**Skip**: CI/CD setup

### Comprehensive Mode (60-90 minutes)
**Actions**:
- Everything from standard mode
- Set up GitHub Actions workflow
- Configure pre-commit hooks
- Deploy documentation
- Generate coverage reports
- Create example documentation

## Output Deliverables

### Files Updated/Created

**Always**:
- ✅ README.md (or README.rst)

**If Sphinx exists**:
- ✅ docs/index.rst (main page)
- ✅ docs/api/ (API reference with autodoc)
- ✅ docs/conf.py (optimized configuration)

**If API exists**:
- ✅ API.md or docs/api/openapi.json
- ✅ Code examples

**Comprehensive mode**:
- ✅ .github/workflows/docs.yml
- ✅ .pre-commit-config.yaml
- ✅ docs/examples/
- ✅ Coverage report

### Summary Report

```markdown
# Documentation Update Summary

## Changes Made
- Created: [new files]
- Updated: [modified files]
- Added sections: [new sections]

## Coverage Improvement
- Before: X% documented
- After: Y% documented
- Improvement: +Z%

## Identified Gaps
### Critical
1. [Gap 1]
2. [Gap 2]

### Next Steps
1. Review generated documentation
2. Add custom examples
3. Set up CI/CD
```

## Quality Verification

### Build Checks

**Sphinx**:
```bash
cd docs && make html  # No errors/warnings
cd docs && make linkcheck  # No broken links
```

**Coverage**:
```bash
interrogate -v src/ --fail-under 80
```

### Success Criteria

✅ All code elements have docstrings/comments
✅ Sphinx builds without errors/warnings
✅ README is comprehensive and current
✅ API documentation covers all endpoints
✅ Examples are working and illustrative
✅ No broken links or references
✅ Documentation coverage > 90%
✅ All recent changes reflected in docs

## Integration with External Docs

This command leverages comprehensive external documentation for implementation details:

- **AST parsing**: Language-specific extractors in [ast-parsing-implementation.md](../docs/code-documentation/ast-parsing-implementation.md)
- **Sphinx setup**: Configuration templates in [sphinx-optimization.md](../docs/code-documentation/sphinx-optimization.md)
- **API docs**: OpenAPI templates in [api-documentation-templates.md](../docs/code-documentation/api-documentation-templates.md)
- **Automation**: CI/CD patterns in [documentation-automation.md](../docs/code-documentation/documentation-automation.md)

Focus on **accuracy**, **completeness**, and **maintainability** through AST-driven analysis and automated quality checks.
