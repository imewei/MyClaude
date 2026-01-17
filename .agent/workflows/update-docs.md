---
description: Update Sphinx docs, README, API documentation with AST-based code analysis
triggers:
- /update-docs
- workflow for update docs
version: 1.0.7
category: code-documentation
command: /update-docs
argument-hint: '[--full] [--sphinx] [--readme] [--api] [--format=<type>]'
color: blue
execution_modes:
  quick: 15-20min
  standard: 30-45min
  comprehensive: 60-90min
allowed-tools: [Read, Task, Bash]
---


# Documentation Update

$ARGUMENTS

Flags: `--full`, `--sphinx`, `--readme`, `--api`, `--format=sphinx|mkdocs`, `--dry-run`

## Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 15-20min | README + critical updates |
| Standard | 30-45min | Full AST analysis + all docs |
| Comprehensive | 60-90min | Everything + CI/CD automation |

## Process

1. **Project Analysis**:
```bash
git log --since="2 weeks ago" --stat | head -200
find . -type f \( -name "*.py" -o -name "*.js" \) | head -100
find . -type f \( -name "*.md" -o -name "*.rst" \) | head -50
```

**Inventory**: README.md/rst, docs/conf.py, docs/index.rst, API.md, package.json, pyproject.toml

2. **AST-Based Code Analysis**:

**Python**:
- Parse all .py files with AST
- Extract modules, classes, functions, docstrings
- Identify type hints and parameters
- Compare with existing Sphinx autodoc

**JavaScript/TypeScript**:
- Parse .js/.ts files
- Extract exports, interfaces, types
- Collect JSDoc comments
- Identify React components and props

3. **Gap Analysis**:
   For each code element:
   - ✓ Documented in Sphinx/API docs?
   - ✓ Signature up-to-date?
   - ✓ All parameters documented?
   - ✓ Return types documented?
   - ✓ Examples provided?

   **Gap Report**:
   - UNDOCUMENTED: [classes, functions, modules]
   - OUTDATED: [signature changes, deprecated]
   - INCOMPLETE: [missing params, returns, examples]

4. **Sphinx Updates**:
```rst
Module Name
===========

.. automodule:: module_name
   :members:
   :undoc-members:
   :show-inheritance:
```

**Build Verification**:
```bash
cd docs && make html  # No errors
make linkcheck        # No broken links
```

5. **README Optimization**:
   1. Project overview (2-3 sentences)
   2. Badges (CI, coverage, version)
   3. Features (bulleted)
   4. Installation (pip, npm, source)
   5. Quick start (simple example)
   6. Documentation links
   7. Configuration (env vars table)
   8. Development setup
   9. Contributing
   10. License

6. **API Documentation** (REST APIs):
   - Extract endpoints from route decorators
   - Generate OpenAPI spec
   - Create code examples (Python, JS, cURL)
   - Document authentication
   - Add request/response examples

7. **Automation** (Comprehensive):

**GitHub Actions**:
```yaml
name: Generate Docs
on: {push: {paths: ['src/**', 'docs/**']}}
jobs:
  generate:
    steps: [{run: sphinx-build -b html docs/source docs/build}]
```

**Pre-commit**:
```yaml
repos:
  - repo: local
    hooks:
      - id: doc-coverage
        entry: interrogate --fail-under 80
```

## Output

| Mode | Files |
|------|-------|
| Always | README.md |
| If Sphinx | docs/index.rst, docs/api/, docs/conf.py |
| If API | API.md or openapi.json |
| Comprehensive | + .github/workflows/docs.yml, .pre-commit-config.yaml |

**Summary**:
```markdown
## Changes
- Created: [new]
- Updated: [modified]

## Coverage
- Before: X% → After: Y% (+Z%)

## Next Steps
1. Review docs
2. Add custom examples
3. Set up CI/CD
```

## Success

- ✅ All code elements have docstrings
- ✅ Sphinx builds without errors
- ✅ README comprehensive
- ✅ API documentation complete
- ✅ No broken links
- ✅ Coverage >90%

## External Docs

- `ast-parsing-implementation.md` (~400 lines)
- `sphinx-optimization.md` (~350 lines)
- `api-documentation-templates.md` (~550 lines)
- `documentation-automation.md` (~300 lines)
