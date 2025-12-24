---
version: "1.0.5"
category: "code-documentation"
command: "/update-docs"
description: Update Sphinx docs, README, and API documentation with AST-based code analysis
argument-hint: [--full] [--sphinx] [--readme] [--api] [--format=<type>]
color: blue
execution_modes:
  quick: "15-20 minutes"
  standard: "30-45 minutes"
  comprehensive: "60-90 minutes"
agents:
  primary:
    - docs-architect
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "sphinx|numpy|scipy|scientific"
    - agent: systems-architect
      trigger: complexity > 50 OR pattern "architecture"
  orchestrated: true
---

# Comprehensive Documentation Update

Update Sphinx, README, and API docs using AST-based code analysis.

## Arguments

$ARGUMENTS

**Flags:** `--full`, `--sphinx`, `--readme`, `--api`, `--format=sphinx|mkdocs`, `--dry-run`

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 15-20 min | README + critical updates |
| Standard (default) | 30-45 min | Full AST analysis + all docs |
| Comprehensive | 60-90 min | Everything + CI/CD automation |

---

## Phase 1: Project Analysis

### Gather Information
```bash
git log --since="2 weeks ago" --stat | head -200
find . -type f \( -name "*.py" -o -name "*.js" \) | head -100
find . -type f \( -name "*.md" -o -name "*.rst" \) | head -50
```

### Existing Docs Inventory
- README.md / README.rst
- docs/conf.py, docs/index.rst
- API.md, package.json, pyproject.toml

---

## Phase 2: AST-Based Code Analysis

### Python
1. Parse all `.py` files with AST
2. Extract modules, classes, functions, docstrings
3. Identify type hints and parameters
4. Compare with existing Sphinx autodoc

### JavaScript/TypeScript
1. Parse `.js`/`.ts` files
2. Extract exports, interfaces, types
3. Collect JSDoc comments
4. Identify React components and props

---

## Phase 3: Gap Analysis

### For Each Code Element
- ✓ Documented in Sphinx/API docs?
- ✓ Signature up-to-date?
- ✓ All parameters documented?
- ✓ Return types documented?
- ✓ Examples provided?

### Gap Report
```
UNDOCUMENTED: [classes, functions, modules]
OUTDATED: [signature changes, deprecated]
INCOMPLETE: [missing params, returns, examples]
```

---

## Phase 4: Sphinx Updates

### API Reference Generation
```rst
Module Name
===========

.. automodule:: module_name
   :members:
   :undoc-members:
   :show-inheritance:
```

### Build Verification
```bash
cd docs && make html  # No errors
make linkcheck        # No broken links
```

---

## Phase 5: README Optimization

### Required Sections
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

---

## Phase 6: API Documentation

### For REST APIs
1. Extract endpoints from route decorators
2. Generate OpenAPI spec
3. Create code examples (Python, JS, cURL)
4. Document authentication
5. Add request/response examples

---

## Phase 7: Automation (Comprehensive)

### GitHub Actions
```yaml
name: Generate Documentation
on:
  push:
    paths: ['src/**', 'docs/**']
jobs:
  generate-docs:
    steps:
      - run: sphinx-build -b html docs/source docs/build
```

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: doc-coverage
        entry: interrogate --fail-under 80
```

---

## Output Deliverables

| Mode | Files Updated |
|------|---------------|
| Always | README.md |
| If Sphinx | docs/index.rst, docs/api/, docs/conf.py |
| If API | API.md or openapi.json |
| Comprehensive | + .github/workflows/docs.yml, .pre-commit-config.yaml |

### Summary Report
```markdown
## Changes Made
- Created: [new files]
- Updated: [modified files]

## Coverage Improvement
- Before: X% → After: Y% (+Z%)

## Next Steps
1. Review generated documentation
2. Add custom examples
3. Set up CI/CD
```

---

## Success Criteria

- ✅ All code elements have docstrings
- ✅ Sphinx builds without errors
- ✅ README is comprehensive
- ✅ API documentation complete
- ✅ No broken links
- ✅ Coverage > 90%

---

## External Documentation

- `ast-parsing-implementation.md` - Language extractors (~400 lines)
- `sphinx-optimization.md` - Config templates (~350 lines)
- `api-documentation-templates.md` - OpenAPI templates (~550 lines)
- `documentation-automation.md` - CI/CD patterns (~300 lines)
