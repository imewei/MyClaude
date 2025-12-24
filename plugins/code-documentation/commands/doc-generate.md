---
version: 1.0.5
category: code-documentation
command: /doc-generate
description: Generate comprehensive documentation from code with AI-powered analysis
argument-hint: [--api] [--readme] [--sphinx] [--full]
color: blue
execution_modes:
  quick: "10-15 minutes"
  standard: "20-30 minutes"
  comprehensive: "40-60 minutes"
agents:
  primary:
    - docs-architect
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "sphinx|numpy|scipy|scientific"
    - agent: fullstack-developer
      trigger: files "package.json|src/components/"
  orchestrated: false
---

# Automated Documentation Generation

Generate comprehensive, maintainable documentation from code using AI-powered analysis.

## Arguments

$ARGUMENTS

**Flags:** `--api`, `--readme`, `--sphinx`, `--full`

---

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| Quick | 10-15 min | README.md, basic API.md |
| Standard (default) | 20-30 min | Complete docs with examples |
| Comprehensive | 40-60 min | Full doc site + CI/CD automation |

---

## External Documentation

| Topic | Reference | Lines |
|-------|-----------|-------|
| API Templates | [api-documentation-templates.md](../docs/code-documentation/api-documentation-templates.md) | ~550 |
| Automation | [documentation-automation.md](../docs/code-documentation/documentation-automation.md) | ~300 |

---

## Phase 1: Code Analysis

| Step | Action |
|------|--------|
| 1 | Identify project type (Python, JS, Go, etc.) |
| 2 | Parse code structure with AST |
| 3 | Extract API endpoints from decorators |
| 4 | Extract schemas (Pydantic, TypeScript) |
| 5 | Analyze dependencies and configuration |

**API extraction:** See [api-documentation-templates.md](../docs/code-documentation/api-documentation-templates.md#api-endpoint-extraction)

---

## Phase 2: API Documentation

| Component | Content |
|-----------|---------|
| OpenAPI 3.0 spec | Generated from endpoints |
| Interactive docs | Swagger UI, Redoc |
| Code examples | Python, JavaScript, cURL, Go |
| Authentication | Methods and requirements |
| Request/response | Example payloads |

**Templates:** [api-documentation-templates.md](../docs/code-documentation/api-documentation-templates.md#complete-openapi-30-template)

---

## Phase 3: README Generation

### Required Sections

| Section | Content |
|---------|---------|
| Badges | CI/CD, coverage, version |
| Overview | 2-3 sentence description |
| Features | Bulleted list |
| Installation | Multiple methods |
| Quick Start | Simple compelling example |
| Configuration | Environment variables table |
| Development | Setup and testing |

**Template:** [documentation-automation.md](../docs/code-documentation/documentation-automation.md#readme-generation)

---

## Phase 4: Architecture Diagrams

| Diagram Type | Purpose |
|--------------|---------|
| System architecture | Components and connections |
| API flow | Request/response paths |
| Database schema | If applicable |
| Deployment | Infrastructure layout |

Use Mermaid for inline diagrams.

---

## Phase 5: Code Examples

| Example Type | Requirements |
|--------------|--------------|
| Basic usage | Common tasks |
| Advanced | Complex scenarios |
| Integration | Other services |
| Error handling | Common errors |
| Best practices | Recommended patterns |

All examples must be complete, runnable, with comments.

---

## Phase 6: Documentation Automation (Comprehensive)

### CI/CD Setup

| Component | Purpose |
|-----------|---------|
| GitHub Actions | Automatic doc generation |
| Pre-commit hooks | Doc linting |
| Coverage checks | interrogate >80% |
| Deployment | GitHub Pages/Netlify/RTD |

**Workflow:** [documentation-automation.md](../docs/code-documentation/documentation-automation.md#github-actions-workflow)

---

## Phase 7: Sphinx Documentation

For Python projects with `--sphinx` or comprehensive mode:

| Step | Action |
|------|--------|
| 1 | Generate conf.py with autodoc |
| 2 | Create index.rst with toctree |
| 3 | Generate API reference with autosummary |
| 4 | Add examples and tutorials |
| 5 | Configure theme and deployment |

---

## Output Structure

### API Projects
```
docs/
├── README.md
├── API.md
├── api/openapi.json, swagger-ui.html
├── examples/
└── diagrams/
```

### Python with Sphinx
```
docs/
├── source/index.rst, api/, examples/
├── conf.py
└── Makefile
```

---

## Quality Standards

- [ ] Accurate and synchronized with code
- [ ] Consistent terminology
- [ ] Practical examples included
- [ ] Searchable and organized
- [ ] Accessible
- [ ] API versioning documented
- [ ] Troubleshooting guides
- [ ] Cross-linked

---

## Success Criteria

- [ ] All public APIs documented
- [ ] README comprehensive and clear
- [ ] Installation instructions tested
- [ ] Code examples runnable
- [ ] API docs include authentication
- [ ] Architecture diagrams accurate
- [ ] Documentation builds without errors
- [ ] No broken links
- [ ] Coverage >80% for public APIs
