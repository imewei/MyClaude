---
description: Generate comprehensive documentation from code with AI-powered analysis
triggers:
- /doc-generate
- workflow for doc generate
version: 1.0.7
category: code-documentation
command: /doc-generate
argument-hint: '[--api] [--readme] [--sphinx] [--full]'
color: blue
execution_modes:
  quick: 10-15min
  standard: 20-30min
  comprehensive: 40-60min
allowed-tools: [Read, Task, Bash]
---


# Documentation Generation

$ARGUMENTS

Flags: `--api`, `--readme`, `--sphinx`, `--full`

## Modes

| Mode | Time | Output |
|------|------|--------|
| Quick | 10-15min | README.md, basic API.md |
| Standard | 20-30min | Complete docs with examples |
| Comprehensive | 40-60min | Full doc site + CI/CD automation |

## External Docs

- `api-documentation-templates.md` (~550 lines)
- `documentation-automation.md` (~300 lines)

## Process

1. **Code Analysis**:
   - Identify project type (Python, JS, Go)
   - Parse structure with AST
   - Extract API endpoints from decorators
   - Extract schemas (Pydantic, TypeScript)
   - Analyze dependencies and config

2. **API Documentation**:
   | Component | Content |
   |-----------|---------|
   | OpenAPI 3.0 | Generated from endpoints |
   | Interactive | Swagger UI, Redoc |
   | Examples | Python, JS, cURL, Go |
   | Authentication | Methods, requirements |
   | Request/response | Example payloads |

3. **README Generation**:
   | Section | Content |
   |---------|---------|
   | Badges | CI/CD, coverage, version |
   | Overview | 2-3 sentence description |
   | Features | Bulleted list |
   | Installation | Multiple methods |
   | Quick Start | Simple compelling example |
   | Configuration | Environment variables table |
   | Development | Setup and testing |

4. **Architecture Diagrams**:
   - System architecture: Components and connections
   - API flow: Request/response paths
   - Database schema: If applicable
   - Deployment: Infrastructure layout

   Use Mermaid for inline diagrams

5. **Code Examples**:
   | Type | Requirements |
   |------|--------------|
   | Basic | Common tasks |
   | Advanced | Complex scenarios |
   | Integration | Other services |
   | Error handling | Common errors |
   | Best practices | Recommended patterns |

   All examples complete, runnable, with comments

6. **Automation** (Comprehensive):
   | Component | Purpose |
   |-----------|---------|
   | GitHub Actions | Automatic doc generation |
   | Pre-commit hooks | Doc linting |
   | Coverage checks | interrogate >80% |
   | Deployment | GitHub Pages/Netlify/RTD |

7. **Sphinx** (Python with `--sphinx` or comprehensive):
   - Generate conf.py with autodoc
   - Create index.rst with toctree
   - Generate API reference with autosummary
   - Add examples and tutorials
   - Configure theme and deployment

## Output Structure

### API Projects
```
docs/
├── README.md
├── API.md
├── api/ (openapi.json, swagger-ui.html)
├── examples/
└── diagrams/
```

### Python with Sphinx
```
docs/
├── source/ (index.rst, api/, examples/)
├── conf.py
└── Makefile
```

## Quality Standards

- [ ] Accurate, synchronized with code
- [ ] Consistent terminology
- [ ] Practical examples
- [ ] Searchable, organized
- [ ] Accessible
- [ ] API versioning documented
- [ ] Troubleshooting guides
- [ ] Cross-linked

## Success

- [ ] All public APIs documented
- [ ] README comprehensive
- [ ] Installation tested
- [ ] Examples runnable
- [ ] API docs include authentication
- [ ] Architecture diagrams accurate
- [ ] Builds without errors
- [ ] No broken links
- [ ] Coverage >80% for public APIs
