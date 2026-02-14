---
version: "2.2.1"
description: Unified documentation management - generate, update, and sync documentation
argument-hint: "<action> [--api] [--readme] [--sphinx] [--full] [--from-git]"
category: quality-suite
color: blue
execution-modes:
  generate-quick: "10-15min"
  generate-standard: "20-30min"
  generate-comprehensive: "40-60min"
  update-quick: "15-20min"
  update-standard: "30-45min"
  sync-quick: "5-10min"
  sync-standard: "10-15min"
allowed-tools: [Bash, Edit, Read, Write, Task, Bash(git:*)]
agents:
  primary:
    - documentation-expert
  conditional:
    - agent: software-architect
      trigger: argument "--full" OR files > 50
---

# Documentation Management

$ARGUMENTS

## Actions

| Action | Description | Use Case |
|--------|-------------|----------|
| `generate` | Create comprehensive docs from code | New projects, major releases |
| `update` | Update existing docs with AST analysis | Regular maintenance |
| `sync` | Sync CLAUDE.md with git changes | After commits, PRs |

**Usage:**
```bash
/docs generate                   # Generate all documentation
/docs generate --api             # API docs only
/docs generate --sphinx          # Python Sphinx docs
/docs update                     # Update existing docs
/docs update --readme            # README only
/docs sync                       # Sync CLAUDE.md from git
/docs sync --force               # Full rebuild of CLAUDE.md
```

---

## Action: Generate

Create comprehensive documentation from scratch using AI-powered code analysis.

### Modes

| Mode | Time | Output |
|------|------|--------|
| Quick | 10-15min | README.md, basic API.md |
| Standard | 20-30min | Complete docs with examples |
| Comprehensive | 40-60min | Full doc site + CI/CD automation |

### Flags

- `--api` - Generate API documentation only
- `--readme` - Generate README only
- `--sphinx` - Generate Sphinx docs (Python)
- `--full` - Generate everything

### Process

1. **Code Analysis**:
   - Identify project type (Python, JS, Go, Rust)
   - Parse structure with AST
   - Extract API endpoints from decorators
   - Extract schemas (Pydantic, TypeScript, OpenAPI)
   - Analyze dependencies and configuration

2. **API Documentation**:
   | Component | Content |
   |-----------|---------|
   | OpenAPI 3.0 | Generated from endpoints |
   | Interactive | Swagger UI, Redoc |
   | Examples | Python, JS, cURL, Go |
   | Authentication | Methods, requirements |
   | Request/Response | Example payloads |

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

4. **Architecture Diagrams** (Mermaid):
   - System architecture: Components and connections
   - API flow: Request/response paths
   - Database schema: If applicable
   - Deployment: Infrastructure layout

5. **Sphinx** (Python with `--sphinx`):
   - Generate conf.py with autodoc
   - Create index.rst with toctree
   - Generate API reference with autosummary
   - Add examples and tutorials
   - Configure theme and deployment

### Output Structure

**API Projects:**
```
docs/
├── README.md
├── API.md
├── api/ (openapi.json, swagger-ui.html)
├── examples/
└── diagrams/
```

**Python with Sphinx:**
```
docs/
├── source/ (index.rst, api/, examples/)
├── conf.py
└── Makefile
```

---

## Action: Update

Update existing documentation using AST-based code analysis to find gaps.

### Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 15-20min | README + critical updates |
| Standard | 30-45min | Full AST analysis + all docs |
| Comprehensive | 60-90min | Everything + CI/CD automation |

### Flags

- `--full` - Update all documentation
- `--sphinx` - Update Sphinx docs only
- `--readme` - Update README only
- `--api` - Update API docs only
- `--dry-run` - Preview changes without applying

### Process

1. **Project Analysis**:
   ```bash
   git log --since="2 weeks ago" --stat | head -200
   ```
   Inventory existing docs: README.md, docs/conf.py, API.md, etc.

2. **AST-Based Code Analysis**:
   - Parse all source files with AST
   - Extract modules, classes, functions, docstrings
   - Identify type hints and parameters
   - Compare with existing documentation

3. **Gap Analysis**:
   For each code element:
   - ✓ Documented in docs?
   - ✓ Signature up-to-date?
   - ✓ All parameters documented?
   - ✓ Return types documented?
   - ✓ Examples provided?

   **Gap Report:**
   - UNDOCUMENTED: [classes, functions, modules]
   - OUTDATED: [signature changes, deprecated]
   - INCOMPLETE: [missing params, returns, examples]

4. **Apply Updates**:
   - Update Sphinx autodoc references
   - Refresh README sections
   - Regenerate API documentation
   - Fix broken links

5. **Verification**:
   ```bash
   cd docs && make html    # No errors
   make linkcheck          # No broken links
   ```

---

## Action: Sync

Synchronize CLAUDE.md with recent git changes.

### Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 5-10min | Last 5 commits |
| Standard | 10-15min | Last 10 commits |
| Force | 15-20min | Complete rebuild |

### Flags

- `--force` - Complete rebuild analyzing entire git history
- `--summary` - Only show summary without updating

### Process

1. **Git Analysis**:
   ```bash
   git log --oneline -10
   git diff HEAD~5 --name-only
   git diff --name-status HEAD~10 | grep "^A"  # New files
   ```

2. **Identify Changes**:
   - New Features: Functionality added
   - API Changes: Endpoints, routes, parameters
   - Config Updates: Build tools, dependencies
   - File Structure: New directories, moved files
   - Database: Models, schemas, migrations
   - Bug Fixes: Important behavioral changes

3. **Update Sections**:
   - Project Overview: Scope, technologies, version
   - Architecture: Patterns, structure, components
   - Setup Instructions: Environment, dependencies
   - API Documentation: Endpoints, auth
   - Development Workflow: Scripts, tools
   - Recent Updates: Timestamped change summary

4. **Smart Content Management**:
   - Don't duplicate existing docs
   - Prioritize developer-impacting changes
   - Keep concise and summarized
   - Maintain existing structure
   - Add timestamps for major updates

---

## Automation (Comprehensive Mode)

### GitHub Actions
```yaml
name: Generate Docs
on:
  push:
    paths: ['src/**', 'docs/**']
jobs:
  generate:
    steps:
      - run: sphinx-build -b html docs/source docs/build
```

### Pre-commit
```yaml
repos:
  - repo: local
    hooks:
      - id: doc-coverage
        entry: interrogate --fail-under 80
```

---

## Quality Standards

- [ ] Accurate and synchronized with code
- [ ] Consistent terminology throughout
- [ ] Practical, runnable examples
- [ ] Searchable and well-organized
- [ ] Accessible language
- [ ] API versioning documented
- [ ] Troubleshooting guides included
- [ ] Cross-linked references
- [ ] No broken links
- [ ] Coverage >80% for public APIs

## Success Criteria

- ✅ All public APIs documented
- ✅ README comprehensive and current
- ✅ Installation instructions tested
- ✅ Examples runnable
- ✅ Architecture diagrams accurate
- ✅ Sphinx/docs builds without errors
- ✅ CLAUDE.md reflects recent changes
