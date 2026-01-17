---
description: Scaffold production-ready Python projects with modern tooling across
  3 execution modes
triggers:
- /python-scaffold
- scaffold production ready python projects
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



# Python Project Scaffolding

Create production-ready Python project: $ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "Which execution mode best fits your project requirements?"
    header: "Execution Mode"
    multiSelect: false
    options:
      - label: "Quick (1-2 hours)"
        description: "Simple script, prototype, or basic CLI tool. Minimal viable structure with ~15 files."

      - label: "Standard (3-6 hours)"
        description: "Production FastAPI/Django web app or distributable library. Complete structure with testing, CI/CD, Docker (~50 files)."

      - label: "Enterprise (1-2 days)"
        description: "Microservices architecture or complex platform. Multi-service setup with K8s, observability, security (~100 files)."
</AskUserQuestion>

## Instructions

### 1. Analyze Project Type

Determine the project type from user requirements:

**Quick Mode**: Simple script, basic CLI, minimal library
**Standard Mode**:
- **FastAPI**: REST APIs, microservices, async applications → [FastAPI Structure](../../plugins/python-development/docs/python-scaffold/fastapi-structure.md)
- **Django**: Full-stack web apps, admin panels, ORM-heavy projects → [Django Structure](../../plugins/python-development/docs/python-scaffold/django-structure.md)
- **Library**: Reusable packages, utilities → [Library Packaging](../../plugins/python-development/docs/python-scaffold/library-packaging.md)
- **CLI**: Command-line tools, automation scripts → [CLI Tools](../../plugins/python-development/docs/python-scaffold/cli-tools.md)
**Enterprise Mode**: All standard types + distributed systems, microservices

### 2. Initialize Project with uv

```bash
# Create new project
uv init <project-name>
cd <project-name>

# Initialize git
git init
cat >> .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env
EOF

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Generate Project Structure

Based on project type and execution mode:

#### Quick Mode
- Basic `pyproject.toml` with minimal dependencies
- Single main file (`main.py`, `cli.py`, or `__init__.py`)
- Simple test file
- README.md and .gitignore

#### Standard Mode
Select appropriate structure:
- **FastAPI Project**: Full structure from [FastAPI Guide](../../plugins/python-development/docs/python-scaffold/fastapi-structure.md)
- **Django Project**: Complete setup from [Django Guide](../../plugins/python-development/docs/python-scaffold/django-structure.md)
- **Library**: Package structure from [Library Guide](../../plugins/python-development/docs/python-scaffold/library-packaging.md)
- **CLI Tool**: Typer-based structure from [CLI Guide](../../plugins/python-development/docs/python-scaffold/cli-tools.md)

#### Enterprise Mode
All Standard features plus:
- Multi-service architecture (if microservices)
- Kubernetes manifests and Helm charts
- Distributed tracing setup (OpenTelemetry)
- Comprehensive observability (Prometheus/Grafana)
- Security hardening (secrets management, RBAC)
- Production deployment configs

### 4. Configure Development Tools

For all modes, set up modern Python tooling:

**See comprehensive configuration**: [Development Tooling Guide](../../plugins/python-development/docs/python-scaffold/development-tooling.md)

**Quick Setup**:
```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 5. Project-Specific Configuration

**Standard/Enterprise**: Add production-ready configurations:
- `.env.example` with all required environment variables
- `Makefile` with common development tasks
- `docker-compose.yml` for local development
- CI/CD workflows (`.github/workflows/tests.yml`)

**Full configurations available**: [Development Tooling Guide](../../plugins/python-development/docs/python-scaffold/development-tooling.md)

## Output Format

Deliver complete project with:

1. **Project Structure**: All files and directories for selected mode
2. **Configuration**: pyproject.toml with dependencies and tool settings
3. **Entry Point**: Main application file (main.py, manage.py, cli.py)
4. **Tests**: Test structure with pytest configuration
5. **Documentation**: README.md with setup and usage instructions
6. **Development Tools** (Standard/Enterprise): Makefile, Docker, CI/CD

## Success Criteria

**Quick Mode**:
- ✅ Project initializes with uv
- ✅ Virtual environment activates
- ✅ Basic tests pass (`pytest`)
- ✅ Type checking passes (`mypy`)
- ✅ Linting passes (`ruff check`)

**Standard Mode**:
- ✅ All Quick criteria met
- ✅ Framework-specific structure complete (FastAPI/Django/Library/CLI)
- ✅ Integration tests pass
- ✅ Docker builds successfully
- ✅ CI pipeline configured

**Enterprise Mode**:
- ✅ All Standard criteria met
- ✅ Multi-service coordination working
- ✅ Observability stack deployed
- ✅ K8s manifests validate
- ✅ Security scanning passes

## External Documentation

- [FastAPI Structure Guide](../../plugins/python-development/docs/python-scaffold/fastapi-structure.md) - Complete FastAPI project templates
- [Django Structure Guide](../../plugins/python-development/docs/python-scaffold/django-structure.md) - Django 5.x project structure
- [Library Packaging Guide](../../plugins/python-development/docs/python-scaffold/library-packaging.md) - PyPI-ready package setup
- [CLI Tools Guide](../../plugins/python-development/docs/python-scaffold/cli-tools.md) - Typer-based command-line tools
- [Development Tooling Guide](../../plugins/python-development/docs/python-scaffold/development-tooling.md) - Makefiles, Docker, CI/CD
