# Development Tooling and Configuration

Modern Python development tools and configuration for linting, formatting, testing, and automation.

## .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# Tools
.ruff_cache/
.mypy_cache/
.dmypy.json
dmypy.json

# Environment
.env
.env.local
*.env

# OS
.DS_Store
Thumbs.db

# Databases
*.db
*.sqlite3
*.sqlite

# Logs
*.log
```

## .env.example

### FastAPI Project

```env
# Application
PROJECT_NAME="FastAPI Project"
VERSION="0.1.0"
DEBUG=True
ENVIRONMENT="development"

# API Configuration
API_V1_PREFIX="/api/v1"
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:8080"]
CORS_ALLOW_CREDENTIALS=True

# Database
DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/dbname"
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Security
SECRET_KEY="your-secret-key-here-change-in-production"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis (optional)
REDIS_URL="redis://localhost:6379/0"

# Logging
LOG_LEVEL="INFO"
LOG_FORMAT="json"

# External Services
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"

# AWS (if needed)
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_REGION="us-east-1"
S3_BUCKET=""
```

### Django Project

```env
# Django
SECRET_KEY="django-insecure-change-this-in-production"
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DJANGO_SETTINGS_MODULE="config.settings.development"

# Database
DATABASE_URL="postgres://user:password@localhost:5432/dbname"

# Celery
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Email
EMAIL_BACKEND="django.core.mail.backends.console.EmailBackend"
EMAIL_HOST="smtp.gmail.com"
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=""
EMAIL_HOST_PASSWORD=""

# Static/Media
STATIC_ROOT="/path/to/staticfiles"
MEDIA_ROOT="/path/to/media"
```

## Makefile

```makefile
.PHONY: help install dev test lint format type-check clean docker-build docker-up docker-down

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync --no-dev

dev: ## Install all dependencies including dev
	uv sync

run: ## Run development server (FastAPI)
	uv run uvicorn src.project_name.main:app --reload --host 0.0.0.0 --port 8000

run-django: ## Run Django development server
	uv run python manage.py runserver 0.0.0.0:8000

migrate: ## Run database migrations
	uv run alembic upgrade head

migrate-django: ## Run Django migrations
	uv run python manage.py migrate

migrations: ## Create new migration (Django)
	uv run python manage.py makemigrations

test: ## Run tests with pytest
	uv run pytest -n auto -v

test-cov: ## Run tests with coverage report
	uv run pytest -n auto -v --cov=src --cov-report=term-missing --cov-report=html

test-watch: ## Run tests in watch mode
	uv run pytest-watch

lint: ## Check code with ruff
	uv run ruff check .

lint-fix: ## Fix linting issues automatically
	uv run ruff check --fix .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check if code is formatted
	uv run ruff format --check .

type-check: ## Run type checking with mypy
	uv run mypy src

audit: ## Check for security vulnerabilities
	uv pip check

clean: ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build

docker-build: ## Build Docker image
	docker build -t project-name:latest .

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker container logs
	docker-compose logs -f

celery-worker: ## Start Celery worker (Django)
	uv run celery -A config worker -l info

celery-beat: ## Start Celery beat scheduler
	uv run celery -A config beat -l info

shell: ## Open Python shell
	uv run python

db-shell: ## Open database shell (Django)
	uv run python manage.py dbshell

superuser: ## Create Django superuser
	uv run python manage.py createsuperuser

seed: ## Seed database with sample data (Django)
	uv run python manage.py seed_data
```

## Docker Configuration

### Dockerfile (Multi-stage for FastAPI)

```dockerfile
# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src ./src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.project_name.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/mydb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./src:/app/src
    command: uvicorn src.project_name.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: mydb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Celery worker (for Django/background tasks)
  celery_worker:
    build: .
    command: celery -A config worker -l info
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/mydb
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

## GitHub Actions CI/CD

### .github/workflows/tests.yml

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]

    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync

      - name: Run linting
        run: |
          uv run ruff check .
          uv run ruff format --check .

      - name: Run type checking
        run: uv run mypy src

      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        run: uv run pytest -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

## Pre-commit Hooks

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, sqlalchemy]
```

Install hooks:
```bash
uv add --dev pre-commit
pre-commit install
```

## VSCode Configuration

### .vscode/settings.json

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.linting.enabled": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "ruff.lint.args": ["--config=${workspaceFolder}/pyproject.toml"],
  "ruff.format.args": ["--config=${workspaceFolder}/pyproject.toml"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.ruff_cache": true,
    "**/.mypy_cache": true
  }
}
```

### .vscode/extensions.json

```json
{
  "recommendations": [
    "charliermarsh.ruff",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "tamasfe.even-better-toml"
  ]
}
```

## Development Workflow

### 1. Setup
```bash
# Clone and setup
git clone <repo-url>
cd project
uv sync
cp .env.example .env
# Edit .env with your values
```

### 2. Daily Development
```bash
# Start development server
make run  # or make run-django

# Run tests in watch mode (separate terminal)
make test-watch

# Format and lint before commit
make format
make lint
```

### 3. Before Commit
```bash
# Run full quality checks
make format
make lint
make type-check
make test-cov

# Or use pre-commit hooks (automatic)
git commit -m "feat: add feature"
```

### 4. Database Workflow (Django)
```bash
# Create migrations
make migrations

# Apply migrations
make migrate-django

# Create superuser
make superuser
```

## Best Practices

### 1. Environment Management
- Use `.env` files for configuration
- Never commit secrets to git
- Provide `.env.example` with all required variables
- Use different `.env` files per environment (dev/staging/prod)

### 2. Dependency Management
- Lock dependencies with `uv.lock`
- Separate dev and production dependencies
- Regularly update dependencies: `uv lock --upgrade`
- Use `uv pip check` to audit security issues

### 3. Code Quality
- Run linting and formatting on save (IDE integration)
- Use pre-commit hooks for automatic checks
- Maintain >80% test coverage
- Run type checking with mypy

### 4. Docker Best Practices
- Use multi-stage builds to reduce image size
- Run as non-root user
- Use `.dockerignore` to exclude unnecessary files
- Pin base image versions

### 5. CI/CD
- Test on multiple Python versions
- Run security audits
- Generate coverage reports
- Cache dependencies for faster builds
