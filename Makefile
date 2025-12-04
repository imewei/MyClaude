# Makefile for Scientific Computing Workflows Marketplace
# Version: 1.0.4

.PHONY: help clean clean-all clean-python clean-docs clean-cache clean-build clean-reports \
        build docs docs-live test lint validate install dev-install plugin-enable-all

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Cleaning

clean: clean-python clean-cache clean-reports ## Clean Python artifacts, cache, and reports
	@echo "✓ Basic cleanup complete"

clean-all: clean clean-docs clean-build ## Deep clean: everything including documentation builds
	@echo "✓ Full cleanup complete"

clean-python: ## Remove Python cache and compiled files
	@echo "Cleaning Python artifacts..."
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name '.coverage' -delete 2>/dev/null || true
	@find . -type d -name 'htmlcov' -exec rm -rf {} + 2>/dev/null || true
	@echo "  ✓ Python artifacts cleaned"

clean-docs: ## Clean Sphinx documentation build artifacts
	@echo "Cleaning documentation build..."
	@if [ -d "docs/_build" ]; then \
		cd docs && make clean; \
		echo "  ✓ Sphinx build cleaned"; \
	else \
		echo "  ℹ No docs/_build directory found"; \
	fi

clean-cache: ## Remove all cache directories (.DS_Store, .cache, etc.)
	@echo "Cleaning cache files..."
	@find . -type f -name '.DS_Store' -delete 2>/dev/null || true
	@find . -type d -name '.cache' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.venv' -prune -o -type d -name 'node_modules' -prune -o -type f -name '*.log' -delete
	@echo "  ✓ Cache files cleaned"

clean-build: ## Remove build artifacts (dist, build directories)
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ .eggs/
	@find . -type d -name 'build' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name 'dist' -exec rm -rf {} + 2>/dev/null || true
	@echo "  ✓ Build artifacts cleaned"

clean-reports: ## Remove auto-generated reports and reviews
	@echo "Cleaning auto-generated reports..."
	@rm -rf reports/ reviews/ 2>/dev/null || true
	@rm -f DOCUMENTATION_UPDATE_SUMMARY.md VERSION_VERIFICATION_REPORT.md 2>/dev/null || true
	@echo "  ✓ Reports cleaned"

##@ Documentation

docs: ## Build Sphinx documentation
	@echo "Building documentation..."
	@cd docs && make html
	@echo "✓ Documentation built: docs/_build/html/index.html"

docs-live: ## Build and serve documentation with auto-reload
	@echo "Starting live documentation server..."
	@if command -v sphinx-autobuild >/dev/null 2>&1; then \
		cd docs && sphinx-autobuild . _build/html --open-browser; \
	else \
		echo "Error: sphinx-autobuild not installed"; \
		echo "Install with: pip install sphinx-autobuild"; \
		exit 1; \
	fi

docs-linkcheck: ## Check documentation for broken links
	@echo "Checking documentation links..."
	@cd docs && make linkcheck

##@ Development

install: ## Install the marketplace and dependencies
	@echo "Installing marketplace..."
	@if [ -f "requirements.txt" ]; then \
		pip install -r requirements.txt; \
	fi
	@if [ -f "docs/requirements.txt" ]; then \
		pip install -r docs/requirements.txt; \
	fi
	@echo "✓ Installation complete"

dev-install: install ## Install development dependencies
	@echo "Installing development dependencies..."
	@pip install sphinx-autobuild pytest pytest-cov ruff mypy black
	@echo "✓ Development environment ready"

##@ Quality

lint: ## Run linters on Python code
	@echo "Running linters..."
	@if command -v ruff >/dev/null 2>&1; then \
		echo "Running ruff..."; \
		ruff check . || true; \
	fi
	@if command -v mypy >/dev/null 2>&1; then \
		echo "Running mypy..."; \
		mypy --ignore-missing-imports tools/ plugins/ || true; \
	fi
	@echo "✓ Linting complete"

format: ## Format Python code with black and ruff
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black tools/ plugins/; \
	fi
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check --fix . || true; \
	fi
	@echo "✓ Formatting complete"

validate: ## Validate plugin metadata and configuration
	@echo "Validating plugins..."
	@if [ -f "tools/metadata-validator.py" ]; then \
		python tools/metadata-validator.py; \
	fi
	@echo "✓ Validation complete"

##@ Testing

test: ## Run tests with pytest
	@echo "Running tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		pytest tests/ -v || true; \
	else \
		echo "pytest not installed. Install with: pip install pytest"; \
	fi

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@if command -v pytest >/dev/null 2>&1; then \
		pytest tests/ --cov=. --cov-report=html --cov-report=term; \
		echo "Coverage report: htmlcov/index.html"; \
	else \
		echo "pytest-cov not installed. Install with: pip install pytest-cov"; \
	fi

##@ Git Operations

git-status: ## Show git status with helpful formatting
	@echo "=== Git Status ==="
	@git status --short
	@echo ""
	@echo "=== Recent Commits ==="
	@git log --oneline -5

git-clean: ## Remove untracked files (dry-run, use git-clean-force to actually delete)
	@echo "Files that would be removed:"
	@git clean -n -d -x

git-clean-force: ## Remove untracked files (WARNING: irreversible)
	@echo "⚠️  This will permanently delete untracked files!"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	@git clean -f -d -x
	@echo "✓ Untracked files removed"

##@ Plugin Operations

plugin-count: ## Count plugins and show breakdown
	@echo "=== Plugin Statistics ==="
	@echo "Total plugins: $$(ls -d plugins/*/ 2>/dev/null | wc -l | tr -d ' ')"
	@echo ""
	@echo "Plugins with plugin.json: $$(find plugins -name 'plugin.json' | wc -l | tr -d ' ')"
	@echo "Plugins with README: $$(find plugins -maxdepth 2 -name 'README.md' | wc -l | tr -d ' ')"
	@echo ""
	@echo "=== Category Breakdown ==="
	@for json in plugins/*/plugin.json; do \
		grep -o '"category"[[:space:]]*:[[:space:]]*"[^"]*"' "$$json" 2>/dev/null; \
	done | cut -d'"' -f4 | sort | uniq -c | sort -rn

plugin-list: ## List all plugins with their versions
	@echo "=== Plugin List ==="
	@for dir in plugins/*/; do \
		plugin=$$(basename "$$dir"); \
		if [ -f "$$dir/plugin.json" ]; then \
			version=$$(grep '"version"' "$$dir/plugin.json" | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/'); \
			printf "%-35s %s\n" "$$plugin" "v$$version"; \
		fi \
	done | sort

plugin-enable-all: ## Enable all plugins in Claude Code (requires restart)
	@echo "Enabling all plugins in Claude Code..."
	@python3 tools/enable-all-plugins.py

##@ Information

info: ## Show repository information
	@echo "=== Repository Information ==="
	@echo "Name: Scientific Computing Workflows Marketplace"
	@echo "Version: 1.0.4"
	@echo "Author: Wei Chen"
	@echo "Documentation: https://myclaude.readthedocs.io/en/latest/"
	@echo "Repository: https://github.com/imewei/MyClaude"
	@echo ""
	@echo "=== Statistics ==="
	@echo "Total Plugins: $$(find plugins -maxdepth 1 -type d | tail -n +2 | wc -l | tr -d ' ')"
	@echo "RST Documentation Files: $$(find docs -name '*.rst' -type f 2>/dev/null | wc -l | tr -d ' ')"
	@echo "Python Scripts: $$(find . -name '*.py' -not -path './.venv/*' -not -path './docs/_build/*' | wc -l | tr -d ' ')"
	@echo "Total Lines of Code: $$(find . -name '*.py' -o -name '*.md' -o -name '*.rst' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $$1}')"

version: ## Show current version
	@echo "v1.0.4"

##@ Advanced Cleaning

clean-test-corpus: ## Remove test corpus directory
	@echo "Removing test corpus..."
	@rm -rf test-corpus/ 2>/dev/null || true
	@echo "  ✓ Test corpus removed"

clean-specs: ## Remove agent-os specs directory
	@echo "Removing specifications..."
	@rm -rf agent-os/ 2>/dev/null || true
	@echo "  ✓ Specifications removed"

clean-tools-output: ## Clean tool output files
	@echo "Cleaning tool outputs..."
	@find tools -name '*.log' -delete 2>/dev/null || true
	@find tools -name '*.tmp' -delete 2>/dev/null || true
	@echo "  ✓ Tool outputs cleaned"

nuke: clean-all clean-test-corpus clean-specs ## Nuclear option: remove everything that can be regenerated
	@echo "⚠️  Nuclear cleanup complete - regenerate docs with 'make docs'"
