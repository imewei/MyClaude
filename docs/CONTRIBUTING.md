# Contributing

## Setup

Prerequisites: Python 3.13+, [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync            # installs dev + docs + science dependency groups
```

## Available Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependency groups (dev + docs + science) |
| `uv run pytest` | Run test suite (tests live in `tools/tests/`) |
| `uv run pytest tools/tests/test_x.py -v` | Run a single test file |
| `uv run ruff check .` | Lint |
| `uv run mypy tools/` | Type-check (excludes `plugins/*/hooks/`, `plugins/*/examples/`) |
<!-- AUTO-GENERATED from Makefile `## ` target comments — regenerate with /ecc:update-docs, do not hand-edit -->
| `make clean` | Clean Python artifacts, cache, and reports |
| `make clean-all` | Deep clean: everything including documentation builds |
| `make docs` | Build Sphinx documentation |
| `make docs-live` | Build and serve documentation with auto-reload |
| `make docs-linkcheck` | Check documentation for broken links |
| `make install` | Install the marketplace and dependencies |
| `make dev-install` | Install development dependencies |
| `make lint` | Run linters on Python code |
| `make format` | Format Python code with black and ruff |
| `make validate` | Validate plugin.json, command file structure, doc cross-links, skill context budget + agent prompt size |
| `make test` | Run tests with pytest |
| `make test-coverage` | Run tests with coverage report |
| `make verify` | Full local CI (lint + validate + tests) — run before push |
| `make verify-fast` | Quick verification (lint + validate only, no tests) |
| `make audit` | Run full audit (deps + secrets + SAST + dead code) |
| `make audit-deps` | pip-audit: known CVEs in Python dependencies |
| `make audit-secrets` | gitleaks: secret-scan working tree (history skipped; working tree only) |
| `make audit-sast` | bandit: static security analysis on production Python sources (tests/ excluded — test code legitimately uses /tmp) |
| `make audit-deadcode` | vulture: dead-code detection (--min-confidence 80) |
| `make plugin-count` | Count plugins and show breakdown |
| `make plugin-list` | List all plugins with their versions |
| `make help` | List all Makefile targets (including clean-*/audit-*/git-* subtargets not listed here) |
<!-- END AUTO-GENERATED -->

Run validation against one plugin directly:

```bash
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/dev-suite/
```

## Testing

- Tests live in `tools/tests/`, not the repo root. Always use `uv run pytest` or `make test` — bare `pytest` misses `testpaths` unless run from a context that respects `pyproject.toml`.

## Code Style

- Lint: `uv run ruff check .` (config in `pyproject.toml` under `[tool.ruff]`).
- Format: `make format` (black + `ruff --fix`).
- Type-check: `uv run mypy tools/` (config in `pyproject.toml` under `[tool.mypy]`).
- No pre-commit hook is configured in this repo; run `make verify-fast` before committing and `make verify` before pushing.

## Plugin Component Rules

- Each suite under `plugins/<suite>/` follows the layout documented in the root `CLAUDE.md`.
- `plugin.json` registers only top-level hub skills — adding a sub-skill does not require a manifest edit.
- Plugin version must stay in sync across `plugin.json`, `pyproject.toml`, and READMEs. `make validate` only checks semver format within each `plugin.json`; `make verify` checks the three `plugin.json` versions agree with each other and with `pyproject.toml`. `marketplace.json`, `Makefile`, `docs/conf.py`, and README/docs references are not checked — grep for the old version by hand.
- Do not add `.. contents::` to RST docs — the Furo Sphinx theme auto-generates the sidebar TOC.

## PR Checklist

- [ ] `make verify` passes locally
- [ ] Plugin version bumped everywhere required if you touched a plugin (checked by hand — no tooling verifies this)
- [ ] New sub-skills are *not* added to `plugin.json` (hub-skill routing only)
- [ ] `CHANGELOG.md` updated for user-visible changes
