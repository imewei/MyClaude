# Contributing

## Setup

Prerequisites: Python 3.13+, [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync            # installs dev + docs + science dependency groups
```

<!-- AUTO-GENERATED: scripts table (source: Makefile, pyproject.toml) -->
## Available Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependency groups (dev + docs + science) |
| `uv run pytest` | Run test suite (tests live in `tools/tests/`) |
| `uv run pytest tools/tests/test_x.py -v` | Run a single test file |
| `uv run ruff check .` | Lint (excludes `test-corpus/`) |
| `uv run mypy tools/` | Type-check (excludes `plugins/*/hooks/` and `plugins/*/examples/`) |
| `make format` | Format with black + `ruff --fix` |
| `make validate` | Validate plugin metadata, command lint, doc cross-links |
| `make verify-fast` | Quick gate: lint + validate |
| `make verify` | Full local CI: lint + validate + tests — run before every push |
| `make audit` | pip-audit + bandit + vulture + gitleaks |
| `make docs` | Build Sphinx docs to `docs/_build/html/` |
| `make docs-live` | Sphinx docs with autobuild + browser reload |
| `make plugin-list` | List all plugins with versions |
| `make plugin-count` | Plugin statistics and category breakdown |
| `make clean` | Remove Python cache, cache dirs, and reports |
| `make help` | List all Makefile targets |

Run validation against one plugin directly:

```bash
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/dev-suite/
```
<!-- /AUTO-GENERATED -->

## Testing

- Tests live in `tools/tests/`, not the repo root. Always use `uv run pytest` or `make test` — bare `pytest` misses `testpaths` unless run from a context that respects `pyproject.toml`.
- `test-corpus/` is excluded from ruff, mypy, and pytest: it holds fixture files that intentionally import scientific libraries (jax, numpyro, ...) not installed in this project.

## Code Style

- Lint: `uv run ruff check .` (config in `pyproject.toml` under `[tool.ruff]`).
- Format: `make format` (black + `ruff --fix`).
- Type-check: `uv run mypy tools/` (config in `pyproject.toml` under `[tool.mypy]`).
- No pre-commit hook is configured in this repo; run `make verify-fast` before committing and `make verify` before pushing.

## Plugin Component Rules

- Each suite under `plugins/<suite>/` follows the layout documented in the root `CLAUDE.md`.
- `plugin.json` registers only top-level hub skills — adding a sub-skill does not require a manifest edit.
- Plugin version must stay in sync across `plugin.json`, `pyproject.toml`, and READMEs; `make validate` catches drift.
- Do not add `.. contents::` to RST docs — the Furo Sphinx theme auto-generates the sidebar TOC.

## PR Checklist

- [ ] `make verify` passes locally
- [ ] Plugin version bumped everywhere required if you touched a plugin (`make validate` confirms)
- [ ] New sub-skills are *not* added to `plugin.json` (hub-skill routing only)
- [ ] `CHANGELOG.md` updated for user-visible changes
