# Runbook

This repo ships a Claude Code plugin marketplace, not a deployed service — there is no server, endpoint, or process to keep alive. This runbook covers the release procedure and the checks that stand in for health checks / rollback here.

## Release Procedure

1. Bump the version everywhere it's tracked. `make verify` checks `plugins/*/plugin.json` against each other and against `pyproject.toml`; the rest is by hand — `grep -rn "<old>" --exclude-dir=_build --exclude-dir=graphify-out .` and update every *current-state* mention (leave changelog/history mentions alone):
   - `pyproject.toml` → `[project].version`
   - `Makefile` → header comment (`# Version: X.Y.Z`) and `info` target
   - `plugins/*/.claude-plugin/plugin.json` → `version` (all three are kept in lockstep)
   - `.claude-plugin/marketplace.json` → root `version`, each plugin entry `version`, `note` prefix
   - `docs/conf.py` → `release`; `docs/index.rst`; `README.md` badge + intro; `docs/reference/{agents,commands,cheatsheet}.md` headers/footers; `docs/suites/*.rst` headers + `:version:` fields; `docs/claude-code-spec-compliance.md`
2. Add a new section to `CHANGELOG.md` describing user-visible changes (see prior entries for format/tone).
3. Run the full gate: `make verify` (lint + validate + tests). Fix anything red before proceeding.
4. Run `make audit` if the release touches dependencies, hooks, or anything security-sensitive.
5. Commit, tag, and push per normal git workflow (not automated by any script in `tools/`).

## "Health Checks" (pre-release verification)

| Command | What it checks |
|---------|-----------------|
| `make validate` | Per-plugin `plugin.json` schema, required fields, semver format (each file checked independently — no cross-file comparison), command file structure, doc cross-links, skill context budget (2%) + agent prompt size (10,000 chars) |
| `make verify-fast` | Lint + validate only (quick gate) |
| `make verify` | Lint + validate + full test suite (run before every push) |
| `make audit` | `pip-audit` (dependency CVEs) + `bandit` (SAST) + `vulture` (dead code) + `gitleaks` (secret scan) |
| `make plugin-count` | Total plugin count, `plugin.json`/README presence, category breakdown — does not count agents, commands, or hub skills |

## Common Issues and Fixes

- **Version drift across files** — `make validate` only checks semver *format* within each `plugin.json`; `make verify` checks the three `plugin.json` agree with each other and with `pyproject.toml`, but `marketplace.json`, `Makefile`, `docs/conf.py`, and README/docs are unchecked. Grep all four locations listed under Release Procedure above before every release.
- **`pytest` finds no tests / fails to collect** — tests live under `tools/tests/`, not the repo root; run `uv run pytest` (respects `testpaths` in `pyproject.toml`), not a bare `pytest` from an unexpected `cwd`.
- **mypy errors inside `plugins/*/hooks/` or `plugins/*/examples/`** — these paths are intentionally excluded (`pyproject.toml` `[tool.mypy].exclude`); if mypy is still flagging them, check the invocation isn't overriding the config.
- **Sphinx build shows duplicate TOC entries** — remove any stray `.. contents::` directive; the Furo theme auto-generates the sidebar.

## Rollback

No deployed runtime to roll back. If a bad version was tagged/pushed:
1. Publish a follow-up patch release with the fix (preferred — plugin consumers pin marketplace refs, not live endpoints).
2. If the tag itself must be removed, coordinate with the repo owner before force-deleting a pushed tag — do not do this unilaterally.

## Escalation

No on-call/paging setup exists for this repo. Open a GitHub issue at the repository (`https://github.com/imewei/MyClaude`, per `make info`) for anything blocking.
