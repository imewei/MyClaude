# Runbook

This repo ships a Claude Code plugin marketplace, not a deployed service — there is no server, endpoint, or process to keep alive. This runbook covers the release procedure and the checks that stand in for health checks / rollback here.

## Release Procedure

1. Bump the version everywhere it's tracked — no tooling checks cross-file consistency, so do this by hand:
   - `pyproject.toml` → `[project].version`
   - `Makefile` → header comment (`# Version: X.Y.Z`) and `info` target
   - `plugins/*/.claude-plugin/plugin.json` → `version` (only plugins whose contents changed)
   - `.claude-plugin/marketplace.json` (root manifest)
2. Add a new section to `CHANGELOG.md` describing user-visible changes (see prior entries for format/tone).
3. Run the full gate: `make verify` (lint + validate + tests). Fix anything red before proceeding.
4. Run `make audit` if the release touches dependencies, hooks, or anything security-sensitive.
5. Commit, tag, and push per normal git workflow (not automated by any script in `tools/`).

## "Health Checks" (pre-release verification)

| Command | What it checks |
|---------|-----------------|
| `make validate` | Per-plugin `plugin.json` schema, required fields, semver format (each file checked independently — no cross-file comparison), command file frontmatter, doc cross-links |
| `make verify-fast` | Lint + validate only (quick gate) |
| `make verify` | Lint + validate + full test suite (run before every push) |
| `make audit` | `pip-audit` (dependency CVEs) + `bandit` (SAST) + `vulture` (dead code) + `gitleaks` (secret scan) |
| `make plugin-count` | Total plugin count, `plugin.json`/README presence, category breakdown — does not count agents, commands, or hub skills |

## Common Issues and Fixes

- **Version drift across files** — no tooling catches this; `make validate` only checks semver *format* within each `plugin.json` independently. Grep all four locations listed under Release Procedure above before every release.
- **`pytest` finds no tests / fails to collect** — tests live under `tools/tests/`, not the repo root; run `uv run pytest` (respects `testpaths` in `pyproject.toml`), not a bare `pytest` from an unexpected `cwd`.
- **mypy errors inside `plugins/*/hooks/` or `plugins/*/examples/`** — these paths are intentionally excluded (`pyproject.toml` `[tool.mypy].exclude`); if mypy is still flagging them, check the invocation isn't overriding the config.
- **ruff flags files under `test-corpus/`** — that directory is fixture content for the skill validator and is excluded in `[tool.ruff].exclude`; don't "fix" imports there.
- **Sphinx build shows duplicate TOC entries** — remove any stray `.. contents::` directive; the Furo theme auto-generates the sidebar.

## Rollback

No deployed runtime to roll back. If a bad version was tagged/pushed:
1. Publish a follow-up patch release with the fix (preferred — plugin consumers pin marketplace refs, not live endpoints).
2. If the tag itself must be removed, coordinate with the repo owner before force-deleting a pushed tag — do not do this unilaterally.

## Escalation

No on-call/paging setup exists for this repo. Open a GitHub issue at the repository (`https://github.com/imewei/MyClaude`, per `make info`) for anything blocking.
