# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MyClaude is a Claude Code plugin marketplace: 3 plugin suites containing 24 agents, 36 commands, and 168 skills. It is **not** a runnable application — it's a collection of markdown-based plugin definitions with Python tooling for validation and maintenance.

## Commands

```bash
# Install dependencies
uv sync

# Run tests (60 tests covering plugin integrity and validation)
uv run pytest tools/tests/ -v

# Run a single test file
uv run pytest tools/tests/test_agent_core_integrity.py -v

# Lint
uv run ruff check .

# Type check
uv run mypy --ignore-missing-imports tools/ plugins/

# Format
uv run black tools/ plugins/

# Validate all plugin metadata
make validate
# Or validate a single plugin:
python3 tools/validation/metadata_validator.py plugins/agent-core

# Check skill context budget (2% limit)
python3 tools/validation/context_budget_checker.py

# Build docs
cd docs && make html

# Clean all artifacts
make clean
```

## Architecture

### Plugin Suites (`plugins/`)

Each suite is a directory with this structure:
```
plugins/<suite-name>/
├── .claude-plugin/
│   └── plugin.json      # Manifest: declares agents, commands, skills, hooks
├── settings.json        # Default agent configuration for the suite
├── agents/              # Agent definitions (markdown with YAML frontmatter)
├── commands/            # Slash command definitions (markdown with frontmatter)
├── skills/              # Skill definitions (directories containing SKILL.md)
├── hooks/               # Hook definitions (hooks.json + Python scripts)
├── output-styles/       # Output style definitions (agent-core only)
└── README.md
```

Suite breakdown:
| Suite | Agents | Commands | Skills | Hooks | Focus |
|-------|--------|----------|--------|-------|-------|
| agent-core | 3 | 6 | 13 | 8 events | Reasoning, orchestration, context engineering, safety |
| dev-suite | 9 | 27 | 49 | 2 events | Full SDLC: architecture, implementation, CI/CD, testing, debugging |
| science-suite | 12 | 3 | 106 | 0 | JAX, Julia, physics, ML/DL/HPC, nonlinear dynamics, research |

### Component File Formats

**Agents** (`agents/<name>.md`): Markdown files with YAML frontmatter containing `name`, `description`, `model` (opus/sonnet/haiku), `effort` (low/medium/high), `memory` (user/project/local), `maxTurns`, and optionally `tools`, `background`, `isolation`, `disallowedTools`.

**Commands** (`commands/<name>.md`): Markdown files with YAML frontmatter containing `name`, `description`, and optionally `argument-hint`, `allowed-tools`, `execution-modes`.

**Skills** (`skills/<name>/SKILL.md`): Each skill lives in its own directory. The main file is always `SKILL.md` with frontmatter containing `name`, `description`.

**Hooks** (`hooks/hooks.json`): JSON object with `description` and `hooks` object keyed by event name. 8 events: SessionStart, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStop, PermissionDenied, TaskCompleted. Python scripts implement hook logic.

### Plugin Manifest (`plugin.json`)

The manifest uses **file-path references** (not inline objects) to point to agents, commands, and skills. Version is declared only in `plugin.json` (not in agent or command frontmatter).

### Python Tooling (`tools/`)

- `tools/validation/` — Validators: metadata, skills, cross-references, context budget, docs
- `tools/tests/` — Pytest suite: one integrity test per suite + validator tests
- `tools/common/` — Shared utilities: loader, models, reporter, timer
- `tools/maintenance/` — Maintenance scripts (e.g., `enable_all_plugins.py`)

## Key Conventions

- **Version sync**: All `plugin.json` files must use the same version string. Version lives only in manifests, not in agent/command/skill frontmatter.
- **Skill budget**: All skills must fit within 2% of the context window. Run `context_budget_checker.py` after adding skills.
- **Skill size governance**: Skills exceeding 3,000 bytes require review before merge. Skills at >80% of their context budget are flagged as at-risk. Skills at >90% should be refactored (split content to dedicated skills). Never add content to a frozen skill — create a new skill instead.
- **Model tiers**: Agents use `opus` (deep reasoning: orchestrator, reasoning-engine, debugger-pro, software-architect, research-expert, statistical-physicist, nonlinear-dynamics-expert, neural-network-master, simulation-expert), `sonnet` (standard tasks), or `haiku` (fast/simple: documentation-expert).
- **No wildcard imports**: `from module import *` is prohibited.
- **Python 3.13+**: Required by `pyproject.toml`.
- **uv only**: Use `uv` for all dependency management. Never install to global/user site-packages.
