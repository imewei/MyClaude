# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MyClaude is a Claude Code plugin marketplace: 4 plugin suites containing 25 agents, 14 registered commands, and 38 registered hub/standalone skills (routing to 179 sub-skills; 217 SKILL.md files total on disk). It is **not** a runnable application — it's a collection of markdown-based plugin definitions with Python tooling for validation and maintenance.

## Commands

```bash
# Install dependencies
uv sync

# Run tests (180 tests covering plugin integrity and validation)
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

# Run pip-audit CVE scan
make audit

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
| Suite | Agents | Registered Cmds | Skills (registered → sub) | Hooks | Focus |
|-------|--------|-----------------|---------------------------|-------|-------|
| agent-core | 3 | 2 | 4 → 13 | 12 events | Reasoning, orchestration, context engineering, safety |
| dev-suite | 9 | 12 | 9 → 49 | 7 events | Full SDLC: architecture, implementation, CI/CD, testing, debugging |
| research-suite | 2 | 0 | 11 → 5 | 0 events | Peer review (`scientific-review`), 8-stage `research-spark` pipeline, methodology hub (`research-practice`) |
| science-suite | 11 | 0 | 14 → 112 | 5 events | JAX, Julia, physics, ML/DL/HPC, nonlinear dynamics (pure computational — research methodology moved to research-suite) |

**Note:** 22 additional commands exist on disk but are not registered in `plugin.json`. They split into two categories:

- **~10 are skill-invoked**: referenced from a skill's routing tree or an agent's body and triggered during workflows, not directly by users as `/slash-commands`.
- **12 are intentional reference templates**: substantial command files (29-318 lines) that are kept on disk for reference and for users to copy/adapt, but are NOT currently invoked from any skill, agent, or other command. The list (verified 2026-04-18 via grep across `plugins/`): `adopt-code`, `agent-build`, `ai-assistant`, `c-project`, `deps`, `monitor-setup`, `onboard`, `paper-review`, `profile-performance`, `run-experiment`, `rust-project`, `scaffold`. To make any of these user-invocable, add `"./commands/<name>.md"` to the suite's `plugin.json` `commands` array.

### Hub-Skill Architecture

Skills use a two-tier **hub-skill routing** system:
1. `plugin.json` declares only **hub skills** (meta-orchestrators).
2. Each hub contains a **routing decision tree** that dispatches to specialized **sub-skills**.
3. Sub-skills are discovered through hub references, not directly from the manifest.

```
plugin.json → hub SKILL.md → routing tree → ../sub-skill/SKILL.md
```

Each hub SKILL.md has a standard structure: YAML frontmatter, Expert Agent reference, Core Skills section with `../` relative links, Routing Decision Tree (code block), and Checklist.

### Component File Formats

**Agents** (`agents/<name>.md`): Markdown files with YAML frontmatter containing `name`, `description`, `model` (opus/sonnet/haiku/inherit), `effort` (low/medium/high), `memory` (user/project/local), `maxTurns`, and optionally `tools`, `background`, `isolation`, `disallowedTools`, `skills` (list of hub skill names to preload), `permissionMode` (default/acceptEdits/auto/dontAsk/bypassPermissions/plan), `hooks`, `color`.

**Commands** (`commands/<name>.md`): Markdown files with YAML frontmatter containing `name`, `description`, and optionally `argument-hint`, `allowed-tools`, `execution-modes`.

**Skills** (`skills/<name>/SKILL.md`): Each skill lives in its own directory. The main file is always `SKILL.md` with frontmatter containing `name`, `description`. Hub skills additionally contain: Expert Agent section, Core Skills with `../` relative links, Routing Decision Tree, and Checklist.

**Hooks** (`hooks/hooks.json`): JSON object with `description` and `hooks` object keyed by event name. We implement 24 handlers across all suites (12 agent-core, 7 dev-suite, 5 science-suite). Events include SessionStart, SessionEnd, PreToolUse, PostToolUse, PreCompact, PostCompact, SubagentStart, SubagentStop, PermissionDenied, TaskCreated, TaskCompleted, StopFailure. Handler types: command (shell), HTTP, prompt, agent. Python scripts implement hook logic. Note: PreSubagentUse, ExecutionError, PermissionPrompt, ContextOverflow, and CostThreshold were removed — not yet supported by the CLI event schema.

### Plugin Manifest (`plugin.json`)

The manifest uses **file-path references** (not inline objects) to point to agents, commands, and skills. Version is declared only in `plugin.json` (not in agent or command frontmatter).

### Python Tooling (`tools/`)

- `tools/validation/` — Validators (run individually or via `make validate`):
  - `metadata_validator.py` — Validate plugin.json structure: `python3 tools/validation/metadata_validator.py plugins/<suite>`
  - `context_budget_checker.py` — Check skill sizes against 2% limit: `python3 tools/validation/context_budget_checker.py [--plugins-dir DIR] [--context-size N]`
  - `skill_validator.py` — Test skill triggering patterns: `python3 tools/validation/skill_validator.py [--plugins-dir DIR] [--plugin NAME] [--corpus-dir DIR]`
  - `xref_validator.py` — Validate cross-plugin references and broken links
  - `doc_checker.py` — Check documentation completeness (README sections, markdown formatting)
  - `plugin_review_script.py` — Full automated plugin review with structured markdown report
- `tools/tests/` — Pytest suite: one integrity test per suite + validator tests
- `tools/common/` — Shared utilities: loader, models, reporter, timer
- `tools/maintenance/` — Maintenance scripts:
  - `enable_all_plugins.py` — Enable all plugins in settings
  - `analyze_ecosystem.py` — Analyze skill/agent ecosystem metrics

## Key Conventions

- **Version sync**: All `plugin.json` files must use the same version string. Version lives only in manifests, not in agent/command/skill frontmatter.
- **Skill budget**: All skills must fit within 2% of the context window. Run `context_budget_checker.py` after adding skills.
- **Skill size governance**: Skills exceeding 3,000 bytes require review before merge. Skills at >80% of their context budget are flagged as at-risk. Skills at >90% should be refactored (split content to dedicated skills). Never add content to a frozen skill — create a new skill instead.
- **Model tiers**: Agents use `opus` (deep reasoning: orchestrator, reasoning-engine, context-specialist, debugger-pro, software-architect, research-expert, research-spark-orchestrator, statistical-physicist, nonlinear-dynamics-expert, neural-network-master, simulation-expert), `sonnet` (standard tasks), or `haiku` (fast/simple: documentation-expert).
- **Command registration**: Only 14 commands are registered in `plugin.json` manifests (2 agent-core, 12 dev-suite). The remaining 22 command files on disk are skill-invoked — do not add them to manifests.
- **Hub routing**: When adding a new sub-skill, it must be referenced by at least one hub's Core Skills section and Routing Decision Tree. Run the orphan check to verify reachability.
- **Team templates**: 10 focused agent teams with 20 variants in `plugins/agent-core/commands/team-assemble.md`. Teams use `--var MODE=x` for specialization. 20 aliases preserve backward compatibility (Step 5). No duplicate agent types per team.
- **Cross-suite delegation**: Agent delegation tables referencing agents from other suites must include a `(suite-name)` annotation, e.g., `ml-expert (science-suite)`.
- **No wildcard imports**: `from module import *` is prohibited.
- **Python 3.13+**: Required by `pyproject.toml`.
- **uv only**: Use `uv` for all dependency management. Never install to global/user site-packages.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- **Before any file exploration, multi-file search, or codebase question**, read graphify-out/GRAPH_REPORT.md. Focus on: "God Nodes" (most-connected abstractions), "Community Hubs" (navigation entry points), and "Surprising Connections" (non-obvious cross-module relationships).
- Prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep/find for any cross-module question — these traverse EXTRACTED + INFERRED edges instead of scanning files.
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost).
