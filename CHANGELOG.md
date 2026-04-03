# Changelog

## v3.1.0 (2026-04-03)

**Knowledge Gap Closure (+28 skills, +3 commands)**

* Added 6 agent-core skills: prompt-engineering-patterns, memory-system-patterns,
  safety-guardrails, tool-use-patterns, agent-evaluation, knowledge-graph-patterns.
* Added 10 dev-suite skills: database-patterns, containerization-patterns,
  cloud-provider-patterns, message-queue-patterns, caching-patterns,
  graphql-patterns, accessibility-testing, websocket-patterns,
  search-patterns, mobile-testing-patterns.
* Added 12 science-suite skills: computer-vision, nlp-fundamentals,
  bioinformatics, time-series-analysis, control-theory, experiment-tracking,
  signal-processing, symbolic-math, reinforcement-learning, quantum-computing,
  federated-learning, advanced-optimization.
* Added 3 science-suite commands: run-experiment, analyze-data, paper-review.
* Deduplicated prompt-engineering-patterns (removed science-suite copy,
  migrated resources to agent-core).
* Total: 24 agents, 36 commands, 169 skills.

**Agent Optimization (24 agents)**

* Added `background: true` to 18 agents for parallel dispatch.
* Upgraded neural-network-master and simulation-expert to opus model tier.
* Upgraded context-specialist and automation-engineer effort to high.
* Right-sized maxTurns on 4 agents.
* Restricted tools on context-specialist and research-expert to read-only.
* Added worktree isolation to systems-engineer and python-pro.
* Added "Use when..." activation triggers to all 24 agent descriptions.
* Added missing CoT frameworks (6 agents) and Constitutional AI principles (6 agents).

**Skill Quality Improvements (170 skills)**

* Added "Use when..." trigger phrases to all 170 skill descriptions (was 143).
* Expanded 9 thin skills from <200 words to 400-800 words.
* Strengthened 27 weak descriptions with proactive trigger language.
* Added domain-specific checklists to 23 skills that lacked them.
* All 170 skills now have: trigger phrases, Expert Agent sections, and checklists.
* Refactored testing-patterns from 96% to under 75% context budget.
* Froze sciml-modern-stack at 78% budget.

**Security Fixes**

* Gated `commit_fixes()` behind `--auto-commit` flag in iterative-error-resolution
  engine (default: dry-run showing diff).
* Added package name validation regex for npm/pip subprocess calls.
* Replaced `git add .` with `git add --update` for safe staging.
* Added CLI argument validation for `gh` subprocess calls.
* Anchored SessionStart hook matcher to `^(startup|resume)$`.

**Governance**

* Added skill size governance policy to CLAUDE.md (>3000 bytes = review required).

## v3.0.0 (2026-04-02)

**Julia ML/DL/HPC Expansion**

* Added `julia-ml-hpc` agent (sonnet) for Julia ML, Deep Learning, and HPC.
  Covers Lux.jl/Flux.jl, MLJ.jl, CUDA.jl, KernelAbstractions.jl, MPI.jl,
  GraphNeuralNetworks.jl, and ReinforcementLearning.jl.
* Added 10 new Julia skills: `julia-neural-networks`, `julia-neural-architectures`,
  `julia-training-diagnostics`, `julia-ad-backends`, `julia-ml-pipelines`,
  `julia-gpu-kernels`, `julia-hpc-distributed`, `julia-model-deployment`,
  `julia-graph-neural-networks`, `julia-reinforcement-learning`.
* Updated 4 existing agents (julia-pro, neural-network-master, ml-expert,
  simulation-expert) with julia-ml-hpc delegation rows.
* Added Julia cross-references to 7 existing Python/JAX skills.

**Nonlinear Dynamics Expansion (2026-03-31)**

* Added `nonlinear-dynamics-expert` agent (opus) for bifurcation theory,
  chaos analysis, network dynamics, and pattern formation.
* Added 8 nonlinear dynamics skills: bifurcation-analysis, chaos-attractors,
  pattern-formation, equation-discovery, network-coupled-dynamics, and more.

**Agent-Skill Synergy (100% Coverage)**

* Added Expert Agent pointers to all 142 skills (was 47% → now 100%).
* Agent-core: 7/7 skills now reference their owning agent with cross-references.
* Dev-suite: 39/39 skills now mapped to 9 domain agents.
* Science-suite: 29 orphan skills assigned to correct agents.

**Architecture Reorganization (5 suites -> 3 suites)**

* Merged engineering-suite + infrastructure-suite + quality-suite into a single **dev-suite**.
* Eliminates 27 cross-suite agent delegation edges — all now intra-suite.
* New structure: agent-core (3 meta-agents), dev-suite (9 agents, 27 commands, 39 skills), science-suite (12 agents, 96 skills).
* Total: 24 agents, 33 commands, 142 skills across 3 suites.

**Skill Consolidation (131 -> 124 skills)**

* Merged 7 overlapping skill pairs with zero function loss:
  - advanced-reasoning + structured-reasoning -> reasoning-frameworks
  - meta-cognitive-reflection + comprehensive-reflection-framework -> reflection-framework
  - ai-assisted-debugging + debugging-strategies -> debugging-toolkit
  - comprehensive-validation-framework merged into comprehensive-validation
  - machine-learning-essentials absorbed into machine-learning
  - parallel-computing-strategy absorbed into parallel-computing
  - python-testing-patterns + javascript-testing-patterns -> testing-patterns

**v2.1.88 Spec Compliance**

* Migrated all manifests to `.claude-plugin/plugin.json` per official plugin spec.
* Removed non-spec `version`/`color` fields from all agent and command frontmatter.
* Version now lives only in `plugin.json` (single source of truth).
* Added explicit `name` field to 27 commands that were missing it.
* Updated metadata validator schema from v2.1.42 to v2.1.88.

**Agent Hardening**

* Added `effort` field (low/medium/high) to all 24 agents.
* Extended `memory: project` to all 24 agents (was 11).
* Added explicit `tools` or `disallowedTools` to all 24 agents (was 1).
* Added `background: true` to 5 research-focused agents.
* Added `isolation: worktree` to app-developer and automation-engineer.

**Model Tier Optimization**

* Assigned Opus to 6 deep-reasoning agents: orchestrator, reasoning-engine, software-architect, debugger-pro, research-expert, statistical-physicist.
* Assigned Haiku to documentation-expert for speed-optimized doc generation.
* Fixed neural-network-master from `inherit` to explicit `sonnet`.

**Hook Expansion (3 -> 10 events)**

* agent-core: Added PostToolUse, PostCompact, SubagentStop, PermissionDenied, TaskCompleted (3 -> 8 events).
* dev-suite: Added PostToolUse (auto-lint for Python/JS/TS) and SubagentStop (2 events).

**New Infrastructure**

* Added `output-styles/` directory to agent-core and dev-suite (terse + verbose modes).
* Added `settings.json` with default agent configuration to all 3 suites.
* Added `.lsp.json` to dev-suite (Pyright + TypeScript language servers).
* Updated marketplace.json for 3-suite architecture.

**Cleanup**

* Removed obsolete `_optimization/` directory (v2.1.42 audit patches all superseded).
* Stripped `version` field from all 121 skill SKILL.md frontmatter files.
* Removed stale suite references from all command body content (13 files).
* Updated all documentation: README, CLAUDE.md, Sphinx docs, cheatsheet, changelog, agent-teams-guide.

## v2.2.1 (2026-02-15)

**Debugging Team Templates**

* Added 5 debugging agent teams (teams 34-38): `debug-gui`, `debug-numerical`, `debug-schema`, `debug-triage`, and `debug-full-audit`.
* Teams use a proven Core Trio pattern (explorer → debugger → python-pro) plus rotating domain specialists.
* Updated team-assemble catalog from 33 to 38 teams.
* Updated all docs to reflect new team count: README, cheatsheet, commands, agent-core RST and README.

**Version Consistency**

* Bumped all 208 versioned files from 2.2.0 to 2.2.1.
* Synchronized marketplace.json, plugin.json, agents, commands, skills, and docs.
* Added `.serena/` to `.gitignore`.
* Added `team-assemble` command to marketplace agent-core entry.

## v2.2.0 (2026-02-14)

**Agent Teams & Core**

* **New Feature**: Added Agent Teams support with `team-assemble` command and guide.
* Added project-level configuration support (`.serena/project.yml`).
* Implemented new hooks system (`pre_task`, `session_start`).

**Suite Updates**

* **Science Suite**: Updated agents and skills to v2.2.0; enhanced metadata validation.
* **Quality Suite**: Comprehensive skill updates and new validation tools.
* **Engineering & Infrastructure**: Full suite updates with new skills and agents.

**Marketplace & Configuration**

* Bumped all marketplace.json versions from 2.1.0 to 2.2.0.
* Added `team-assemble` command to agent-core marketplace entry.
* Added `.serena/` to `.gitignore`.

**Maintenance**

* Updated dependencies and lockfile.
* Added context budget checker tool.

## v2.1.0 (2026-01-20)

**Suite Consolidation**

* Consolidated 31 legacy plugins into 5 powerful suites:

  - **agent-core**: Multi-agent coordination, reasoning, and LLM applications
  - **engineering-suite**: Full-stack development and platform-specific implementations
  - **infrastructure-suite**: CI/CD automation, observability, and Git workflows
  - **quality-suite**: Code quality, testing, debugging, and documentation
  - **science-suite**: Scientific computing, HPC, JAX/Julia, and data science

**Flattened Skills Architecture**

* Restructured all skills to a **flat directory structure** for reliable auto-discovery.
* Previously nested skills (e.g., `jax-mastery/jax-bayesian-pro`) are now peers
  at the same directory level (e.g., `skills/jax-bayesian-pro/`).
* Meta-skills (`jax-mastery`, `julia-mastery`) remain as aggregators linking
  to related skills.
* Science suite now contains 80 flattened skills for comprehensive coverage.

**Agent Improvements**

* Standardized agent metadata with consistent colors, versions, and examples.
* Enhanced agent system prompts with explicit activation rules.
* Aligned all agents and skills to version 2.1.0.

**Command Updates**

* Renamed `feature-dev` to `eng-feature-dev` to prevent conflicts with
  the core `feature-dev` plugin.
* Added version metadata to all commands.

**Documentation**

* Updated documentation system for the new 5-suite architecture.
* Auto-generated suite RST files from plugin.json manifests.

## v2.0.0 (2025-12-15)

* Initial release of the consolidated architecture.
