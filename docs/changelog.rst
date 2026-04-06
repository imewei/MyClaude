Changelog
=========

v3.1.0 (2026-04-03)
-------------------

**Hub-Skill Architecture**

* Introduced :term:`Hub Skill` routing: 26 hub skills route to 167 :term:`sub-skills <Sub-Skill>` via
  :term:`routing decision trees <Routing Decision Tree>`. Hubs are declared in ``plugin.json``; sub-skills
  are discovered through hub routing. Eliminates ambiguous flat-list skill matching.
* agent-core: 3 hubs (agent-systems, reasoning-and-memory, llm-engineering) → 12 sub-skills.
* dev-suite: 9 hubs (backend-patterns, frontend-and-mobile, architecture-and-infra,
  testing-and-quality, ci-cd-pipelines, observability-and-sre, python-toolchain,
  data-and-security, dev-workflows) → 49 sub-skills.
* science-suite: 14 hubs → 106 sub-skills.
* Total: 24 agents, 14 registered commands, 26 hubs → 167 sub-skills (193 total).

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
* Added 3 science-suite commands: ``run-experiment``, ``analyze-data``, ``paper-review``.
* Deduplicated ``prompt-engineering-patterns`` (science-suite copy removed, migrated to agent-core).

**Agent Optimization (24 agents)**

* Added ``background: true`` to 18 agents for parallel dispatch.
* Upgraded neural-network-master and simulation-expert to opus model tier.
* 9 opus agents: orchestrator, reasoning-engine, software-architect, debugger-pro,
  research-expert, statistical-physicist, nonlinear-dynamics-expert,
  neural-network-master, simulation-expert.
* Added "Use when..." activation triggers to all 24 agent descriptions.
* Fixed cross-suite delegation annotations and invalid agent references.

**Skill Quality & Integrity**

* All 193 skills have: trigger phrases, Expert Agent sections, and checklists.
* Fixed 235 broken relative links across 38 files.
* Zero orphaned skills — all 193 reachable via hub routing.
* Resolved routing overlaps (ml-and-data-science/ml-deployment/deep-learning-hub triangle).
* Refactored testing-patterns from 96% to under 75% context budget.
* 193/193 skills within 2% context budget.

**Security Fixes**

* Gated ``commit_fixes()`` behind ``--auto-commit`` flag (default: dry-run).
* Added package name validation regex for npm/pip subprocess calls.
* Replaced ``git add .`` with ``git add --update`` for safe staging.
* Anchored SessionStart hook matcher to ``^(startup|resume)$``.

**Documentation**

* Rewrote all reference docs for hub architecture.
* Added :term:`Hub Skill`, :term:`Sub-Skill`, :term:`Routing Decision Tree`, and
  :term:`Agent Team` to glossary.
* Updated all workflow guides with hub → sub notation.
* Docs build with zero warnings. 60/60 tests pass.

**Governance**

* Added skill size governance policy (>3000 bytes = review required).
* 14 commands intentionally registered; 22 skill-invoked by design.

v3.0.0 (2026-04-02)
-------------------

**Julia ML/DL/HPC Expansion**

* Added ``julia-ml-hpc`` agent (sonnet) for Julia ML, Deep Learning, and HPC.
  Covers Lux.jl/Flux.jl, MLJ.jl, CUDA.jl, MPI.jl, GraphNeuralNetworks.jl, and
  ReinforcementLearning.jl. Delegates SciML/ODE work to ``julia-pro``.
* Added 10 new Julia skills: ``julia-neural-networks``, ``julia-neural-architectures``,
  ``julia-training-diagnostics``, ``julia-ad-backends``, ``julia-ml-pipelines``,
  ``julia-gpu-kernels``, ``julia-hpc-distributed``, ``julia-model-deployment``,
  ``julia-graph-neural-networks``, ``julia-reinforcement-learning``.
* Updated 4 existing agents with ``julia-ml-hpc`` delegation rows.

**Nonlinear Dynamics Expansion (2026-03-31)**

* Added ``nonlinear-dynamics-expert`` agent (opus) for bifurcation theory,
  chaos analysis, network dynamics, and pattern formation.
* Added 8 nonlinear dynamics skills: bifurcation-analysis, chaos-attractors,
  pattern-formation, equation-discovery, network-coupled-dynamics, and more.

**Agent-Skill Synergy (100% coverage)**

* Added Expert Agent pointers to all 142 skills (47% → 100%).
* Dev-suite: 39/39 skills mapped to 9 domain agents.
* Science-suite: 29 orphan skills assigned to correct agents.

**Architecture Reorganization (5 → 3 suites)**

* Merged engineering-suite + infrastructure-suite + quality-suite into ``dev-suite``.
  Eliminates 27 cross-suite agent delegation edges.
* New structure: agent-core (3 meta-agents), dev-suite (9 agents, 27 commands,
  39 skills), science-suite (12 agents, 96 skills).

**v2.1.88 Spec Compliance**

* Migrated all manifests to ``.claude-plugin/plugin.json`` per official plugin spec.
* Removed non-spec ``version``/``color`` fields from all agent and command frontmatter.
* Version now lives only in ``plugin.json`` (single source of truth).

**Agent Hardening**

* Added ``effort``, ``memory``, ``tools``/``disallowedTools`` fields to all 24 agents.
* Added ``isolation: worktree`` to app-developer and automation-engineer.

**Model Tier Optimization**

* Assigned Opus to 6 deep-reasoning agents; Haiku to documentation-expert.
* Fixed neural-network-master from ``inherit`` to explicit ``sonnet``.

**Hook Expansion (3 → 10 events)**

* agent-core: Added PostToolUse, PostCompact, SubagentStop, PermissionDenied,
  TaskCompleted (3 → 8 events).
* dev-suite: Added PostToolUse and SubagentStop (2 events).

**Skill Consolidations (7 merges)**

* advanced-reasoning + structured-reasoning → reasoning-frameworks
* meta-cognitive-reflection + comprehensive-reflection-framework → reflection-framework
* ai-assisted-debugging + debugging-strategies → debugging-toolkit
* comprehensive-validation-framework → comprehensive-validation
* machine-learning-essentials → machine-learning
* parallel-computing-strategy → parallel-computing
* python-testing-patterns + javascript-testing-patterns → testing-patterns

v2.2.1 (2026-02-15)
-------------------

**Debugging Team Templates**

* Added 5 debugging :term:`agent teams <Agent Team>`: debug-gui, debug-numerical,
  debug-schema, debug-triage, and debug-full-audit.
* Teams use a Core Trio pattern (explorer → debugger → python-pro) plus rotating specialists.
* Consolidated from 35 to 21 team templates (40% reduction): merged 5 overlapping pairs
  (pr-review, quality-security, sci-compute, md-simulation, docs-publish), removed 7 niche
  teams, added alias table for backward compatibility. Total: 21 team templates.

**Agent Teams System**

* New ``/team-assemble`` command with pre-built team configurations.
* Teams span 5 categories: Development & Operations, Scientific Computing,
  Cross-Suite Specialized, Official Plugin Integration, and Debugging.
* Integrated 20 official plugin agents (pr-review-toolkit, feature-dev,
  coderabbit, plugin-dev, hookify, huggingface-skills, agent-sdk-dev, superpowers).
* Quality Gate Enhancers for adding review agents to any team.
* Comprehensive reference guide at ``docs/agent-teams-guide.md``.

**Agent Enhancements**

* Added adaptive thinking references to reasoning-engine agent.
* Integrated Agent Teams coordination into orchestrator agent.
* Added ``memory`` frontmatter to 11 key agents for persistent context.

**Hooks Infrastructure**

* Added hooks support to agent-core suite (``SessionStart``, ``PreToolUse``).
* New ``hooks/hooks.json`` configuration in agent-core plugin manifest.

**Tooling**

* Added context budget checker tool (``tools/validation/context_budget_checker.py``).

v2.2.0 (2026-02-14)
-------------------

* Added Agent Teams support with ``team-assemble`` command and guide.
* Updated all suites to v2.2.0 for Claude Opus 4.6 compatibility.
* Added context budget checker tool.

v2.1.0 (2026-01-20)
-------------------

**Suite Consolidation**

* Consolidated 31 legacy plugins into 5 suites: agent-core, engineering-suite,
  infrastructure-suite, quality-suite, science-suite.

**Flattened Skills Architecture**

* Restructured all skills to a flat directory structure for reliable auto-discovery.
* Science suite: 80 flattened skills for comprehensive coverage.

**Agent & Command Updates**

* Standardized agent metadata with consistent colors, versions, and examples.
* Renamed ``feature-dev`` to ``eng-feature-dev`` to prevent conflicts.

v2.0.0 (2025-12-15)
-------------------

* Initial release of the consolidated architecture.
