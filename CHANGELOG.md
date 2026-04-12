# Changelog

## v3.1.6 (2026-04-11)

**Julia ↔ Python Parity Polish**

* **Julia → Python handoff for nonlinear time-series tools.** Added a new
  "Calling Python nonlinear-dynamics tools from Julia via PythonCall.jl"
  section in `chaos-attractors` covering nolds / antropy / IDTxl / pyEDM /
  pyunicorn / teaspoon (no native Julia equivalents). Canonical
  `PythonCall.jl` + `CondaPkg.jl` import pattern with a concrete `lyap_r`
  example and GIL-under-`@threads` caveats. Completes the Julia ↔ Python
  interop story with v3.1.5's reciprocal juliacall bifurcation path.
  Pointer edits in `nonlinear-dynamics` (hub ecosystem-selection table)
  and `time-series-analysis`.
* **BAR free-energy worked example** bridging Langevin ensemble and
  non-equilibrium theory. Added a 4-stage pipeline to
  `non-equilibrium-theory`: (1) JAX Langevin ensemble + `jax.lax.scan`
  to accumulate forward/reverse work samples, (2) BAR fit via
  `pymbar.other_estimators.bar` (Context7-verified — pymbar 4.0 moved
  it out of top level), (3) variance comparison vs Jarzynski cumulant
  expansion, (4) multi-state MBAR pointer with `alchemlyb` ecosystem
  wrapper for LAMMPS/GROMACS/NAMD/AMBER/OpenMM alchemical output.
  One-sentence cross-link added to `stochastic-dynamics` to preserve
  its 74% budget cap.
* **freud ecosystem for physical correlations.** Added a "Python freud
  ecosystem" section to `correlation-physical-systems` covering
  `freud.density.RDF`, `StaticStructureFactorDebye` /
  `StaticStructureFactorDirect` (with API-drift warning: Debye takes
  `num_k_values` not `bins`), Steinhardt `Q_l`, Hexatic, Nematic, and
  the `SolidLiquid` phase classifier. `freud.density.IntermediateScattering`
  tagged `[unverified]` (absent in freud 3.5.0) with a `numpy.fft` +
  MDAnalysis fallback. Algorithmic notes in
  `correlation-computational-methods` (AABBQuery neighbor-list reuse,
  `reset=False` multi-frame averaging, CuPy breakeven N≈10⁴, the
  "MDAnalysis/MDTraj as iterator, freud as analyzer" production pattern).
  One-line hub pointer in `correlation-analysis` with a
  `PythonCall.jl` handoff note for Julia users.

**Deferred**

* Item A (ML-FF CLI spot-check) — explicitly deferred until the user
  resumes active MLIP training.

**Validator State**

* metadata_validator: 0 errors / 0 warnings across all 3 plugins.
* xref_validator: 519/519 references valid.
* context_budget_checker: 205/205 skills fit. `non-equilibrium-theory`
  at 73.8% (under 75% Commit C gate, fallback #2 fired — Stage 3
  collapsed from for-loop to prose). `correlation-physical-systems`
  at 74.45% (under 75% Commit D gate, fallbacks #1+#2 fired — caveats
  compressed to 2 bullets, intermediate scattering as prose).
  `chaos-attractors` at 79% (under 80% at-risk line, compressed
  caveats).
* skill_validator: EXCELLENT on science-suite + agent-core.
* pytest: 118/118 passing.
* ruff: all checks pass. mypy: 0 errors (42 source files).

**Known forward items for v3.1.7+**

* `equation-discovery` at 88% budget (pre-existing from external
  commit, not a v3.1.6 regression) — flagged for v3.1.7 Bayesian
  SINDy extraction split.
* `freud.density.IntermediateScattering` presence in newer freud
  releases — re-verify when the correlation skills are next touched.

## v3.1.5 (2026-04-11)

**Julia/Python Parity Pass**

* **Fokker-Planck direct PDE methods** added to `stochastic-dynamics`:
  finite-difference / spectral discretization of the Fokker-Planck
  equation, boundary condition patterns, and cross-links to Langevin
  sampling.
* **Python bifurcation continuation escape hatch** added to
  `bifurcation-analysis` and `nonlinear-dynamics` — since
  `BifurcationKit.jl` is blocked on Julia 1.12, documented the
  `juliacall` path for calling Julia bifurcation routines from a
  Python-primary codebase, plus PyDSTool and AUTO-07p as
  Python-native alternatives.
* **Modern ML force fields** expansion in `ml-force-fields`:
  equivariant GNNs (NequIP, MACE, Allegro), Julia ACE stack
  (`ACEpotentials.jl`), training loops, active learning, energy-
  and-force loss balance. Budget-tight at 78%.
* **Julia Monte Carlo idioms** for `statistical-physics`: Metropolis
  sampler patterns with `@inbounds` / `@fastmath`, SIMD inner loops,
  parallel tempering via `Distributed.jl`, and tuning heuristics.

**Tooling & Dependencies**

* Added `types-PyYAML` dev dependency for mypy stubs.
* Tightened `self-improving-ai` triggers so they no longer overlap
  with `dspy-basics`.
* Added `tools/validation/command_file_linter.py` — a targeted
  structural linter for Claude Code command files with 5 rules
  (`fence-unbalanced` error, `heading-skip` warning,
  `step-ref-broken` error, `trailing-whitespace` info,
  `heading-duplicate` warning). Importable API + standalone CLI.
  Wired into `make validate`; non-blocking warnings. Caught a
  pre-existing duplicate `## Metrics` H2 in
  `dev-suite/commands/tech-debt.md`.
* Added 15 new linter tests; pytest suite grew from 60 → 103.

**Validator State**

* metadata_validator: 0/0/0 on all 3 plugins.
* xref_validator: 515/515 references valid (+3 from v3.1.4).
* context_budget_checker: 204/204 skills fit; 2 headroom warnings
  (`bifurcation-analysis` 79%, `ml-force-fields` 78%) — both under
  the 80% at-risk line.
* skill_validator: EXCELLENT on science-suite + agent-core.
* pytest: 103 passing (60 pre-existing + 12 non-English handling + 15 linter + 16 team-assemble).
* ruff clean. mypy: 0 errors with new `types-PyYAML` stub.

## v3.1.4 (2026-04-11)

**Research-Focus Optimization Pass (science-suite)**

* Aligned agents and skills with research in Bayesian MCMC
  (NUTS / Consensus MC / Pigeons), Universal Differential Equations,
  SINDy, nonlinear dynamics, time series, rare events / avalanche
  dynamics, non-equilibrium statistical physics, and point / jump
  processes.
* **9 new sub-skills** (none registered in `plugin.json` — discovered
  via hub references per the hub-skill architecture):
  `consensus-mcmc-pigeons` (non-reversible parallel tempering via
  Pigeons.jl, distinguished from Scott-2016 divide-and-conquer
  Consensus Monte Carlo), `bayesian-ude-workflow` (Turing + DiffEq +
  Lux staged pipeline), `bayesian-ude-jax` (Python/JAX counterpart
  via Diffrax + Equinox + NumPyro), `bayesian-pinn`
  (BNNODE/BayesianPINN split out of `neural-pde` to keep it under
  budget — neural-pde dropped from 78% → 65%), `point-processes`
  (Hawkes / HSGP / Julia PointProcesses.jl), `rare-events-sampling`
  (large-deviation / cloning / avalanche statistics),
  `self-improving-ai` (research overview for autonomous improvement),
  `dspy-basics` (DSPy programmatic prompts depth-skill), and
  `rlaif-training` (Constitutional AI / RLAIF / DPO depth-skill).
* **1 new agent-core skill**: `self-improving-agents` under the
  `reasoning-and-memory` hub — operational counterpart to
  science-suite's `self-improving-ai` (agents inside Claude Code vs
  research framework overview). Cross-linked with
  `prompt-engineering-patterns`.

**Research Audit Remediation**

* Closed 8 findings from the 2026-04-11 research audit:
  - Added `extreme-value-statistics` skill (GEV/GPD/Hill/Pickands/POT,
    return levels, non-stationary EVT) and wired it into
    `statistical-physics-hub`.
  - Wired the orphaned `robust-testing` sub-skill into the
    `research-and-domains` hub so it is reachable from hub routing.
  - Extended `rare-events-sampling` triggers to cover SOC, sandpile /
    Bak-Tang-Wiesenfeld, crackling noise, and avalanche-size
    distributions.
  - Resolved jump-diffusion routing: `stochastic-dynamics` now owns
    general physics jump-diffusion SDEs (Lévy flights, shot noise,
    regime-switching Langevin); `catalyst-reactions` stays scoped
    to biochemical reaction networks.
  - Added Bayesian SINDy coverage to `equation-discovery` (horseshoe-
    prior NumPyro, ensemble SINDy, UQ-SINDy via Turing) — flagged
    for v3.1.7 extraction when it pushed `equation-discovery` to 88%.
  - Disambiguated `sciml-modern-stack` vs `sciml-and-diffeq` by
    rewriting hub routing without touching the frozen
    `sciml-modern-stack` body.
  - Added missing trigger keywords (ADF, KPSS, Phillips-Perron, PELT,
    BinSeg, renewal processes, non-parametric Hawkes EM) to
    `time-series-analysis` and `point-processes`.

**Agent Updates**

* `julia-pro`: Bayesian stack upgraded to Turing + Pigeons; sensealg
  table rewritten with `GaussAdjoint` as the modern default and the
  ForwardDiff-bypasses-sensealg factual fix; decision tree adds UDE
  and multimodal branches.
* `julia-ml-hpc`: Related Skills table extended; Bayesian UDE
  delegation.
* `statistical-physicist`: Related Skills extended with
  `stochastic-dynamics`, `non-equilibrium-theory`, `point-processes`,
  `rare-events-sampling`, `correlation-math-foundations`,
  `correlation-physical-systems`.
* `ai-engineer`, `jax-pro`, `simulation-expert` also aligned.

**Codebase-Aware /team-assemble (agent-core)**

* Major rework of `/team-assemble`: static catalog → codebase-aware
  recommender / adapter / validator. **21 → 25 team templates.**
* **4 new teams**: `nonlinear-dynamics` (bifurcation, chaos, coupled
  oscillators, pattern formation — wires `nonlinear-dynamics-expert`
  to its documented delegation targets for the first time),
  `julia-ml` (Lux.jl/Flux.jl/MLJ.jl + CUDA.jl/MPI.jl distributed
  training), `multi-agent-systems` (orchestrator + reasoning-engine
  + context-specialist + ai-engineer for production multi-agent
  apps), `sci-desktop` (PyQt6/PySide6 + JAX scientific desktop apps
  with view/logic decoupling invariants).
* `ai-engineering` team swaps `reasoning-architect` for
  `context-architect` as the default 4th teammate (RAG/memory is
  the more common production need).
* Closed drift: **0 unused local agents** (down from 4 — orchestrator,
  context-specialist, julia-ml-hpc, nonlinear-dynamics-expert were
  previously unreferenced).
* New capabilities: Step 1.5 Codebase Detection (4-tier signal
  gathering with efficiency gates), Step 2.5 Signal → Team Mapping
  (25-row canonical fingerprint table), Step 2.6 Rank & Recommend
  (rule-based scoring with confidence labels), Step 2.6a Validation +
  2.6b Auto-fill, five new invocation modes (no-arg recommendation,
  `--no-detect` escape hatch).
* **Session cache**: Tier 0 cache at
  `/tmp/team-assemble-cache/<sanitized-abspath>.json` with mtime-based
  invalidation (15 min TTL); `--no-cache` bypass flag; never caches
  `project_type=unknown` results.
* **S1 prompt-injection safeguards** for README probes (HIGH
  severity): character neutralization, `<untrusted_readme_excerpt>`
  wrapping, 9 refusal-trigger patterns. Non-English README
  hardening via `language_hint` classification and auto-fill trust
  tiers (`latin`/`non-latin`/`mixed`/`empty` → `standard`/`low`/
  `very_low`). Non-Latin content is still wrapped and emitted, but
  surfaced under a dedicated review header.

**Tooling Hardening**

* Added `sys.path.insert` to 5 validators (`metadata_validator`,
  `xref_validator`, `skill_validator`, `doc_checker`,
  `plugin_review_script`) so CLI invocation via
  `python tools/validation/X.py` works without `PYTHONPATH=.`.
* `PluginLoader` consolidates YAML frontmatter parsing via new
  helpers (`_read_frontmatter`, `_normalize_component_entry`,
  `_normalize_component_list`); `skill_validator` and
  `xref_validator` consume normalized dicts.
* `xref_validator` gains disk-discovery of sub-skills so the
  hub-skill architecture validates cleanly: sub-skills not
  registered in `plugin.json` no longer false-positive as broken
  references. New regex extractors for relative and absolute skill
  links.
* Restored `requires = ["maturin>=1.0,<2.0"]` in rust-extensions
  scaffold — maturin is the actual PEP 517 build backend for PyO3
  projects; any future maturin 2.x must be an opt-in upgrade.
* `pyproject.toml`: excluded `test-corpus/` from mypy and ruff
  (scientific fixture files that are not installable).

**Key Refactors**

* Split `neural-pde` → `bayesian-pinn` to keep `neural-pde` under
  budget (65% of 200K tokens, down from 78%).
* `time-series-analysis` points to dedicated `point-processes` for
  event-time content.
* `stochastic-dynamics` gains JAX Langevin ensemble + SDE library
  selector + Fokker-Planck numerical path.
* `non-equilibrium-theory` gains BAR / Jarzynski / entropy
  production compute patterns + large-deviation theory with
  avalanche statistics.
* `catalyst-reactions` adds PDMP / jump-diffusion / `JumpProcesses.jl`
  beyond pure Gillespie.
* `equation-discovery` expands SINDy ecosystem (PySINDy + PyDMD +
  PySR).

**Validator State**

* metadata 0/0/0, xref 512/512 valid, context budget 204/204,
  pytest 60/60, ruff clean.

## v3.1.3 (2026-04-10)

**New Skill: thinkfirst (agent-core)**

* Added `thinkfirst` as a sub-skill under the `llm-engineering` hub.
  Interview-first workflow that clarifies vague user intent through
  a **Seven Dimensions framework** before any prompt is drafted —
  addresses the common failure mode where Claude generates polished
  prompts from ambiguous requirements.
* Positioned as the first branch in the `llm-engineering` routing
  tree so users with brain dumps hit clarification before reaching
  for production templates.
* Cross-linked with `prompt-engineering-patterns` to make the
  upstream/downstream relationship explicit: `thinkfirst` handles
  intent clarification, `prompt-engineering-patterns` handles
  production-grade refinement.

## v3.1.2 (2026-04-06)

**Bug Fixes**

* Removed duplicate `hooks` manifest entries from agent-core and dev-suite
  `plugin.json`. The `hooks/hooks.json` file is auto-discovered by convention;
  declaring it explicitly caused duplicate-load errors at startup.
* Fixed dev-suite `.lsp.json` structure to match expected schema.

**Documentation**

* Updated plugin READMEs to use hub→sub-skill notation matching CLAUDE.md
  (agent-core 13→15, dev-suite 49→58, science-suite 107→120 total skills).
* Rewrote `tools/README.md` to reflect current tooling (removed references to
  deleted `tools/generation/` directory and pre-v3.0 plugin names).

## v3.1.1 (2026-04-06)

**Bug Fixes**

* Set `strict: true` in marketplace.json to resolve conflicting manifests
  when both marketplace.json and individual plugin.json files declare components.
* Fixed agent teams guide reference in README (34 → 21 teams).

## v3.1.0 (2026-04-03)

**Hub-Skill Architecture**

* Introduced hub-skill routing: 26 hub skills route to 167 sub-skills via
  decision trees. Hubs are declared in plugin.json; sub-skills are discovered
  through hub routing. Eliminates ambiguous flat-list skill matching.
* agent-core: 3 hubs (agent-systems, reasoning-and-memory, llm-engineering) → 12 sub-skills.
* dev-suite: 9 hubs (backend-patterns, frontend-and-mobile, architecture-and-infra,
  testing-and-quality, ci-cd-pipelines, observability-and-sre, python-toolchain,
  data-and-security, dev-workflows) → 49 sub-skills.
* science-suite: 14 hubs (nonlinear-dynamics, jax-computing, julia-language,
  julia-ml-and-dl, sciml-and-diffeq, correlation-analysis, statistical-physics-hub,
  deep-learning-hub, ml-and-data-science, llm-and-ai, ml-deployment,
  simulation-and-hpc, research-and-domains, bayesian-inference) → 106 sub-skills.
* Each hub has: YAML frontmatter, Expert Agent reference, Core Skills with
  relative links, Routing Decision Tree, and Checklist.
* 14 registered commands (2 agent-core, 12 dev-suite, 0 science-suite).
  22 additional commands on disk are skill-invoked, not user-facing.
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
* Added 3 science-suite commands: run-experiment, analyze-data, paper-review.
* Deduplicated prompt-engineering-patterns (removed science-suite copy,
  migrated resources to agent-core).

**Agent Optimization (24 agents)**

* Added `background: true` to 18 agents for parallel dispatch.
* Upgraded neural-network-master and simulation-expert to opus model tier.
* 9 opus agents: orchestrator, reasoning-engine, software-architect, debugger-pro,
  research-expert, statistical-physicist, nonlinear-dynamics-expert,
  neural-network-master, simulation-expert.
* Right-sized maxTurns on 4 agents.
* Added "Use when..." activation triggers to all 24 agent descriptions.
* Fixed cross-suite delegation annotations (added `(dev-suite)` / `(science-suite)`).
* Fixed invalid `security-auditor` → `quality-specialist` in context-specialist.
* Fixed duplicate `ml-expert` row → `devops-architect (dev-suite)` in simulation-expert.

**Skill Quality & Integrity**

* All 193 skills have: trigger phrases, Expert Agent sections, and checklists.
* Fixed 235 broken relative links (`./` → `../`, `.../` → `../`) across 38 files.
* Zero orphaned skills — all 193 reachable via hub routing.
* Resolved routing overlaps: ml-and-data-science/ml-deployment/deep-learning-hub
  triangle disambiguated; error-handling-patterns scoped to Python;
  iterative-error-resolution scoped to CI/CD.
* Reduced sciml-modern-stack from 84% → 79% budget (NeuralPDE.jl dedup).
* Refactored testing-patterns from 96% to under 75% context budget.
* 193/193 skills within 2% context budget.

**Security Fixes**

* Gated `commit_fixes()` behind `--auto-commit` flag in iterative-error-resolution
  engine (default: dry-run showing diff).
* Added package name validation regex for npm/pip subprocess calls.
* Replaced `git add .` with `git add --update` for safe staging.
* Added CLI argument validation for `gh` subprocess calls.
* Anchored SessionStart hook matcher to `^(startup|resume)$`.

**Team Consolidation (35 → 21 teams)**

* Consolidated `/team-assemble` from 35 to 21 templates (40% reduction).
* Merged 5 overlapping pairs: pr-review + full-pr-review, quality-audit +
  security-harden + code-health, sci-pipeline + dl-research, md-campaign +
  ml-forcefield, docs-sprint + reproducible-research.
* Removed 7 niche/broken teams: duplicate feature-dev, hf-ml-publish (broken ref),
  hpc-interop, monorepo-refactor, codebase-archaeology, agent-sdk-build,
  stat-phys, debug-full-audit.
* Added alias resolution table for backward compatibility with old team names.
* Fixed duplicate agent types per team, sharpened role names, added debug
  cross-references, improved placeholder specificity.

**Documentation**

* Rewrote all reference docs for hub architecture: agents.md, commands.md,
  cheatsheet.md, 3 suite RST files, README.md, CLAUDE.md, index.rst, changelog.
* Rewrote agent-teams-guide.md for 21-team catalog with updated quick reference
  and detailed sections.
* Docs build with zero warnings. 60/60 tests pass.

**Governance**

* Added skill size governance policy to CLAUDE.md (>3000 bytes = review required).
* 14 commands intentionally registered; 22 skill-invoked commands on disk by design.

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
