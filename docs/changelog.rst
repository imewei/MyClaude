Changelog
=========

v3.1.6 (2026-04-11)
-------------------

**Julia ↔ Python Parity Polish**

* **Julia → Python handoff for nonlinear time-series tools.** New section
  in ``chaos-attractors`` covering ``nolds``, ``antropy``, ``IDTxl``,
  ``pyEDM``, ``pyunicorn``, ``teaspoon`` (no native Julia equivalents)
  via the canonical ``PythonCall.jl`` + ``CondaPkg.jl`` import pattern
  with a concrete ``lyap_r`` example and GIL-under-``@threads`` caveats.
  Pointer edits in ``nonlinear-dynamics`` hub ecosystem-selection table
  and ``time-series-analysis``. Completes the Julia ↔ Python interop
  story with v3.1.5's reciprocal ``juliacall`` bifurcation path.
* **BAR free-energy worked example** bridging Langevin ensemble and
  non-equilibrium theory. Added a 4-stage pipeline to
  ``non-equilibrium-theory``: (1) JAX Langevin ensemble + ``jax.lax.scan``
  to accumulate forward/reverse work samples, (2) BAR fit via
  ``pymbar.other_estimators.bar`` (Context7-verified — pymbar 4.0 moved
  it out of top level), (3) variance comparison vs Jarzynski cumulant
  expansion, (4) multi-state MBAR pointer with ``alchemlyb`` ecosystem
  wrapper. One-sentence cross-link added to ``stochastic-dynamics`` to
  preserve its 74% budget cap.
* **freud ecosystem for physical correlations.** Added a "Python freud
  ecosystem" section to ``correlation-physical-systems`` covering
  ``freud.density.RDF``, ``StaticStructureFactorDebye`` /
  ``StaticStructureFactorDirect`` (with API-drift warning:
  ``StaticStructureFactorDebye`` takes ``num_k_values``, not ``bins``),
  Steinhardt ``Q_l``, Hexatic, Nematic, and ``SolidLiquid`` phase
  classifier. ``freud.density.IntermediateScattering`` tagged
  ``[unverified]`` (absent in freud 3.5.0) with a ``numpy.fft`` +
  MDAnalysis fallback. Algorithmic notes in
  ``correlation-computational-methods`` (AABBQuery neighbor-list reuse,
  ``reset=False`` multi-frame averaging, CuPy breakeven N ≈ 10⁴,
  "MDAnalysis/MDTraj as iterator, freud as analyzer" production
  pattern). One-line hub pointer in ``correlation-analysis`` with a
  ``PythonCall.jl`` handoff note for Julia users.

**Deferred**

* Item A (ML-FF CLI spot-check) — explicitly deferred until the user
  resumes active MLIP training.

**Validator State**

* metadata_validator: 0/0/0 across all 3 plugins.
* xref_validator: 519/519 references valid.
* context_budget_checker: 205/205 skills fit.
  ``non-equilibrium-theory`` at 73.8% (under 75% Commit C gate),
  ``correlation-physical-systems`` at 74.45% (under 75% Commit D
  gate), ``chaos-attractors`` at 79% (under 80% at-risk line).
* skill_validator EXCELLENT; pytest 118/118; ruff clean; mypy 0 errors.

**Known forward items for v3.1.7+**

* ``equation-discovery`` at 88% — flagged for Bayesian SINDy
  extraction split.
* ``freud.density.IntermediateScattering`` presence in newer freud
  releases — re-verify when correlation skills are next touched.

v3.1.5 (2026-04-11)
-------------------

**Julia/Python Parity Pass**

* **Fokker-Planck direct PDE methods** in ``stochastic-dynamics``:
  finite-difference / spectral discretization, boundary-condition
  patterns, cross-links to Langevin sampling.
* **Python bifurcation continuation escape hatch** in
  ``bifurcation-analysis`` and ``nonlinear-dynamics``: documented the
  ``juliacall`` path to Julia bifurcation routines (since
  ``BifurcationKit.jl`` is blocked on Julia 1.12), plus PyDSTool and
  AUTO-07p as Python-native alternatives.
* **Modern ML force fields** expansion in ``ml-force-fields``:
  equivariant GNNs (NequIP, MACE, Allegro), Julia ACE stack
  (``ACEpotentials.jl``), training loops, active learning,
  energy-and-force loss balance. Budget-tight at 78%.
* **Julia Monte Carlo idioms** in ``statistical-physics``: Metropolis
  sampler patterns with ``@inbounds`` / ``@fastmath``, SIMD inner
  loops, parallel tempering via ``Distributed.jl``, tuning heuristics.

**Tooling**

* Added ``types-PyYAML`` dev dependency for mypy stubs.
* Tightened ``self-improving-ai`` triggers so they no longer overlap
  with ``dspy-basics``.
* Added ``tools/validation/command_file_linter.py`` — a targeted
  structural linter for Claude Code command files with 5 stable
  rule IDs (``fence-unbalanced``, ``heading-skip``, ``step-ref-broken``,
  ``trailing-whitespace``, ``heading-duplicate``). Importable API +
  standalone CLI, wired into ``make validate`` (errors block, warnings
  non-blocking). Caught a pre-existing duplicate ``## Metrics`` H2
  in ``dev-suite/commands/tech-debt.md``. 15 new tests.

**Validator State**

* metadata 0/0/0; xref 515/515 valid (+3 from v3.1.4); context budget
  204/204 (``bifurcation-analysis`` 79%, ``ml-force-fields`` 78%,
  both under 80%). pytest 103 passing, ruff clean, mypy 0 errors.

v3.1.4 (2026-04-11)
-------------------

**Research-Focus Optimization Pass (science-suite)**

* Aligned agents and skills with research in Bayesian MCMC
  (NUTS / Consensus MC / Pigeons), Universal Differential Equations,
  SINDy, nonlinear dynamics, time series, rare events / avalanche
  dynamics, non-equilibrium statistical physics, and point / jump
  processes.
* **9 new sub-skills** (hub-discovered, not registered in
  ``plugin.json``): ``consensus-mcmc-pigeons`` (non-reversible
  parallel tempering via Pigeons.jl, now distinguished from Scott-2016
  divide-and-conquer Consensus Monte Carlo), ``bayesian-ude-workflow``
  (Turing + DiffEq + Lux staged pipeline), ``bayesian-ude-jax``
  (Python/JAX counterpart via Diffrax + Equinox + NumPyro),
  ``bayesian-pinn`` (BNNODE/BayesianPINN extracted from ``neural-pde``
  which drops from 78% → 65% of budget), ``point-processes``
  (Hawkes / HSGP / Julia PointProcesses.jl), ``rare-events-sampling``
  (large-deviation / cloning / avalanche statistics),
  ``self-improving-ai`` (research overview), ``dspy-basics``
  (DSPy programmatic prompts depth-skill), ``rlaif-training``
  (Constitutional AI / RLAIF / DPO depth-skill).
* **1 new agent-core skill**: ``self-improving-agents`` under the
  ``reasoning-and-memory`` hub — operational counterpart to
  science-suite's ``self-improving-ai`` (agents inside Claude Code
  vs research framework overview). Covers closed-loop
  reflection-refine-validate, self-consistency ensembles, DSPy and
  TextGrad automatic prompt optimization, evolutionary prompt
  search, and constitutional self-critique.

**Research Audit Remediation**

* Added ``extreme-value-statistics`` skill (GEV/GPD/Hill/Pickands/POT,
  return levels, non-stationary EVT) and wired into
  ``statistical-physics-hub``.
* Wired the orphaned ``robust-testing`` sub-skill into the
  ``research-and-domains`` hub (was on disk but unreachable).
* Extended ``rare-events-sampling`` triggers to cover SOC, sandpile /
  Bak-Tang-Wiesenfeld, crackling noise, and avalanche-size
  distributions; cross-linked to ``extreme-value-statistics``.
* Resolved jump-diffusion routing: ``stochastic-dynamics`` owns
  general physics jump-diffusion SDEs (Lévy flights, shot noise,
  regime-switching Langevin); ``catalyst-reactions`` stays scoped
  to biochemical reaction networks.
* Added Bayesian SINDy coverage to ``equation-discovery``
  (horseshoe-prior NumPyro, ensemble SINDy, UQ-SINDy via Turing) —
  pushed to 88% budget and flagged for v3.1.7 extraction.
* Disambiguated ``sciml-modern-stack`` vs ``sciml-and-diffeq`` by
  rewriting hub routing without touching the frozen
  ``sciml-modern-stack`` body.
* Added missing trigger keywords (ADF, KPSS, Phillips-Perron, PELT,
  BinSeg, renewal processes, non-parametric Hawkes EM) to
  ``time-series-analysis`` and ``point-processes``.

**Agent Updates**

* ``julia-pro``: Bayesian stack upgraded to Turing + Pigeons;
  sensealg table rewritten with ``GaussAdjoint`` as modern default
  and the ForwardDiff-bypasses-sensealg factual fix; decision tree
  adds UDE and multimodal branches.
* ``julia-ml-hpc``, ``statistical-physicist``, ``ai-engineer``,
  ``jax-pro``, ``simulation-expert`` also aligned with research-focus
  delegation updates.

**Codebase-Aware /team-assemble (agent-core)**

* Major rework: static catalog → codebase-aware recommender /
  adapter / validator. **21 → 25 team templates.**
* **4 new teams**: ``nonlinear-dynamics`` (bifurcation, chaos,
  coupled oscillators, pattern formation — first wiring of
  ``nonlinear-dynamics-expert`` to its documented delegation
  targets), ``julia-ml`` (Lux.jl/Flux.jl/MLJ.jl + CUDA.jl/MPI.jl
  distributed training), ``multi-agent-systems``, ``sci-desktop``
  (PyQt6/PySide6 + JAX scientific desktop apps).
* ``ai-engineering`` team swaps ``reasoning-architect`` for
  ``context-architect`` as default 4th teammate.
* Closed drift: **0 unused local agents** (down from 4).
* New capabilities: Step 1.5 codebase detection (4-tier signal
  gathering with efficiency gates), Step 2.5 fingerprint table,
  Step 2.6 rule-based ranking with confidence labels, Step 2.6a/b
  validation + auto-fill, five new invocation modes.
* **Session cache**: Tier 0 cache at
  ``/tmp/team-assemble-cache/<sanitized-abspath>.json`` with
  mtime-based invalidation (15 min TTL); ``--no-cache`` bypass flag.
* **S1 prompt-injection safeguards** for README probes (HIGH):
  character neutralization, ``<untrusted_readme_excerpt>`` wrapping,
  9 refusal-trigger patterns. Non-English README hardening via
  ``language_hint`` classification and auto-fill trust tiers.

**Tooling Hardening**

* Added ``sys.path.insert`` to 5 validators so CLI invocation
  works without ``PYTHONPATH=.``.
* ``PluginLoader`` consolidates YAML frontmatter parsing via
  normalized component helpers.
* ``xref_validator`` gains disk-discovery of sub-skills so
  hub-architecture sub-skills no longer false-positive as broken
  references.
* Restored ``requires = ["maturin>=1.0,<2.0"]`` in rust-extensions
  scaffold (maturin is the real PEP 517 build backend for PyO3).
* ``pyproject.toml``: excluded ``test-corpus/`` from mypy and ruff.

**Validator State**

* metadata 0/0/0; xref 512/512 valid; context budget 204/204;
  pytest 60/60; ruff clean.

v3.1.3 (2026-04-10)
-------------------

**New Skill: thinkfirst (agent-core)**

* Added ``thinkfirst`` as a sub-skill under the ``llm-engineering`` hub.
  Interview-first workflow that clarifies vague user intent through
  a Seven Dimensions framework before any prompt is drafted.
* Positioned as the first branch in the ``llm-engineering`` routing
  tree so users with brain dumps hit clarification before reaching
  for production templates.
* Cross-linked with ``prompt-engineering-patterns``: ``thinkfirst``
  handles intent clarification, ``prompt-engineering-patterns``
  handles production-grade refinement.

v3.1.2 (2026-04-06)
-------------------

**Bug Fixes**

* Removed duplicate ``hooks`` manifest entries from agent-core and dev-suite
  ``plugin.json``. The ``hooks/hooks.json`` file is auto-discovered by convention;
  declaring it explicitly caused duplicate-load errors at startup.
* Fixed dev-suite ``.lsp.json`` structure to match expected schema.

**Documentation**

* Updated plugin READMEs to use hub→sub-skill notation matching CLAUDE.md.
* Rewrote ``tools/README.md`` to reflect current tooling structure.

v3.1.1 (2026-04-06)
-------------------

**Bug Fixes**

* Set ``strict: true`` in marketplace.json to resolve conflicting manifests
  when both marketplace.json and individual plugin.json files declare components.
* Fixed agent teams guide reference in README (34 → 21 teams).

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
