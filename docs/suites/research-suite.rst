Research Suite
==============

Scientific research workflows: peer review, idea-to-plan refinement, and methodology orchestration. Three complementary tracks — ``scientific-review`` (manuscripts from other authors → .docx referee report), ``research-spark`` (own rough idea → 8-stage artifact-gated plan), and ``research-practice`` (general methodology hub).

**Version:** 3.4.0 | **2 Agents** | **0 Registered Commands** | **11 Registered Skills (6 pipeline stages + 2 hubs + 2 standalones + 1 resource) → 5 methodology sub-skills** | **0 Hook Events**

Created in v3.4.0 by extracting ``research-expert`` plus 5 methodology skills from ``science-suite`` and adding the research-spark pipeline (new 8-stage orchestrator + 7 stage-specialist skills + ``_research-commons`` resource hub).

Agents
------

.. agent:: research-expert
   :description: Unified specialist for research methodology, evidence synthesis (PRISMA/GRADE), statistical-rigor assessment, IMRaD structuring, paper-to-code reproduction, and publication-quality visualization. For one-off methodology tasks, not pipeline-driven work.
   :model: opus
   :version: 3.4.0

.. agent:: research-spark-orchestrator
   :description: Autonomous driver for the 8-stage research-spark refinement pipeline. Owns ``_state.yaml``, enforces the artifact contract, fans out to parallel sub-agents at Stage 2 (literature layers), Stage 6 (validation passes), and Stage 8 (reviewer archetypes).
   :model: opus
   :version: 3.4.0

Commands
--------

This suite registers **zero slash commands** — every workflow is skill-driven. The legacy ``/paper-review`` command was removed because ``scientific-review`` auto-triggers on "review this paper" phrasings and produces a strictly better ``.docx`` deliverable (journal-adapted Six-Lens analysis with Confidential Comments to Editor).

Stand-alone Workflows
---------------------

Two top-level skills that are complete workflows, not hubs or stage specialists:

Skill: scientific-review
^^^^^^^^^^^^^^^^^^^^^^^^

Peer review of a single manuscript (PDF / DOCX / pasted text) with six competencies: domain expertise, methodological rigor, critical thinking, constructive communication, ethical integrity, and time-efficient delivery. Produces a downloadable ``.docx`` referee report (markdown fallback if ``python-docx`` is unavailable). If the user names a target journal, performs a live web search for that journal's reviewer guidelines before structuring the output.

Skill: _research-commons
^^^^^^^^^^^^^^^^^^^^^^^^

Shared assets for the research-spark skill stack — not a standalone workflow. Other skills reference files here for writing style (``style/writing_constraints.md`` with banned-vocabulary list), code architecture rules (``code_architecture/jax_first_rules.md``), shared templates (``templates/heilmeier.md``, ``templates/reviewer2_persona.md``, ``templates/abstract.md``, ``templates/onepage.md``, ``templates/project_log.md``), and utility scripts (``scripts/style_lint.py``, ``scripts/formalism_code_reconcile.py``, ``scripts/concept_extractor.py``, ``scripts/latex_sanity.py``, ``scripts/artifact_diff.py``).

research-spark Pipeline (8 stages)
----------------------------------

Pipeline orchestrator plus seven stage-specialist skills. Each stage writes one canonical artifact at a canonical path; the next stage consumes it as authoritative input. State lives in ``_state.yaml`` at the project root.

Hub: research-spark
^^^^^^^^^^^^^^^^^^^

Orchestrator for the 8-stage refinement pipeline. Detects current stage from ``_state.yaml`` + user cue, loads the right stage specialist, enforces canonical paths, preserves prior-stage artifacts when a completed stage is re-entered, and logs depth-gate overrides (e.g., the 8-steelmanned-papers rule in Stage 2).

- ``spark-articulator`` — **Stage 1.** Rough idea → 3-5 sentence articulation naming the spark, its novelty, and the observation that would confirm it. Writes ``01_spark.md``.
- ``landscape-scanner`` — **Stage 2.** Three-layer literature scan (foundational / recent / adjacent), steelmanning each paper, gap matrix, Reviewer 2 adversarial pass. Writes ``02_landscape.md``.
- ``falsifiable-claim`` — **Stage 3.** Claim + Heilmeier catechism + kill criterion + Reviewer 2 challenge. Writes ``03_claim.md``.
- ``theory-scaffold`` — **Stages 4-5** (merged). Stepwise derivation protocol → LaTeX formalism. Blocks multi-step symbolic leaps, identifies governing dimensionless groups, checks known limits. Writes ``04_theory.md`` + ``05_formalism.tex``.
- ``numerical-prototype`` — **Stage 6.** JAX-based solver + three validation passes (analytic-limit recovery, synthetic benchmark, convergence study) → concrete predicted observable. Writes ``06_prototype.md`` + ``code/``.
- ``experiment-designer`` — **Stage 7.** Instrument capability map (3× margin rule per dimension), DoE matrix, formal power analysis, pre-registered success metrics, risk register. Writes ``07_plan.md``.
- ``premortem-critique`` — **Stage 8.** Failure narrative, root-cause clustering, cheapest early-warning signals fed back into Stage 7, simulated-reviewer critique across archetypes. Writes ``08_premortem.md``.

research-practice Hub (methodology)
-----------------------------------

For free-form methodology questions that are neither structured pipelines nor peer reviews.

Hub: research-practice (5 sub-skills)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Meta-orchestrator for the research lifecycle. Routes to the appropriate phase-aligned specialist.

- ``research-methodology`` — **Design phase.** Hypothesis formulation, power analysis, sample-size justification, ablation planning, statistical-test selection *before* data collection.
- ``research-quality-assessment`` — **Evaluate phase.** Score existing work against CONSORT/STROBE/PRISMA/MOOSE, detect red flags (p-hacking, HARKing, selective reporting, circular analysis). Not a .docx deliverable — use ``scientific-review`` for that.
- ``research-paper-implementation`` — **Reproduce phase.** Translate a published paper's methods + appendix into runnable code.
- ``scientific-communication`` — **Write-up phase.** IMRaD structure, abstracts (see ``_research-commons/templates/abstract.md``), posters, technical reports.
- ``evidence-synthesis`` — **Synthesize phase.** PRISMA systematic reviews, meta-analysis (effect-size pooling, I²/Q heterogeneity), GRADE evidence grading.

Phase ↔ research-spark stage mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the user is inside an active research-spark project, the pipeline stage supersedes the generic methodology sub-skill — the stage version enforces tighter artifact contracts.

+--------------------------------+----------------------------------------+
| research-practice phase        | research-spark stage                   |
+================================+========================================+
| Design (research-methodology)  | Stage 7 — experiment-designer          |
+--------------------------------+----------------------------------------+
| Reproduce (paper-impl)         | Stage 6 — numerical-prototype          |
+--------------------------------+----------------------------------------+
| Synthesize (evidence-synth)    | Stage 2 — landscape-scanner            |
+--------------------------------+----------------------------------------+
| Write-up (sci-comm)            | Stage 1 — spark-articulator            |
+--------------------------------+----------------------------------------+
| Evaluate (quality-assessment)  | Stage 8 — premortem-critique           |
+--------------------------------+----------------------------------------+

Three adversarial patterns worth knowing
-----------------------------------------

These exist because they catch failures the non-adversarial workflow misses.

- **Reviewer 2 persona** (Stages 2-3). Adversarial reviewer argues the gap isn't real / tractable / impact-bearing, or the claim is physically impossible / mathematically unsound / already solved. Each rebuttal must cite a specific paper.
- **Stepwise derivation protocol** (Stages 4-5). One conceptual step per invocation, with a verification pass (dimensional check, limit check, sanity argument) before the next step. Blocks multi-step symbolic leaps.
- **Instrument capability margin** (Stage 7). For each measurable quantity, compute the margin between predicted signal and instrument capability on each axis. Margin < 3× → high-risk measurement requiring explicit mitigation.

Style enforcement
-----------------

Every emitted markdown artifact passes ``_research-commons/scripts/style_lint.py``: no em dashes, no banned vocabulary (*innovative, state-of-the-art, transformative, novel, groundbreaking, cutting-edge*), quantified language preferred.

Cross-suite delegation
----------------------

The ``research-spark-orchestrator`` delegates across suite boundaries at natural fan-out points:

+-----------------------------------+-----------------------------------------------------------------------+
| Delegate to (suite)               | When                                                                  |
+===================================+=======================================================================+
| ``jax-pro`` (science-suite)       | Stage 6 JAX implementation details (JIT, vmap, integrator choice)     |
+-----------------------------------+-----------------------------------------------------------------------+
| ``julia-pro`` (science-suite)     | Stage 6 SciML/DifferentialEquations.jl, SINDy, stiff-ODE alternatives |
+-----------------------------------+-----------------------------------------------------------------------+
| ``nonlinear-dynamics-expert``     | Stages 4-5 when theory involves bifurcation, chaos, pattern formation |
| (science-suite)                   |                                                                       |
+-----------------------------------+-----------------------------------------------------------------------+
| ``statistical-physicist``         | Stages 4-5 for correlation functions, Langevin/Fokker-Planck,         |
| (science-suite)                   | critical phenomena                                                    |
+-----------------------------------+-----------------------------------------------------------------------+
| ``simulation-expert``             | Stage 6 when the prototype is MD or Monte Carlo                       |
| (science-suite)                   |                                                                       |
+-----------------------------------+-----------------------------------------------------------------------+

Hooks
-----

This suite ships **zero hook events**. Adversarial patterns and style linting are enforced inside skill workflows rather than via harness hooks, so they run deterministically without depending on CLI event schemas.

Requirements
------------

**scientific-review**
   ``python-docx`` for .docx generation (markdown fallback), ``pandoc`` for DOCX ingestion, ``pdftotext`` / ``pypdf`` / ``pymupdf`` for PDF.

**research-spark stack**
   Python 3.12+, ``uv`` for dependency resolution. Stage-specific: ``sympy`` (theory-scaffold), ``scipy`` + ``pyyaml`` (experiment-designer), ``jax`` + ``jaxlib`` (numerical-prototype), ``pdflatex`` (latex_compile_check.sh). All scripts install locally via ``uv add``, never globally.

Project directory layout (research-spark)
------------------------------------------

.. code-block:: text

   <workspace>/<idea-slug>/
   ├── _state.yaml
   ├── project_log.md
   ├── artifacts/
   │   ├── 01_spark.md
   │   └── ... 08_premortem.md
   └── code/                    # emerges at Stage 6
       ├── pyproject.toml
       ├── src/<slug>/
       └── tests/
