---
name: _research-commons
description: Internal-only resource directory for the research-spark skill stack. Do not select this skill directly for user prompts. Load it only when another research-spark skill (spark-articulator, landscape-scanner, falsifiable-claim, theory-scaffold, numerical-prototype, experiment-designer, premortem-critique) explicitly references its files for writing style, code architecture rules, shared templates, or utility scripts.
---

# research-commons

Shared assets for the research-spark skill stack. Not a standalone workflow.

## Contents

- `style/`: writing conventions enforced across all emitted artifacts
  - `writing_constraints.md`: banned vocabulary, em-dash prohibition, quantified-over-qualitative preference
  - `citation_style.md`: APS PRL-style conventions
  - `figure_standards.md`: font, size, colormap conventions

- `code_architecture/`: conventions every code-emitting skill must follow
  - `jax_first_rules.md`: no Python loops in physics cores, vmap/jit discipline, PRNGkey handling
  - `env_conventions.md`: Python 3.12+, uv, pyproject.toml layout
  - `testing_conventions.md`: pytest, property-based tests for invariants
  - `repo_layout.md`: standard directory structure matching homodyne/heterodyne/RheoJax
  - `julia_first_rules.md`: Julia counterpart to the four files above (env choice, type stability, package layout, testing, juliacall bridge), used when Stage 6 picks Julia over JAX

- `templates/`: cross-cutting templates used by more than one skill
  - `reviewer2_persona.md`: adversarial-reviewer prompt pattern (used by landscape-scanner and falsifiable-claim)
  - `heilmeier.md`: six-question catechism (used by falsifiable-claim)
  - `onepage.md`: one-page summary format (used by premortem-critique)
  - `abstract.md`: shared abstract structure with length variants (used anywhere an abstract is written)
  - `project_log.md`: orchestrator's log format
  - `hostile_self_audit.md`: solo adversarial audit (kill switch, artifact check, bottleneck route), used at the core-completion checkpoint
  - `proposal_assembly.md`: reverse-order draft assembly into `proposal_draft.md` (used at the core-completion checkpoint)

- `scripts/`: utilities invoked by multiple skills
  - `style_lint.py`: flags banned vocabulary and em dashes in emitted markdown
  - `latex_sanity.py`: LaTeX compilation and undefined-reference check
  - `artifact_diff.py`: compares iterations of an artifact across stages
  - `concept_extractor.py`: extracts key concepts from Stage 1 for Stage 2 handoff
  - `formalism_code_reconcile.py`: checks symbol correspondence between Stage 5 LaTeX and Stage 6 code

## How other skills use this directory

Skills reference files here with `../_research-commons/` relative paths from their own SKILL.md:
```
../_research-commons/style/writing_constraints.md
../_research-commons/code_architecture/jax_first_rules.md
../_research-commons/scripts/style_lint.py
```

(When this stack was developed against `~/.claude/skills/`, the references were flat `_research-commons/...`; the `../` prefix is the plugin-layout equivalent.)

Every code-emitting skill (theory-scaffold, numerical-prototype) loads `code_architecture/` before writing any code. Every markdown-emitting skill runs `scripts/style_lint.py` against its output before finalizing.
