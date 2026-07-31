# agent-core Retirement + dev-suite Trim & Reposition

**Date:** 2026-07-30
**Status:** Draft — pending user review
**Supersedes:** The "Out of Scope" exclusions for dev-suite and agent-core in
`2026-05-05-plugin-scicomp-redesign.md` (that spec was implemented — verified
`jax-pro`/`julia-pro` are `opus`, `ml-expert` is `haiku`, and the 5 new
commands it added exist). This spec is the first sub-project of a larger
marketplace redesign; science-suite and research-suite "expand" specs follow
separately and will build on the already-implemented 2026-05-05 baseline.

---

## 1. Objective

MyClaude's unique value is scientific computing (JAX-first Python, Julia
SciML) and academic research workflow — nothing else on this machine covers
that ground. Generic software-engineering and agent-orchestration content is
now redundant: this Claude Code install also loads `ecc` (per-language
reviewers, build-fixers, and pattern skills for essentially every stack),
`ruflo-core`/`ruflo-swarm` (agent orchestration, coordination, memory),
`superpowers` (TDD, debugging, planning process skills), `pr-review-toolkit`,
`code-modernization`, and `plugin-dev`. Where MyClaude's `agent-core` and
`dev-suite` duplicate that installed set, they add trigger-matching noise
and context budget cost without new capability.

This spec cuts that duplication and repositions what remains around
scientific-computing specificity — matching the pattern three dev-suite
commands (`smart-debug`, `test-generate`, `eng-feature-dev`) already use:
"for general X, use ecosystem-plugin Y; this command handles
scientific-computing-specific Z."

**Dedupe baseline:** the actual installed plugin set on this machine (not a
hypothetical minimal install for other marketplace users), per explicit user
decision.

## 2. Scope

In scope: `plugins/agent-core/`, `plugins/dev-suite/`.
Out of scope: `plugins/science-suite/`, `plugins/research-suite/` (separate
expand specs); deciding a version-numbering policy beyond recording the
required bump (§8) — the mechanics are handled at execution time via
`make validate`.

## 3. agent-core: retire

Delete `plugins/agent-core/` entirely. Rationale:

- Its 2 registered commands (`ultra-think`, `team-assemble`) are not part of
  the user's actual workflow (confirmed — not selected as in-use). 4 more
  command files exist on disk but were never registered in `plugin.json`
  (`agent-build`, `ai-assistant`, `docs-lookup`, `reflection`) — same
  zero-risk-deletion logic as the unregistered dev-suite commands in §4, and
  removed by the same `git rm -r`.
- Its 3 agents (orchestrator, context-specialist, reasoning-engine) and 18
  skill directories — 5 registered hubs (`agent-hub`, `agent-systems`,
  `llm-engineering`, `reasoning-and-memory`, `thinkfirst`) plus 13 sub-skills
  (`multi-agent-coordination`, `reflection-framework`,
  `self-improving-agents`, `memory-system-patterns`, etc.) — are fully
  covered by `ruflo-core`, `ruflo-swarm`, `superpowers`, and ecc's `agent-*`
  skill family (`agent-architecture-audit`, `agent-eval`,
  `agent-harness-construction`, `agentic-engineering`, `agentic-os`,
  `harness-audit`, `continuous-learning-v2`, `prompt-optimizer`).
- No content from agent-core is scientific-computing-specific, so there is
  nothing to migrate into science-suite or dev-suite.

**Action:** `git rm -r plugins/agent-core`. Remove all references to
`agent-core` from root README, docs cross-links, and any `xref_validator`
target lists.

## 4. dev-suite: command surface

**Keep (10, already registered in `plugin.json`):** `smart-debug`,
`double-check`, `run-all-tests`, `test-generate`, `docs`, `modernize`,
`workflow-automate`, `fix-commit-errors`, `merge-all`, `eng-feature-dev`.
User-confirmed as actually used.

**Cut (15, never registered — currently unreachable via `plugin.json`, so
this is a zero-risk deletion):** `adopt-code`, `c-project`, `code-analyze`,
`code-explain`, `deps`, `fix-imports`, `github-assist`, `monitor-setup`,
`multi-platform`, `onboard`, `profile-performance`, `rust-project`,
`scaffold`, `slo-implement`, `tech-debt`. Each has a direct installed-plugin
equivalent (ecc's per-language `*-build-resolver` agents, `api-connector-builder`,
`github-ops`, `dashboard-builder`, `database-migrations`, `pm2`).

## 5. dev-suite: agent surface

9 → 6.

**Keep, with scope rewritten to scientific-computing lens:**

| Agent | New scope |
|---|---|
| `documentation-expert` | Docs for numerical/ML/SciML codebases — API specs for JAX/Julia interfaces, Sphinx integration, notebook-to-doc pipelines. For general documentation, defer to `ecc:update-docs`. |
| `software-architect` | Numerical/ML/simulation system architecture — JAX pipeline boundaries, SciML module design, data/compute separation for scientific workloads. For general system architecture, defer to `ecc`'s architecture skills. |
| `app-developer` | Scientific application development — PyQt/PySide6 scientific GUIs, JAX/Julia app integration. For general app development, defer to `ecc` per-framework reviewers. |
| `automation-engineer` | Scientific workflow automation — experiment pipelines, Airflow/data-pipeline orchestration for numerical workloads. For general CI/CD automation, defer to `ecc:deployment-patterns`/`ecc:docker-patterns`. |
| `quality-specialist` | Scientific-computing validation — numerical precision, property-based mathematical invariants, reproducibility checks. For general test coverage, defer to `ecc:test-coverage`. |
| `sre-expert` | Reliability for long-running scientific workloads — HPC job monitoring, GPU/cluster observability, simulation checkpoint/resume. For general SRE, defer to `ruflo-observability:observe`. |

**Cut (3, zero command references found in any of the 10 kept commands):**
`debugger-pro`, `devops-architect`, `systems-engineer`. Generic ground
already covered by `mattpocock-skills:diagnosing-bugs`, `ecc:build-fix`
family, `ecc:homelab-*`/`ecc:kubernetes-patterns`.

Rewrite is scoped to each agent's frontmatter `description` and system
prompt intro — not a full rewrite of technical content, which may already be
reusable where it discusses e.g. Sphinx or Airflow in a way that's already
scientific-computing-adjacent.

## 6. dev-suite: skill surface

**Cut — stack-specific, fully duplicated by ecc's per-language skills (9):**
`frontend-and-mobile`, `frontend-mobile-engineering`, `graphql-patterns`,
`mobile-testing-patterns`, `modern-javascript-patterns`,
`nodejs-backend-patterns`, `typescript-advanced-types`,
`typescript-project-scaffolding`, `websocket-patterns`. Of these,
`frontend-and-mobile` is one of dev-suite's 12 registered hub entries in
`plugin.json`'s `skills` array — its removal means deleting that manifest
entry directly. The other 8 are unregistered sub-skill directories, reachable
only through hub routing per this repo's manifest convention, so their
removal touches no manifest.

**Cut — duplicate of an actively maintained equivalent (1):**
`plugin-syntax-validator` (near-identical to
`ruflo-plugin-creator:validate-plugin`).

**Move to science-suite, merging with existing overlap rather than
duplicating (5):** `async-python-patterns`, `python-packaging`,
`python-performance-optimization`, `python-toolchain`,
`uv-package-manager`. Of these, `python-toolchain` is likewise one of
dev-suite's 12 registered hubs — moving it requires deleting its
`plugin.json` entry, not just relocating a directory; the other 4 are
unregistered sub-skills. science-suite already has `python-development` and
`python-packaging-advanced` — the science-suite expand spec must reconcile
these into one Python-tooling home rather than landing 5 more files
alongside near-duplicates. Tracked as an input to that spec, not resolved
here.

**Keep as-is, backs kept commands/agents, no clean 1:1 ecosystem
duplicate identified (remaining ~46 skills — 10 hubs + ~36 sub-skills, i.e.
the 61 skill directories on disk minus the 9+1 cuts and 5 moves above):**
e.g. `debugging-toolkit`,
`comprehensive-validation`, `git-workflow`, `ci-cd-pipelines`,
`architecture-and-infra`, `testing-and-quality`, `observability-and-sre`,
`database-patterns`, `security-ci-template`, `documentation-standards`,
`modernization-migration`, `dev-hub`, `three-brain`, etc. These were not
individually re-litigated against every ecc equivalent in this pass — flagged
as a possible follow-up audit (see §8), but not blocking this spec, since
the concrete, evidenced deletions above already remove the highest-confidence
duplication.

**`dev-hub` (routing hub) description:** must be edited to drop keywords for
everything cut above (GraphQL, Node.js/Express/Fastify, frontend
accessibility for generic web, mobile cross-platform testing) so its
trigger surface matches the trimmed skill set.

**`three-brain`:** kept, unresolved. Possibly overlaps with `ecc:multi-plan`/
`multi-execute`/`multi-workflow` and with `ccg`'s built-in Codex/Gemini
backend routing (this session's `ccg-session` context shows
`Default (frontend=gemini, backend=codex)`). Flagged as a needs-inspection
follow-up rather than decided blind in this spec.

## 7. Cross-reference check before execution

Before deleting anything, run `xref_validator.py` and `doc_checker.py`
against the full repo to enumerate what science-suite and research-suite
skills reference in dev-suite or agent-core (e.g. `eng-feature-dev`
explicitly bridges to science-suite/research-suite; check the reverse
direction too). Any live cross-reference into a cut file must be updated or
the reference removed as part of the same change, not left dangling.

## 8. Validation / acceptance criteria

- `make validate` passes with zero errors (warnings pre-existing and
  unrelated are acceptable).
- `PYTHONPATH=. python3 tools/validation/context_budget_checker.py` shows no
  new oversized skills and a lower total skill count.
- `PYTHONPATH=. python3 tools/validation/xref_validator.py` shows no
  dangling cross-references into deleted agent-core/dev-suite paths.
- `uv run pytest` passes unchanged (no test currently targets agent-core or
  the cut dev-suite surface by name — verify during implementation).
- `plugins/dev-suite/.claude-plugin/plugin.json` `commands`/`agents` arrays
  match the §4/§5 keep-lists exactly. The `skills` array (which per this
  repo's convention lists hub skills only, never sub-skills) drops exactly
  the two cut/moved entries that are registered hubs — `frontend-and-mobile`
  and `python-toolchain` — leaving 10 of the original 12; the remaining §6
  cuts and moves are unregistered sub-skill directories and require no
  array edit.
- Root README and any doc cross-links to `agent-core` removed.
- Version bump: this is a breaking removal of agents/commands/skills, which
  warrants a major bump under semver, not the minor bump originally proposed
  here. `agent-core`, `dev-suite`, `science-suite`, and `research-suite`
  currently share one synced version (`3.5.2` in every `plugin.json` and in
  `pyproject.toml` today) enforced by `make validate`'s drift check, so the
  bump is repo-wide — there is no independent "`dev-suite`-only" version to
  bump.
- `uv.lock`'s pre-existing modification (present before this session) stays
  out of any commit this spec produces.

## 9. Out of scope / follow-ups

- `three-brain` vs `ecc:multi-*` vs `ccg` routing comparison — separate
  follow-up, not blocking.
- Itemized audit of the remaining ~46 "keep as-is" dev-suite skills against
  every installed ecc equivalent — optional deeper pass if the user wants it
  after this spec ships.
- science-suite Python-tooling consolidation (receiving the 5 moved skills)
  — input to the science-suite expand spec, not resolved here.
- science-suite and research-suite "optimize for Opus 5 / Sonnet 5, expand
  coverage" — separate specs, next in sequence.
