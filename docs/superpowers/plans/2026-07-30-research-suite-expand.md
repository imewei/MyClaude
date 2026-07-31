# research-suite Expand & Optimize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "SEE ALSO" overlap framing against ecc's generic research tools to 4 research-suite files, and wire `paper-implement`/`replicate`/`research-expert`'s routing to the 4 new/extended science-suite specialists (`continuum-mechanics-engineer`, extended `statistical-physicist`, `pinn-engineer`, `simulation-expert`) alongside the existing `jax-pro`/`julia-pro` routing.

**Architecture:** Plugin-content repo — no code changes, only markdown routing-table and description edits. "Tests" are `make validate` and `xref_validator.py`.

**Tech Stack:** Markdown (command/skill/agent files).

## Global Constraints

- No new research-suite domains, tracks, agents, or commands. No model-tier changes (`research-expert` and `research-spark-orchestrator` stay opus). (Spec §1, §2, §7)
- This plan's Task 2 depends on the science-suite plan's Task 3 (`continuum-mechanics-engineer` must exist) and Task 6 (`statistical-physicist`'s extended scope) having landed — `xref_validator.py` will fail on a reference to an agent file that doesn't exist yet. (Spec §6)
- Cross-references to `continuum-mechanics-engineer` must use a form `xref_validator.py`'s regexes actually match: `` agent `continuum-mechanics-engineer` `` or the literal path `plugins/science-suite/agents/continuum-mechanics-engineer.md` — a bare backtick mention does not register as a cross-reference and would silently fail to be checked at all. (Spec §5)
- `ecc:*` and `ruflo-goals:*` referenced in Task 1 are external plugins from other marketplaces, not part of this repo's `.claude-plugin/marketplace.json` — their names cannot be repo-validated; the SEE ALSO text is informational framing, not a hard dependency. (Spec §3)

---

### Task 1: Add SEE ALSO overlap framing to 4 files

**Files:**
- Modify: `plugins/research-suite/commands/lit-review.md`
- Modify: `plugins/research-suite/skills/research-quality-assessment/SKILL.md`
- Modify: `plugins/research-suite/skills/evidence-synthesis/SKILL.md`
- Modify: `plugins/research-suite/skills/landscape-scanner/SKILL.md`

**Interfaces:**
- Produces: each file has a `> **SEE ALSO:**` blockquote following the exact 2-line format used in `plugins/dev-suite/commands/smart-debug.md`.

- [ ] **Step 1: Add SEE ALSO to lit-review.md**

Read `plugins/dev-suite/commands/smart-debug.md` first to confirm the exact blockquote format (2 lines, first starts `> **SEE ALSO:**`, second starts `> Use this command for`).

Read `plugins/research-suite/commands/lit-review.md`. After the `# /lit-review — Literature Review` heading and before the `Routes to research-expert via research-suite:research-practice hub.` line, insert:

```markdown
> **SEE ALSO:** For a general literature search without structured claim extraction or PRISMA/GRADE synthesis, use `ecc:scientific-thinking-literature-review` or `ecc:deep-research`.
> Use this command for research-suite's structured topic-scan pipeline (claim extraction, evidence synthesis, gap identification) via the `research-practice` hub.
```

- [ ] **Step 2: Add SEE ALSO to research-quality-assessment**

Read `plugins/research-suite/skills/research-quality-assessment/SKILL.md`. After the `# Research Quality Assessment` heading and before the introductory paragraph, insert:

```markdown
> **SEE ALSO:** For general scholarly-work feedback, use `ecc:scientific-thinking-scholar-evaluation`.
> Use this skill for research-suite's specific quality rubrics (PRISMA/GRADE/CONSORT/STROBE) and red-flag detection.
```

- [ ] **Step 3: Add SEE ALSO to evidence-synthesis**

Read `plugins/research-suite/skills/evidence-synthesis/SKILL.md`. After the `# Evidence Synthesis` heading and before the introductory paragraph, insert:

```markdown
> **SEE ALSO:** For general multi-source research synthesis outside the research-spark pipeline's artifact contract, use `ruflo-goals:deep-researcher` or `ruflo-goals:research-synthesize`.
> Use this skill for systematic reviews (PRISMA), meta-analyses, and formal evidence grading (GRADE).
```

- [ ] **Step 4: Add SEE ALSO to landscape-scanner**

Read `plugins/research-suite/skills/landscape-scanner/SKILL.md`. After the `# landscape-scanner` heading and before the "## Why this stage exists" section, insert:

```markdown
> **SEE ALSO:** For general multi-source research synthesis outside an active research-spark project, use `ruflo-goals:deep-researcher` or `ruflo-goals:research-synthesize`.
> Use this skill for Stage 2 of the research-spark pipeline specifically — the three-layer scan, steelmanning, and Reviewer 2 pass tied to that pipeline's artifact contract.
```

- [ ] **Step 5: Verify**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -c "SEE ALSO" plugins/research-suite/commands/lit-review.md \
  plugins/research-suite/skills/research-quality-assessment/SKILL.md \
  plugins/research-suite/skills/evidence-synthesis/SKILL.md \
  plugins/research-suite/skills/landscape-scanner/SKILL.md
```
Expected: `1` for each of the 4 files.

```bash
PYTHONPATH=. python3 tools/validation/command_file_linter.py plugins/research-suite/commands/lit-review.md 2>&1 || true
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/research-suite 2>&1 | grep -A3 -E "research-quality-assessment|evidence-synthesis|landscape-scanner"
```
Expected: no new errors on any of the 4 files.

- [ ] **Step 6: Commit**

```bash
git add plugins/research-suite/commands/lit-review.md plugins/research-suite/skills/research-quality-assessment/SKILL.md \
  plugins/research-suite/skills/evidence-synthesis/SKILL.md plugins/research-suite/skills/landscape-scanner/SKILL.md
git commit -m "$(cat <<'EOF'
docs(research-suite): add SEE ALSO framing against ecc/ruflo generic research tools

Matches the pattern already proven in dev-suite's smart-debug/test-generate/
eng-feature-dev. research-suite's differentiator (structured, artifact-gated
pipelines tied to JAX/Julia implementation) isn't duplicated anywhere, but
nothing told a trigger-matcher that until now. Text-only, no functional
change.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire cross-suite routing to the 4 new/extended science-suite specialists

**⚠️ Do not start this task until the science-suite plan's Task 3 (`continuum-mechanics-engineer` created) and Task 6 (`statistical-physicist` extended) are committed.** Check first:

```bash
cd /home/wei/Documents/GitHub/MyClaude
test -f plugins/science-suite/agents/continuum-mechanics-engineer.md && echo "continuum-mechanics-engineer: EXISTS" || echo "continuum-mechanics-engineer: MISSING -- STOP, run the science-suite plan first"
test -f plugins/science-suite/skills/glass-and-collective-dynamics/SKILL.md && echo "statistical-physicist extension: EXISTS" || echo "statistical-physicist extension: MISSING -- STOP, run the science-suite plan first"
```

**Files:**
- Modify: `plugins/research-suite/commands/paper-implement.md` (routing prose only, per Spec §4's finding — no `argument-hint`/`description` frontmatter change needed)
- Modify: `plugins/research-suite/commands/replicate.md` (routing prose only)
- Modify: `plugins/research-suite/agents/research-expert.md` (add a new section to the `## Delegation Strategy` table)

**Interfaces:**
- Consumes: `continuum-mechanics-engineer` (science-suite plan Task 3), `statistical-physicist`'s extended scope (science-suite plan Task 6), `pinn-engineer` (existing), `simulation-expert` (existing).
- Produces: all 3 files list all 6 candidate implementation specialists (`jax-pro`, `julia-pro`, `continuum-mechanics-engineer`, `statistical-physicist`, `pinn-engineer`, `simulation-expert`) with a one-line disambiguation each; references use the `` agent `name` `` form so `xref_validator.py` recognizes them.

- [ ] **Step 1: Update paper-implement.md's routing prose**

Read `plugins/research-suite/commands/paper-implement.md`. Find:
```markdown
Routes to `research-expert` for methodology parsing, then cross-delegates to `jax-pro` (JAX) or `julia-pro` (Julia) for implementation.
```
Replace with:
```markdown
Routes to `research-expert` for methodology parsing, then cross-delegates to the specialist matching the paper's method: agent `jax-pro` (general JAX numerics), agent `julia-pro` (general Julia numerics), agent `continuum-mechanics-engineer` (FEM/FEA, constitutive modeling, rheology/DMA, transient networks, nanocomposites), agent `statistical-physicist` (phase transitions, correlations, glass/collective phenomena, physical learning), agent `pinn-engineer` (physics-informed neural networks, NeuralPDE), or agent `simulation-expert` (MD/HPC particle simulation).
```

Also find, further down:
```markdown
## Framework Delegation

`research-expert` owns methodology parsing. Implementation delegated to `jax-pro` (JAX) or `julia-pro` (Julia) via cross-suite call.
```
Replace with:
```markdown
## Framework Delegation

`research-expert`'s methodology-parsing step determines the paper's domain and picks the specialist (see the routing list above); `--framework` (where applicable) only disambiguates JAX vs. Julia within a numerics-based specialist, not which specialist handles the paper.
```

- [ ] **Step 2: Update replicate.md's routing prose**

Read `plugins/research-suite/commands/replicate.md`. Find:
```markdown
Routes to `research-expert` (claim extraction and replication design) → `jax-pro` or `julia-pro` (implementation) → `quality-specialist` (numerical validation gates).
```
Replace with:
```markdown
Routes to `research-expert` (claim extraction and replication design) → the specialist matching the paper's method (agent `jax-pro`, agent `julia-pro`, agent `continuum-mechanics-engineer`, agent `statistical-physicist`, agent `pinn-engineer`, or agent `simulation-expert` — implementation) → `quality-specialist` (numerical validation gates).
```

Also find, in "## What This Does":
```markdown
3. `jax-pro` / `julia-pro` implement the core method
```
Replace with:
```markdown
3. The specialist matching the paper's method implements the core method (agent `jax-pro`/agent `julia-pro` for general numerics, agent `continuum-mechanics-engineer` for FEM/rheology/materials, agent `statistical-physicist` for stat-mech/glass/physical-learning, agent `pinn-engineer` for physics-informed neural PDEs, agent `simulation-expert` for MD/HPC particle simulation)
```

- [ ] **Step 3: Add the delegation table entries to research-expert.md**

Read `plugins/research-suite/agents/research-expert.md`. Find the existing `## Delegation Strategy` table:
```markdown
## Delegation Strategy

| Delegate To | When |
|-------------|------|
| ml-expert | Implementing advanced ML models for analysis |
| simulation-expert | Generating data from physics simulations, HPC experiments |
| sci-workflow-engineer | Building interactive research dashboards, LLM synthesis |
| python-pro | Performance optimization, systems architecture |
```
Replace with (adding 4 new rows for paper-implementation routing, since this table currently has no JAX/Julia/implementation-specialist entries at all):
```markdown
## Delegation Strategy

| Delegate To | When |
|-------------|------|
| ml-expert | Implementing advanced ML models for analysis |
| simulation-expert | Generating data from physics simulations, HPC experiments; or implementing a paper's MD/HPC particle-simulation method for /paper-implement or /replicate |
| sci-workflow-engineer | Building interactive research dashboards, LLM synthesis |
| python-pro | Performance optimization, systems architecture |
| agent `jax-pro` | Implementing a paper's method in JAX (general numerics) for /paper-implement or /replicate |
| agent `julia-pro` | Implementing a paper's method in Julia (general numerics) for /paper-implement or /replicate |
| agent `continuum-mechanics-engineer` | Implementing a paper's FEM/FEA, constitutive-modeling, rheology/DMA, transient-network, or nanocomposite method |
| agent `statistical-physicist` | Implementing a paper's stat-mech, phase-transition, correlation-function, glass/collective-phenomena, or physical-learning method |
| agent `pinn-engineer` | Implementing a paper's physics-informed neural network or NeuralPDE-style method |
```

- [ ] **Step 4: Verify all 3 files list all 6 specialists**

```bash
cd /home/wei/Documents/GitHub/MyClaude
for f in plugins/research-suite/commands/paper-implement.md plugins/research-suite/commands/replicate.md plugins/research-suite/agents/research-expert.md; do
  echo "=== $f ==="
  grep -c -E "jax-pro|julia-pro|continuum-mechanics-engineer|statistical-physicist|pinn-engineer|simulation-expert" "$f"
done
```
Expected: non-zero counts in all 3 files, with all 6 names present in each (spot-check with individual greps if the combined count looks low).

- [ ] **Step 5: Confirm the reference form matches xref_validator's patterns**

```bash
grep -c 'agent `continuum-mechanics-engineer`' plugins/research-suite/commands/paper-implement.md \
  plugins/research-suite/commands/replicate.md plugins/research-suite/agents/research-expert.md
```
Expected: at least `1` in each of the 3 files (confirms the `` agent `name` `` form, not a bare backtick mention, per Spec §5's warning).

- [ ] **Step 6: Run xref_validator and full validation**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/xref_validator.py 2>&1 | grep -A10 "research-suite\|continuum-mechanics-engineer"
make validate
uv run pytest 2>&1 | tail -20
```
Expected: `xref_validator.py` shows the new `continuum-mechanics-engineer` references resolving (not dangling) — this only passes if the science-suite plan's Task 3 has actually landed, per this task's opening check. `make validate` exits 0. `uv run pytest` passes.

- [ ] **Step 7: Commit**

```bash
git add plugins/research-suite/commands/paper-implement.md plugins/research-suite/commands/replicate.md \
  plugins/research-suite/agents/research-expert.md
git commit -m "$(cat <<'EOF'
feat(research-suite): route paper reproduction to all 6 implementation specialists

paper-implement/replicate previously only knew about jax-pro/julia-pro --
papers on FEM/rheology/materials, stat-mech/glass/physical-learning, or
physics-informed neural PDEs would silently fall through to the wrong
specialist. research-expert's delegation table gains its first
implementation-specialist entries. --framework now means "JAX vs Julia
within the chosen specialist," not "which specialist."

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** §1/§3 (overlap framing) → Task 1. §1/§4 (cross-suite routing) → Task 2. §2 (no tier changes) → correctly not actioned. §5/§6 (validation, sequencing) → Task 2's opening dependency check and Step 6. §7 (out of scope) → correctly not actioned (no new domains/agents/commands, no tier changes).
- **Placeholder scan:** every SEE ALSO insertion and every routing-line replacement gives exact before/after text; the cross-plan dependency check is a runnable `test -f` command, not a vague "make sure science-suite is done" instruction.
- **Type consistency:** the 6 specialist names (`jax-pro`, `julia-pro`, `continuum-mechanics-engineer`, `statistical-physicist`, `pinn-engineer`, `simulation-expert`) are spelled identically across all 3 modified files in Task 2, matching exactly the names used in the science-suite plan (verified against that plan's Task 3/Task 2 content).
