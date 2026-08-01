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
> Use this skill for research-suite's specific quality rubrics (PRISMA/GRADE/CONSORT/STROBE).
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
for f in plugins/research-suite/commands/lit-review.md \
  plugins/research-suite/skills/research-quality-assessment/SKILL.md \
  plugins/research-suite/skills/evidence-synthesis/SKILL.md \
  plugins/research-suite/skills/landscape-scanner/SKILL.md; do
  count=$(grep -c "SEE ALSO" "$f")
  echo "$f: $count"
  [ "$count" -eq 1 ] || { echo "FAIL: expected exactly 1 SEE ALSO block in $f, got $count"; exit 1; }
done
```
Expected: `1` printed for each of the 4 files; the loop exits nonzero (not just prints a mismatched number) if any file is missing its block or has it duplicated.

```bash
PYTHONPATH=. python3 tools/validation/command_file_linter.py plugins/research-suite/commands/lit-review.md
PYTHONPATH=. python3 tools/validation/skill_validator.py --plugin research-suite
```
Expected: `command_file_linter.py` exits 0 (do not suppress its exit status with `|| true` — a real lint failure must be visible). `skill_validator.py` takes `--plugin <name>`, not a positional path — it exits 2 with "unrecognized arguments" otherwise. `skill_validator.py` itself exits 0 and prints `✓ research-suite: N skills` with no `⚠` warning lines — do not pipe its output through a `grep -A3` for individual skill names: without `--corpus-dir` (not used here) it only ever prints the aggregate schema-load summary, never per-skill lines, so such a grep matches nothing and exits 1 on every run, including a fully passing one (confirmed by direct execution against this repo).

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
test -f plugins/science-suite/agents/continuum-mechanics-engineer.md || { echo "continuum-mechanics-engineer: MISSING -- STOP, run the science-suite plan first"; exit 1; }
echo "continuum-mechanics-engineer: EXISTS"
test -f plugins/science-suite/skills/glass-and-collective-dynamics/SKILL.md || { echo "statistical-physicist extension (glass-and-collective-dynamics): MISSING -- STOP, run the science-suite plan first"; exit 1; }
echo "statistical-physicist extension (glass-and-collective-dynamics): EXISTS"
test -f plugins/science-suite/skills/physical-learning-systems/SKILL.md || { echo "statistical-physicist extension (physical-learning-systems): MISSING -- STOP, run the science-suite plan first"; exit 1; }
echo "statistical-physicist extension (physical-learning-systems): EXISTS"
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
Routes to `research-expert` for methodology parsing, then cross-delegates to the specialist matching the paper's method: agent `jax-pro` (general JAX numerics), agent `julia-pro` (general Julia numerics), agent `continuum-mechanics-engineer` (FEM/FEA, constitutive modeling, rheology/DMA, transient networks, nanocomposites), agent `statistical-physicist` (phase transitions, correlations, MCMC diagnostics, glass/jamming/collective phenomena, physical learning), agent `pinn-engineer` (physics-informed neural networks, NeuralPDE), or agent `simulation-expert` (MD/HPC/agent-based simulation, as distinct from the differentiable-programming implementations that route to `jax-pro`/`julia-pro`/`continuum-mechanics-engineer`).
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
3. The specialist matching the paper's method implements the core method (agent `jax-pro`/agent `julia-pro` for general numerics, agent `continuum-mechanics-engineer` for FEM/FEA, constitutive modeling, rheology/DMA, transient-network (CAN/vitrimer) theory, or nanocomposite mechanics, agent `statistical-physicist` for phase transitions, correlations, MCMC diagnostics, glass/jamming/disordered systems, or physical learning, agent `pinn-engineer` for physics-informed neural PDEs, agent `simulation-expert` for MD/HPC/agent-based simulation as distinct from differentiable-programming implementations)
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
| agent `simulation-expert` | Generating data from physics simulations, HPC experiments; or implementing a paper's MD/HPC/agent-based simulation method for /paper-implement or /replicate (distinct from differentiable-programming implementations, which route to jax-pro/julia-pro/continuum-mechanics-engineer) |
| sci-workflow-engineer | Building interactive research dashboards, LLM synthesis |
| python-pro | Performance optimization, systems architecture |
| agent `jax-pro` | Implementing a paper's method in JAX (general numerics) for /paper-implement or /replicate |
| agent `julia-pro` | Implementing a paper's method in Julia (general numerics) for /paper-implement or /replicate |
| agent `continuum-mechanics-engineer` | Implementing a paper's FEM/FEA, constitutive-modeling, rheology/DMA, transient-network, or nanocomposite method |
| agent `statistical-physicist` | Implementing a paper's stat-mech, phase-transition, correlation-function, MCMC-diagnostics, glass/jamming/collective-phenomena, or physical-learning method |
| agent `pinn-engineer` | Implementing a paper's physics-informed neural network or NeuralPDE-style method |
```

- [ ] **Step 4: Verify all 3 files list all 6 specialists**

```bash
cd /home/wei/Documents/GitHub/MyClaude
fail=0
for f in plugins/research-suite/commands/paper-implement.md plugins/research-suite/commands/replicate.md plugins/research-suite/agents/research-expert.md; do
  echo "=== $f ==="
  for name in jax-pro julia-pro continuum-mechanics-engineer statistical-physicist pinn-engineer simulation-expert; do
    count=$(grep -c -- "$name" "$f")
    echo "  $name: $count"
    if [ "$count" -eq 0 ]; then echo "  MISSING: $name not found in $f"; fail=1; fi
  done
done
[ "$fail" -eq 0 ] || { echo "FAIL: one or more specialist names missing — see MISSING lines above"; exit 1; }
```
Expected: a non-zero count for each of the 6 names in each of the 3 files (18 checks total, none reporting MISSING, loop exits 0). A combined `grep -c -E "name1|name2|..."` only proves *some* specialist name occurs per line and cannot prove all six are present, so this checks each name individually instead — and now actually fails the step (`exit 1`) if any is absent, rather than only printing a MISSING line.

- [ ] **Step 5: Confirm the reference form matches xref_validator's patterns**

```bash
cd /home/wei/Documents/GitHub/MyClaude
for f in plugins/research-suite/commands/paper-implement.md \
  plugins/research-suite/commands/replicate.md \
  plugins/research-suite/agents/research-expert.md; do
  count=$(grep -c 'agent `continuum-mechanics-engineer`' "$f")
  echo "$f: $count"
  [ "$count" -ge 1 ] || { echo "FAIL: $f has no \`agent \`continuum-mechanics-engineer\`\` reference in the xref_validator-recognized form"; exit 1; }
done
```
Expected: at least `1` in each of the 3 files (confirms the `` agent `name` `` form, not a bare backtick mention, per Spec §5's warning). Checked per-file in a loop rather than one combined multi-file `grep -c` — a combined grep's exit status only reflects whether *any* of the three files matched, not whether *each* one did, and would silently pass if one file's reference form were wrong.

- [ ] **Step 6: Run xref_validator and full validation**

```bash
cd /home/wei/Documents/GitHub/MyClaude
set -o pipefail
PYTHONPATH=. python3 tools/validation/xref_validator.py | tee /tmp/xref-out.txt
grep -q "Broken: 0" /tmp/xref-out.txt || { echo "FAIL: xref_validator.py reported broken references — see reports/xref-validation.md"; exit 1; }
plugin_json_diff=$(git diff --stat -- plugins/research-suite/.claude-plugin/plugin.json)
[ -z "$plugin_json_diff" ] || { echo "FAIL: plugin.json changed — violates the no-structural-additions invariant:"; echo "$plugin_json_diff"; exit 1; }
make validate
uv run pytest 2>&1 | tee /tmp/pytest-out.txt | tail -20
```
Expected: `xref_validator.py`'s stdout summary line reads `Broken: 0` — this is the only reliable pass/fail signal the tool exposes on stdout (it never prints individual reference names, agent or otherwise, to stdout or to `reports/xref-validation.md`; both only ever show aggregate Total/Valid/Broken counts and per-plugin totals, confirmed by direct execution against this repo). `Broken: 0` only holds if the science-suite plan's Task 3 has actually landed, per this task's opening check — if `continuum-mechanics-engineer` doesn't exist yet, the new agent references in these 3 files become the broken references this check catches. `$plugin_json_diff` must be empty (asserted, not just printed) — confirms the Global Constraint / Spec §5 "no structural additions" invariant: this task only edits routing prose, never `plugin.json`'s agent/command/skill arrays. `make validate` exits 0. `uv run pytest` passes — piped through `tee` (not just `tail`) with `set -o pipefail` already active from the top of this block, so a real pytest failure fails the script instead of being masked by `tail`'s own (always-zero) exit status.

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

- **Spec coverage:** §1/§3 (overlap framing) → Task 1. §1/§4 (cross-suite routing) → Task 2. §2 (no tier changes) → correctly not actioned. §5/§6 (validation, sequencing) → Task 2's opening dependency check and Step 6, including the §5 "no `plugin.json` structural change" invariant check added to Step 6. §7 (out of scope) → correctly not actioned (no new domains/agents/commands, no tier changes).
- **Placeholder scan:** every SEE ALSO insertion and every routing-line replacement gives exact before/after text; the cross-plan dependency check is a runnable `test -f` command, not a vague "make sure science-suite is done" instruction.
- **Type consistency:** the 6 specialist names (`jax-pro`, `julia-pro`, `continuum-mechanics-engineer`, `statistical-physicist`, `pinn-engineer`, `simulation-expert`) are spelled identically across all 3 modified files in Task 2, matching exactly the names used in the science-suite plan (verified against that plan's Task 3/Task 2 content).
