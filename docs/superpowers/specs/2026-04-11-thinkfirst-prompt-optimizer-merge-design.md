# thinkfirst ↔ prompt-optimizer merge — design spec

**Date:** 2026-04-11
**Scope:** Consolidate `/Users/b80985/Downloads/prompt-optimizer` into
`plugins/agent-core/skills/thinkfirst/` with **full functional parity** —
no functions from prompt-optimizer may be lost, and no Claude Code
scaffolding from thinkfirst may be lost.
**Status:** Approved (user approved section order 2026-04-11).

## Context

`thinkfirst` is the agent-core sub-skill added in v3.1.3 (commit
`5ac6724a`) that routes from the `llm-engineering` hub. Its current
body is a single-mode (Craft) workflow: brain dump → Seven Dimensions
interview → reasoning strategy suggestion → draft → iterate.

`prompt-optimizer` is an external (Downloads) skill that shares the
same file structure (SKILL.md + `examples/worked-example.md` + 3
reference files) and the same Craft Mode content, but adds a second
mode — **Optimize Mode** — for users who already have an existing
prompt and want it rewritten.

Three of the five files are byte-identical between the two sources:
`examples/worked-example.md`, `references/reasoning-strategies.md`,
`references/seven-dimensions.md`. The work is confined to `SKILL.md`
and `references/prompt-structure.md`.

## Decision

**Adopt Option A from the brainstorming session:** add Optimize Mode
to `thinkfirst` in-place, keeping the name, keeping Claude Code hub
scaffolding, and pulling in the 10 optimization techniques as a new
section of `references/prompt-structure.md`.

Name retention (`thinkfirst`, not `prompt-optimizer`) is load-bearing.
The name is already cited in CHANGELOG.md (v3.1.3 entry written
earlier in this session), `docs/changelog.rst`, `docs/suites/agent-core.rst`,
the llm-engineering hub's routing tree, and agent-core README's new
skills list. A rename would cascade across ~10 files and invalidate
the v3.1.3 release history.

## Parity checklist

Every prompt-optimizer feature has an explicit destination in the
merged thinkfirst. Every thinkfirst scaffolding element is retained.

### From prompt-optimizer `SKILL.md`

| # | Feature | Destination |
|---|---|---|
| 1 | Description triggers: "help me write a prompt", "turn this into a prompt", brain dumps, rough ideas, "I want to build/create X", unstructured notes, vague requirements, AI session prep | Merged `description` frontmatter |
| 2 | **NEW triggers:** "optimize this prompt", "make this prompt better", "my prompt isn't working" | Appended to merged `description` frontmatter |
| 3 | Slash command mention | `/thinkfirst` only (drop `/prompt-optimizer` — not a registered command) |
| 4 | Intro: "AI is the most articulate thing in the room" | Unchanged (already in thinkfirst) |
| 5 | Framing: "prompt engineering specialist who happens to be a great listener" | Unchanged |
| 6 | **Mode Selection table** with Craft/Optimize/Ambiguous signals | **NEW section** between Expert Agent and Craft Mode |
| 7 | Craft Mode — Core Principle | `## Craft Mode` → existing content |
| 8 | Calibrate step (skim `examples/worked-example.md`) | Preserved under Craft Mode |
| 9 | Phase 1 — Receive the Brain Dump | Craft Mode Phase 1 |
| 10 | Phase 2 — Seven Dimensions (Outcome, Stakes, Success Criteria, Failure Modes, Hidden Context, Components, The Hard Part) + reference pointer | Craft Mode Phase 2 |
| 11 | Phase 3 — Reasoning Strategies (Chain of Thought / Validation Gates / Confidence Signaling) + reference pointer | Craft Mode Phase 3 |
| 12 | Phase 3 — Draft with standard sections + reference pointer | Craft Mode Phase 3 |
| 13 | Phase 4 — Present and Iterate | Craft Mode Phase 4 |
| 14 | **Optimize Mode** intro: "faster, more direct — but still confirm intent before delivering" | **NEW `## Optimize Mode` section** |
| 15 | **Step 1: Diagnose** — 8-weakness internal checklist (vague task / missing audience / no tone / no output structure / no examples / needs step-by-step / missing role / failure modes not addressed) | NEW Optimize Mode subsection |
| 16 | **"If intent unclear, ask exactly ONE clarifying question"** | NEW, within Step 1 |
| 17 | **Step 2: Apply Optimization Techniques** + reference pointer into `references/prompt-structure.md` | NEW Optimize Mode subsection |
| 18 | **Step 3: Deliver** — output format template (`## Optimized Prompt` / rewritten prompt in code block / `**What changed:**` bullets / offer to iterate) | NEW Optimize Mode subsection, **verbatim template** |
| 19 | Non-Negotiable Rules (7 rules, "both modes") | Existing `## Non-Negotiable Rules` section, re-scoped to "both modes" |
| 20 | Reference Files listing with per-file load guidance | `## Additional Resources` updated to mention both modes where relevant |

### thinkfirst-only scaffolding retained

| # | Feature | Status |
|---|---|---|
| 21 | `## Expert Agent` — delegation to `context-specialist` | **Retained verbatim** |
| 22 | `## Related Skills` — cross-refs to `prompt-engineering-patterns` and `reasoning-frameworks` | **Retained verbatim** |
| 23 | Skill name `thinkfirst` | **Retained** (cascade cost) |
| 24 | **All** of thinkfirst's Craft Mode prose and Non-Negotiable Rules wording | **Retained verbatim** — see "Prose parity rule" below |

### Prose parity rule (critical — applies to SKILL.md)

The self-review found that prompt-optimizer's SKILL.md is not just
tighter editorially — it drops **~17 specific pieces of guidance**
that thinkfirst preserves. An abbreviated list:

- Intro: "never pause, never stumble, never say 'let me think about that'"
- Intro closing: "Listen, ask smart questions, and draft only when the user's intent is genuinely understood"
- Core Principle: "a 30-second check saves a 5-minute rewrite"
- Phase 1: the numbered 3-step internal procedure, "so the user can confirm or correct", "Not even a 'rough draft.' The temptation to jump ahead is strong — resist it"
- Phase 2: "do not drag the conversation out for its own sake", "Each dimension has its own questioning style, its own signal for when to probe harder, and its own mapping to the final prompt structure"
- Phase 3: "(high-stakes from Dimension 2)" explicit dimension linkage, "Do not force strategies", the inline "(multi-persona debate, adversarial self-review, reference-class priming, constrained-then-expand)" technique enumeration, "Once the user accepts (or declines) strategy suggestions, announce readiness to draft"
- Phase 4: "Most prompts benefit from 1–2 rounds of refinement. Common adjustments: tone, added or refined examples, tightened or loosened constraints, newly surfaced edge cases"
- Rule 1: "and even then, confirm before finalizing"
- Rule 2: "Ask one question, wait for the answer, then decide what to ask next based on what was said"
- Rule 3: "They know what they want, even if they cannot yet articulate it"
- Rule 4: "Do not lay out options and ask them to pick"
- Rule 5: "If they write in Chinese, produce everything in Chinese. If English, produce in English"
- Rule 6: "Everything the AI needs to understand the task must live inside the prompt"
- Rule 7: "Not every prompt needs XML tags, examples, role definitions, and reasoning strategies. A prompt for a simple task should be simple"

**Rule:** For every paragraph/rule/phase that already exists in
thinkfirst, **the merged SKILL.md keeps thinkfirst's wording
verbatim**. The merge is purely additive at the content level —
prompt-optimizer contributes only:

1. **Widened `description` frontmatter** (picks up "optimize this prompt", "make this prompt better", "my prompt isn't working" triggers)
2. **Updated H1** ("thinkfirst — Prompt Crafter & Optimizer")
3. **Mode Selection table** (new section)
4. **`## Craft Mode` wrapper** around the existing phases (heading-level reorganization only — the phase content stays as-is)
5. **Entire `## Optimize Mode` section** (new — Step 1 diagnose table + Step 2 + Step 3 output template)
6. **"(both modes)" scope suffix** on the Non-Negotiable Rules heading
7. **Minor wording in Additional Resources** to mention both modes where applicable

Everything else — Expert Agent, Core Principle, Calibrate, Phase 1–4
bodies, all 7 Non-Negotiable Rules, Related Skills — stays at
thinkfirst's verbatim wording. Zero substance loss.

### From prompt-optimizer `references/prompt-structure.md`

| # | Feature | Destination |
|---|---|---|
| 25 | Title → "Prompt Structure, Best Practices, and Optimization Techniques" | Updated heading |
| 26 | Intro: "Use this reference when drafting (Craft Mode Phase 3) or rewriting (Optimize Mode Step 2)" | Replaces current intro |
| 27 | Standard Prompt Sections template | Use prompt-optimizer's wording throughout **except for two parity restorations** — see "Parity restorations" below |
| 28 | When to Skip Sections — **table form** | Replaces thinkfirst's bullet form |
| 29 | Mapping Seven Dimensions to Prompt Sections | Use prompt-optimizer's tighter intro; table content is identical |
| 30 | **Optimization Techniques — 10 techniques with before/after examples**: (1) Clarity & Specificity, (2) Output Structure Definition, (3) Audience Definition, (4) Tone & Style Guidance, (5) Role/Persona Assignment, (6) Step-by-Step Thinking (Chain of Thought), (7) Examples (Few-Shot), (8) Document Q&A — Specificity + Citations, (9) Uncertainty Acknowledgment, (10) Iterative Structure (Constrained then Expand) | **NEW major section** — verbatim |
| 31 | Best Practices Checklist — prompt-optimizer's 9-item tightened version | **Do NOT adopt wholesale.** Keep thinkfirst's existing 9-item version instead — see "Parity restorations" below. Re-frame the intro as "Apply when drafting or rewriting:" to match both-modes scope. |
| 32 | Final-Draft Checklist — 8 items with "both modes" scope | Use prompt-optimizer's version — it is strictly better (mode-agnostic wording + adds the 8th item "Complexity of prompt matches complexity of task — no over-engineering"). No thinkfirst substance is lost. |

### Parity restorations (the self-review found these)

The self-review found three places where prompt-optimizer's tighter
wording **drops substance** that thinkfirst preserves. To honor
"no functions lost", the merge must not blindly use prompt-optimizer's
shorter version in these spots:

**R1 — Standard Prompt Sections → Output Format:** prompt-optimizer
drops the `("don't use bullet points")` parenthetical example. Merged
version keeps thinkfirst's longer form:

> Tell the AI what to do ("write in flowing prose") rather than what not to do ("don't use bullet points").

**R2 — Standard Prompt Sections → Constraints / Guardrails:** this
is the one place where prompt-optimizer's wording is actually
*better* — it inlines a concrete positive-framing example that
thinkfirst only has in the Best Practices section. **Use
prompt-optimizer's version here** (no substance loss either way,
and the inline example aids comprehension during drafting).

**R3 — Best Practices Checklist:** prompt-optimizer's 9-item tight
version drops multiple pieces of thinkfirst's specific guidance:
- Item 1: drops "Specificity beats vagueness every time"
- Item 2: drops "Explaining the why helps the AI generalize correctly"
- Item 3: drops `<instructions>` tag + "Use consistent, descriptive tag names"
- Item 4: drops "(3–5)" example count + "Make examples relevant, diverse, and structured"
- Item 5: drops "If the prompt involves large inputs" framing
- Item 6: drops "Tell the AI what to do, not what to avoid" closing
- Item 8: softens "persona focuses its behavior and tone" → "expertise"

**Decision: keep thinkfirst's 9-item Best Practices Checklist
verbatim**, but re-frame the intro line ("Follow these prompt
engineering principles when writing the draft:") as ("Apply when
drafting or rewriting:") so it scopes to both modes. This preserves
every piece of thinkfirst's specific guidance while still serving
Optimize Mode. Zero substance loss.

### Byte-identical files — no work needed

- `examples/worked-example.md` (6393 bytes, identical)
- `references/reasoning-strategies.md` (6362 bytes, identical)
- `references/seven-dimensions.md` (5558 bytes, identical)

## Merged `SKILL.md` section order

```
1. Frontmatter
   - name: thinkfirst (unchanged)
   - description: expanded to cover Craft triggers + Optimize triggers
2. H1: "# thinkfirst — Prompt Crafter & Optimizer"
3. Intro paragraphs (two: "AI is most articulate thing", "specialist
   who happens to be a great listener")
4. ## Expert Agent                  [retained from thinkfirst]
5. ## Mode Selection                 [NEW — 3-row table]
6. ## Craft Mode                     [wraps current Phases 1-4]
   - Core Principle
   - Calibrate Before Responding
   - Phase 1 — Receive the Brain Dump
   - Phase 2 — Clarify Through the Seven Dimensions
   - Phase 3 — Suggest Reasoning Strategies and Draft
   - Phase 4 — Present and Iterate
7. ## Optimize Mode                  [NEW — 3 steps]
   - Step 1: Diagnose (8-weakness checklist + one-clarifying-question rule)
   - Step 2: Apply Optimization Techniques (pointer to reference)
   - Step 3: Deliver (output format template, verbatim)
8. ## Non-Negotiable Rules (both modes)  [7 rules, merged wording]
9. ## Additional Resources           [reference file list + example]
10. ## Related Skills                 [retained from thinkfirst]
```

## Merged `references/prompt-structure.md` section order

```
1. H1: "Prompt Structure, Best Practices, and Optimization Techniques"
2. Intro (both-modes framing)
3. ## Standard Prompt Sections
4. ## When to Skip Sections (table)
5. ## Mapping Seven Dimensions to Prompt Sections
6. ## Optimization Techniques        [NEW major section, 10 subsections]
7. ## Best Practices Checklist       [9 items]
8. ## Final-Draft Checklist          [8 items, both-modes]
```

## Small defaults (approved)

1. **H1 title**: `# thinkfirst — Prompt Crafter & Optimizer` (reflects
   dual capability)
2. **Rule wording**: where thinkfirst and prompt-optimizer differ, the
   more detailed/instructive version wins (thinkfirst typically)
3. **Slash trigger**: `/thinkfirst` only (drop `/prompt-optimizer`)

## Budget math

Revised after the Prose Parity Rule. Since thinkfirst's existing
prose is kept verbatim (not shortened to prompt-optimizer's tighter
form), the merged SKILL.md is ~800 bytes larger than my original
estimate.

| File | Before | After (est) | % of 4000-token budget | Status |
|---|---|---|---|---|
| `SKILL.md` | 9196 bytes (~2299 tok) | ~11400 bytes (~2850 tok) | **~71%** | Under 80% at-risk line ✓ |
| `references/prompt-structure.md` | 5378 bytes | ~8700 bytes | N/A | Reference files don't count against main skill budget ✓ |

Size breakdown of additions on top of thinkfirst's current 9196 bytes:
- Widened `description` frontmatter: +~300 bytes
- H1 suffix " & Optimizer": +12 bytes
- `## Mode Selection` section + table: +~350 bytes
- `## Craft Mode` wrapper heading + 1-line intro: +~100 bytes
- Horizontal rules between sections: +~50 bytes
- `## Optimize Mode` section (intro + 8-weakness diagnose table + Step 1 one-clarifying-question rule + Step 2 with reference pointer + Step 3 with output template): +~1400 bytes
- "(both modes)" scope suffix on Rules heading: +~14 bytes

**Total: ~9196 + 2226 ≈ 11422 bytes ≈ 2855 tokens ≈ 71% of budget.**

Must pre-check with `context_budget_checker.py` after the edit. If
it comes in above 75%, that's a signal to push some content (e.g.
the optional "common adjustments" list in Phase 4 that's already in
thinkfirst) into a reference file. If above 80%, hard stop and
escalate.

No risk of pushing thinkfirst into the >90% "must refactor" band
per CLAUDE.md §Key Conventions.

## Verification gates

After the two edits land:

1. `make validate` → metadata_validator 0/0/0 on all 3 plugins
2. `python3 tools/validation/context_budget_checker.py` →
   `thinkfirst` reported under 80% budget, no new >80% warnings
3. `uv run pytest tools/tests/ -q` → 118/118 passing (no new tests
   added, no existing behavior changed)
4. `cd docs && sphinx-build -b html . _build/html` → build succeeded
   (no broken references)

## Rollback plan

All changes are confined to two files inside one skill directory.
`git checkout plugins/agent-core/skills/thinkfirst/SKILL.md
plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`
reverts the merge entirely. No cascading dependencies.

## Out of scope (deliberately)

- `/thinkfirst` as a **registered** slash command (it's a natural-
  language trigger phrase; registering it would cascade into
  `plugin.json` manifests and the `cheatsheet.md` "2 registered
  commands" count for agent-core).
- Updating the v3.1.3 changelog entry written earlier in this session
  to mention Optimize Mode — that entry documents what v3.1.3
  actually shipped, not what will exist after this merge.
- Adding a corresponding v3.1.7 or later changelog entry for the
  merge — the merge is a polish, not a release. If the user wants
  it in a release, that's a separate decision.
