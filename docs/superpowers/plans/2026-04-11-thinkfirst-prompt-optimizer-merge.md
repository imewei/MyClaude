# thinkfirst ↔ prompt-optimizer merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge prompt-optimizer into thinkfirst with full functional parity — add Mode Selection + Optimize Mode + 10 Optimization Techniques, without dropping any thinkfirst prose.

**Architecture:** Two surgical file edits inside
`plugins/agent-core/skills/thinkfirst/`. The merge is purely additive at
the content level — thinkfirst's existing text is kept verbatim, with
prompt-optimizer's new sections layered on top. No other files touched.
No cascading renames. Three byte-identical files
(`examples/worked-example.md`, `references/reasoning-strategies.md`,
`references/seven-dimensions.md`) are skipped entirely.

**Tech Stack:** Markdown, Claude Code plugin architecture. Validators:
`metadata_validator`, `command_file_linter`, `context_budget_checker`,
pytest, `sphinx-build`.

**Spec:** `docs/superpowers/specs/2026-04-11-thinkfirst-prompt-optimizer-merge-design.md`

---

## File Structure

Two files modified. No new files created.

| Path | Role | Change |
|---|---|---|
| `plugins/agent-core/skills/thinkfirst/SKILL.md` | Skill entry point — frontmatter, body, rules, references, cross-links | Purely additive: widened description, updated H1, new Mode Selection section, `## Craft Mode` wrapper, new `## Optimize Mode` section, "(both modes)" Rules scope, Additional Resources both-modes note. Keeps ALL existing thinkfirst prose verbatim. |
| `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md` | On-demand reference loaded during Phase 3 (Craft) or Step 2 (Optimize) | Purely additive: new title, new both-modes intro, R1+R2 restorations in Standard Prompt Sections, new `## Optimization Techniques` major section (10 techniques verbatim), Best Practices Checklist scope re-framed, Final-Draft Checklist replaced with prompt-optimizer's 8-item both-modes version. |

**Files NOT touched:**
- `plugins/agent-core/skills/thinkfirst/examples/worked-example.md` (byte-identical)
- `plugins/agent-core/skills/thinkfirst/references/reasoning-strategies.md` (byte-identical)
- `plugins/agent-core/skills/thinkfirst/references/seven-dimensions.md` (byte-identical)
- `plugins/agent-core/.claude-plugin/plugin.json` (thinkfirst is a sub-skill, discovered via hub routing — no manifest entry)
- `plugins/agent-core/skills/llm-engineering/SKILL.md` (hub routing already points to thinkfirst by name; Optimize Mode is internal to thinkfirst so no routing change needed)
- Any docs/changelog — the v3.1.3 changelog entry describes what v3.1.3 shipped; this merge is a polish after the fact, not a release bump

---

## Task 1: Baseline snapshot

**Files:**
- Read: `plugins/agent-core/skills/thinkfirst/SKILL.md`
- Read: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`
- Read: `/Users/b80985/Downloads/prompt-optimizer/SKILL.md` (source)
- Read: `/Users/b80985/Downloads/prompt-optimizer/references/prompt-structure.md` (source)

- [ ] **Step 1: Confirm working tree clean**

Run: `cd /Users/b80985/Projects/MyClaude && git status -s`
Expected: empty output (no uncommitted changes). If not empty, stash or commit first.

- [ ] **Step 2: Confirm starting byte counts**

Run:
```bash
wc -c plugins/agent-core/skills/thinkfirst/SKILL.md \
       plugins/agent-core/skills/thinkfirst/references/prompt-structure.md
```
Expected:
- `SKILL.md`: 9196 bytes
- `references/prompt-structure.md`: 5378 bytes

Any deviation means thinkfirst has drifted since the spec was written. Stop and re-run the spec-review step in that case.

- [ ] **Step 3: Confirm source files exist**

Run:
```bash
wc -c /Users/b80985/Downloads/prompt-optimizer/SKILL.md \
      /Users/b80985/Downloads/prompt-optimizer/references/prompt-structure.md
```
Expected:
- `SKILL.md`: 7732 bytes
- `references/prompt-structure.md`: 8651 bytes

If the Downloads files have moved or been deleted, stop and ask the user.

- [ ] **Step 4: Baseline validator run**

Run: `make validate`
Expected: metadata 0/0/0 on all 3 plugins, command-file-linter "1 total issue(s); 0 at >= error" (pre-existing `tech-debt.md` heading-duplicate warning, not a blocker).

Any new errors? Stop — something drifted since the last session.

---

## Task 2: SKILL.md — widen frontmatter description

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md:1-4`

- [ ] **Step 1: Replace frontmatter with widened description**

Current (line 1-4):
```markdown
---
name: thinkfirst
description: This skill should be used when the user asks to "help me write a prompt", "turn this into a prompt", shares a "brain dump", "rough idea", or unstructured notes about what they need from AI, says "I want to build/create X" (when the goal is to craft the prompt), wants to prepare for a work session, or invokes /thinkfirst. Transforms rough ideas into professional, well-structured prompts through a clarifying interview before drafting. This is the recommended starting point before any significant AI-assisted work — use it even when the user has not explicitly asked for "a prompt."
---
```

New:
```markdown
---
name: thinkfirst
description: Use this skill whenever the user wants to write, improve, or optimize a prompt. Trigger on "help me write a prompt", "turn this into a prompt", "optimize this prompt", "make this prompt better", "my prompt isn't working", brain dumps and rough ideas about what they need from AI, "I want to build/create X" (when the goal is crafting the prompt), unstructured notes or vague requirements, requests to prepare for an AI work session, or /thinkfirst. This skill has two modes — Craft (build from a brain dump) and Optimize (rewrite an existing prompt) — and selects the right one automatically. Use it even when the user has not explicitly asked for "a prompt."
---
```

Note: flattened prompt-optimizer's multi-line folded-scalar (`description: >`) form to a single line. YAML parses both; single-line matches the convention used by every other SKILL.md in agent-core.

---

## Task 3: SKILL.md — update H1

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md:6`

- [ ] **Step 1: Replace H1**

Current (line 6):
```markdown
# thinkfirst — Prompt Crafter
```

New:
```markdown
# thinkfirst — Prompt Crafter & Optimizer
```

---

## Task 4: SKILL.md — add Mode Selection section

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md` — insert between the `## Expert Agent` block and the `## Core Principle` heading

- [ ] **Step 1: Insert Mode Selection table after Expert Agent**

Find the last line of the Expert Agent block:
```markdown
  - *When to delegate*: When the user's target task is high-stakes, multi-stage, or requires specialized prompt-engineering judgment beyond what the Seven Dimensions interview surfaces.
```

After that line and before `## Core Principle`, insert:

```markdown

## Mode Selection

On receiving the first message, select a mode:

| Signal | Mode |
|---|---|
| User shares a rough idea, brain dump, or vague goal — no existing prompt | **Craft Mode** |
| User pastes an existing prompt and asks to improve, fix, or rewrite it | **Optimize Mode** |
| Ambiguous | Default to **Craft Mode**; a 30-second check is cheaper than a bad rewrite |
```

---

## Task 5: SKILL.md — wrap existing phases under `## Craft Mode`

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md` — change section headings only, no prose changes

- [ ] **Step 1: Change "## Core Principle" → "## Craft Mode" + subsection**

Current (was at line 20-22 pre-edit; may have shifted after Tasks 2-4):
```markdown
## Core Principle

Do not produce the prompt before clarifying. Even when the request seems obvious, confirm understanding first — a 30-second check saves a 5-minute rewrite. The only exception: if the user's request is already extremely detailed and specific, a draft may come sooner, but still confirm before calling it final.
```

New:
```markdown
## Craft Mode

For brain dumps, rough ideas, and "I want to create X" requests. Full four-phase process.

### Core Principle

Do not produce the prompt before clarifying. Even when the request seems obvious, confirm understanding first — a 30-second check saves a 5-minute rewrite. The only exception: if the user's request is already extremely detailed and specific, a draft may come sooner, but still confirm before calling it final.
```

- [ ] **Step 2: Change "## Calibrate Before Responding" → "### Calibrate Before Responding"**

Current:
```markdown
## Calibrate Before Responding

Before replying to the first user message, consider skimming `examples/worked-example.md`...
```

New:
```markdown
### Calibrate Before Responding

Before replying to the first user message, consider skimming `examples/worked-example.md`...
```

(Only the heading `##` → `###` changes. The paragraph is unchanged.)

- [ ] **Step 3: Remove "## The Four-Phase Process" heading**

Current:
```markdown
## The Four-Phase Process

### Phase 1 — Receive the Brain Dump
```

New (delete the `## The Four-Phase Process` line and the blank line after it):
```markdown
### Phase 1 — Receive the Brain Dump
```

The four Phase headings stay at `###` level — they now sit directly under `## Craft Mode`.

Note: Phase 1, Phase 2, Phase 3, Phase 4 bodies are unchanged. Do not touch the prose inside them.

---

## Task 6: SKILL.md — add `## Optimize Mode` section

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md` — insert between Phase 4 and `## Non-Negotiable Rules`

- [ ] **Step 1: Insert the entire Optimize Mode section**

Find the last line of Phase 4:
```markdown
Present the draft and invite feedback: "Here is the draft — does this capture what you are going for? Anything feel off or missing?" Be ready to iterate. Most prompts benefit from 1–2 rounds of refinement. Common adjustments: tone, added or refined examples, tightened or loosened constraints, newly surfaced edge cases. When the user is satisfied, present the final prompt in a clean, copyable code block.
```

After that line and before `## Non-Negotiable Rules`, insert:

````markdown

## Optimize Mode

For users who paste a prompt and want it improved. Faster, more direct — but still confirm intent before delivering.

### Step 1: Diagnose

Read the prompt and internally check for these weaknesses:

| Weakness | Check |
|---|---|
| Vague task (no specifics on output, length, format) | |
| Missing audience or context | |
| No tone or style guidance | |
| No output structure defined | |
| No examples where they would help | |
| Complex task that would benefit from step-by-step thinking | |
| Missing role or persona | |
| Failure modes not addressed | |

If the user's intent is unclear from the prompt alone, ask **one** clarifying question before rewriting. Do not ask multiple.

### Step 2: Apply Optimization Techniques

Apply whichever techniques improve the prompt. Not every prompt needs all of them.

**For the full technique guide with before/after examples, consult `references/prompt-structure.md` — see the "Optimization Techniques" section.**

### Step 3: Deliver

Output the rewritten prompt in a clearly labeled code block, followed by a brief explanation of what changed and why (3–5 bullets, concise). Close with an offer to iterate.

**Output format:**

```
## Optimized Prompt

[rewritten prompt in a code block]

**What changed:**
- [change and reason]
- [change and reason]
- ...

*Want me to adjust anything — tone, length, format, or add an example?*
```
````

---

## Task 7: SKILL.md — scope Non-Negotiable Rules to both modes

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md` — one heading change

- [ ] **Step 1: Update Rules heading**

Current:
```markdown
## Non-Negotiable Rules
```

New:
```markdown
## Non-Negotiable Rules (both modes)
```

(The 7 rule bodies are unchanged — keep thinkfirst's detailed wording per the prose parity rule.)

---

## Task 8: SKILL.md — mention both modes in Additional Resources

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/SKILL.md` — one reference description

- [ ] **Step 1: Update prompt-structure.md reference description**

Current:
```markdown
- **`references/prompt-structure.md`** — Standard prompt sections, best-practices checklist, dimension-to-section mapping, and the final-draft checklist. Load this when entering Phase 3 to draft.
```

New:
```markdown
- **`references/prompt-structure.md`** — Standard prompt sections, best-practices checklist, dimension-to-section mapping, optimization techniques with before/after examples, and the final-draft checklist. Load this when entering Craft Mode Phase 3 (to draft) or Optimize Mode Step 2 (to rewrite).
```

---

## Task 9: SKILL.md — budget check

**Files:**
- Inspect: `plugins/agent-core/skills/thinkfirst/SKILL.md`

- [ ] **Step 1: Check file size**

Run: `wc -c plugins/agent-core/skills/thinkfirst/SKILL.md`
Expected: ~11400 bytes (spec target).

- [ ] **Step 2: Run context_budget_checker**

Run:
```bash
python3 tools/validation/context_budget_checker.py --plugins-dir plugins
```
Expected: thinkfirst fits under 80%. Pass condition: no new `> 80%` warning for thinkfirst.

**Stop condition:** If thinkfirst is reported >75%, flag it in the summary (spec says 71% target). If >80%, STOP, revert Task 2-8 edits with `git checkout plugins/agent-core/skills/thinkfirst/SKILL.md`, and escalate to the user.

---

## Task 10: SKILL.md — commit

**Files:**
- Commit: `plugins/agent-core/skills/thinkfirst/SKILL.md`

- [ ] **Step 1: Stage the file**

Run: `git add plugins/agent-core/skills/thinkfirst/SKILL.md`

- [ ] **Step 2: Commit with message**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(agent-core): add Optimize Mode to thinkfirst skill

Merges the external prompt-optimizer skill into thinkfirst with full
functional parity. Purely additive at the content level — every piece
of thinkfirst's existing Craft Mode prose and Non-Negotiable Rules is
kept verbatim. The merge contributes:

- Widened description frontmatter covering "optimize this prompt",
  "make this prompt better", "my prompt isn't working" triggers
- Updated H1 to "thinkfirst — Prompt Crafter & Optimizer"
- New Mode Selection table (Craft / Optimize / Ambiguous)
- "## Craft Mode" wrapper around the existing four phases
- Full "## Optimize Mode" section with Step 1 (8-weakness diagnose
  table + one-clarifying-question rule), Step 2 (techniques pointer),
  Step 3 (output format template verbatim)
- "(both modes)" scope on the Non-Negotiable Rules heading
- Updated Additional Resources to mention both modes

Design spec: docs/superpowers/specs/2026-04-11-thinkfirst-prompt-optimizer-merge-design.md

Budget impact: thinkfirst SKILL.md grows 9196 → ~11400 bytes
(~71% of 4000-token budget, under the 80% at-risk line).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

Run: `git log --oneline -1`
Expected: the new commit at HEAD with the expected subject.

---

## Task 11: prompt-structure.md — title and intro

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md:1-3`

- [ ] **Step 1: Replace the title and intro paragraph**

Current (lines 1-3):
```markdown
# Prompt Structure and Best Practices

Use this reference in Phase 3 of the `thinkfirst` workflow, once the Seven Dimensions are clear enough to draft. A professional prompt does not need every section below — adapt the structure to fit the task. Rigid templates produce rigid outputs.
```

New:
```markdown
# Prompt Structure, Best Practices, and Optimization Techniques

Use this reference when drafting (Craft Mode Phase 3) or rewriting (Optimize Mode Step 2). A professional prompt does not need every section — adapt the structure to fit the task. Rigid templates produce rigid outputs.
```

---

## Task 12: prompt-structure.md — Standard Prompt Sections restorations

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md` — the Standard Prompt Sections code block, specifically the Constraints paragraph

- [ ] **Step 1: Update Constraints / Guardrails**

Per spec R2, use prompt-optimizer's version (has inline positive-framing example).

Current Constraints block:
```
[Constraints / Guardrails]
Boundaries, things to avoid (framed as positive alternatives), edge case handling.
Include failure modes the user identified — these become explicit constraints.
```

New:
```
[Constraints / Guardrails]
Boundaries and edge case handling.
Include failure modes the user identified — these become explicit constraints.
Frame positively: "use accessible language" not "avoid jargon."
```

Note: The current thinkfirst Output Format block already has the `("don't use bullet points")` parenthetical, so R1 requires no change — it's already present. Verify by reading lines 28-31 of the file:
```
What the output should look like. Be specific about structure, length, tone, audience.
Tell the AI what to do ("write in flowing prose") rather than what not to do
("don't use bullet points").
```
If present — good, no edit. If drifted — restore to this form.

---

## Task 13: [MERGED INTO TASK 14]

When-to-Skip replacement happens inside Task 14's atomic middle-block rewrite. Skip this task.

---

## Task 14: prompt-structure.md — reorder middle + insert Optimization Techniques (atomic)

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md` — insert between Mapping Seven Dimensions (end) and Best Practices Checklist (start)

- [ ] **Step 1: Find the insertion point**

The insertion point is immediately after this paragraph (end of Mapping section):
```markdown
Use the table as a checklist while drafting — every clarified dimension should appear somewhere in the prompt.
```

And immediately before the next section heading. Note: the current file has `## When to Skip Sections` AFTER the Mapping section, but Task 13 already moved it. So after Task 13 the structure is:

```
## Standard Prompt Sections (unchanged)
...
## Best Practices to Apply (will be renamed in Task 15)
...
## Mapping Seven Dimensions to Prompt Sections (unchanged)
...
## When to Skip Sections (table now)
...
## Final-Draft Checklist (will be replaced in Task 16)
```

Wait — the current file actually has the order: Standard Sections → Best Practices → Mapping → When to Skip → Final-Draft. The spec's target order is: Standard Sections → When to Skip → Mapping → Optimization Techniques → Best Practices → Final-Draft. So Tasks 13-16 need to reorder as well.

**Revised approach for Task 14:** insert the Optimization Techniques section directly before `## Best Practices to Apply`. Task 15 then reorders by moving Best Practices down and Mapping/When-to-Skip up. Actually easier: do the full reorder in one step here.

**Replace the entire middle of the file in one operation.** Find the block from `## Best Practices to Apply` through and including `## When to Skip Sections` and the final `Do not force sections...` line. Replace with the new ordered block:

Current middle (approximately lines 45-88):
```markdown
## Best Practices to Apply

Follow these prompt engineering principles when writing the draft:

- **Be clear and direct.** Specificity beats vagueness every time. If the user wants thorough, detailed output, say so explicitly rather than hoping the AI infers it.
- **Explain the why.** When adding a constraint or instruction, briefly explain the reasoning. "Write for a general audience because this will be published on our public blog" is better than just "Write for a general audience." Explaining the why helps the AI generalize correctly.
- **Use XML tags for structure.** Tags like `<instructions>`, `<context>`, `<example>`, `<input>` help the AI parse complex prompts without ambiguity. Use consistent, descriptive tag names.
- **Show, do not just tell.** A few well-crafted examples (3–5) are more reliable than paragraphs of explanation. Make examples relevant, diverse, and structured.
- **Put long data at the top.** If the prompt involves large inputs, place them above the instructions. Queries and instructions at the end improve response quality.
- **Frame positively.** Instead of "don't use jargon," write "use language accessible to a general audience." Tell the AI what to do, not what to avoid.
- **Match prompt style to desired output.** For prose output, write the prompt in prose. For structured data output, structure the prompt accordingly.
- **Give a role when it helps.** A single sentence setting the AI's persona focuses its behavior and tone.
- **Use step-by-step instructions** for sequential tasks. Numbered lists help when order and completeness matter.

---

## Mapping Seven Dimensions to Prompt Sections

The clarification phase produces content that maps cleanly into the prompt sections:

| Seven-Dimension answer      | Prompt section it feeds              |
|-----------------------------|--------------------------------------|
| Outcome                     | Task (opening sentence)              |
| Stakes                      | Context / Background (motivation)    |
| Success Criteria            | Output Format                        |
| Failure Modes               | Constraints / Guardrails             |
| Hidden Context              | Context / Background                 |
| Components                  | Task (numbered steps or sub-tasks)   |
| The Hard Part               | Constraints + Examples (where to be tightest) |

Use the table as a checklist while drafting — every clarified dimension should appear somewhere in the prompt.

---

## When to Skip Sections

Match the complexity of the prompt to the complexity of the task:

- **Simple one-off task** (e.g., "rewrite this paragraph more formally") — `Task` + brief `Output Format`. Skip the rest.
- **Repeated workflow** — add `Role`, `Input Specification` with `{{placeholders}}`, and 1–2 `Examples`.
- **High-stakes deliverable** — include all sections, especially `Constraints` and `Examples`, and consider embedding a reasoning strategy (see `reasoning-strategies.md`).

Do not force sections that the task does not need. A prompt for a simple task should be simple.
```

**Important:** Tasks 12 and 13 already target parts of this block. Execute Tasks 12-14 as a single sequence. After Task 12 updates Constraints and Task 13 replaces the When-to-Skip form, the structure is still Best-Practices → Mapping → When-to-Skip. Task 14 then does the bigger reorder + insertion in one atomic Edit.

**Replace the whole middle block** with:

```markdown
## When to Skip Sections

| Task type | Sections needed |
|---|---|
| Simple one-off ("rewrite this paragraph formally") | Task + Output Format only |
| Repeated workflow | Add Role + Input Specification with `{{placeholders}}` + 1–2 Examples |
| High-stakes deliverable | All sections + reasoning strategy (see `reasoning-strategies.md`) |

Do not force sections the task does not need.

---

## Mapping Seven Dimensions to Prompt Sections

Every clarified dimension should appear somewhere in the prompt:

| Seven-Dimension answer | Prompt section it feeds |
|---|---|
| Outcome | Task (opening sentence) |
| Stakes | Context / Background (motivation) |
| Success Criteria | Output Format |
| Failure Modes | Constraints / Guardrails |
| Hidden Context | Context / Background |
| Components | Task (numbered steps or sub-tasks) |
| The Hard Part | Constraints + Examples (where to be tightest) |

---

## Optimization Techniques

Apply these when rewriting an existing prompt. Use judgment — not every prompt needs every technique.

### 1. Clarity & Specificity
The single highest-impact improvement. Replace vague verbs with action verbs; add counts, purpose, scope, and constraints.

**Before:** "Help me with a presentation."
**After:** "Create a 10-slide outline for our quarterly sales meeting covering Q2 performance, top-selling products, and Q3 targets. Provide 3–4 key points per slide."

**Checklist:**
- Replace "help me with" → "write / analyze / outline / compare"
- Specify quantities (number of slides, word count, number of ideas)
- State the purpose and end use
- Add constraints (audience, format, length)

---

### 2. Output Structure Definition
Tell the AI exactly how to format the response.

**Before:** "Analyze our sales data."
**After:** "Present the analysis as: (1) Executive Summary (2–3 sentences), (2) Key Metrics table with quarterly totals and top performer, (3) three notable trends with brief explanations, (4) three data-driven recommendations with rationale."

**Patterns:** numbered sections, tables for comparisons, explicit word/section limits.

---

### 3. Audience Definition
Different audiences require different vocabulary, depth, and tone.

**Before:** "Write about cybersecurity."
**After:** "Write a 1000-word blog post about cybersecurity for small business owners who are not tech-savvy. Use plain language, practical actionable tips, and a slightly humorous tone."

**Ask:** Technical or non-technical? Senior or junior? Internal or external? Consumer or professional?

---

### 4. Tone & Style Guidance
Specify the voice. Reference a style guide if one exists.

**Before:** "Write a product description."
**After:** "Write a ~200-word product description in a professional but engaging tone. Brand voice: friendly, innovative, health-conscious. Highlight ergonomic features, health benefits, sustainable materials, and close with a call to action."

**Tone options:** formal / casual / friendly / technical / humorous / empathetic

---

### 5. Role / Persona Assignment
Assigning an expert role unlocks more nuanced, domain-aware responses.

**Before:** "Help me prepare for a negotiation."
**After:** "You are a fabric supplier for my backpack manufacturing company. As the supplier, provide three objections to a 10% price reduction request, a counterargument for each, and two alternative proposals. Then switch roles and advise me as the buyer on how to approach this negotiation."

**Strong patterns:** "Act as a senior [role] with 15 years of experience in [domain]."

---

### 6. Step-by-Step Thinking (Chain of Thought)
For complex or analytical tasks, ask the AI to reason through the problem.

**Before:** "How can I improve team productivity?"
**After:** "Think through this step-by-step: (1) identify current productivity blockers, (2) suggest solutions for each, (3) identify implementation challenges, (4) suggest measurement methods. Explain your reasoning at each step and summarize at the end."

**Trigger phrases:** "think step-by-step," "walk me through your reasoning," "analyze each factor in order."

---

### 7. Examples (Few-Shot)
Show the desired style, tone, or format. Works especially well for emails, creative writing, and structured content.

**Before:** "Write a professional email."
**After:** "Here is a similar email I sent before: [example]. Draft a new email in the same tone for a situation where we are delayed by one month due to supply chain issues."

---

### 8. Document Q&A — Specificity + Citations
When asking the AI to analyze a document, specify the focus and request citations.

**Before:** "Summarize this report."
**After:** "From the attached 'Tech Industry Trends 2023' report, provide a 2-paragraph summary focused on AI and ML trends. Then answer these three questions: [questions]. Cite the specific section or page number for each answer."

---

### 9. Uncertainty Acknowledgment
For fact-heavy tasks, explicitly give the AI permission to say it does not know. Reduces confident hallucinations.

**Add to any research or factual prompt:** "If you are unsure about specific figures or claims, say so rather than guessing."

---

### 10. Iterative Structure (Constrained Draft then Expand)
When the user is not sure what they want and iteration cost is high, ask for a short version first, confirm direction, then expand.

**Pattern:** "First write a one-paragraph version. Once confirmed, expand to full length."

---

## Best Practices to Apply

Apply when drafting or rewriting:

- **Be clear and direct.** Specificity beats vagueness every time. If the user wants thorough, detailed output, say so explicitly rather than hoping the AI infers it.
- **Explain the why.** When adding a constraint or instruction, briefly explain the reasoning. "Write for a general audience because this will be published on our public blog" is better than just "Write for a general audience." Explaining the why helps the AI generalize correctly.
- **Use XML tags for structure.** Tags like `<instructions>`, `<context>`, `<example>`, `<input>` help the AI parse complex prompts without ambiguity. Use consistent, descriptive tag names.
- **Show, do not just tell.** A few well-crafted examples (3–5) are more reliable than paragraphs of explanation. Make examples relevant, diverse, and structured.
- **Put long data at the top.** If the prompt involves large inputs, place them above the instructions. Queries and instructions at the end improve response quality.
- **Frame positively.** Instead of "don't use jargon," write "use language accessible to a general audience." Tell the AI what to do, not what to avoid.
- **Match prompt style to desired output.** For prose output, write the prompt in prose. For structured data output, structure the prompt accordingly.
- **Give a role when it helps.** A single sentence setting the AI's persona focuses its behavior and tone.
- **Use step-by-step instructions** for sequential tasks. Numbered lists help when order and completeness matter.
```

This atomic replacement achieves the reorder (When-to-Skip and Mapping swap positions relative to Best Practices) AND the insertion of Optimization Techniques AND the Best Practices intro rewording from "Follow these prompt engineering principles when writing the draft:" to "Apply when drafting or rewriting:".

**Note:** This atomic replacement subsumes the old Tasks 13 and 15 (When-to-Skip table form and Best Practices intro rescope are part of the new block).

---

## Task 15: prompt-structure.md — verify atomic replacement landed

**Files:**
- Inspect: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`

- [ ] **Step 1: Verify When-to-Skip is now in table form**

Run: `grep -c "^| Simple one-off" plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`
Expected: `1`

- [ ] **Step 2: Verify Best Practices intro was rescoped**

Run: `grep -n "Apply when drafting or rewriting" plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`
Expected: one match, under the `## Best Practices to Apply` heading.

- [ ] **Step 3: Verify Optimization Techniques section exists**

Run: `grep -c "^## Optimization Techniques" plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`
Expected: `1`

- [ ] **Step 4: Verify all 10 technique subheadings are present**

Run:
```bash
grep -c "^### [0-9]\+\." plugins/agent-core/skills/thinkfirst/references/prompt-structure.md
```
Expected: `10`

If any of these checks fails, Task 14 didn't land as expected. Re-run it.

---

## Task 16: prompt-structure.md — replace Final-Draft Checklist

**Files:**
- Modify: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md` — the Final-Draft Checklist section at end of file

- [ ] **Step 1: Replace the Final-Draft Checklist**

Current:
```markdown
## Final-Draft Checklist

Before presenting the prompt to the user in Phase 4, confirm:

1. The outcome is stated explicitly in the Task section.
2. At least one failure mode from Phase 2 appears in Constraints.
3. Every piece of hidden context is captured in Context / Background.
4. The output format specifies structure, length, tone, and audience.
5. If the user accepted a reasoning strategy in Phase 2, it is woven into the draft (see `reasoning-strategies.md`).
6. The prompt is self-contained — it would work when pasted into any AI tool without additional context.
7. The prompt matches the language the user was writing in.
```

New:
```markdown
## Final-Draft Checklist

Before presenting the prompt (both modes), confirm:

1. The outcome is stated explicitly in the Task section.
2. At least one failure mode from clarification appears in Constraints.
3. All hidden context is captured in Context / Background.
4. Output format specifies structure, length, tone, and audience.
5. If a reasoning strategy was accepted, it is woven in (see `reasoning-strategies.md`).
6. The prompt is self-contained — works when pasted into any AI tool without extra context.
7. The prompt matches the language the user was writing in.
8. Complexity of prompt matches complexity of task — no over-engineering.
```

---

## Task 17: prompt-structure.md — commit

**Files:**
- Commit: `plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`

- [ ] **Step 1: Stage the file**

Run: `git add plugins/agent-core/skills/thinkfirst/references/prompt-structure.md`

- [ ] **Step 2: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
docs(agent-core): expand thinkfirst prompt-structure with optimization techniques

Adds the 10 Optimization Techniques with before/after examples from
prompt-optimizer (Clarity & Specificity, Output Structure Definition,
Audience Definition, Tone & Style Guidance, Role/Persona Assignment,
Step-by-Step Thinking, Examples, Document Q&A, Uncertainty
Acknowledgment, Iterative Structure). Preserves thinkfirst's full
Best Practices Checklist verbatim (9 items) and adopts
prompt-optimizer's stronger 8-item Final-Draft Checklist with
both-modes scope.

Minor restorations per spec parity rule: inline "use accessible
language" positive-framing example in Constraints / Guardrails;
title updated to mention Optimization Techniques; intro rewritten
for both-modes framing; When to Skip Sections converted to table
form; Mapping Seven Dimensions table reordered above Optimization
Techniques for logical flow.

Design spec: docs/superpowers/specs/2026-04-11-thinkfirst-prompt-optimizer-merge-design.md

File grows 5378 → ~8700 bytes. Reference files do not count against
the main skill context budget.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

Run: `git log --oneline -2`
Expected: last two commits are (1) SKILL.md merge from Task 10 and (2) prompt-structure.md merge from Task 17.

---

## Task 18: Verify — make validate

**Files:** (none edited)

- [ ] **Step 1: Run validators**

Run: `make validate`
Expected:
```
# Metadata Validation Report: agent-core
**Status:** ✅ VALID - No issues found
...
# Metadata Validation Report: dev-suite
**Status:** ✅ VALID - No issues found
...
# Metadata Validation Report: science-suite
**Status:** ✅ VALID - No issues found
...
Linting command files...
plugins/dev-suite/commands/tech-debt.md:128: [warning] heading-duplicate: ...
1 total issue(s); 0 at >= error
✓ Validation complete
```

The pre-existing tech-debt.md warning is not a blocker. Zero errors is the pass condition.

---

## Task 19: Verify — context budget

**Files:** (none edited)

- [ ] **Step 1: Run context budget checker**

Run: `python3 tools/validation/context_budget_checker.py --plugins-dir plugins`
Expected: `thinkfirst` reported as fitting the budget. No new `> 80%` warnings. thinkfirst's percentage should be ~71% (spec estimate).

If >75%: note in the final summary that the merge is closer to the at-risk line than estimated.
If >80%: STOP — revert the SKILL.md commit with `git revert HEAD~1 --no-edit` (the HEAD~1 target assumes prompt-structure commit is HEAD), run validators again, and escalate to the user with the measured size.

---

## Task 20: Verify — pytest and sphinx

**Files:** (none edited)

- [ ] **Step 1: Run pytest**

Run: `uv run pytest tools/tests/ -q`
Expected: `118 passed`. No new tests were added, no existing tests should have changed behavior.

- [ ] **Step 2: Run sphinx-build**

Run: `cd docs && sphinx-build -b html . _build/html 2>&1 | tail -5`
Expected: `build succeeded.`

A successful build means the reference-file links in the merged skill don't point to anything Sphinx thinks is broken. (Sphinx doesn't index skill files, but a spurious broken-reference warning would surface here.)

---

## Task 21: Summary

- [ ] **Step 1: Confirm all changes landed**

Run:
```bash
git log --oneline -3
```
Expected: the two new commits at HEAD and HEAD~1, whatever was at HEAD before at HEAD~2.

- [ ] **Step 2: Final byte-count diff**

Run:
```bash
wc -c plugins/agent-core/skills/thinkfirst/SKILL.md \
      plugins/agent-core/skills/thinkfirst/references/prompt-structure.md
```
Expected:
- `SKILL.md`: ~11400 bytes (from 9196, +~2200)
- `references/prompt-structure.md`: ~8700 bytes (from 5378, +~3300)

- [ ] **Step 3: Report**

Report to the user:
- Both commits landed with hashes `<X>` and `<Y>`
- Budget measurement: thinkfirst at X% (spec estimate was 71%)
- All validators green, 118/118 tests pass, sphinx build succeeded
- Functionality gained: Optimize Mode for users with an existing prompt to rewrite, 10 Optimization Techniques reference material, broadened trigger phrases
- Zero prose dropped from thinkfirst (prose parity rule honored)

---

## Rollback

If anything goes wrong and the merge needs to be reverted:

```bash
git revert HEAD --no-edit        # revert prompt-structure.md commit
git revert HEAD --no-edit        # revert SKILL.md commit
```

Or for a clean discard before commits land:

```bash
git checkout plugins/agent-core/skills/thinkfirst/SKILL.md \
             plugins/agent-core/skills/thinkfirst/references/prompt-structure.md
```

No other files or branches are affected. The `/Users/b80985/Downloads/prompt-optimizer/` source directory is untouched throughout.
