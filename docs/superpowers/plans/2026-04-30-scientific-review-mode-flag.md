# scientific-review --mode flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--mode` flag (simple / standard / comprehensive) to `scientific-review` so reference files load conditionally, cutting token use by 39–85% depending on mode.

**Architecture:** SKILL.md becomes a lean base (~4,200 bytes, always loaded). `references/review_structure.md` loads for `standard` and `comprehensive`. `references/integrity_checks.md` loads for `comprehensive` only. All six analysis lenses and Phase 0 run in every mode. Default mode is `standard`.

**Tech Stack:** Markdown only — no code changes. Validate with `python3 tools/validation/context_budget_checker.py` and `uv run pytest tools/tests/ -v`.

---

### Task 1: Record baseline metrics

**Files:**
- Read: `plugins/research-suite/skills/scientific-review/SKILL.md`
- Read: `plugins/research-suite/skills/scientific-review/references/review_structure.md`
- Read: `plugins/research-suite/skills/scientific-review/references/integrity_checks.md`

- [ ] **Step 1: Run context budget checker and save output**

```bash
python3 tools/validation/context_budget_checker.py 2>&1 | grep -A1 "scientific-review"
```

Expected output (approximate):
```
| scientific-review | research-suite | 2609 | pass | pass | 104 |
```

Note: 2,609 tokens at 104% of the 2% context budget. This is the before-state.

- [ ] **Step 2: Record raw byte counts**

```bash
wc -c plugins/research-suite/skills/scientific-review/SKILL.md \
       plugins/research-suite/skills/scientific-review/references/review_structure.md \
       plugins/research-suite/skills/scientific-review/references/integrity_checks.md
```

Expected (approximate):
```
  10489 SKILL.md
  11711 review_structure.md
   6771 integrity_checks.md
  28971 total
```

Keep this output handy — Task 5 compares against it.

---

### Task 2: Rewrite SKILL.md

Replace the entire file with the compressed content below. This covers simple mode fully and routes richer modes to the reference files.

**Files:**
- Modify: `plugins/research-suite/skills/scientific-review/SKILL.md` (full replacement)

- [ ] **Step 1: Replace SKILL.md with the following content**

```markdown
---
name: scientific-review
description: "Use this skill whenever the user wants to review a scientific manuscript, write peer review comments, prepare a referee report, or critique a research paper. Trigger on phrases like 'review this paper', 'review this manuscript', 'peer review', 'referee report', 'reviewer comments', 'critique this paper', 'evaluate this manuscript', or when the user provides a scientific paper (PDF, DOCX, or pasted text) and asks for feedback, assessment, or structured comments. Also trigger when the user asks to assess novelty, check experimental design, verify statistical claims, check for plagiarism or data integrity concerns, or produce a formal review for a journal submission. If the user names a specific journal, this skill performs a web search for that journal's reviewer guidelines before structuring the output. The final deliverable is a downloadable .docx review report (markdown fallback if python-docx is unavailable)."
---

# Scientific Manuscript Review

Produce a rigorous peer review as a downloadable `.docx` (fallback: `.md`, convert with `pandoc review.md -o review.docx`).

## Mode

Pass `--mode` to control depth. Default: `standard`.

| `--mode`        | Loads additionally                      | Use when                        |
|-----------------|-----------------------------------------|---------------------------------|
| `simple`        | nothing                                 | quick triage, short papers      |
| `standard`      | `references/review_structure.md`        | typical review work             |
| `comprehensive` | + `references/integrity_checks.md`      | integrity-sensitive reviews     |

## Phase 0: Reviewer responsibilities

Ask all three in one message; skip if already answered this session:

- **COI.** Co-authorship ≤5 yr, active collaboration, advisor–advisee, direct competitor, or financial stake → recommend declining.
- **Confidentiality & AI policy.** Manuscript is privileged. Journal AI policies vary; check if user names a journal (fold into Phase 2 search).
- **Expertise match.** Flag areas outside reviewer's competence; note them in the review rather than critique confidently.

## Phase 1: Ingest

Extract full text: DOCX → `pandoc <path> -t markdown`; PDF → `pdftotext` first, escalate to `pypdf`/`pymupdf` if figures matter. Build inventory: title, authors (redacted?), journal, section structure, field/sub-field, and what is missing (Limitations? Data availability? Funding disclosure?).

## Phase 2: Journal adaptation

If journal named, search current reviewer guidelines (WebSearch or Context7). Check: structure (free-form vs. structured fields), recommendation categories, journal-specific criteria, ethics requirements, AI policy.

Without a journal — `standard`/`comprehensive`: load `references/review_structure.md` for the full template. `simple`: use default section order: Summary → Overall Assessment → Major Issues → Minor Issues → Questions for Authors → Statistical Notes → Recommendation → Confidential Comments to Editor (if needed).

## Phase 3: Six-lens analysis

All lenses run in every mode. Tie every observation to a specific section, figure, equation, or claim — generic comments are useless.

**Lens 1 — Domain expertise.**
- Contribution genuinely new, or derivative of prior work?
- Literature comprehensive; missing citations or competing frameworks?
- Notation consistent; dimensional balance correct in equations?
- Vocabulary matches sub-field terms-of-art.

**Lens 2 — Methodological rigor.**
- Controls adequate; variables isolated; confounds addressed?
- Statistical tests appropriate; sample size justified; multiple-comparison corrections applied?
- Error bars defined (SD/SEM/CI); reported statistics support stated conclusions?
- Methods reproducible: reagents (vendor/catalog), instruments, software versions, code/data links.

**Lens 3 — Critical thinking.**
- Conclusion follows from data, or is there an unsupported leap?
- Alternative explanations addressed or silently ignored?
- Abstract matches paper; claimed impact proportionate to evidence?

**Lens 4 — Results and data presentation.**
- Figures clear, labeled, axis units complete, significance indicators present?
- Numbers in text match tables?
- Raw or supplementary data accessible?

**Lens 5 — Writing quality.**
- Organization, clarity, notation discipline, abbreviations defined on first use.
- Flag serious issues only; do not copy-edit.

**Lens 6 — Ethical integrity.**
- `comprehensive` mode: load `references/integrity_checks.md` now for full red-flag checklist.
- All modes: verify IRB/IACUC approval, COI disclosure, funding statement, data availability statement.
- Integrity concerns → Confidential Comments to Editor only. Phrase as "features that may warrant the editor's attention."

## Phase 4: Quantitative verification

- **Direct:** Recompute reported statistics, percentages, derived quantities, unit conversions. Show the arithmetic.
- **Flagging:** Note missing error bars, unreported sample sizes, unspecified fitting method, "significantly different" without stated tests or corrections.

Frame discrepancies diplomatically: *"The reported p = 0.03 appears inconsistent with t = 1.8 at n = 15; the authors may wish to verify."*

## Phase 5: Compose the report

`standard`/`comprehensive`: load `references/review_structure.md` for the full template, section guidance, format spec, and tone examples.

Save to `./reviews/<short-title>.docx`. Keep the chat message to one paragraph: main issues + recommendation only. If `python-docx` is unavailable, save as `.md`.

## Principles

- Specific beats comprehensive — five sharp observations over twenty generic ones.
- Separate the paper's logic from your own preferences.
- Every problem identified needs a path forward.
- Acknowledge genuine strengths.
- Distinguish publication-blocking from publication-improving issues.
- Time budget ~3–6 hours; focus on what affects publishability.
- Surface concerns to the editor; do not pronounce guilt.
```

- [ ] **Step 2: Verify byte count reduced**

```bash
wc -c plugins/research-suite/skills/scientific-review/SKILL.md
```

Expected: ≤ 4,400 bytes (down from 10,489).

- [ ] **Step 3: Commit**

```bash
git add plugins/research-suite/skills/scientific-review/SKILL.md
git commit -m "refactor(scientific-review): compress SKILL.md, add --mode routing table"
```

---

### Task 3: Update review_structure.md

Three changes: (1) add the docx format spec section moved from SKILL.md, (2) cut two of the four tone example pairs, (3) trim ~20% of per-section prose.

**Files:**
- Modify: `plugins/research-suite/skills/scientific-review/references/review_structure.md`

- [ ] **Step 1: Trim the Reviewer's statement section**

Replace:

```markdown
### Reviewer's statement (optional)

A one- or two-sentence statement on the reviewer's expertise relative to the manuscript. Useful when the manuscript spans multiple methodologies — it tells the editor which parts of the critique carry the most weight.

**Example:**
> "This review focuses primarily on the experimental polymer physics and surface force measurements, which are within the reviewer's core expertise. The statistical treatment of the QCM-D data was assessed to the best of the reviewer's ability but may benefit from additional input from a reviewer with specific QCM-D modeling expertise."

Omit if the manuscript is entirely within the reviewer's area.
```

With:

```markdown
### Reviewer's statement (optional)

One or two sentences on expertise match relative to the manuscript. Omit if the manuscript is entirely within the reviewer's area.
```

- [ ] **Step 2: Add document format section after the opening paragraph**

Insert after the line `This is the default report template. Use it unless the user has specified a journal whose guidelines differ, in which case adapt accordingly.` and before `## Section order`:

```markdown
## Document format

Arial 11pt body, Heading 1 for the report title, Heading 2 for each section. Small metadata table at the top: Title, Date, Reviewer (leave blank for user to fill), Journal. Numbered issues with a bolded short label line followed by prose. US Letter, 1-inch margins, restrained color on headings only. If `python-docx` is unavailable, produce `.md` with identical structure.

```

- [ ] **Step 3: Trim the Summary section guidance**

Replace:

```markdown
### Summary of the Manuscript

One or two tight paragraphs. Cover: what the authors did, what they found, and what they claim it means. This demonstrates to the editor that the reviewer has genuinely read and understood the paper, and gives the authors a chance to correct you if your summary reveals a misread. Do not editorialize — save judgment for later sections.
```

With:

```markdown
### Summary of the Manuscript

One or two tight paragraphs: what the authors did, what they found, what they claim it means. Do not editorialize — save judgment for later sections.
```

- [ ] **Step 4: Trim the Overall Assessment section guidance**

Replace:

```markdown
### Overall Assessment

A short paragraph (3–6 sentences) giving the headline verdict. Lead with strengths briefly, then the most significant concerns. The editor often reads only this section before triaging the paper, so it should stand on its own.
```

With:

```markdown
### Overall Assessment

3–6 sentences: strengths briefly, then the most significant concerns. Stands alone — the editor often reads only this section before triaging.
```

- [ ] **Step 5: Trim the Major Issues section guidance**

Replace:

```markdown
### Major Issues

Numbered list. Each item has:
- A **bolded short label** (5–10 words) identifying the issue
- A paragraph explaining (a) what the problem is, (b) why it matters, and (c) what the authors could do about it

Major issues threaten the paper's validity or core claims. Examples:
- Missing controls
- Statistical analysis that doesn't support the stated conclusion
- Alternative explanations that haven't been addressed
- Reagents or methods that aren't adequately characterized
- Reproducibility gaps severe enough that the work can't be replicated
- Overreaching conclusions not supported by the data

Aim for 3–7 major issues. Fewer than 3 is rare for a paper that needs review at all; more than 7 usually indicates the reviewer is lumping minor issues into the major category.
```

With:

```markdown
### Major Issues

Numbered list. Each item: **bolded short label** (5–10 words) + paragraph covering (a) what the problem is, (b) why it matters, (c) what the authors can do. Aim for 3–7 items. More than 7 usually means minor issues are being lumped in.

Major issues threaten validity or core claims: missing controls, unsupported statistical conclusions, unaddressed alternative explanations, inadequate characterization, reproducibility gaps, overreaching conclusions.
```

- [ ] **Step 6: Trim the Minor Issues and Questions sections**

Replace:

```markdown
### Minor Issues

Same numbered format. Minor issues are improvements that should be made but don't threaten the paper's validity. Examples:
- Typos, grammatical errors, notation inconsistencies
- Figures with missing axis labels or units
- Abbreviations not defined on first use
- References in the wrong format
- A paragraph that would read better reorganized
- A small piece of characterization data that should be reported

Don't exhaustively catalogue every typo — editors have copy-editors for that. Flag representative examples and suggest a proofreading pass if the writing needs broader attention.

### Questions for the Authors

Genuine questions, not rhetorical ones. If the paper leaves you uncertain about something, ask. Authors often clarify in the response, and sometimes the clarification reveals that the manuscript itself needs revision. Good question topics:
- Reversibility, temperature sensitivity, or other parameter dependencies not explored
- Physical or biological interpretation of a parameter or plateau
- Relationship between different experimental conditions
```

With:

```markdown
### Minor Issues

Same numbered format. Improvements that don't threaten validity: typos, missing axis labels/units, undefined abbreviations, wrong reference format, reorganization suggestions. Don't catalogue every typo — flag representative examples and suggest a proofreading pass.

### Questions for the Authors

Genuine questions only. Good topics: parameter dependencies not explored, physical or biological interpretation of a result, relationship between experimental conditions.
```

- [ ] **Step 7: Trim the Recommendation section**

Replace:

```markdown
### Recommendation

A single line stating the recommendation, followed by a short paragraph of justification that ties back to the major issues. The four standard categories:

- **Accept** — The paper is ready for publication as-is. Rare on first review; signal a high bar.
- **Minor Revision** — The work is sound and the claims are supported; some clarifications, added references, or figure improvements are needed.
- **Major Revision** — The core work is valuable but significant issues must be addressed. The reviewer should be willing to re-review.
- **Reject** — Either the work has fatal flaws that cannot be addressed without a fundamentally different study, or it is not appropriate for this journal.

Don't hedge the recommendation itself. Editors want a clear call even if the reasoning is nuanced.
```

With:

```markdown
### Recommendation

One line + short justification tied to the major issues. Four categories:

- **Accept** — Ready as-is. Rare on first review.
- **Minor Revision** — Sound work; clarifications or figure improvements needed.
- **Major Revision** — Valuable but significant issues must be addressed; reviewer willing to re-review.
- **Reject** — Fatal flaws or wrong journal.

Don't hedge. Editors want a clear call.
```

- [ ] **Step 8: Trim the Confidential Comments section**

Replace:

```markdown
### Confidential Comments to Editor (if needed)

This section is for things the reviewer wants the editor to see but not the authors. Most journals have a separate field for this — format it as a visually distinct block at the end of the report so the reviewer can easily copy-paste into that field.

Appropriate content:
- Integrity concerns identified in Lens 6 (see `integrity_checks.md` for specific framings)
- Observations about the reviewer's expertise that would be inappropriate in the authors-facing review
- A stronger or weaker recommendation than the authors-facing review implies, if the reviewer feels the tone for the authors should be gentler than the substantive judgment
- Potential conflicts of interest the reviewer wants to disclose
- Suggestions about which other reviewers or expertise areas would be valuable

Do not use this section to:
- Insult the authors personally
- Register general gripes unrelated to the specific manuscript
- Speculate about authors' motivations

If no integrity or editor-specific concerns exist, omit the section entirely.
```

With:

```markdown
### Confidential Comments to Editor (if needed)

Visually distinct block at the end — most journals have a separate submission field for this. Include: integrity concerns from Lens 6, expertise caveats inappropriate for the authors, a stronger/weaker recommendation if warranted, COI disclosures, suggestions for other reviewers. Do not insult authors or speculate about motives. Omit entirely if nothing editor-specific to say.
```

- [ ] **Step 9: Trim the Tone principles section**

Replace:

```markdown
## Tone principles

**Write as a colleague, not a judge.** The reviewer is a peer trying to help produce better science — not a gatekeeper rendering verdicts. Treat the authors as capable scientists who can respond to specific, well-reasoned feedback.

**Hedge appropriately.** The reviewer is working from a manuscript, not a complete record of the work. "This appears to be" and "the reviewer notes that" are honest acknowledgments of limited information, not weakness.

**Be concrete.** Every observation should be specific enough that the authors know exactly what section, figure, or sentence you're referring to. Page numbers, figure numbers, and equation numbers are your friends.

**Don't demand rewrites of what isn't yours to rewrite.** Suggest changes; don't dictate them. "The authors may wish to consider reorganizing Section 3.2" is better than "Section 3.2 must be reorganized."

**Acknowledge your own uncertainty.** If a claim is in a sub-field where you're less expert, say so. Editors weight reviews by expertise.

**Use field-appropriate technical language.** A review that reads as though written by a peer in the sub-field carries more weight than one that reads as generic. This means using the correct terms-of-art ("Manning condensation" rather than "some counterions sticking to the chain"), referencing the right classical results ("this is inconsistent with the Milner–Witten–Cates prediction" rather than "this doesn't match theory"), and matching the notation conventions of the field. When unsure, search recent literature for current usage.
```

With:

```markdown
## Tone principles

**Colleague, not judge.** Treat the authors as capable scientists; you are helping produce better science, not rendering a verdict.

**Hedge appropriately.** "This appears to be" and "the reviewer notes that" are honest, not weak — you are working from a manuscript, not a complete record.

**Be concrete.** Every observation must name a section, figure, equation, or page number. Authors cannot act on "Section 3 is weak."

**Suggest, don't dictate.** "The authors may wish to consider" not "Section 3.2 must be reorganized."

**Acknowledge uncertainty.** Flag sub-fields outside your expertise; editors weight reviews by expertise.

**Use field-appropriate language.** Use correct terms-of-art and cite the right classical results — a review that reads as written by a sub-field peer carries more weight.
```

- [ ] **Step 10: Cut two tone example pairs**

The current `## Tone examples` section has four before/after pairs. Remove pairs 2 ("Too vague") and 3 ("Too deferential"). Keep pairs 1 ("Too harsh") and 4 ("Too accusatory (integrity)").

Replace the entire `## Tone examples` section:

```markdown
## Tone examples

**Too harsh:**
> "The authors clearly have no understanding of polymer brush theory. Their interpretation is obviously wrong."

**Better:**
> "The interpretation on p. 8 appears to conflict with the predictions of Alexander–de Gennes theory for brushes in this regime. The authors should either reconcile their interpretation with existing theory or explicitly argue why it does not apply here."

**Too accusatory (integrity):**
> "Figure 3B has been fabricated."

**Better (in Confidential Comments to Editor):**
> "Figure 3B, panel (ii), shows a vertical discontinuity between lanes 4 and 5 that appears inconsistent with a single continuous gel image. The editor may wish to request the original source file from the authors."
```

- [ ] **Step 11: Verify byte count reduced**

```bash
wc -c plugins/research-suite/skills/scientific-review/references/review_structure.md
```

Expected: ≤ 8,000 bytes (down from 11,711).

- [ ] **Step 12: Commit**

```bash
git add plugins/research-suite/skills/scientific-review/references/review_structure.md
git commit -m "refactor(scientific-review): trim review_structure.md, move format spec, cut 2 tone examples"
```

---

### Task 4: Update integrity_checks.md

Two changes: remove the final "How to write" section (already covered in SKILL.md), and compress the per-category "How to flag" paragraphs to one-liners.

**Files:**
- Modify: `plugins/research-suite/skills/scientific-review/references/integrity_checks.md`

- [ ] **Step 1: Compress the plagiarism "How to flag" paragraph**

Replace:

```markdown
How to flag: identify the specific passage and the suspected source (if known) in Confidential Comments to Editor. Suggest the editor run similarity software on the passage. Do not claim plagiarism in Comments to Authors.
```

With:

```markdown
How to flag: name the passage and suspected source in Confidential Comments to Editor; suggest similarity-software check. Never claim plagiarism in Comments to Authors.
```

- [ ] **Step 2: Compress the data/image integrity "How to flag" paragraph**

Replace:

```markdown
How to flag: describe what you see and where (figure and panel), and ask the editor to request raw data or source files. Phrase as "features worth verifying" rather than "evidence of manipulation."
```

With:

```markdown
How to flag: describe what you see and where (figure/panel); ask editor to request raw data or source files. Phrase as "features worth verifying."
```

- [ ] **Step 3: Remove the final section entirely**

Delete from `## How to write integrity concerns in the report` to the end of the file (lines 88–99 in the original):

```markdown
## How to write integrity concerns in the report

The Confidential Comments to Editor section is where these go. Three principles:

1. **Describe what you observed, not what you conclude.** "Figure 3B shows a vertical discontinuity between lanes 4 and 5 that may indicate splicing" is verifiable. "The authors have fabricated data" is an accusation the reviewer cannot support.

2. **Cite the specific location.** Page, figure, panel, paragraph. The editor may not see what you see without direction.

3. **Suggest the next step.** "The editor may wish to request original source files for Figure 3" or "The editor may wish to run similarity software on the Introduction, particularly the passage on p. 2."

Keep this section measured. Over-zealous integrity accusations from reviewers harm the system as much as under-vigilant ones.
```

The file should now end after the `## Authorship concerns` section.

- [ ] **Step 4: Verify byte count reduced**

```bash
wc -c plugins/research-suite/skills/scientific-review/references/integrity_checks.md
```

Expected: ≤ 5,600 bytes (down from 6,771).

- [ ] **Step 5: Commit**

```bash
git add plugins/research-suite/skills/scientific-review/references/integrity_checks.md
git commit -m "refactor(scientific-review): compress integrity_checks.md, remove redundant how-to-write section"
```

---

### Task 5: Validate and verify targets

- [ ] **Step 1: Run context budget checker**

```bash
python3 tools/validation/context_budget_checker.py 2>&1 | grep -A1 "scientific-review"
```

Expected: token count ≤ 1,100 (down from 2,609), percentage ≤ 45%.

- [ ] **Step 2: Check all three file byte counts**

```bash
wc -c plugins/research-suite/skills/scientific-review/SKILL.md \
       plugins/research-suite/skills/scientific-review/references/review_structure.md \
       plugins/research-suite/skills/scientific-review/references/integrity_checks.md
```

Expected targets:

| File | Before | Target |
|------|--------|--------|
| SKILL.md | 10,489 | ≤ 4,400 |
| review_structure.md | 11,711 | ≤ 8,000 |
| integrity_checks.md | 6,771 | ≤ 5,600 |
| **Total** | **28,971** | **≤ 18,000** |

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tools/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass (no regressions in cross-suite invariants or integrity checks).

- [ ] **Step 4: Smoke-test mode routing logic**

Read SKILL.md and verify:
- The `## Mode` table is present with all three modes and their file loads
- Phase 3 Lens 6 references `--mode=comprehensive` for loading integrity_checks.md
- Phase 5 references `references/review_structure.md` for standard/comprehensive
- No mention of Arial/margins/format spec in SKILL.md (moved to review_structure.md)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "refactor(scientific-review): add --mode flag, reduce token use 39-85% by mode"
```
