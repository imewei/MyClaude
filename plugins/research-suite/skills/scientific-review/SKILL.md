---
name: scientific-review
description: "Use this skill whenever the user wants to review a scientific manuscript, write peer review comments, prepare a referee report, or critique a research paper. Trigger on phrases like 'review this paper', 'review this manuscript', 'peer review', 'referee report', 'reviewer comments', 'critique this paper', 'evaluate this manuscript', or when the user provides a scientific paper (PDF, DOCX, or pasted text) and asks for feedback, assessment, or structured comments. Also trigger when the user asks to assess novelty, check experimental design, verify statistical claims, check for plagiarism or data integrity concerns, or produce a formal review for a journal submission. If the user names a specific journal, this skill performs a web search for that journal's reviewer guidelines before structuring the output. The final deliverable is a downloadable .docx review report (markdown fallback if python-docx is unavailable)."
---

# Scientific Manuscript Review

Produce a rigorous peer review as a downloadable `.docx` (fallback: `.md`, convert with `pandoc review.md -o review.docx`).

## Mode

Default: `standard`.

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
