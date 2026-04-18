---
name: scientific-review
description: "Use this skill whenever the user wants to review a scientific manuscript, write peer review comments, prepare a referee report, or critique a research paper. Trigger on phrases like 'review this paper', 'review this manuscript', 'peer review', 'referee report', 'reviewer comments', 'critique this paper', 'evaluate this manuscript', or when the user provides a scientific paper (PDF, DOCX, or pasted text) and asks for feedback, assessment, or structured comments. Also trigger when the user asks to assess novelty, check experimental design, verify statistical claims, check for plagiarism or data integrity concerns, or produce a formal review for a journal submission. If the user names a specific journal, this skill performs a web search for that journal's reviewer guidelines before structuring the output. The final deliverable is a downloadable .docx review report (markdown fallback if python-docx is unavailable)."
---

# Scientific Manuscript Review

Produce a rigorous peer review as a downloadable `.docx`. The skill covers six competencies:
domain expertise, methodological rigor, critical thinking, constructive communication,
ethical integrity, and time-efficient delivery. The output is a thorough first-pass draft the
reviewer refines with their own judgment — not a replacement for the reviewer's professional
responsibility.

## Reference files to load when relevant

- `references/review_structure.md` — report template, section guidance, tone examples. Load
  before composing the report.
- `references/integrity_checks.md` — specific red flags for plagiarism, data/image
  manipulation, and other integrity concerns. Load during Phase 3, Lens 6.

Document generation uses `python-docx` (`pip install python-docx` or `uv add python-docx`
if missing). If the user's environment cannot install it, fall back to a markdown
`.md` file with the same structure and tell the user to convert via `pandoc review.md -o
review.docx` at their convenience.

PDF ingestion uses `pdftotext` (poppler) for plain text or `pypdf` / `pymupdf` when figure
captions and layout matter. DOCX ingestion uses `pandoc <path> -t markdown`.

## Phase 0: Reviewer responsibilities

Before analyzing, surface a short checklist and wait for confirmation. Ask the three
questions below in a single message so the user can answer them together. Skip re-asking on
subsequent uses in the same session when context is already clear.

1. **Conflict of interest.** Recent co-authorship (3–5 years), current collaboration,
   advisor–advisee relationship, direct competition on the same question, or financial stake
   typically requires recusal. If a COI exists, recommend declining the review.
2. **Confidentiality and AI-assistance policy.** Manuscripts under review are privileged —
   the reviewer should not share contents or use them to inform unpublished research. Journal
   policies on AI-assisted review vary (some permit, some prohibit, some require disclosure).
   If the user names a journal, fold this check into the Phase 2 web search.
3. **Expertise match.** Areas the reviewer flags as outside their competence get noted in the
   review as requiring additional human expertise rather than given confident critique.

## Phase 1: Ingest

Extract the full text. For DOCX, `pandoc <path> -t markdown`. For PDF, start with
`pdftotext` and escalate to `pypdf` / `pymupdf` if figures matter. Build a mental inventory:
title, authors (redacted?), target journal, section structure, figures/tables/equations, and
the field/sub-field for applying domain norms. Note what's missing — Limitations section?
Data availability statement? Funding disclosure?

## Phase 2: Journal-specific adaptation

If the user names a journal, search for its current reviewer guidelines (WebSearch or
Context7 for well-documented publishers). Look for: review structure (free-form vs.
structured fields; separate Comments-to-Editor field?), recommendation categories,
journal-specific criteria (*Cell* weights broad impact, *PRL* weights surprise and brevity,
clinical journals weight CONSORT-style reporting), ethics requirements, and AI-assistance
policy. Adapt the output to match. Without a named journal, use the default structure in
`references/review_structure.md`.

## Phase 3: Six-lens analysis

Work through all six lenses. Observations should be specific — tied to a particular section,
figure, equation, or claim. Generic comments are the hallmark of a lazy review; the
difference between a useful review and a weak one is whether the authors can act on each
point.

**Lens 1 — Domain expertise.** Is the contribution genuinely new or derivative? Is the
literature coverage comprehensive? Any obvious missing citations or competing frameworks the
authors ignored? Engage with the sub-field's actual vocabulary: use the correct terms-of-art
("Manning condensation," "Voigt viscoelastic model") rather than generic paraphrases, check
notation consistency, verify dimensional balance in equations. A review that reads as
written by a peer in the sub-field carries more weight than a generic one. Search recent
literature when unsure about current usage.

**Lens 2 — Methodological rigor.** Are controls adequate and variables isolated? Could the
effect arise from confounds? Are statistical tests appropriate for the data type and
distribution? Is sample size justified? Multiple-comparison corrections where needed? Do
reported statistics actually support the conclusions? Are error bars defined (SD vs. SEM vs.
CI)? Could a competent peer replicate the work from the Methods section alone — reagents
with vendor/catalog, instrument models, software versions, parameters, code and data
availability?

**Lens 3 — Critical thinking.** Does the conclusion follow from the data or is there a leap?
Are alternative explanations addressed or silently ignored? Does the abstract match the
paper? Is the claimed impact proportionate to the evidence? Is the motivation clear in the
introduction and revisited in the conclusion?

**Lens 4 — Results and data presentation.** Figures clear and properly labeled? Axis units
complete? Error bars and significance indicators where expected? Raw or supplementary data
accessible? Do numbers in the text match the tables?

**Lens 5 — Writing quality.** Organization, clarity, terminology consistency, notation
discipline, abbreviations defined on first use. Flag serious issues but don't drift into
copy-editing.

**Lens 6 — Ethical integrity.** Load `references/integrity_checks.md`. At minimum, assess:
plagiarism and self-plagiarism (stylistic shifts, uncited passages that track a specific
prior paper), data/image integrity (gel splicing artifacts, repeated background texture,
implausible precision), selective reporting and p-hacking, dual submission, ethics approvals
(IRB/IACUC/trial registration where applicable), COI and funding disclosure, authorship
statements. Integrity concerns go to the *editor* (Confidential Comments to Editor), not to
the authors. Phrase as "features that may warrant the editor's attention," not accusations —
the reviewer surfaces concerns; the editor adjudicates.

## Phase 4: Quantitative verification

Use the Bash tool or a short Python script. This is one of the highest-value contributions
a careful reviewer can make. Two modes:

**Direct verification.** Do reported statistics match given the stated df? Do percentages,
ratios, derived quantities recompute correctly? Do unit conversions and dimensional analyses
check out? For regression/fit exponents, are uncertainties and R² reported with the fitting
method and data range? Show the work in the review (e.g., *"Recomputing from Δm = 35 ng/cm²
and Mₙ ≈ 12,700 g/mol gives Γ ≈ 0.017 chains/nm², consistent with the authors' value"*).

**Flagging.** Missing error bars, unreported sample sizes, unspecified fitting methodology,
"significantly different" without stated tests or corrections.

Frame apparent discrepancies diplomatically. You're working from the manuscript alone —
*"The reported p = 0.03 appears inconsistent with t = 1.8 at n = 15; the authors may wish to
verify"* is right; *"The p-value is wrong"* is not.

## Phase 5: Compose the report

Load `references/review_structure.md` for the template and tone.

Format: Arial 11pt body, Heading 1 for title, Heading 2 for sections, small metadata table
at the top (Title, Date, Reviewer, Journal), numbered issues with a bolded short label line
followed by prose, black body text with restrained color on headings, US Letter with 1-inch
margins.

If integrity concerns surfaced in Lens 6, place them in a clearly labeled "Confidential
Comments to Editor" section at the end, visually separated. Most journals treat this as a
separate submission field.

Save to the user's working directory (or a subdirectory like `./reviews/`) with a
descriptive filename (e.g., `Peer_Review_<short-title>.docx`) and report the absolute path
to the user. Keep the chat message brief: a one-paragraph summary of the main issues and
the recommendation — not a recap.

## Principles that matter more than the template

- **Specific beats comprehensive.** Five sharp observations beat twenty generic ones. If a
  comment can't be tied to a section, figure, or equation, reconsider whether it's useful.
- **Separate the paper's logic from your preferences.** The question is whether the authors'
  framework is internally consistent and defensible, not whether it matches yours.
  Disagreements with framing go in Questions or Discussion, not Major Issues — unless the
  framing actually breaks the argument. Guard against the bias of rejecting work that
  challenges your own.
- **Every problem gets a path forward.** "This section is weak" is useless; "this section
  would be strengthened by adding a control at condition X" is actionable.
- **Acknowledge strengths honestly.** A review that is all criticism signals disengagement.
- **Distinguish publication-blocking from publication-improving.** Don't lump minor issues
  into Major.
- **Time is finite.** Typical budget is 3–6 hours. Focus on what affects whether the work
  should be published, not every typographical imperfection. This skill exists to produce a
  thorough first-pass draft so the reviewer spends their time on high-judgment refinement.
- **The reviewer is not a lawyer; the manuscript is not a contract.** Flag concerns, suggest
  improvements, trust the editor.
