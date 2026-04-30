# Review Report Structure

This is the default report template. Use it unless the user has specified a journal whose guidelines differ, in which case adapt accordingly.

## Document format

Arial 11pt body, Heading 1 for the report title, Heading 2 for each section. Small metadata table at the top: Title, Date, Reviewer (leave blank), Journal. Numbered issues with a bolded short label line followed by prose. US Letter, 1-inch margins, restrained color on headings only. If `python-docx` is unavailable, produce `.md` with identical structure.

## Section order

1. **Metadata header** — Manuscript title, review date, reviewer (leave blank for the user to fill), journal
2. **Reviewer's statement** (optional) — A brief line on expertise match and any sections the reviewer flagged as outside their competence
3. **Summary of the Manuscript** — Shows you read the paper
4. **Overall Assessment** — Headline judgment
5. **Major Issues** — Publication-blocking concerns
6. **Minor Issues** — Improvements that should be made but don't block publication
7. **Questions for the Authors** — Clarifications needed
8. **Statistical and Quantitative Notes** — Arithmetic checks, missing uncertainties, fit quality concerns (omit if nothing quantitative to say)
9. **Recommendation** — Accept / Minor Revision / Major Revision / Reject, with brief justification
10. **Confidential Comments to Editor** (if integrity concerns were identified) — Visually separated from the rest of the review

Sections 3–9 are the "Comments to Authors" block. Section 10, if present, must be visually distinct because most journals treat it as a separate submission field.

## Section-by-section guidance

### Reviewer's statement (optional)

One or two sentences on expertise match relative to the manuscript. Omit if the manuscript is entirely within the reviewer's area.

### Summary of the Manuscript

One or two tight paragraphs: what the authors did, what they found, what they claim it means. Do not editorialize — save judgment for later sections.

### Overall Assessment

3–6 sentences: strengths briefly, then the most significant concerns. Stands alone — the editor often reads only this section before triaging.

**Good example:**
> "This is a well-conceived study that addresses a genuine gap in the literature on X. The experimental design is thoughtful and the combination of techniques A, B, and C provides complementary evidence for the central claim. However, the manuscript has significant issues that must be addressed: the characterization of the key reagent is incomplete, the power-law fit lacks theoretical grounding, and several conclusions are asserted rather than demonstrated."

**Weak example (avoid):**
> "The paper is okay but has problems."

### Major Issues

Numbered list. Each item: **bolded short label** (5–10 words) + paragraph covering (a) what the problem is, (b) why it matters, (c) what the authors can do. Aim for 3–7 items. More than 7 usually means minor issues are being lumped in.

Major issues threaten validity or core claims: missing controls, unsupported statistical conclusions, unaddressed alternative explanations, inadequate characterization, reproducibility gaps, overreaching conclusions.

**Good example of a major issue:**

> **2. Power-law exponent lacks theoretical grounding.** The reported scaling exponent of approximately −7.2 for W_ad vs. L_c/L_0 is described as "unusually steep," but no attempt is made to derive or justify this value from a physical model. Is this exponent expected from any existing polymer brush theory (e.g., Alexander–de Gennes, Milner–Witten–Cates)? How many data points were used in the fit, and what is the R²? The interpretation that it reflects "exceptional sensitivity to confinement" is qualitative. A more rigorous treatment — either by connecting the exponent to a scaling argument or by explicitly stating that no existing model predicts it — would significantly strengthen this finding.

### Minor Issues

Same numbered format. Improvements that don't threaten validity: typos, missing axis labels/units, undefined abbreviations, wrong reference format, reorganization suggestions. Don't catalogue every typo — flag representative examples and suggest a proofreading pass.

### Questions for the Authors

Genuine questions only. Good topics: parameter dependencies not explored, physical or biological interpretation of a result, relationship between experimental conditions.

### Statistical and Quantitative Notes

Use this section for concrete arithmetic and statistical observations. Show your work where independent calculations were done — one of the most valuable contributions a careful reviewer can make.

**Good example:**

> **2. Surface coverage ratio arithmetic.** The calculation shows Γ_PVBTMAC/σ = 0.017/0.076 ≈ 0.22, giving ~1 PVBTMAC per 5 PSS chains. Recomputing from the reported Δm = 35 ng/cm² and an estimated M_n ≈ 12,700 g/mol (using DP ≈ 60 and VBTMAC monomer MW ≈ 211.7 g/mol) gives Γ ≈ 1.66 × 10¹² chains/cm² = 0.0166 chains/nm², consistent with the reported value. The arithmetic checks out.

Frame any apparent discrepancies with hedged language — you may be missing context the authors have. "The reported p = 0.03 appears inconsistent with t = 1.8 at n = 15; the authors may wish to verify" is better than "The p-value is wrong."

If the manuscript provides nothing quantitative to check, omit this section entirely rather than padding it.

### Recommendation

One line + short justification tied to the major issues. Four categories:

- **Accept** — Ready as-is. Rare on first review.
- **Minor Revision** — Sound work; clarifications or figure improvements needed.
- **Major Revision** — Valuable but significant issues must be addressed; reviewer willing to re-review.
- **Reject** — Fatal flaws or wrong journal.

Don't hedge. Editors want a clear call.

### Confidential Comments to Editor (if needed)

Visually distinct block at the end — most journals have a separate submission field for this. Include: integrity concerns from Lens 6, expertise caveats inappropriate for the authors, a stronger/weaker recommendation if warranted, COI disclosures, suggestions for other reviewers. Do not insult authors or speculate about motives. Omit entirely if nothing editor-specific to say.

## Tone principles

**Colleague, not judge.** Treat the authors as capable scientists; you are helping produce better science, not rendering a verdict.

**Hedge appropriately.** "This appears to be" and "the reviewer notes that" are honest, not weak — you are working from a manuscript, not a complete record.

**Be concrete.** Every observation must name a section, figure, equation, or page number. Authors cannot act on "Section 3 is weak."

**Suggest, don't dictate.** "The authors may wish to consider" not "Section 3.2 must be reorganized."

**Acknowledge uncertainty.** Flag sub-fields outside your expertise; editors weight reviews by expertise.

**Use field-appropriate language.** Use correct terms-of-art and cite the right classical results — a review that reads as written by a sub-field peer carries more weight.

## Tone examples

**Too harsh:**
> "The authors clearly have no understanding of polymer brush theory. Their interpretation is obviously wrong."

**Better:**
> "The interpretation on p. 8 appears to conflict with the predictions of Alexander–de Gennes theory for brushes in this regime. The authors should either reconcile their interpretation with existing theory or explicitly argue why it does not apply here."

**Too accusatory (integrity):**
> "Figure 3B has been fabricated."

**Better (in Confidential Comments to Editor):**
> "Figure 3B, panel (ii), shows a vertical discontinuity between lanes 4 and 5 that appears inconsistent with a single continuous gel image. The editor may wish to request the original source file from the authors."
