# Review Report Structure

This is the default report template. Use it unless the user has specified a journal whose guidelines differ, in which case adapt accordingly.

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

A one- or two-sentence statement on the reviewer's expertise relative to the manuscript. Useful when the manuscript spans multiple methodologies — it tells the editor which parts of the critique carry the most weight.

**Example:**
> "This review focuses primarily on the experimental polymer physics and surface force measurements, which are within the reviewer's core expertise. The statistical treatment of the QCM-D data was assessed to the best of the reviewer's ability but may benefit from additional input from a reviewer with specific QCM-D modeling expertise."

Omit if the manuscript is entirely within the reviewer's area.

### Summary of the Manuscript

One or two tight paragraphs. Cover: what the authors did, what they found, and what they claim it means. This demonstrates to the editor that the reviewer has genuinely read and understood the paper, and gives the authors a chance to correct you if your summary reveals a misread. Do not editorialize — save judgment for later sections.

### Overall Assessment

A short paragraph (3–6 sentences) giving the headline verdict. Lead with strengths briefly, then the most significant concerns. The editor often reads only this section before triaging the paper, so it should stand on its own.

**Good example:**
> "This is a well-conceived study that addresses a genuine gap in the literature on X. The experimental design is thoughtful and the combination of techniques A, B, and C provides complementary evidence for the central claim. However, the manuscript has significant issues that must be addressed: the characterization of the key reagent is incomplete, the power-law fit lacks theoretical grounding, and several conclusions are asserted rather than demonstrated."

**Weak example (avoid):**
> "The paper is okay but has problems."

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

**Good example of a major issue:**

> **2. Power-law exponent lacks theoretical grounding.** The reported scaling exponent of approximately −7.2 for W_ad vs. L_c/L_0 is described as "unusually steep," but no attempt is made to derive or justify this value from a physical model. Is this exponent expected from any existing polymer brush theory (e.g., Alexander–de Gennes, Milner–Witten–Cates)? How many data points were used in the fit, and what is the R²? The interpretation that it reflects "exceptional sensitivity to confinement" is qualitative. A more rigorous treatment — either by connecting the exponent to a scaling argument or by explicitly stating that no existing model predicts it — would significantly strengthen this finding.

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

### Statistical and Quantitative Notes

Use this section for concrete arithmetic and statistical observations. Show your work where independent calculations were done — one of the most valuable contributions a careful reviewer can make.

**Good example:**

> **2. Surface coverage ratio arithmetic.** The calculation shows Γ_PVBTMAC/σ = 0.017/0.076 ≈ 0.22, giving ~1 PVBTMAC per 5 PSS chains. Recomputing from the reported Δm = 35 ng/cm² and an estimated M_n ≈ 12,700 g/mol (using DP ≈ 60 and VBTMAC monomer MW ≈ 211.7 g/mol) gives Γ ≈ 1.66 × 10¹² chains/cm² = 0.0166 chains/nm², consistent with the reported value. The arithmetic checks out.

Frame any apparent discrepancies with hedged language — you may be missing context the authors have. "The reported p = 0.03 appears inconsistent with t = 1.8 at n = 15; the authors may wish to verify" is better than "The p-value is wrong."

If the manuscript provides nothing quantitative to check, omit this section entirely rather than padding it.

### Recommendation

A single line stating the recommendation, followed by a short paragraph of justification that ties back to the major issues. The four standard categories:

- **Accept** — The paper is ready for publication as-is. Rare on first review; signal a high bar.
- **Minor Revision** — The work is sound and the claims are supported; some clarifications, added references, or figure improvements are needed.
- **Major Revision** — The core work is valuable but significant issues must be addressed. The reviewer should be willing to re-review.
- **Reject** — Either the work has fatal flaws that cannot be addressed without a fundamentally different study, or it is not appropriate for this journal.

Don't hedge the recommendation itself. Editors want a clear call even if the reasoning is nuanced.

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

## Tone principles

**Write as a colleague, not a judge.** The reviewer is a peer trying to help produce better science — not a gatekeeper rendering verdicts. Treat the authors as capable scientists who can respond to specific, well-reasoned feedback.

**Hedge appropriately.** The reviewer is working from a manuscript, not a complete record of the work. "This appears to be" and "the reviewer notes that" are honest acknowledgments of limited information, not weakness.

**Be concrete.** Every observation should be specific enough that the authors know exactly what section, figure, or sentence you're referring to. Page numbers, figure numbers, and equation numbers are your friends.

**Don't demand rewrites of what isn't yours to rewrite.** Suggest changes; don't dictate them. "The authors may wish to consider reorganizing Section 3.2" is better than "Section 3.2 must be reorganized."

**Acknowledge your own uncertainty.** If a claim is in a sub-field where you're less expert, say so. Editors weight reviews by expertise.

**Use field-appropriate technical language.** A review that reads as though written by a peer in the sub-field carries more weight than one that reads as generic. This means using the correct terms-of-art ("Manning condensation" rather than "some counterions sticking to the chain"), referencing the right classical results ("this is inconsistent with the Milner–Witten–Cates prediction" rather than "this doesn't match theory"), and matching the notation conventions of the field. When unsure, search recent literature for current usage.

## Tone examples

**Too harsh:**
> "The authors clearly have no understanding of polymer brush theory. Their interpretation is obviously wrong."

**Better:**
> "The interpretation on p. 8 appears to conflict with the predictions of Alexander–de Gennes theory for brushes in this regime. The authors should either reconcile their interpretation with existing theory or explicitly argue why it does not apply here."

**Too vague:**
> "The statistics need work."

**Better:**
> "The reported p-values in Table 2 are given without stated tests, degrees of freedom, or effect sizes. For the key comparisons in rows 3 and 7, please specify the test used and report the effect size so the practical significance can be assessed."

**Too deferential:**
> "I may be missing something, but perhaps the authors could possibly consider, if they think it appropriate, clarifying the methods slightly."

**Better:**
> "The Methods section does not specify the PVBTMAC concentration used in the QCM-D measurements (Figure 5). Please state the concentration explicitly."

**Too accusatory (integrity):**
> "Figure 3B has been fabricated."

**Better (in Confidential Comments to Editor):**
> "Figure 3B, panel (ii), shows a vertical discontinuity between lanes 4 and 5 that appears inconsistent with a single continuous gel image. The editor may wish to request the original source file from the authors."
