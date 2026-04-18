<!-- style_lint:ignore-file -->

# Writing constraints

These rules apply to every markdown, LaTeX, and prose artifact emitted by any skill in the research-spark stack. Enforced by `../_research-commons/scripts/style_lint.py`.

## Punctuation

**No em dashes.** Replace with commas, semicolons, colons, parentheses, or restructure the sentence.

- Incorrect: "The method is fast: and accurate."
- Correct: "The method is fast, and it is accurate."
- Correct: "The method is fast; it is also accurate."
- Correct: "The method is fast (and accurate)."

En dashes (–) are allowed only for numerical ranges: "pages 4–7", "Stages 4–5".

Regular hyphens (-) in compound modifiers are fine: "well-defined", "state-variable-free".

## Banned vocabulary

The following words must not appear in any DOE proposal context, and should be avoided generally in research writing:

- innovative
- state-of-the-art
- transformative
- sustainable
- novel
- groundbreaking
- cutting-edge
- revolutionary
- paradigm-shifting
- seamless

These words signal marketing rather than substance. If the work is genuinely new, quantify what is new. If the method is genuinely better, quantify the margin.

## Quantified over qualitative

Replace qualitative claims with quantified ones wherever possible.

- Weak: "significantly faster"
- Strong: "3.2× faster at N=10^6 particles"

- Weak: "improved accuracy"
- Strong: "reduced mean absolute error from 0.041 to 0.012"

- Weak: "wide applicability"
- Strong: "applicable across Péclet numbers from 10^-2 to 10^3"

If a number is not available, flag it as needing one rather than reaching for a qualitative substitute.

## Voice

Active voice is the default. Passive voice is allowed only when the actor is genuinely unknown or irrelevant.

- Weak: "The samples were prepared according to..."
- Strong: "We prepared samples according to..." (or "Samples: prepared according to..." in methods lists)

## Character-with-space limits

Many funding-agency forms impose character limits that include spaces. When a character count is mentioned in a proposal template (e.g., "2000 characters including spaces"), honor it exactly. Use `wc -m` or Python `len(text)` to verify, not word-count heuristics.

## Acronyms and defined terms

Define every acronym on first use. After first use, use the acronym. Do not re-define it later in the same document. For acronyms that are universal in the subfield (XPCS, SAXS, DMA), a single definition in the methods section suffices; do not redefine in every figure caption.
