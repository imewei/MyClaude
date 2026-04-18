# Three-layer search strategy

A surface search returns recent papers. This protocol returns the papers that matter.

## Layer 1: Foundational

The papers the subfield cites without re-reading. Usually 10-40 years old.

**How to find them:**
- Look at recent review articles in the subfield. Note which citations are repeated across reviews; those are the foundational ones.
- Ask the user directly: "What are the two or three papers that define this subfield? The ones everyone in your group cites without looking at again."
- Check the first 10 citations in a recent (last 2 years) strong paper on the topic. The foundational papers tend to cluster here.

**How many:** 3-6 papers. Err on the side of fewer; foundational papers are usually thoroughly understood by the user already.

**Why this layer:** Foundational papers set the vocabulary and the baseline. Missing them makes the synthesis look uninformed even if the recent coverage is strong.

## Layer 2: Recent (trailing 3 years)

Papers that have moved the state of the art on the specific phenomenon.

**How to find them:**
- Search databases (arXiv, Web of Science, Semantic Scholar) with queries built from the Stage 1 artifact's named mechanisms and phenomena.
- Follow forward citations from Layer 1 papers: what recent work cites the foundational papers AND addresses the same phenomenon.
- Check the preprints on arXiv with last-month and last-quarter filters. If the user's spark is active in the field, there is often something fresh.

**Search query construction:** Combine one system term with one phenomenon term. Adding more terms filters too aggressively. Example: "battery slurry flocculation" beats "battery slurry flocculation XPCS early warning spectral".

**How many:** 3-5 papers. These are the hardest to choose; be selective.

**Why this layer:** The recent layer is where the user's contribution will actually land. The contribution must be positioned against these papers explicitly.

## Layer 3: Adjacent

Fields that have encountered the same phenomenon from a different angle.

**How to find them:**
- Identify what the core phenomenon is abstractly. "Early warning signals of a transition" is studied in ecology, finance, condensed matter, and climate. "Bond-exchange kinetics" is studied in organic chemistry, polymer physics, and metal-organic frameworks.
- Search the abstracted term in a field not your own. Example: for spectral-gap early warning in rheology, search "critical slowing down" in ecology.
- Ask the user: "What fields outside yours have you seen touch this phenomenon, even tangentially?"

**How many:** 2-4 papers. Adjacent papers need not be closely related; they exist to anchor the abstraction.

**Why this layer:** Adjacent-field results often show that a phenomenon has a general structure beyond the specific system. This changes how the claim is framed in Stage 3, and often how the method is built in Stage 4.

## Query hygiene

- Do not search with compound AND queries of 5+ terms. Too narrow. Start broad, skim, narrow.
- Do not trust keyword matches alone. Read the abstract. If the abstract is ambiguous, read the introduction.
- Do not accept papers into the bibliography on the strength of the title alone. Title promises often overstate content.

## When the literature is genuinely thin

If the user's spark is in a niche where Layer 2 returns only one or two papers, document that fact and proceed. The depth gate (8 steelmanned papers) can be overridden, with the override logged. A thin Layer 2 is also informative: either the field is genuinely open, or the spark is positioned poorly relative to where the activity actually is. Both are worth knowing before Stage 3.
