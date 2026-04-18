# Citation style

Default conventions for the research-spark stack. These apply to draft manuscripts, proposal narratives, and internal artifacts unless a specific venue requires otherwise. Override by venue: if submitting to PRL, use their exact style; if to a Nature journal, use theirs; these defaults approximate APS style because most soft-matter work lands there.

## In-text citation format

Author-year, parenthetical by default. Use numbered citations only when the venue explicitly requires them.

- One author: (Koga, 2012) or Koga (2012) when the name is a syntactic subject
- Two authors: (Koga and Baker, 2012)
- Three or more: (Koga et al., 2012)
- Multiple papers: (Koga et al., 2012; Chen et al., 2021; Park, 2022), ordered chronologically oldest to newest

Same year, same first author: suffix with a, b, c (Chen et al., 2021a; Chen et al., 2021b).

## Journal name abbreviations

Use ISO 4 abbreviations when the venue accepts them. Common abbreviations in our subfield:

| Full journal | ISO 4 abbreviation |
|---|---|
| Physical Review Letters | Phys. Rev. Lett. |
| Physical Review E | Phys. Rev. E |
| Physical Review Research | Phys. Rev. Res. |
| Journal of Rheology | J. Rheol. |
| Soft Matter | Soft Matter (no abbreviation) |
| Macromolecules | Macromolecules (no abbreviation) |
| Nature | Nature |
| Nature Physics | Nat. Phys. |
| Nature Materials | Nat. Mater. |
| Proceedings of the National Academy of Sciences | Proc. Natl. Acad. Sci. U.S.A. |
| Journal of Chemical Physics | J. Chem. Phys. |
| Journal of the American Chemical Society | J. Am. Chem. Soc. |
| Advanced Materials | Adv. Mater. |
| ACS Nano | ACS Nano |
| Synchrotron Radiation News | Synchrotron Radiat. News |
| Journal of Synchrotron Radiation | J. Synchrotron Radiat. |

When unsure, check https://woodward.library.ubc.ca/research-help/journal-abbreviations/ or the venue's own list.

## Author name format in the bibliography

APS PRL style:

```
N. Koga, R. Tatsumi-Koga, G. Liu, R. Xiao, T. Acton, G. Montelione, and D. Baker, Nature 491, 222 (2012).
```

Conventions:
- Initials with periods and spaces: `N. Koga`
- Surname follows initials
- "and" separates last author
- Journal name after authors, then volume (italic usually, handled by `.bst`), page, year in parentheses
- DOI optional in PRL; preferred in preprints and supplementary material

For Nature-style journals, the format is similar but often uses full first names and different punctuation. Match the specific journal's template when submitting.

## DOIs

Include a DOI for every reference where one exists. Store as plain text:

```
doi: 10.1038/nature11600
```

In BibTeX:

```bibtex
doi = {10.1038/nature11600}
```

Never hide DOIs behind ambiguous URL shorteners in research artifacts.

## arXiv preprints

Cite as:

```
A. Author and B. Other, arXiv:2301.12345 [cond-mat.soft] (2023).
```

Include the category tag `[cond-mat.soft]`, `[physics.bio-ph]`, `[stat.ML]` as appropriate; it helps readers find related work.

If a preprint has been published, cite the published version and note the preprint in a footnote or supplementary material only when relevant to priority claims.

## Self-citation discipline

A citation to your own prior work is a citation like any other. Cite when it supports the current claim; do not pad. Three or more self-citations in a five-citation paragraph is a reader flag.

## BibTeX hygiene

- Every entry has a citekey of the form `FirstAuthorSurnameYearKeyword` (e.g., `Koga2012Principles`, `Chen2021Flocculation`)
- No duplicates (use `../_research-commons/scripts/... ` or `landscape-scanner/scripts/dedupe_refs.py`)
- Every entry has title, author, year, journal (or booktitle), and DOI where available
- Use `{Curly Braces}` around proper nouns and chemical formulas that should not be case-corrected: `Ca{C}l_2`, `{B}rownian`

## Venue override

When a target venue's style differs, the venue wins. Note the override at the top of the manuscript's main LaTeX file so downstream tooling (like the style linter) can be told to skip citation-related checks:

```latex
% style: Nature main-text; see journal guidelines
```

## What not to do

- Mix author-year and numbered citations within a single manuscript
- Cite review articles as the primary source of a specific result when the original paper is accessible
- Cite conference abstracts without page numbers as if they were peer-reviewed papers
- Use "private communication" citations except when there is genuinely no other reference and the communication is substantive
- Pad the bibliography with uncited entries to inflate apparent depth
