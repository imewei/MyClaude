# Integrity Checks

Concrete signals to look for when assessing ethical integrity in Lens 6. This file is a checklist of red flags, not an accusation manual. The reviewer's role is to *surface* concerns to the editor, never to pronounce guilt.

## Plagiarism and self-plagiarism

Most journals run similarity software (iThenticate, Turnitin) on submissions, but these tools miss paraphrased material and can be defeated by light rewording. A careful reader can sometimes catch what software misses.

**Text-level warning signs:**
- Abrupt stylistic shifts mid-section — e.g., a methods paragraph in plain prose suddenly switches to dense, ornate academic phrasing, then switches back
- Passages in the target paper that closely track the structure of a specific other paper the reviewer happens to know (same paragraph order, same examples, same transitional phrases)
- Unusually formal or archaic phrasing in a paper otherwise written in modern style
- Quoted definitions or long descriptive passages without citation
- Background sections that appear to summarize a specific review article without attribution

**Self-plagiarism warning signs:**
- Methods sections reused from the authors' prior papers nearly verbatim without citation
- Results that appear in the manuscript and also in a recently published paper by the same group, without clear differentiation
- Figures that appear similar or identical to figures in the authors' prior work

How to flag: name the passage and suspected source in Confidential Comments to Editor; suggest similarity-software check. Never claim plagiarism in Comments to Authors.

## Data and image integrity

Image-based fabrication and duplication are among the most common forms of research misconduct caught by reviewers.

**In gel/blot images:**
- Unusual sharp boundaries or discontinuities between lanes (possible splicing)
- Regions of identical background texture or noise patterns (suggests copy-paste)
- Bands that appear unnaturally uniform in shape, brightness, or spacing
- Contrast or brightness that appears to have been adjusted nonlinearly (e.g., only around bands of interest)
- Loading controls that do not match the intensity variation of the experimental bands

**In microscopy and biological images:**
- Repeated features (cells, organelles) at identical orientation across supposedly different fields
- Background patterns that repeat across "independent" samples
- Images that appear in this paper and also in a prior paper by the same group with different labels

**In quantitative data:**
- Standard deviations or error bars that are implausibly small for the stated sample size
- N-values that fluctuate unexplainedly between experiments
- Data points that cluster too tightly on theoretical curves (can indicate model-fitting rather than measurement)
- Last-digit distributions that violate Benford's law in numerical tables (possible fabrication indicator, but only suggestive)

How to flag: describe what you see and where (figure/panel); ask editor to request raw data or source files. Phrase as "features worth verifying."

## Selective reporting and p-hacking

These are about what was *not* reported.

- Outcome measures in the results that do not appear in the stated hypotheses or methods
- Subgroup analyses that appear only when significant
- Fine-grained statistical thresholds that produce "p = 0.048" in a pattern across multiple tests
- No correction for multiple comparisons when many tests were performed
- No pre-registration referenced for work in fields where pre-registration is standard (clinical trials, some social science)
- Null or negative results mentioned in the introduction but not revisited

## Dual submission and duplicate publication

- Substantive overlap with a recent publication by the same author group — not just topical overlap, but shared data, shared figures, or shared conclusions
- A "new" study whose methods section references experiments that are clearly the same as those in a recent paper
- Results posted in a preprint that differ substantively from the submitted manuscript in ways that suggest the preprint was intended as a separate publication

A quick search of the senior author's recent publications (Google Scholar, ORCID) can reveal overlap.

## Ethics approvals and disclosures

These are checklist items, but commonly forgotten. Absence is not proof of wrongdoing — it may just be a reporting oversight — but it should be flagged.

- **Human subjects research** → IRB or ethics committee approval, with institution name and approval number; informed consent statement; registration (for clinical trials) on a public registry
- **Animal research** → IACUC or equivalent approval; adherence to ARRIVE guidelines or equivalent
- **Identifiable human data** → consent for publication; data protection compliance
- **Secondary data** → licensing and use permissions
- **Conflict of interest** → author disclosure statement
- **Funding** → funding sources acknowledged
- **Data availability** → statement of where raw data can be accessed (many journals now require this)
- **Code availability** → for computational work, code repository link
- **Author contributions** → CRediT-style statement where required

## Authorship concerns

- Authorship order or contributions that seem implausible given the described work (e.g., a first author listed on a project that clearly required expertise outside their field, with no corresponding senior contributor)
- Gift authorship or ghost authorship signals (uneven distribution of stated contributions)
- Senior authors whose stated contribution is limited to "supervision" on a highly technical paper — this can be normal but is sometimes a tell

These are hard to judge from the manuscript alone; flag only if something is notably off.
