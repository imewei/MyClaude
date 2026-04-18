# Gap matrix

A two-dimensional table that maps the field. Rows are phenomena or sub-questions. Columns are methods or approaches. Each cell describes the current state of knowledge, with a citation.

## Format

```markdown
| Phenomenon / Sub-question       | Method A       | Method B       | Method C       |
|---------------------------------|----------------|----------------|----------------|
| [sub-question 1]                | [state + cite] | [state + cite] | [gap]          |
| [sub-question 2]                | [state + cite] | [gap]          | [state + cite] |
| [sub-question 3]                | [gap]          | [gap]          | [state + cite] |
```

Cell contents:
- State-of-knowledge cells: one sentence + author-year citation
- Gap cells: write "GAP" in caps, followed by why the cell is empty

## Worked example (condensed)

| Sub-question                       | Bulk rheology (G', G'') | Scattering (SAXS) | Microscopy |
|------------------------------------|-------------------------|-------------------|------------|
| Onset of flocculation, time-resolved | Crossover detection, tens of seconds late (Smith 2019) | Structure-factor growth, correlated with onset (Chen 2021) | Direct visualization, slow acquisition (Lee 2018) |
| Spectral signature of flocculation onset | GAP: ensemble averaging obscures spectral structure | GAP: static SAXS has no dynamic signature | GAP: no operator-valued extraction in standard pipelines |
| Real-time intervention on flocculation | Shear-history protocols (Park 2022), blind to spectral gap | Not applicable (offline) | Not applicable (slow) |

## How to use it

1. After building the matrix, scan down each column: where are the gap cells? A column with multiple gaps indicates a method that is underused on this problem.
2. Scan across each row: where are the gap cells? A row with multiple gaps indicates a sub-question that no existing method resolves.
3. The intersection of a gap row and a gap column is a candidate research question. Verify it is real (not just underreported), tractable (some route to a measurement exists), and impact-bearing (resolving it changes something downstream).

## What not to do

- Do not invent gaps by omitting relevant papers. If a method has been applied to a sub-question, cite it, even if the application was incomplete.
- Do not write "GAP" in a cell to describe a method that is routine but unreported in the literature you have seen. Search harder first.
- Do not compress multiple papers into a single cell by listing them separated by commas. If two papers disagree about the state of knowledge, they belong in separate rows or deserve their own discussion outside the matrix.
