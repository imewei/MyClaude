# Steelman note template

One note per paper in the bibliography. Every field must be non-empty. If a field cannot be filled, the paper is not yet fully surveyed and should be marked as such rather than left with placeholder text.

## Format

```
### [Author(s), Year]: Short paper title

**DOI:** [10.xxxx/yyyy]  
**Layer:** [foundational | recent | adjacent]  
**Read depth:** [abstract | introduction + conclusion | full read]

**Strongest claim** (authors' own framing):
[1-2 sentences. Phrase the claim as the authors would, at its most ambitious.
Not "they claim that X", but X stated as if you were presenting their work.]

**Conditions under which it breaks:**
[Specific regime, system, parameter range, or assumption where the claim fails 
or becomes inapplicable. MUST be specific, not generic. 
"Fails at high Péclet" is specific. "Not exact" is not specific.]

**Residual uncertainty:**
[What the paper leaves unresolved that bears on our spark. 
1-2 sentences. If nothing bears on our spark, say so and explain why the paper 
is still worth including.]

**Relevance to our spark:**
[1 sentence linking this paper to a specific sub-question in our work.]
```

## Worked example

```
### Koga et al., 2012: Principles for designing ideal protein structures

**DOI:** 10.1038/nature11600  
**Layer:** foundational  
**Read depth:** full read

**Strongest claim:**
Ideal protein folds obey simple rules linking secondary-structure length, 
loop geometry, and chirality; by encoding these rules, de novo design can 
produce folds that refold correctly in solution. The five designed proteins 
(Di-I through Di-V) validate the framework and fold as predicted by NMR.

**Conditions under which it breaks:**
The rules were derived for small (<100 residue) all-alpha or alpha/beta folds 
with well-defined secondary structure. Larger folds, folds with significant 
disorder, and folds with non-canonical secondary structure (e.g., beta-barrels, 
PPII-rich) are outside the training set. The rules predict folding success 
but not functional-site geometry.

**Residual uncertainty:**
The framework is about fold topology, not function. How well the same rules 
transfer to de novo design of catalytic or binding proteins is addressed in 
later Baker-lab work but not here.

**Relevance to our spark:**
Provides the canonical reference for what "ideal" means in de novo design, 
which sets the baseline against which our extension must show added value.
```

## Common failure modes

- Writing the strongest claim as a neutral summary rather than as the authors would phrase it. The steelman requires presenting the work at its best, so that the eventual critique is earned.
- Leaving "conditions under which it breaks" generic. "Uses approximations" is not a break condition. "Breaks when the interparticle distance falls below 2×radius of gyration" is a break condition.
- Skipping "residual uncertainty" because "the paper is solid." Every paper has residual uncertainty; if none is visible, the paper has not been read carefully enough.
