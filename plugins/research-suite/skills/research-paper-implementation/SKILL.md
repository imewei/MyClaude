---
name: research-paper-implementation
description: Translate published papers into working code by extracting architecture specs, hyperparameters, and training procedures from methods + appendix + supplementary material. This skill should be used when the user asks to "reproduce this paper", "implement the architecture from X", "extract hyperparameters from this paper", "map the notation to code", "recover missing details from the appendix", "rebuild the baseline from [paper]", "port this algorithm to PyTorch / JAX / Julia", or provides a paper reference and wants a runnable implementation. For translating the user's own Stage 4-5 formalism into a JAX prototype inside an active research-spark project, use `numerical-prototype` instead.
---

# Research Paper Implementation

## Expert Agent

For translating research papers into production implementations, delegate to:

- **`research-expert`**: Research methodology, paper analysis, and systematic implementation.
  - *Location*: `plugins/research-suite/agents/research-expert.md`

Systematic approach to translating papers into working code.

---

## 6-Step Translation Framework

| Step | Focus | Key Questions |
|------|-------|---------------|
| 1. Core Contribution | Novel idea | What's new? Why better? |
| 2. Mathematics | Formulas, notation | Key equations? Assumptions? |
| 3. Architecture | Specifications | Layers? Dimensions? Hyperparams? |
| 4. Experiments | Validation | Datasets? Baselines? Ablations? |
| 5. Implementation | Code guidance | Essential vs optional? Pitfalls? |
| 6. Adaptation | Practical use | How to adapt? Trade-offs? |

---

## Paper Types

| Type | Focus | Priority |
|------|-------|----------|
| Architecture | Novel network design | Get structure exactly right |
| Training Method | Optimization technique | Replicate training procedure |
| Theoretical | Understanding/guarantees | Test theoretical predictions |
| Application | Solving specific problem | Domain-specific adaptations |

---

## Example: Transformer Implementation

**Step 1 - Core**: Self-attention replaces recurrence for parallelization

**Step 2 - Math**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

**Step 3 - Architecture**:
- Encoder: 6 layers × (Multi-head attn + FFN)
- d_model=512, h=8 heads, d_ff=2048
- Positional encoding: sin/cos

**Step 5 - Essential**:
- Multi-head self-attention
- Positional encoding
- Residual connections + LayerNorm

---

## Information Sources

| Source | Find |
|--------|------|
| Main text | Core ideas, high-level architecture |
| Appendix | Implementation details, hyperparameters |
| Supplementary | Extended results, code snippets |
| GitHub | Reference implementations |
| Author responses | Clarifications on issues |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Missing details in main text | Check appendix and supplementary |
| Unclear notation | Find notation table, compare to code |
| Hyperparameters not specified | Search GitHub issues, author responses |
| Results don't reproduce | Check subtle details, LR schedules |
| Theory vs practice gap | Expect engineering beyond paper |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Read appendix first | Hidden details there |
| Find reference code | Compare to official implementation |
| Start with ablations | Identify essential components |
| Log everything | Track deviations from paper |
| Test incrementally | Validate each component |

---

## Related skills

- `numerical-prototype` (research-spark Stage 6) — when implementing the user's *own* Stage 4-5 formalism rather than someone else's paper. Enforces three validation passes (analytic-limit recovery, synthetic benchmark, convergence study) as canonical artifacts.
- `research-quality-assessment` — before investing heavy implementation effort, score whether the paper's reported results are likely reproducible (red-flag check, sample-size sanity, code-availability audit).
- `scientific-review` — for peer-reviewing a paper rather than re-implementing it.

## Checklist

- [ ] Core contribution understood
- [ ] Key equations identified
- [ ] Architecture specs extracted
- [ ] Hyperparameters documented
- [ ] Essential vs optional separated
- [ ] Reference implementation reviewed
- [ ] Incremental testing planned
