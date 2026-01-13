---
name: research-paper-implementation
version: "1.0.7"
maturity: "5-Expert"
specialization: Paper-to-Code Translation
description: Translate research papers into production implementations through systematic analysis and architecture extraction. Use when implementing novel architectures, reproducing experiments, or adapting state-of-the-art methods.
---

# Research Paper Implementation

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

## Checklist

- [ ] Core contribution understood
- [ ] Key equations identified
- [ ] Architecture specs extracted
- [ ] Hyperparameters documented
- [ ] Essential vs optional separated
- [ ] Reference implementation reviewed
- [ ] Incremental testing planned

---

**Version**: 1.0.5
