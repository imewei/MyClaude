---
name: research-paper-implementation
description: Translate research papers into production-ready implementations through systematic analysis, architecture extraction, experiment reproduction, and practical adaptation of state-of-the-art deep learning methods. Use this skill when implementing novel architectures from recent papers (transformers, diffusion models, vision transformers, BERT variants, GPT models). Apply when extracting mathematical formulations, algorithmic pseudocode, or architecture specifications from academic papers. Use when reproducing experimental results, baselines, or ablation studies from published research. Apply when adapting research ideas to specific applications or datasets beyond the original paper scope. Use when understanding theoretical contributions, novel training techniques, or optimization methods from papers. Apply when locating implementation details often hidden in appendices or supplementary materials. Use when finding hyperparameters, learning rate schedules, or data preprocessing steps not explicitly stated. Use when working with arXiv preprints, conference papers (NeurIPS, ICML, ICLR, CVPR), or journal articles requiring implementation. Apply when analyzing reference implementations on GitHub or comparing multiple papers' approaches. Use when translating novel loss functions, attention mechanisms, or architectural components into code.
---

# Research Paper Implementation

Systematic approach to translating research papers into production-ready implementations.

## When to use this skill

- When implementing state-of-the-art architectures from recent papers (Vision Transformers, Stable Diffusion, CLIP, GPT variants, BERT models)
- When translating mathematical formulations or algorithmic pseudocode from papers into executable PyTorch/JAX/TensorFlow code
- When extracting precise architecture specifications (layer dimensions, attention heads, hidden sizes) from paper descriptions
- When reproducing experimental results, baselines, or ablation studies to validate implementations
- When understanding novel training techniques (curriculum learning, contrastive learning, self-supervised methods)
- When locating critical implementation details often relegated to appendices or supplementary materials
- When finding hyperparameters not explicitly stated in main text (learning rates, weight decay, warmup schedules)
- When adapting research ideas to new domains or datasets beyond original paper scope
- When comparing multiple papers' approaches to the same problem (e.g., different transformer variants)
- When implementing novel loss functions, regularization techniques, or optimization algorithms from papers
- When translating theoretical contributions into practical code (attention mechanisms, normalization schemes)
- When working with arXiv preprints or conference papers (NeurIPS, ICML, ICLR, CVPR, ECCV) requiring implementation
- When analyzing and learning from reference implementations on GitHub or official repositories
- When understanding architectural innovations (efficient attention, sparse transformers, neural architecture search results)
- When extracting data preprocessing, augmentation, or evaluation protocols from research
- When identifying which components are essential vs optional for reproducing core results
- When debugging why reproduced results don't match paper claims (common implementation pitfalls)
- When implementing papers for transfer learning, fine-tuning, or domain adaptation scenarios

## 6-Step Paper Translation Framework

### Step 1: Core Contribution Extraction
- Identify the novel idea (architecture, training method, theory)
- Understand problem being solved
- Recognize why it's better than prior work
- Extract theoretical motivations

### Step 2: Mathematical Foundation Analysis
- Study mathematical framework and notation
- Identify assumptions and constraints
- Review proofs (if theoretical paper)
- Understand theoretical guarantees

### Step 3: Architecture/Algorithm Detailed Analysis
- Extract precise architecture specifications
- Identify critical design choices
- List all hyperparameters
- Note implementation details from appendix

### Step 4: Experimental Validation Understanding
- Review datasets and benchmarks used
- Analyze baselines and comparisons
- Study ablation studies (what components matter)
- Note computational requirements

### Step 5: Implementation Guidance
- Separate essential from optional components
- Identify common pitfalls from paper
- Find good starting hyperparameters
- Locate reference implementations if available

### Step 6: Practical Adaptation
- Adapt to target domain/dataset
- Consider computational constraints
- Understand when method works well vs poorly
- Make necessary trade-offs

## Workflow Example: Implementing "Attention is All You Need"

**Step 1 - Core Contribution:**
- Self-attention mechanism replaces recurrence
- Solves parallelization problem in RNNs
- Handles long-range dependencies effectively

**Step 2 - Mathematics:**
- Attention formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Scaling by √d_k prevents dot products from growing large
- Multi-head: parallel attention in different subspaces

**Step 3 - Architecture:**
- Encoder: 6 layers of (Multi-head attention + FFN)
- Decoder: 6 layers of (Masked attention + Cross attention + FFN)
- d_model=512, h=8 heads, d_ff=2048
- Positional encoding: sin/cos functions

**Step 4 - Experiments:**
- WMT English-German translation
- 4.5M sentence pairs
- Beam search with beam size 4

**Step 5 - Essential Components:**
- Must-have: Multi-head self-attention, positional encoding, residual connections, layer norm
- Nice-to-have: Specific hyperparameter values (can be tuned)

**Step 6 - Adaptation:**
- For classification: Add output head, remove decoder
- For small data: Reduce model size, add regularization
- For efficiency: Use efficient attention variants

## Common Paper Types

### Architecture Papers
- Focus: Novel network design
- Key sections: Architecture diagram, ablation studies
- Implementation priority: Get architecture exactly right

### Training Method Papers
- Focus: New optimization/training technique
- Key sections: Algorithm pseudocode, hyperparameters
- Implementation priority: Replicate training procedure

### Theoretical Papers
- Focus: Understanding or guarantees
- Key sections: Theorems, proofs, experiments validating theory
- Implementation priority: Test theoretical predictions

### Application Papers
- Focus: Solving specific problem
- Key sections: Data preprocessing, evaluation metrics
- Implementation priority: Domain-specific adaptations

## Quick Reference: Common Pitfalls

- **Missing details in main text** → Check appendix and supplementary materials
- **Unclear notation** → Look for notation table, compare with reference code
- **Hyperparameters not specified** → Search GitHub issues, author responses
- **Results don't reproduce** → Check for subtle implementation details, learning rate schedules
- **Theory vs practice gap** → Papers often simplify; real implementations need engineering

---

*Systematically translate research papers into working implementations with proper understanding of contributions, mathematics, and practical considerations.*
