---
name: research-and-domains
description: Meta-orchestrator for research methodology and specialized domains. Routes to research methods, paper implementation, quality assessment, scientific writing, evidence synthesis, Python systems, and domain-specific skills (quantum, bio, CV, RL, symbolic math). Use when conducting systematic research, implementing papers, assessing research quality, writing scientific reports, or working with specialized domains (quantum, bio, CV, RL).
---

# Research and Domains

Orchestrator for research methodology and specialized scientific domains. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`research-expert`**: Specialist for research methodology, literature synthesis, and domain-specific scientific computing.
  - *Location*: `plugins/science-suite/agents/research-expert.md`
  - *Capabilities*: Research design, paper implementation, quality assessment, scientific communication, and domain expertise.

## Core Skills

| Category | Skill | Purpose |
|----------|-------|---------|
| Research | [Research Methodology](../research-methodology/SKILL.md) | Study design, literature review |
| Research | [Research Paper Implementation](../research-paper-implementation/SKILL.md) | Reproduce academic paper results |
| Research | [Research Quality Assessment](../research-quality-assessment/SKILL.md) | Rigor, reproducibility, statistics |
| Research | [Scientific Communication](../scientific-communication/SKILL.md) | Papers, reports, presentations |
| Research | [Evidence Synthesis](../evidence-synthesis/SKILL.md) | Meta-analysis, systematic reviews |
| Python | [Python Development](../python-development/SKILL.md) | Idiomatic Python, software engineering |
| Python | [Python Packaging Advanced](../python-packaging-advanced/SKILL.md) | PyPI, build backends |
| Python | [Rust Extensions](../rust-extensions/SKILL.md) | PyO3/maturin high-perf extensions |
| Python | [Type-Driven Design](../type-driven-design/SKILL.md) | Type safety, protocols |
| Python | [Modern Concurrency](../modern-concurrency/SKILL.md) | asyncio, threading, multiprocessing |
| Domain | [Quantum Computing](../quantum-computing/SKILL.md) | Qiskit, PennyLane, VQE/QAOA |
| Domain | [Bioinformatics](../bioinformatics/SKILL.md) | Genomics, proteomics, BioPython |
| Domain | [Computer Vision](../computer-vision/SKILL.md) | Image processing, detection |
| Domain | [Reinforcement Learning](../reinforcement-learning/SKILL.md) | RL algorithms, policy optimization |
| Domain | [Symbolic Math](../symbolic-math/SKILL.md) | SymPy, CAS, algebraic solvers |

## Routing Decision Tree

```
What is the task category?
|
+-- Research process (design, review, write, evaluate)?
|   --> research-methodology / research-paper-implementation
|   --> research-quality-assessment / scientific-communication
|   --> evidence-synthesis
|
+-- Python systems / packaging / performance?
|   (These are co-located here for scientific Python workflows;
|    for general Python toolchain, see dev-suite python-toolchain hub)
|   --> python-development / python-packaging-advanced
|   --> rust-extensions / type-driven-design / modern-concurrency
|
+-- Specialized scientific domain?
    +-- Quantum circuits / VQE / QAOA? --> quantum-computing
    +-- Genomics / proteomics?         --> bioinformatics
    +-- Images / detection?            --> computer-vision
    +-- RL agents / environments?      --> reinforcement-learning
    +-- Symbolic / algebraic math?     --> symbolic-math
```

## Checklist

- [ ] Use routing tree to identify task category before selecting a sub-skill
- [ ] For paper implementation: read methods section fully before coding
- [ ] Validate Python packaging locally with `pip install -e .` before publishing
- [ ] Test Rust extensions with `cargo test` before building Python wheels
- [ ] Check quantum circuit depth and gate count against target hardware limits
- [ ] Confirm bioinformatics pipelines handle both DNA and protein alphabets
- [ ] Use `mypy --strict` when applying type-driven design patterns
- [ ] Validate RL agents against known baselines before novel environment testing
