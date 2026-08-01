---
name: research-and-domains
description: Meta-orchestrator for specialized scientific domains, scientific Python systems, and AI-research methods not covered by the main JAX, Julia, ML, simulation, or research methodology hubs. Routes to self-improving-AI research surveys, DSPy scientific prompt programs, RLAIF training, Python packaging/Rust/type/concurrency/testing patterns, and domain skills such as quantum computing, bioinformatics, computer vision, reinforcement learning, and symbolic math. Use when the user asks about AI-research techniques or scientific Python/domain work; for general study design, literature synthesis, or paper methodology, use research-suite research-practice.
---

# Research and Domains

Orchestrator for scientific Python systems, AI-research methods, and specialized scientific domains. Routes problems to the appropriate specialized skill.

> **General research methodology has moved.** Study design, paper implementation, quality assessment, scientific writing (IMRaD), and evidence synthesis (PRISMA, meta-analysis, GRADE) now live in the `research-practice` hub in `research-suite`. This hub retains AI-research methods (autonomous-agent research, scientific DSPy programs, RLAIF training) and Python/domain specializations that pair naturally with scientific computing.

## Core Skills

| Category | Skill | Purpose |
|----------|-------|---------|
| AI Research | [Self-Improving AI](../self-improving-ai/SKILL.md) | Four families: inference-time scaling, self-refinement, autonomous research loops, evolutionary search |
| AI Research | [DSPy Basics](../dspy-basics/SKILL.md) | Programmatic prompts: Signatures, Modules, MIPROv2, BootstrapFewShot, ReAct |
| AI Research | [RLAIF Training](../rlaif-training/SKILL.md) | DPO / KTO / PPO with `trl`; Constitutional AI; AI-as-judge preference generation |
| Python | [Python Development](../python-development/SKILL.md) | Idiomatic Python, software engineering |
| Python | [Python Packaging Advanced](../python-packaging-advanced/SKILL.md) | PyPI, build backends |
| Python | [Rust Extensions](../rust-extensions/SKILL.md) | PyO3/maturin high-perf extensions |
| Python | [Type-Driven Design](../type-driven-design/SKILL.md) | Type safety, protocols |
| Python | [Modern Concurrency](../modern-concurrency/SKILL.md) | asyncio, threading, multiprocessing |
| Python | [Robust Testing](../robust-testing/SKILL.md) | Property-based testing (Hypothesis), mutation testing, advanced pytest fixtures for scientific code |
| Domain | [Quantum Computing](../quantum-computing/SKILL.md) | Qiskit, PennyLane, VQE/QAOA |
| Domain | [Bioinformatics](../bioinformatics/SKILL.md) | Genomics, proteomics, BioPython |
| Domain | [Computer Vision](../computer-vision/SKILL.md) | Image processing, detection |
| Domain | [Reinforcement Learning](../reinforcement-learning/SKILL.md) | RL algorithms, policy optimization |
| Domain | [Symbolic Math](../symbolic-math/SKILL.md) | SymPy, CAS, algebraic solvers |

## Routing Decision Tree

```
What is the task category?
|
+-- General research methodology (study design, paper write-up, lit review, meta-analysis)?
|   --> (out of hub) research-suite:research-practice
|
+-- Self-improving AI taxonomy / autonomous research loops / four-families overview?
|   --> science-suite:self-improving-ai
|
+-- DSPy scientific prompt programs / MIPROv2 / BootstrapFewShot / ReAct tools?
|   --> science-suite:dspy-basics
|
+-- RLAIF / Constitutional AI / DPO / KTO / PPO with `trl`?
|   --> science-suite:rlaif-training
|
+-- Python systems / packaging / performance?
|   (Canonical Python-tooling home for this marketplace as of the
|    dev-suite trim -- covers general uv/packaging/async/profiling
|    as well as scientific-specific concerns)
|   --> science-suite:python-development / science-suite:python-packaging-advanced
|   --> science-suite:rust-extensions / science-suite:type-driven-design / science-suite:modern-concurrency
|   --> science-suite:robust-testing (Hypothesis property-based tests, mutation testing)
|
+-- Specialized scientific domain?
    +-- Quantum circuits / VQE / QAOA? --> science-suite:quantum-computing
    +-- Genomics / proteomics?         --> science-suite:bioinformatics
    +-- Images / detection?            --> science-suite:computer-vision
    +-- RL agents / environments?      --> science-suite:reinforcement-learning
    +-- Symbolic / algebraic math?     --> science-suite:symbolic-math
|
+-- None of the above / concern is ambiguous?
    --> Delegate to the most domain-relevant expert agent (e.g., `jax-pro` for
        Python AI research, `julia-pro` for Julia domains, `simulation-expert`
        for quantum/bioinformatics) for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Checklist

- [ ] Use routing tree to identify task category before selecting a sub-skill
- [ ] For general research methodology questions, delegate to `research-suite:research-practice`
- [ ] Validate Python packaging locally with `pip install -e .` before publishing
- [ ] Test Rust extensions with `cargo test` before building Python wheels
- [ ] Check quantum circuit depth and gate count against target hardware limits
- [ ] Confirm bioinformatics pipelines handle both DNA and protein alphabets
- [ ] Use `mypy --strict` when applying type-driven design patterns
- [ ] Validate RL agents against known baselines before novel environment testing
