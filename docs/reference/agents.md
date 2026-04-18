# Agent Reference

**25 Agents** across 4 suites | **Version:** 3.4.0

Agents are specialized AI personas with defined model tiers, tool access, and domain expertise. Each agent runs at a specific model tier: **opus** (deep reasoning), **sonnet** (standard tasks), or **haiku** (fast/simple).

---

## Agent Core Suite (`agent-core`) — 3 Agents

Core orchestration, reasoning, and context engineering.

| Agent | Model | Description |
|-------|-------|-------------|
| `orchestrator` | opus | Multi-agent orchestrator for workflow coordination, agent team assembly, and task allocation |
| `reasoning-engine` | opus | Advanced reasoning, prompt design, and cognitive tasks. Masters Chain-of-Thought and structured frameworks |
| `context-specialist` | sonnet | Context engineering specialist for dynamic context management, vector databases, and memory systems |

---

## Dev Suite (`dev-suite`) — 9 Agents

Full-stack engineering, infrastructure, CI/CD, quality assurance, and debugging.

| Agent | Model | Description |
|-------|-------|-------------|
| `software-architect` | opus | Scalable backend systems, microservices, and high-performance APIs (REST/GraphQL/gRPC) |
| `debugger-pro` | opus | AI-assisted debugging, log correlation, and complex root cause analysis across distributed systems |
| `app-developer` | sonnet | Web, iOS, and Android applications. Masters React, Next.js, Flutter, and React Native |
| `automation-engineer` | sonnet | Software delivery pipelines and Git collaboration. Masters GitHub Actions and GitLab CI |
| `devops-architect` | sonnet | Multi-cloud architecture (AWS/Azure/GCP), Kubernetes, and Infrastructure as Code (Terraform/Pulumi) |
| `quality-specialist` | sonnet | Code reviews, security audits, and test automation strategies |
| `sre-expert` | sonnet | System reliability, observability (monitoring, logging, tracing), and incident response |
| `systems-engineer` | sonnet | Low-level systems programming (C, C++, Rust, Go) and production-grade CLI tools |
| `documentation-expert` | haiku | Technical documentation, manuals, and tutorials |

---

## Research Suite (`research-suite`) — 2 Agents

Scientific research workflows: peer review, idea-to-plan refinement, and methodology orchestration. New in v3.4.0 (split from `science-suite`).

| Agent | Model | Description |
|-------|-------|-------------|
| `research-expert` | opus | Unified specialist for research methodology, evidence synthesis (PRISMA/GRADE), statistical rigor, IMRaD structuring, paper-to-code reproduction, and publication-quality visualization (one-off tasks) |
| `research-spark-orchestrator` | opus | Autonomous driver for the 8-stage research-spark refinement pipeline; owns `_state.yaml`, enforces artifact contract, fans out to sub-agents at Stages 2/6/8 |

---

## Science Suite (`science-suite`) — 11 Agents

Scientific computing, HPC, physics simulations, ML/DL, and nonlinear dynamics. `research-expert` moved to `research-suite` in v3.4.0.

| Agent | Model | Description |
|-------|-------|-------------|
| `neural-network-master` | opus | Deep learning authority: architecture design, theory, and implementation (Transformers, CNNs, diagnostics) |
| `nonlinear-dynamics-expert` | opus | Bifurcation analysis, chaos, coupled networks, pattern formation, and equation discovery (SINDy/UDE) |
| `simulation-expert` | opus | Molecular dynamics, statistical mechanics, and numerical methods (HPC/GPU) |
| `statistical-physicist` | opus | Correlation functions, non-equilibrium dynamics, and ensemble theory |
| `ai-engineer` | sonnet | Production-ready LLM applications, RAG systems, and intelligent agents |
| `jax-pro` | sonnet | JAX-based scientific computing, functional transformations, and high-performance numerical kernels |
| `julia-ml-hpc` | sonnet | Julia ML, Deep Learning, and HPC (Lux.jl, MLJ.jl, CUDA.jl, MPI.jl) |
| `julia-pro` | sonnet | Julia programming, SciML (DifferentialEquations.jl, ModelingToolkit.jl), and Turing.jl |
| `ml-expert` | sonnet | Classical ML algorithms, MLOps pipelines, and data engineering |
| `prompt-engineer` | sonnet | Advanced prompt engineering techniques and LLM performance optimization |
| `python-pro` | sonnet | Python systems engineering: type-driven development, Rust extensions, and performance |

---

## Model Tier Summary

| Tier | Count | Agents |
|------|-------|--------|
| **opus** | 11 | orchestrator, reasoning-engine, context-specialist, software-architect, debugger-pro, neural-network-master, nonlinear-dynamics-expert, research-expert, research-spark-orchestrator, simulation-expert, statistical-physicist |
| **sonnet** | 13 | app-developer, automation-engineer, devops-architect, quality-specialist, sre-expert, systems-engineer, ai-engineer, jax-pro, julia-ml-hpc, julia-pro, ml-expert, prompt-engineer, python-pro |
| **haiku** | 1 | documentation-expert |

---

## Cross-Suite Delegation

Agents delegate across suite boundaries when tasks require multiple domains. Key patterns:

| From | To | Boundary |
|------|----|----------|
| `software-architect` | `devops-architect` | Architecture ↔ Infrastructure |
| `julia-pro` | `julia-ml-hpc` | SciML/ODE ↔ ML training/GPU/HPC |
| `neural-network-master` | `julia-ml-hpc` | DL theory ↔ Julia implementation |
| `nonlinear-dynamics-expert` | `jax-pro` / `julia-pro` | Theory ↔ Implementation |
| `statistical-physicist` | `jax-pro` | Theory ↔ JAX implementation |

See the [Integration Map](../integration-map.rst) for full delegation patterns and MCP server roles.

---

## Resources

- [Commands Reference](commands.md)
- [Quick Reference Cheatsheet](cheatsheet.md)
- [Integration Map](../integration-map.rst) — Suite dependencies and skill coverage
- [Agent Teams Guide](../agent-teams-guide.md) — 10 focused team configurations with 20 variants
- [Glossary](../glossary.rst) — Key terms (Hub Skill, Sub-Skill, Agent Team)

*Generated from v3.4.0 validated marketplace data.*
