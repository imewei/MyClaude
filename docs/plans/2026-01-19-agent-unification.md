# Plugin Consolidation (Phase 3: Agent Unification) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Phase 3 of the consolidation strategy by unifying fragmented agents into consolidated, domain-specific experts within the 5 Mega-Plugins. This reduces redundancy and creates more capable generalist agents.

**Architecture:** We will read source agent definitions, synthesize their capabilities into unified prompts, and write the new agent definitions to the target suites. We will also update `plugin.json` for each suite to register these agents.

**Tech Stack:** Markdown (agent definitions), JSON (manifests).

---

### Task 1: Engineering Suite Agent Unification

**Files:**
- Source: `plugins/backend-development/agents/backend-architect.md`
- Source: `plugins/agent-orchestration/agents/systems-architect.md`
- Source: `plugins/ai-reasoning/agents/ai-systems-architect.md`
- Source: `plugins/multi-platform-apps/agents/*.md`
- Target: `plugins/engineering-suite/agents/software-architect.md`
- Target: `plugins/engineering-suite/agents/app-developer.md`
- Target: `plugins/engineering-suite/agents/systems-engineer.md`
- Modify: `plugins/engineering-suite/plugin.json`

**Step 1: Unify Software Architect**
Combine `backend-architect`, `systems-architect`, and `ai-systems-architect` into `software-architect`.
*Action:* Create a comprehensive prompt covering API design, distributed systems, and AI architecture.

**Step 2: Unify App Developer**
Combine `frontend-developer`, `mobile-developer`, `ios-developer`, `flutter-expert` into `app-developer`.
*Action:* Create a prompt for a full-stack client-side expert (Web + Mobile).

**Step 3: Unify Systems Engineer**
Combine `golang-pro`, `rust-pro`, `cpp-pro`, `c-pro` into `systems-engineer`.
*Action:* Create a prompt for low-level systems programming and high-performance computing.

**Step 4: Update Manifest**
Update `plugins/engineering-suite/plugin.json` to register these 3 agents and their specializations.

**Step 5: Commit**
```bash
git add plugins/engineering-suite/
git commit -m "feat(engineering): unify agents into software-architect, app-developer, systems-engineer"
```

---

### Task 2: Infrastructure Suite Agent Unification

**Files:**
- Source: `plugins/cicd-automation/agents/*.md`
- Source: `plugins/observability-monitoring/agents/*.md`
- Target: `plugins/infrastructure-suite/agents/devops-architect.md`
- Target: `plugins/infrastructure-suite/agents/sre-expert.md`
- Target: `plugins/infrastructure-suite/agents/automation-engineer.md`
- Modify: `plugins/infrastructure-suite/plugin.json`

**Step 1: Unify DevOps Architect**
Combine `cloud-architect`, `kubernetes-architect`, `terraform-specialist` into `devops-architect`.
*Action:* Create a prompt for cloud infrastructure, IaC, and platform engineering.

**Step 2: Unify SRE Expert**
Combine `observability-engineer`, `performance-engineer`, `database-optimizer`, `network-engineer` into `sre-expert`.
*Action:* Create a prompt for reliability, observability, and performance optimization.

**Step 3: Unify Automation Engineer**
Combine `deployment-engineer`, `devops-troubleshooter` into `automation-engineer`.
*Action:* Create a prompt for CI/CD pipelines, release automation, and troubleshooting.

**Step 4: Update Manifest**
Update `plugins/infrastructure-suite/plugin.json`.

**Step 5: Commit**
```bash
git add plugins/infrastructure-suite/
git commit -m "feat(infra): unify agents into devops-architect, sre-expert, automation-engineer"
```

---

### Task 3: Science Suite Agent Unification

**Files:**
- Source: `plugins/hpc-computing/agents/*.md`
- Source: `plugins/jax-implementation/agents/*.md`
- Source: `plugins/molecular-simulation/agents/*.md`
- Source: `plugins/machine-learning/agents/*.md`
- Target: `plugins/science-suite/agents/simulation-expert.md`
- Target: `plugins/science-suite/agents/ml-expert.md`
- Target: `plugins/science-suite/agents/research-expert.md`
- Modify: `plugins/science-suite/plugin.json`

**Step 1: Unify Simulation Expert**
Combine `hpc-numerical-coordinator`, `jax-scientist`, `simulation-expert`, `non-equilibrium-expert` into `simulation-expert`.
*Action:* Create a prompt for physics simulations, HPC, and numerical methods.

**Step 2: Unify ML Expert**
Combine `ml-engineer`, `data-scientist`, `neural-architecture-engineer` into `ml-expert`.
*Action:* Create a prompt for scientific ML, deep learning, and data science.

**Step 3: Unify Research Expert**
Combine `research-intelligence`, `visualization-interface` into `research-expert`.
*Action:* Create a prompt for research methodology, literature analysis, and visualization.

**Step 4: Update Manifest**
Update `plugins/science-suite/plugin.json`.

**Step 5: Commit**
```bash
git add plugins/science-suite/
git commit -m "feat(science): unify agents into simulation-expert, ml-expert, research-expert"
```

---

### Task 4: Quality Suite Agent Unification

**Files:**
- Source: `plugins/codebase-cleanup/agents/code-reviewer.md`
- Source: `plugins/git-pr-workflows/agents/code-reviewer.md`
- Source: `plugins/comprehensive-review/agents/security-auditor.md`
- Source: `plugins/debugging-toolkit/agents/debugger.md`
- Source: `plugins/code-documentation/agents/docs-architect.md`
- Target: `plugins/quality-suite/agents/quality-specialist.md`
- Target: `plugins/quality-suite/agents/debugger-pro.md`
- Target: `plugins/quality-suite/agents/documentation-expert.md`
- Modify: `plugins/quality-suite/plugin.json`

**Step 1: Unify Quality Specialist**
Combine `code-reviewer`, `security-auditor`, `test-automator` into `quality-specialist`.
*Action:* Create a prompt for holistic code quality, security, and testing.

**Step 2: Unify Debugger Pro**
Refine `debugger` into `debugger-pro`.
*Action:* Create a prompt for advanced root cause analysis and troubleshooting.

**Step 3: Unify Documentation Expert**
Combine `docs-architect`, `tutorial-engineer` into `documentation-expert`.
*Action:* Create a prompt for technical documentation and knowledge management.

**Step 4: Update Manifest**
Update `plugins/quality-suite/plugin.json`.

**Step 5: Commit**
```bash
git add plugins/quality-suite/
git commit -m "feat(quality): unify agents into quality-specialist, debugger-pro, documentation-expert"
```

---

### Task 5: Agent Core Unification

**Files:**
- Source: `plugins/agent-orchestration/agents/multi-agent-orchestrator.md`
- Source: `plugins/agent-orchestration/agents/context-manager.md`
- Source: `plugins/llm-application-dev/agents/prompt-engineer.md`
- Target: `plugins/agent-core/agents/orchestrator.md`
- Target: `plugins/agent-core/agents/context-specialist.md`
- Target: `plugins/agent-core/agents/reasoning-engine.md`
- Modify: `plugins/agent-core/plugin.json`

**Step 1: Unify Orchestrator**
Refine `multi-agent-orchestrator` into `orchestrator`.
*Action:* Create a prompt for managing agent workflows and delegation.

**Step 2: Unify Context Specialist**
Refine `context-manager` into `context-specialist`.
*Action:* Create a prompt for managing conversation context and memory.

**Step 3: Create Reasoning Engine**
Combine `prompt-engineer` and reasoning capabilities into `reasoning-engine`.
*Action:* Create a prompt for advanced reasoning, prompt design, and cognitive tasks.

**Step 4: Update Manifest**
Update `plugins/agent-core/plugin.json`.

**Step 5: Commit**
```bash
git add plugins/agent-core/
git commit -m "feat(core): unify agents into orchestrator, context-specialist, reasoning-engine"
```
