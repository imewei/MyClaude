# MyClaude Plugin Ecosystem Redundancy Analysis Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Analyze the 31-plugin ecosystem to identify overlapping capabilities, redundant agents/skills, and propose a consolidation strategy to improve maintainability and user experience.

**Architecture:** Systematic analysis of manifest files (`plugin.json`), agent definitions (`agents/*.md`), and skill definitions (`skills/*/SKILL.md`) to map capabilities and identify clusters of redundancy.

**Tech Stack:** Python (for analysis scripts), Markdown (for reporting), JSON (manifest parsing).

---

### Task 1: Capability Mapping Script

**Files:**
- Create: `scripts/analyze_ecosystem.py`

**Step 1: Write the analysis script**

Create a Python script that:
1.  Recursively scans `plugins/` directory.
2.  Parses `plugin.json` for keywords and categories.
3.  Parses `agents/*.md` for `specialization` and `description`.
4.  Parses `skills/*/SKILL.md` for `specialization` and `description`.
5.  Generates a capability matrix mapping components to functional domains (e.g., "Testing", "Optimization", "Security").
6.  Outputs a JSON report `ecosystem_capabilities.json`.

**Step 2: Run the script**

Run: `python3 scripts/analyze_ecosystem.py`
Expected: Generate `ecosystem_capabilities.json` with structured data.

**Step 3: Commit**

```bash
git add scripts/analyze_ecosystem.py
git commit -m "feat(analysis): add script to map plugin ecosystem capabilities"
```

---

### Task 2: Redundancy Identification Report

**Files:**
- Create: `docs/plans/redundancy_findings.md`
- Read: `ecosystem_capabilities.json`

**Step 1: Analyze capability clusters**

Manually or programmatically analyze the JSON output to identify clusters.
*Hypothesis:*
- **Optimization Cluster:** `agent-orchestration` (multi-agent-optimize) vs `codebase-cleanup` (refactor-clean) vs `observability-monitoring` (performance-engineer).
- **Testing Cluster:** `unit-testing` vs `quality-engineering` vs `cicd-automation` (test steps).
- **Scientific Cluster:** `hpc-computing` vs `jax-implementation` vs `deep-learning` vs `molecular-simulation` (likely distinct but potentially overlapping in basics).
- **Migration Cluster:** `code-migration` vs `framework-migration`.

**Step 2: Document specific overlaps**

Create a report detailing:
1.  **Duplicate Agents:** Agents with >80% overlap in role (e.g., is `systems-architect` in multiple places?).
2.  **Overlapping Commands:** Commands that perform similar actions (e.g., `/refactor` vs `/optimize`).
3.  **Redundant Skills:** Skills that cover the same domain (e.g., general python patterns vs specific framework patterns).

**Step 3: Commit**

```bash
git add docs/plans/redundancy_findings.md
git commit -m "docs(analysis): document initial redundancy findings"
```

---

### Task 3: Consolidation Strategy

**Files:**
- Create: `docs/plans/consolidation_strategy.md`

**Step 1: Propose Consolidation Architecture**

Design a consolidated structure (e.g., reducing 31 plugins to ~10-12 "Mega-Plugins"):
1.  **Infrastructure & Ops**: Combine `cicd`, `observability`, `git-pr` into `infrastructure-suite`.
2.  **Software Engineering**: Combine `backend`, `frontend`, `python`, `javascript`, `systems`, `cli` into `engineering-suite`.
3.  **Scientific Computing**: Combine `hpc`, `jax`, `molecular`, `statistical`, `deep-learning` into `science-suite`.
4.  **Quality & Maintenance**: Combine `quality`, `testing`, `cleanup`, `migration`, `documentation` into `quality-suite`.
5.  **Agent Core**: Combine `orchestration`, `reasoning` into `agent-core`.

**Step 2: Define Migration Path**

For each proposed merger:
- List source plugins.
- Define new directory structure.
- Strategy for merging `plugin.json` manifests.
- Strategy for deduplicating agents (e.g., merging `backend-architect` and `systems-architect`).

**Step 3: Commit**

```bash
git add docs/plans/consolidation_strategy.md
git commit -m "docs(plan): propose plugin consolidation strategy"
```
