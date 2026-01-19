# Plugin Consolidation (Phase 1 & 2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the first two phases of the plugin consolidation strategy: verify the skeleton structures of the 5 new "Mega-Plugins" and migrate/deduplicate skills from the 31 fragmented plugins into these new suites.

**Architecture:** We are moving from a fragmented 31-plugin architecture to a consolidated 5-suite architecture (Engineering, Infrastructure, Science, Quality, Agent Core). This plan covers checking the skeletons and populating the `skills/` directories by merging and deduplicating existing skills.

**Tech Stack:** Markdown (skills), JSON (manifests), file system operations.

---

### Task 1: Verify Engineering Suite Skeleton

**Files:**
- Verify: `plugins/engineering-suite/plugin.json`
- Verify: `plugins/engineering-suite/skills/`

**Step 1: Verify plugin.json validity**
Run: `cat plugins/engineering-suite/plugin.json`
Check: Ensure it has valid JSON structure, correct name "engineering-suite", and appropriate keywords/categories.

**Step 2: Verify skill directory structure**
Run: `ls -F plugins/engineering-suite/skills/`
Check: Ensure subdirectories exist for `backend-engineering`, `frontend-mobile-engineering`, `language-mastery`, `systems-cli-engineering`, `modernization-migration`.

**Step 3: Create missing skill categories if needed**
If any are missing, create them with `mkdir -p`.

---

### Task 2: Migrate Engineering Skills

**Files:**
- Source: `plugins/backend-development/skills/*`
- Source: `plugins/frontend-mobile-development/skills/*`
- Source: `plugins/python-development/skills/*`
- Source: `plugins/javascript-typescript/skills/*`
- Target: `plugins/engineering-suite/skills/*`

**Step 1: Migrate Backend Skills**
Copy unique skills from `plugins/backend-development/skills/` to `plugins/engineering-suite/skills/backend-engineering/`.
*Action:* Identify unique skills (e.g., API design, DB schema) and copy their content to new skill files in the target directory.

**Step 2: Migrate Language Skills**
Copy Python and JS/TS skills to `plugins/engineering-suite/skills/language-mastery/`.
*Action:* Merge "python-best-practices" and "typescript-patterns" into consolidated language skills if possible, or keep as distinct files within the category.

**Step 3: Commit Engineering Skills**
```bash
git add plugins/engineering-suite/skills/
git commit -m "feat(engineering): populate skills from legacy plugins"
```

---

### Task 3: Verify Infrastructure Suite Skeleton & Migrate Skills

**Files:**
- Verify: `plugins/infrastructure-suite/plugin.json`
- Source: `plugins/cicd-automation/skills/*`
- Source: `plugins/observability-monitoring/skills/*`
- Target: `plugins/infrastructure-suite/skills/*`

**Step 1: Verify structure**
Run: `ls -R plugins/infrastructure-suite/`

**Step 2: Migrate CI/CD & Observability Skills**
- Copy content from `plugins/cicd-automation/skills` to `plugins/infrastructure-suite/skills/deployment-pipelines/`
- Copy content from `plugins/observability-monitoring/skills` to `plugins/infrastructure-suite/skills/observability/`

**Step 3: Commit Infrastructure Skills**
```bash
git add plugins/infrastructure-suite/skills/
git commit -m "feat(infra): populate skills from legacy plugins"
```

---

### Task 4: Verify Science Suite Skeleton & Migrate Skills

**Files:**
- Verify: `plugins/science-suite/plugin.json`
- Source: `plugins/hpc-computing/skills/*`
- Source: `plugins/jax-implementation/skills/*`
- Source: `plugins/molecular-simulation/skills/*`
- Target: `plugins/science-suite/skills/*`

**Step 1: Verify structure**
Run: `ls -R plugins/science-suite/`

**Step 2: Migrate Scientific Skills**
- Consolidate `hpc-computing` and `parallel-computing` skills into `plugins/science-suite/skills/parallel-computing/`.
- Move `jax` and `julia` specific skills to `plugins/science-suite/skills/jax-mastery` and `julia-mastery`.
- Move domain-specifics (molecular, physics) to `plugins/science-suite/skills/advanced-simulations`.

**Step 3: Commit Science Skills**
```bash
git add plugins/science-suite/skills/
git commit -m "feat(science): populate skills from legacy plugins"
```

---

### Task 5: Verify Quality Suite Skeleton & Migrate Skills

**Files:**
- Verify: `plugins/quality-suite/plugin.json`
- Source: `plugins/quality-engineering/skills/*`
- Source: `plugins/unit-testing/skills/*`
- Source: `plugins/codebase-cleanup/skills/*`
- Target: `plugins/quality-suite/skills/*`

**Step 1: Verify structure**
Run: `ls -R plugins/quality-suite/`

**Step 2: Migrate Quality Skills**
- Merge testing skills into `plugins/quality-suite/skills/test-automation/`.
- Merge review/cleanup skills into `plugins/quality-suite/skills/code-review/`.
- Move debugging skills to `plugins/quality-suite/skills/debugging-strategies/`.

**Step 3: Commit Quality Skills**
```bash
git add plugins/quality-suite/skills/
git commit -m "feat(quality): populate skills from legacy plugins"
```

---

### Task 6: Verify Agent Core Skeleton & Migrate Skills

**Files:**
- Verify: `plugins/agent-core/plugin.json`
- Source: `plugins/agent-orchestration/skills/*`
- Source: `plugins/ai-reasoning/skills/*`
- Target: `plugins/agent-core/skills/*`

**Step 1: Verify structure**
Run: `ls -R plugins/agent-core/`

**Step 2: Migrate Core Skills**
- Move orchestration skills to `plugins/agent-core/skills/agent-orchestration/`.
- Move reasoning skills to `plugins/agent-core/skills/advanced-reasoning/`.

**Step 3: Commit Core Skills**
```bash
git add plugins/agent-core/skills/
git commit -m "feat(core): populate skills from legacy plugins"
```

---

### Task 7: Cleanup & Validation

**Step 1: Verify no empty skill directories**
Run: `find plugins/*-suite/skills -type d -empty`
Action: Delete or populate any empty directories.

**Step 2: Final Verification**
Run: `python3 scripts/analyze_ecosystem.py` (optional, if script updated to point to new locs)
Or manual spot check of 2-3 migrated skills.

**Step 3: Commit Cleanup**
```bash
git add plugins/
git commit -m "chore(cleanup): remove empty directories and verify migration"
```
