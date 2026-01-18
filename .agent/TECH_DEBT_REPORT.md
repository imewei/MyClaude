# Technical Debt Report: `.agent` System

## 1. Executive Summary
The `.agent` system is **Mechanically Sound but Architecturally Fragmented**.

Recent validation efforts (Phases 1-3) confirm the system maintains **100% referential integrity** (0 orphaned files, 0 broken links across 672 files) and strictly adheres to the schema. However, the system suffers from significant **Cognitive Overload** and **Functional Duplication**. With **190 discrete skills**, the agent faces a "Split-Brain" problem where capabilities are fractured across too many overlapping personas (e.g., 6+ distinct mobile development skills), making tool selection non-deterministic and inefficient. Furthermore, the complete lack of runtime telemetry means we are "flying blind" regarding skill efficacy.

**Assessment:** Feature-rich but architecturally fragile due to sprawl.

---

## 2. Debt Inventory & Scoring

| Category | Severity | Issue | Description | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | **CRITICAL** | **Split-Brain / Skill Bloat** | The registry contains **190 skills**, with high overlap. <br>• **JAX**: `jax-pro`, `jax-scientist`, `jax-core`, `jax-physics` (4 overlapping).<br>• **Mobile**: `flutter-dev`, `flutter-expert`, `mobile-dev`, `ios-dev` (4 overlapping).<br>This forces the agent to "guess" which specialized persona to invoke. | **P0** |
| **Infrastructure** | **HIGH** | **Zero Telemetry** | No mechanism exists to track which skills are invoked, their success rates, or latency. We cannot optimize what we cannot measure. | **P1** |
| **Code** | **MEDIUM** | **Static Validation Limits** | Current scripts (`deep_validate.py`) check *existence* and *syntax* but not *semantic quality* or *logic*. A valid file can still be hallucination-prone. | **P2** |
| **Policy** | **MEDIUM** | **Framework Strictness** | PyTorch imports are flagged as warnings, not errors. Given the **JAX-First** mandate, this represents "Policy Debt" that allows non-compliant code to creep in. | **P3** |

---

## 3. ROI Analysis (Quick Wins)

These high-impact tasks offer the best return on engineering time:

1.  **Consolidate "Split-Brain" Clusters (Impact: High, Effort: Low)**
    *   **Task**: Merge `jax-*` skills into a single `scientific-computing` super-skill and `mobile-*` into `multi-platform-mobile`.
    *   **Gain**: Reduces context window usage and improves agent routing accuracy.

2.  **Unified CI Gate (Impact: Medium, Effort: Low)**
    *   **Task**: Combine `deep_validate_agent.py` and `validate_skills.py` into a single `make check` target or `health_check.py` script.
    *   **Gain**: Simplifies developer workflow and ensures atomic validation.

3.  **Enforce JAX Policy (Impact: Medium, Effort: Low)**
    *   **Task**: Upgrade the "PyTorch detected" warning in validation scripts to a hard failure (unless `# allow-torch` is present).
    *   **Gain**: Prevents "Shadow IT" (non-JAX code) from accumulating in the codebase.

---

## 4. Remediation Roadmap

### Immediate (Sprint 1): Hygiene & Consolidation
*   **Goal**: Stop the bleeding and simplify tooling.
*   [x] **Action**: Merge overlapping Validation Scripts into a single CI entry point (`validate_agent.py`).
*   [x] **Action**: Hard-block PyTorch imports in CI (enforce JAX-First).
*   [ ] **Action**: Archive or merge the top 10 most redundant skills (focusing on Mobile/JAX clusters).

### Medium Term (Q1): Refactoring & Migration
*   **Goal**: Architecture cleanup.
*   [ ] **Action**: **Plugin Migration**: Move standardized skills (e.g., `git`, `docker`) out of `.agent` and into specialized MCP servers to reduce noise.
*   [ ] **Action**: **Taxonomy Redesign**: Restructure `skills_index.json` to use a hierarchical categorization (e.g., `domain: [skills]`) rather than a flat list.

### Long Term (Q2): Observability & Automation
*   **Goal**: Data-driven evolution.
*   [ ] **Action**: **Implement Telemetry**: Add a "Usage Logger" sidecar to track tool invocation counts and outcome ratings.
*   [ ] **Action**: **Dynamic Loading**: Refactor the agent to load skills dynamically based on context, rather than loading the full index, to improve "Thinking" time and token efficiency.
