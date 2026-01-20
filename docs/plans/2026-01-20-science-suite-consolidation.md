# Science Suite Consolidation & Optimization Plan

## 1. Redundancy Analysis Summary

### A. Machine Learning vs. Deep Learning
*   **Issue**: `machine-learning` currently claims scope over Deep Learning (PyTorch/JAX), creating ambiguity with the dedicated `deep-learning` skill.
*   **Resolution**:
    *   Refactor `machine-learning` to focus exclusively on **Classical ML** (Scikit-learn, XGBoost, Tabular) and **MLOps** (Deployment, Pipelines).
    *   Move all Neural Network architecture and training content to `deep-learning`.
    *   Clarify `ml-expert` vs `neural-network-master`: `ml-expert` handles the *pipeline* and *production*, `neural-network-master` handles the *theory* and *architecture*.

### B. Simulations vs. Physics vs. Parallel Computing
*   **Issue**: `advanced-simulations` overlaps with `statistical-physics` (theory) and `parallel-computing` (solvers).
*   **Resolution**:
    *   `statistical-physics` becomes the "Theory & Analysis" home (Active matter, Correlations).
    *   `advanced-simulations` becomes "Computational Physics Workflows" (MD setup, Multiscale coupling).
    *   `parallel-computing` retains "Numerical Implementation" (Solvers, GPU kernels).

### C. Circular Delegations
*   **Issue**: Agents delegate to each other in cycles (e.g., `ai-engineer` ↔ `prompt-engineer`).
*   **Resolution**: Establish a strict hierarchy.
    *   **Architects**: `ai-engineer`, `neural-network-master`, `simulation-expert`.
    *   **Specialists**: `research-expert`, `statistical-physicist`, `ml-expert`.
    *   **Toolsmiths**: `jax-pro`, `julia-pro`, `python-pro`, `prompt-engineer`.
    *   *Rule*: Architects delegate to Specialists/Toolsmiths. Toolsmiths do not delegate back up for core tasks.

## 2. Implementation Steps

### Phase 1: Skill Refactoring
1.  **Refine `machine-learning/SKILL.md`**: Remove "Deep Learning" sections. Focus on Scikit-learn, XGBoost, and MLOps.
2.  **Refine `deep-learning/SKILL.md`**: Consolidate NN architecture and math here.
3.  **Merge Physics Content**: Move any theoretical stochastic dynamics from `advanced-simulations` to `statistical-physics`.

### Phase 2: Agent Optimization
1.  **`ml-expert`**: Update description to emphasize MLOps and Classical ML. Remove "Deep Learning Architecture" from primary responsibilities (delegate to `neural-network-master`).
2.  **`neural-network-master`**: Solidify as the primary Deep Learning Architect.
3.  **`prompt-engineer`**: Remove delegation to `ai-engineer`. It should be a leaf node focused purely on optimization.
4.  **`jax-pro` / `julia-pro`**: Focus on being the "Compute Engines" for other agents.

### Phase 3: Triggering & Synergy
1.  **Keyword Optimization**: Ensure unique trigger phrases for each agent in `plugin.json` (implied) and their system prompts.
2.  **Cross-Linking**: Update `SKILL.md` files to explicitly link to related skills (e.g., ML linking to Python Pro for tooling).

## 3. Success Metrics
*   Zero circular delegations.
*   Clear separation of concerns between ML and DL.
*   No duplicate "Numerical Methods" sections across skills.
