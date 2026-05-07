# Plugin Hooks

> 8 nodes · cohesion 0.32

## Key Concepts

- **main()** (7 connections) — `agent-core/hooks/post_tool_use.py`
- **Log file modifications for potential auto-linting.** (4 connections) — `agent-core/hooks/post_tool_use.py`
- **check_numerical_integrity()** (3 connections) — `science-suite/hooks/post_tool_use.py`
- **Suggest linting after file modifications.** (2 connections) — `dev-suite/hooks/post_tool_use.py`
- **Check Bash output for numerical issues.** (2 connections) — `science-suite/hooks/post_tool_use.py`
- **post_tool_use.py** (2 connections) — `science-suite/hooks/post_tool_use.py`
- **post_tool_use.py** (1 connections) — `agent-core/hooks/post_tool_use.py`
- **post_tool_use.py** (1 connections) — `dev-suite/hooks/post_tool_use.py`

## Relationships

- [[HMC-ECS Advanced Sampling]] (4 shared connections)

## Source Files

- `agent-core/hooks/post_tool_use.py`
- `dev-suite/hooks/post_tool_use.py`
- `science-suite/hooks/post_tool_use.py`

## Audit Trail

- EXTRACTED: 22 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*