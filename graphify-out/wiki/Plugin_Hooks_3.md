# Plugin Hooks

> 10 nodes · cohesion 0.22

## Key Concepts

- **main()** (6 connections) — `agent-core/hooks/task_completed.py`
- **has_uncommitted_changes()** (3 connections) — `dev-suite/hooks/task_completed.py`
- **has_uncommitted_changes** (3 connections) — `plugins/dev-suite/hooks/task_completed.py`
- **task_completed.py** (2 connections) — `dev-suite/hooks/task_completed.py`
- **Acknowledge task completion.** (2 connections) — `agent-core/hooks/task_completed.py`
- **Check if there are uncommitted changes to suggest committing.** (2 connections) — `dev-suite/hooks/task_completed.py`
- **Remind about validation and suggest commit after task completion.** (2 connections) — `dev-suite/hooks/task_completed.py`
- **Task Completed** (2 connections) — `plugins/dev-suite/hooks/task_completed.py`
- **task_completed.py** (1 connections) — `agent-core/hooks/task_completed.py`
- **task_completed.py** (1 connections) — `research-suite/hooks/task_completed.py`

## Relationships

- [[HMC-ECS Advanced Sampling]] (4 shared connections)

## Source Files

- `agent-core/hooks/task_completed.py`
- `dev-suite/hooks/task_completed.py`
- `plugins/dev-suite/hooks/task_completed.py`
- `research-suite/hooks/task_completed.py`

## Audit Trail

- EXTRACTED: 24 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*