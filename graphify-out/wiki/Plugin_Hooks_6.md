# Plugin Hooks

> 7 nodes · cohesion 0.29

## Key Concepts

- **main()** (6 connections) — `agent-core/hooks/subagent_stop.py`
- **Log science subagent completion.** (2 connections) — `science-suite/hooks/subagent_stop.py`
- **Log subagent completion for orchestration tracking.** (2 connections) — `agent-core/hooks/subagent_stop.py`
- **subagent_stop.py** (1 connections) — `agent-core/hooks/subagent_stop.py`
- **subagent_stop.py** (1 connections) — `dev-suite/hooks/subagent_stop.py`
- **subagent_stop.py** (1 connections) — `research-suite/hooks/subagent_stop.py`
- **subagent_stop.py** (1 connections) — `science-suite/hooks/subagent_stop.py`

## Relationships

- [[HMC-ECS Advanced Sampling]] (2 shared connections)

## Source Files

- `agent-core/hooks/subagent_stop.py`
- `dev-suite/hooks/subagent_stop.py`
- `research-suite/hooks/subagent_stop.py`
- `science-suite/hooks/subagent_stop.py`

## Audit Trail

- EXTRACTED: 14 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*