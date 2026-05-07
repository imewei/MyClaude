# Plugin Hooks

> 20 nodes · cohesion 0.19

## Key Concepts

- **main()** (10 connections) — `agent-core/hooks/session_end.py`
- **get_recent_commits()** (7 connections) — `agent-core/hooks/session_end.py`
- **get_uncommitted_files()** (7 connections) — `agent-core/hooks/session_end.py`
- **Get recent git commits.** (5 connections) — `science-suite/hooks/session_end.py`
- **write_progress()** (5 connections) — `agent-core/hooks/session_end.py`
- **Session End** (5 connections) — `plugins/dev-suite/hooks/session_end.py`
- **write_progress** (5 connections) — `plugins/agent-core/hooks/session_end.py`
- **session_end.py** (4 connections) — `agent-core/hooks/session_end.py`
- **session_end.py** (4 connections) — `dev-suite/hooks/session_end.py`
- **Persist science session progress.** (4 connections) — `science-suite/hooks/session_end.py`
- **get_uncommitted_files** (4 connections) — `plugins/dev-suite/hooks/session_end.py`
- **get_test_status()** (3 connections) — `dev-suite/hooks/session_end.py`
- **session_end.py** (3 connections) — `science-suite/hooks/session_end.py`
- **get_test_status** (3 connections) — `plugins/dev-suite/hooks/session_end.py`
- **Get commits made during this session (last N).** (2 connections) — `agent-core/hooks/session_end.py`
- **List uncommitted changes.** (2 connections) — `science-suite/hooks/session_end.py`
- **Write structured progress summary for next session.** (2 connections) — `agent-core/hooks/session_end.py`
- **Persist dev session progress.** (2 connections) — `dev-suite/hooks/session_end.py`
- **Persist progress summary and log session end.** (2 connections) — `agent-core/hooks/session_end.py`
- **List uncommitted changes.** (1 connections) — `agent-core/hooks/session_end.py`

## Relationships

- [[HMC-ECS Advanced Sampling]] (8 shared connections)

## Source Files

- `agent-core/hooks/session_end.py`
- `dev-suite/hooks/session_end.py`
- `plugins/agent-core/hooks/session_end.py`
- `plugins/dev-suite/hooks/session_end.py`
- `science-suite/hooks/session_end.py`

## Audit Trail

- EXTRACTED: 80 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*