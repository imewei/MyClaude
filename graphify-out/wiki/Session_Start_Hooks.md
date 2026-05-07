# Session Start Hooks

> 32 nodes · cohesion 0.10

## Key Concepts

- **main()** (12 connections) — `agent-core/hooks/session_start.py`
- **read_progress_file()** (8 connections) — `agent-core/hooks/session_start.py`
- **Session Start** (8 connections) — `plugins/dev-suite/hooks/session_start.py`
- **get_session_context()** (6 connections) — `agent-core/hooks/session_start.py`
- **get_session_context** (6 connections) — `plugins/agent-core/hooks/session_start.py`
- **session_start.py** (5 connections) — `agent-core/hooks/session_start.py`
- **read_progress_file** (5 connections) — `plugins/dev-suite/hooks/session_start.py`
- **session_start.py** (3 connections) — `dev-suite/hooks/session_start.py`
- **detect_compute_env()** (3 connections) — `science-suite/hooks/session_start.py`
- **detect_research_artifacts()** (3 connections) — `research-suite/hooks/session_start.py`
- **detect_stack()** (3 connections) — `dev-suite/hooks/session_start.py`
- **read_git_summary()** (3 connections) — `agent-core/hooks/session_start.py`
- **read_uncommitted_status()** (3 connections) — `agent-core/hooks/session_start.py`
- **session_start.py** (3 connections) — `science-suite/hooks/session_start.py`
- **detect_compute_env** (3 connections) — `plugins/science-suite/hooks/session_start.py`
- **detect_research_artifacts** (3 connections) — `plugins/research-suite/hooks/session_start.py`
- **detect_stack** (3 connections) — `plugins/dev-suite/hooks/session_start.py`
- **read_git_summary** (3 connections) — `plugins/agent-core/hooks/session_start.py`
- **read_uncommitted_status** (3 connections) — `plugins/agent-core/hooks/session_start.py`
- **Detect project stack from file presence.** (2 connections) — `dev-suite/hooks/session_start.py`
- **Detect available compute resources.** (2 connections) — `science-suite/hooks/session_start.py`
- **Get recent git activity summary.** (2 connections) — `agent-core/hooks/session_start.py`
- **Detect research-spark stage artifacts present in the working tree.** (2 connections) — `research-suite/hooks/session_start.py`
- **Read the most recent session progress summary if it exists.** (2 connections) — `agent-core/hooks/session_start.py`
- **Check for uncommitted changes.** (2 connections) — `agent-core/hooks/session_start.py`
- *... and 7 more nodes in this community*

## Relationships

- [[HMC-ECS Advanced Sampling]] (9 shared connections)

## Source Files

- `agent-core/hooks/session_start.py`
- `dev-suite/hooks/session_start.py`
- `plugins/agent-core/hooks/session_start.py`
- `plugins/dev-suite/hooks/session_start.py`
- `plugins/research-suite/hooks/session_start.py`
- `plugins/science-suite/hooks/session_start.py`
- `research-suite/hooks/session_start.py`
- `science-suite/hooks/session_start.py`

## Audit Trail

- EXTRACTED: 111 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*