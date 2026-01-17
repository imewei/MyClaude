---
description: Intelligent debugging with multi-mode execution and automated RCA
triggers:
- /smart-debug
- intelligent debugging with multi mode
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<error-description> [--quick-triage|--standard-debug|--deep-rca] [--production]`
The agent should parse these arguments from the user's request.

# AI-Assisted Debugging

$ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Modes

| Mode | Time | Output |
|------|------|--------|
| `--quick-triage` | 5-10min | 3-5 hypotheses |
| standard | 15-30min | Root cause + fix |
| `--deep-rca` | 30-60min | Full RCA + prevention |

## Workflow

### 1. Triage (Analysis)
Parse error, reproduction, environment (dev/prod), pattern (flaky/consistent)
Match: NullPointer, Timeout, MemLeak, Race, Deadlock, Auth, RateLimit, JSON, FileIO, InfiniteLoop, Injection, TypeCoercion, Config, Async, CORS
Output: Severity (P0-P3), top 3 causes, strategy

### 2. Observability & Hypotheses (Parallel Execution)

> **Orchestration Note**: Execute log gathering and hypothesis generation concurrently.

**Observability (Production):**
Gather: Error tracking (Sentry/Rollbar), APM (Datadog/NewRelic p95/p99), Traces (Jaeger/Zipkin), Logs (ELK/Splunk)
Query window: incident ± 30min, trace by correlation ID

**Hypotheses:**
Generate 3-5 ranked hypotheses: probability score, evidence, falsification criteria
Use: 5 Whys, Fault Tree, Timeline, Fishbone
Categories: Logic, State, Integration, Resources, Config, Data

🚨 **Quick-Triage exits** - deliver hypotheses

### 4. Strategy (Planning)
Reproducible locally? → Interactive (VS Code/pdb/DevTools)
Production-only? → Observability (Sentry/Datadog/Jaeger)
Complex state? → Time-Travel (rr/Redux/git bisect)
Load-dependent? → Chaos (Monkey/Gremlin)
Small %? → Statistical (Delta debugging)

### 5. Instrumentation (Implementation)
Points: Entry, decisions (if/switch), state mutations, external boundaries, error paths
Production-safe: Feature flags, 1% sampling, read-only endpoints, gradual traffic shift

### 6. RCA (Analysis)
Analyze: Execution path, dependencies, timing, code smells, similar patterns
Document: Immediate cause, root cause, contributing factors, evidence

### 7. Fix (Implementation)
Code changes, impact, risk (Low/Med/High), test coverage needs
🚨 **Standard-Debug exits** - root cause + fix + validation

### 8. Validation (Parallel Execution)
- Run tests
- Load test
- Perf comparison
- Canary deploy
Success: 100% pass, no regression, error rate ↓

### 9. Prevention (Deep only)
Regression tests, monitoring alerts, knowledge base, runbook, coding standards
🎯 **Deep-RCA complete**

## Output

**Issue**: Error, frequency, impact, environment, severity
**Root Cause**: Immediate, root, factors, evidence
**Fix**: Changes, impact, risk, tests
**Validation**: Tests, perf, deployment
**Prevention** (Deep only): Tests, monitoring, docs, process
