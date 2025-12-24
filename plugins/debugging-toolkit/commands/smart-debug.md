---
version: "1.0.6"
category: debugging
purpose: AI-assisted debugging with automated root cause analysis, error pattern recognition, and production-safe debugging
description: Intelligent debugging orchestration with multi-mode execution and automated RCA workflows

execution_time:
  quick-triage: "5-10 minutes"
  standard-debug: "15-30 minutes"
  deep-rca: "30-60 minutes"

external_docs:
  - debugging-patterns-library.md
  - rca-frameworks-guide.md
  - observability-integration-guide.md
  - debugging-tools-reference.md

tags: [debugging, rca, observability, production-debugging, error-analysis]
allowed-tools: Bash(python:*), Bash(node:*), Bash(pytest:*), Bash(jest:*), Bash(git:*), Bash(docker:*), Bash(kubectl:*), Bash(go:*), Bash(rust:*)
argument-hint: <error-or-issue-description> [--quick-triage|--standard-debug|--deep-rca] [--production] [--performance]
color: red

agents:
  primary:
    - debugging-toolkit:debugger
  conditional:
    - agent: observability-monitoring:observability-engineer
      trigger: argument "--production" OR pattern "production|incident|outage"
    - agent: observability-monitoring:performance-engineer
      trigger: argument "--performance" OR pattern "slow|timeout|memory.*leak"
    - agent: cicd-automation:kubernetes-architect
      trigger: pattern "pod|container|k8s|kubernetes"
  orchestrated: true
---

# AI-Assisted Debugging Specialist

Expert debugging specialist for automated root cause analysis and production-safe debugging.

## Issue

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Steps | Output |
|------|----------|-------|--------|
| `--quick-triage` | 5-10 min | 1-3 | 3-5 ranked hypotheses |
| standard (default) | 15-30 min | 1-8 | Root cause + fix proposal |
| `--deep-rca` | 30-60 min | 1-10 | Full RCA + prevention strategy |

---

## Workflow

### Step 1: Initial Triage

**Parse issue for:**
- Error messages/stack traces
- Reproduction steps
- Affected components
- Environment (dev/staging/prod)
- Pattern (intermittent/consistent)

**Match against 15 common patterns:**
- NullPointer/Null Reference
- Connection Timeout
- Memory Leak
- Race Condition
- Database Deadlock
- Auth/Authz Failures
- Rate Limiting
- JSON Parsing
- File I/O
- Infinite Loop/Hang
- SQL Injection
- Type Coercion
- Environment Config
- Async Operation
- CORS

**Output:** Pattern match, Severity (P0-P3), Top 3 likely causes, Recommended strategy

---

### Step 2: Observability Data Collection

**For production issues, gather:**

| Source | Data |
|--------|------|
| Error tracking | Sentry, Rollbar - frequency, trends, user cohorts |
| APM | Datadog, New Relic - latency p50/p95/p99, error rates |
| Traces | Jaeger, Zipkin - request flow, span timing |
| Logs | ELK, Splunk - structured queries, timeline |

**Query strategy:**
1. Time window: incident ± 30 min
2. Trace failed requests by correlation ID
3. Compare metrics before/during/after

---

### Step 3: Hypothesis Generation

**Generate 3-5 ranked hypotheses with:**
- Probability score (0-100%)
- Supporting evidence (logs, traces, code)
- Falsification criteria
- Testing approach

**Use frameworks:** 5 Whys, Fault Tree, Timeline Reconstruction, Fishbone

**Categories:** Logic errors, State management, Integration failures, Resource exhaustion, Config drift, Data corruption

🚨 **Quick-Triage exits here** - deliver hypotheses and strategy recommendation.

---

### Step 4: Strategy Selection

**Decision tree:**
```
Reproducible locally? → Interactive Debugging
├─ No → Production-only? → Observability-Driven
│       ├─ No → Load/timing-dependent? → Chaos Engineering
│       └─ No → Statistical Debugging
```

| Strategy | Tools | Use Case |
|----------|-------|----------|
| Interactive | VS Code, pdb, Chrome DevTools | Local reproduction |
| Observability-Driven | Sentry, Datadog, Jaeger | Production incidents |
| Time-Travel | rr, Redux DevTools, git bisect | Complex state |
| Chaos Engineering | Chaos Monkey, Gremlin | Intermittent under load |
| Statistical | Delta debugging, A/B | Small % of cases |

---

### Step 5: Instrumentation

**Strategic points:**
1. Entry points of affected functionality
2. Decision nodes (if/switch)
3. State mutations
4. External boundaries (API, DB, file I/O)
5. Error handling paths

**Production-safe:**
- Feature-flagged debug logging
- Sampling-based profiling (1%)
- Read-only debug endpoints
- Gradual traffic shifting

---

### Step 6: Production-Safe Techniques

**Zero/minimal customer impact:**
- Feature flags for debug logging by user_id
- Dark launches (parallel execution, discard result)
- Canary deployments (1% → 10% → 100%)
- Read-only admin endpoints with rate limiting

---

### Step 7: Root Cause Analysis

**Analyze:**
- Execution path reconstruction with timing
- External dependency responses
- Timing/sequence diagram
- Code smell detection
- Similar bug patterns in codebase

**Document:**
```markdown
## Root Cause

**Immediate Cause**: [Proximate trigger]
**Root Cause**: [Underlying issue]
**Contributing Factors**: [What allowed this]
**Evidence**: [Logs, metrics, code diff]
```

---

### Step 8: Fix Implementation

**Fix proposal template:**
- Code changes (file, lines, before/after)
- Impact assessment (benefits, risks)
- Risk level (Low/Medium/High)
- Test coverage needs (new tests, updates)

🚨 **Standard-Debug exits here** - deliver root cause, fix, validation plan.

---

### Step 9: Validation

**Post-fix verification:**
- Run test suite with coverage
- Load testing
- Performance comparison (before/after)
- Canary deployment monitoring

**Success criteria:**
- Tests pass (100%)
- No performance regression
- Error rate unchanged or decreased
- Code review approved

---

### Step 10: Prevention

**Prevent recurrence:**
1. **Regression tests** - specific to this bug pattern
2. **Monitoring alerts** - early detection
3. **Knowledge base** - document pattern and solution
4. **Runbook** - incident response steps
5. **Coding standards** - prevent similar issues

🎯 **Deep-RCA complete** - full report with prevention strategy.

---

## Output Format

### Issue Summary
- Error, Frequency, Impact, Environment, Severity

### Root Cause
- Immediate cause, Root cause, Contributing factors, Evidence

### Fix Proposal
- Code changes, Impact assessment, Risk level, Test coverage

### Validation Plan
- Test execution, Performance comparison, Deployment strategy

### Prevention (Deep-RCA only)
- Regression tests, Monitoring, Documentation, Process improvements

---

**Start debugging workflow for the provided issue.**
