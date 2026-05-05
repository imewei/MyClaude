---
name: smart-debug
category: debugging
purpose: AI-assisted debugging with automated RCA, pattern recognition, production-safe techniques
description: Scientific computing debugger for NaN/inf propagation, JAX JIT trace failures, Julia dispatch ambiguities, shape mismatches, numerical instability. For general software bugs, use superpowers:systematic-debugging first.
execution-modes:
  quick-triage: "5-10min"
  standard-debug: "15-30min"
  deep-rca: "30-60min"
external-docs: [debugging-patterns-library.md, rca-frameworks-guide.md, observability-integration-guide.md]
tags: [debugging, rca, observability, production]
argument-hint: <error-description> [--quick-triage|--standard-debug|--deep-rca] [--production]
allowed-tools: [Read, Bash, Edit, Task, Monitor]
---

# AI-Assisted Debugging (Scientific)

> **SEE ALSO:** For general software bugs (null pointer, timeout, auth failures, race conditions), use `superpowers:systematic-debugging` — it enforces structured pre-fix root-cause discipline.
> Use this command for **scientific computing failures**: NaN/inf propagation, JAX JIT compilation errors (`TracerBoolConversionError`, `ConcretizationTypeError`), Julia dispatch ambiguities, numerical instability, shape/dtype mismatches, MCMC divergence, gradient explosion, GPU OOM errors, and domain-specific correctness failures.

$ARGUMENTS

## Modes

| Mode | Time | Output |
|------|------|--------|
| `--quick-triage` | 5-10min | 3-5 hypotheses |
| standard | 15-30min | Root cause + fix |
| `--deep-rca` | 30-60min | Full RCA + prevention |

## Examples

```bash
# Quick triage of a specific error
/smart-debug "TypeError: Cannot read property 'id' of undefined in UserService.getProfile" --quick-triage

# Standard debug with production context
/smart-debug "API latency spike on /api/orders endpoint, p99 jumped from 200ms to 3s" --standard-debug --production

# Deep root cause analysis
/smart-debug "Intermittent 502 errors under load, ~5% of requests fail" --deep-rca --production
```

## Workflow

### 1. Triage
Parse error, reproduction, environment (dev/prod), pattern (flaky/consistent)
Match: NullPointer, Timeout, MemLeak, Race, Deadlock, Auth, RateLimit, JSON, FileIO, InfiniteLoop, Injection, TypeCoercion, Config, Async, CORS
Scientific: NaN/Inf, ShapeMismatch, TypeInstability, JITTraceError, DispatchAmbiguity, NumericalOverflow, MCMCDivergence, GradientExplosion, OOMError, SeedNonReproducibility
Output: Severity (P0-P3), top 3 causes, strategy

### 2. Observability (Production)
Gather: Error tracking (Sentry/Rollbar), APM (Datadog/NewRelic p95/p99), Traces (Jaeger/Zipkin), Logs (ELK/Splunk)
Query window: incident ± 30min, trace by correlation ID

### 3. Hypotheses
Generate 3-5 ranked hypotheses: probability score, evidence, falsification criteria
Use: 5 Whys, Fault Tree, Timeline, Fishbone
Categories: Logic, State, Integration, Resources, Config, Data
🚨 **Quick-Triage exits** - deliver hypotheses

### 4. Strategy
Reproducible locally? → Interactive (VS Code/pdb/DevTools)
Production-only? → Observability (Sentry/Datadog/Jaeger)
Complex state? → Time-Travel (rr/Redux/git bisect)
Load-dependent? → Chaos (Monkey/Gremlin)
Small %? → Statistical (Delta debugging)

### 5. Instrumentation
Points: Entry, decisions (if/switch), state mutations, external boundaries, error paths
Production-safe: Feature flags, 1% sampling, read-only endpoints, gradual traffic shift

### 6. RCA
Analyze: Execution path, dependencies, timing, code smells, similar patterns
Document: Immediate cause, root cause, contributing factors, evidence

### 7. Fix
Code changes, impact, risk (Low/Med/High), test coverage needs
🚨 **Standard-Debug exits** - root cause + fix + validation

### 8. Validation
Run tests, load test, perf comparison, canary deploy
Success: 100% pass, no regression, error rate ↓

### 9. Prevention
Regression tests, monitoring alerts, knowledge base, runbook, coding standards
🎯 **Deep-RCA complete**

## Output

**Issue**: Error, frequency, impact, environment, severity
**Root Cause**: Immediate, root, factors, evidence
**Fix**: Changes, impact, risk, tests
**Validation**: Tests, perf, deployment
**Prevention** (Deep only): Tests, monitoring, docs, process
