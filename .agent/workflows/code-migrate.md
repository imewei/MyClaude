---
description: Orchestrate systematic code migration between frameworks with test-first
  discipline
triggers:
- /code-migrate
- orchestrate systematic code migration
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<source-path> [--target <framework>] [--strategy <pattern>] [--mode quick|standard|deep]`
The agent should parse these arguments from the user's request.

# Code Migration

$ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

**Options:** `--target react-hooks|python3|angular15`, `--strategy big-bang|strangler-fig|branch-by-abstraction`

## Phase 1: Assessment (Analysis)

**Analysis:**
- Current framework/language versions
- Architectural patterns
- Dependencies and integration points
- Complexity metrics

**Risk:** Breaking changes, API compatibility, performance, security, skill gaps

**Strategy selection:**
- Complexity >7/10 → Strangler Fig (incremental)
- Timeline <2 weeks → Big Bang (full cutover)
- Otherwise → Branch by Abstraction (feature-by-feature)

| Strategy | Pros | Cons |
|----------|------|------|
| Big Bang | Fast, no dual system | High risk, hard rollback |
| Strangler Fig | Zero downtime, instant rollback | Dual system complexity |
| Branch by Abstraction | Feature-by-feature, continuous deploy | Abstraction overhead |

🚨 **Quick mode:** Deliver assessment + strategy, exit

## Phase 2: Test Coverage (Parallel Execution)

> **Orchestration Note**: Execute test creation and baseline capture concurrently.

**Characterization tests:** Golden master (workflows), snapshots (UI), contracts (APIs), behavior (logic)

**Contract tests:** API schemas, DB queries, external services, events/messages

**Performance baseline:**
```bash
npm run benchmark > baseline-performance.json
```
Capture: p50/p95/p99 latency, throughput, memory, CPU

**Success:** >80% coverage, all integrations tested, baseline documented

## Phase 3: Migration (Iterative)

**Codemods:**
- React Class→Hooks: `npx react-codemod class-to-hooks`
- Python 2→3: `2to3 -w src/`
- Angular: `ng update`
- Vue 2→3: `npx @vue/compat-migration`

**Manual (complex):**
1. Read original
2. Extract business rules (separate from framework)
3. Implement in target
4. Preserve identical behavior
5. Update tests if syntax changes

**Strangler Fig:** API gateway routing, feature flags, monitoring both systems, rollback procedures

**Success:** Compiles, characterization tests pass, no console errors

## Phase 4: Validation (Parallel Execution)

```bash
npm test && npm run test:integration && npm run test:e2e
npm run benchmark > migrated-performance.json
diff baseline-performance.json migrated-performance.json
```

**Acceptable:** Response time <110%, memory <120% of baseline

**Security:** Input validation, auth patterns, dependency vulns, OWASP compliance

**Success:** Tests passing, performance acceptable, no security regressions

🚨 **Standard mode complete**

## Phase 5: Deployment (Sequential)

**Progressive rollout:**
- Stage 1: 5%, 24h → Monitor errors/latency
- Stage 2: 25%, 24h → Business metrics
- Stage 3: 50%, 24h → Validate at scale
- Stage 4: 100%

**Rollback triggers:** Error >5%, p95 latency >2x baseline, data corruption, critical functionality broken

**Docs:** README, architecture, API, deployment procedures

## Phase 6: Post-Migration (Deep)

**Optimization:**
- Leverage target framework features
- Caching strategies
- Bundle optimization
- Code splitting

Target: 20-30% performance improvement

**Migration playbook:** Lessons learned, reusable patterns, gotchas

**Team training:** Workshops, pair programming, code review guidelines, best practices

## Safety

- Characterization tests before changes
- Backward compatibility
- Instant rollback
- Performance validation
- Security audit

## Examples

```bash
/code-migrate src/components/Dashboard.jsx --target react-hooks
/code-migrate src/legacy/ --target python3
/code-migrate src/ --target react18 --strategy strangler-fig
/code-migrate src/ --target nextjs --mode deep --test-first
```
