---
version: "2.1.0"
description: Unified code migration and legacy modernization with Strangler Fig pattern
argument-hint: <action> <path> [options]
category: engineering-suite
execution-time:
  quick: "30-60m: Assessment + strategy"
  standard: "1-2w: Component migration"
  deep: "2-6mo: Enterprise transformation"
color: blue
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep, WebFetch, Bash(uv:*)]
external-docs:
  - migration-patterns-library.md
  - strangler-fig-playbook.md
  - testing-strategies.md
  - rollback-procedures.md
tags: [migration, modernization, strangler-fig, refactoring, technical-debt]
---

# Code Modernization

$ARGUMENTS

## Actions

| Action | Description |
|--------|-------------|
| `migrate` | Framework/language migration (React→Hooks, Python2→3, Angular upgrade) |
| `legacy` | Full legacy system modernization with Strangler Fig pattern |
| `assess` | Assessment only - analyze complexity, risks, strategy recommendation |

**Examples:**
```bash
/modernize migrate src/components --target react-hooks
/modernize legacy src/old-system --strategy strangler-fig
/modernize assess src/ --mode quick
```

## Options

- `--target <framework>`: react-hooks, python3, angular15, nextjs, vue3
- `--strategy <pattern>`: strangler-fig (incremental), big-bang (full cutover), branch-by-abstraction (feature-by-feature)
- `--mode <depth>`: quick (assessment only), standard (single component), deep (enterprise)
- `--parallel-systems`: Keep both systems running indefinitely
- `--by-feature`: Migrate by feature vs technical components
- `--database-first`: Modernize DB before app layer
- `--api-first`: Modernize API while maintaining legacy backend

## Phase 1: Assessment

**Analysis:**
- Current framework/language versions and architectural patterns
- Technical debt inventory (outdated deps, deprecated APIs, security vulns)
- Component complexity scores (1-10), dependency mapping
- Integration points, DB coupling, circular dependencies

**Risk Assessment:**
- Business criticality (revenue impact), user traffic, data sensitivity
- Priority: (Business Value × 0.4) + (Technical Risk × 0.3) + (Quick Win × 0.3)
- Define rollback strategies for each component

**Strategy Selection:**
| Criteria | Recommended Strategy |
|----------|---------------------|
| Complexity >7/10 | Strangler Fig (incremental, zero downtime) |
| Timeline <2 weeks | Big Bang (fast, requires coordination) |
| Otherwise | Branch by Abstraction (feature-by-feature) |

| Strategy | Pros | Cons |
|----------|------|------|
| Strangler Fig | Zero downtime, instant rollback | Dual system complexity |
| Big Bang | Fast, no dual system | High risk, hard rollback |
| Branch by Abstraction | Continuous deploy, feature isolation | Abstraction overhead |

**Quick mode:** Deliver assessment + strategy recommendation, exit

## Phase 2: Test Coverage

**Characterization Tests:**
- Golden master tests (capture current workflows)
- Snapshot tests (UI components)
- Contract tests (API schemas, DB queries, external services)
- Behavior tests (business logic validation)

**Coverage Requirements:**
- Target >80% coverage before migration
- For <40% coverage, generate characterization tests first
- Create test harness for safe refactoring

**Performance Baseline:**
```bash
npm run benchmark > baseline-performance.json
```
Capture: p50/p95/p99 latency, throughput, memory, CPU

## Phase 3: Migration

**Automated Codemods:**
- React Class→Hooks: `npx react-codemod class-to-hooks`
- Python 2→3: `2to3 -w src/`
- Angular: `ng update`
- Vue 2→3: `npx @vue/compat-migration`

**Manual Migration (complex components):**
1. Read original implementation
2. Extract business rules (separate from framework)
3. Implement in target framework
4. Preserve identical behavior
5. Update tests for syntax changes

**Strangler Fig Infrastructure:**
- API gateway for traffic routing
- Feature flags for gradual rollout
- Proxy layer with routing rules (URL patterns, headers, user segments)
- Circuit breakers and fallbacks
- Observability dashboard for dual-system monitoring

**Security Checklist:**
- OAuth 2.0/JWT auth, RBAC
- Input validation/sanitization
- SQL injection prevention, XSS protection
- Secrets management, OWASP top 10 compliance

## Phase 4: Validation

```bash
npm test && npm run test:integration && npm run test:e2e
npm run benchmark > migrated-performance.json
diff baseline-performance.json migrated-performance.json
```

**Acceptable Thresholds:**
- Response time <110% of baseline
- Memory <120% of baseline
- P95 latency ≤110% baseline

**Security Audit:** Input validation, auth patterns, dependency vulns, OWASP compliance

**Standard mode complete**

## Phase 5: Progressive Rollout

**Rollout Stages:**
| Stage | Traffic | Duration | Focus |
|-------|---------|----------|-------|
| 1 | 5% | 24h | Error rates, latency |
| 2 | 25% | 24h | Business metrics |
| 3 | 50% | 24h | Scale validation |
| 4 | 100% | - | Full deployment |

**Auto-Rollback Triggers:**
- Error rate >5% (>1% for critical systems)
- P95 latency >2x baseline
- Data corruption detected
- Critical functionality broken
- Business metric degradation

## Phase 6: Completion (Deep Mode)

**Optimization:**
- Leverage target framework features
- Caching strategies (Redis/Memcached)
- Bundle optimization, code splitting
- DB query optimization with indexing
- Target: 20-30% performance improvement

**Legacy Decommissioning:**
- Verify no dependencies via traffic analysis (30d at 0%)
- Archive legacy code with functionality docs
- Update CI/CD to remove legacy builds
- Clean unused DB tables, remove deprecated APIs

**Documentation:**
- Architecture diagrams (before/after)
- API docs with migration guides
- Runbooks for dual-system operation
- Lessons learned and migration playbook
- Developer onboarding guide (<1w ramp-up target)

## Success Criteria

- All target components migrated with >80% test coverage
- Zero unplanned downtime during migration
- Performance maintained/improved (P95 ≤110% baseline)
- Security vulnerabilities reduced >90%
- Technical debt score improved >60%
- Successful 30d operation post-migration
- Complete documentation enabling rapid onboarding
