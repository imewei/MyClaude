---
description: Comprehensive legacy system modernization using Strangler Fig pattern
  with multi-agent orchestration
triggers:
- /legacy-modernize
- comprehensive legacy system modernization
allowed-tools: [Bash, Read, Write, Edit, Task, Glob, Grep]
version: 1.0.0
---



## User Input
Input arguments pattern: `<legacy-system-path> [--strategy strangler-fig|big-bang|branch-by-abstraction] [--mode quick|standard|deep]`
The agent should parse these arguments from the user's request.

# Legacy Code Modernization

$ARGUMENTS

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

Strangler fig pattern for gradual replacement of legacy components with continuous business operations.

## Phase 1: Assessment (Sequential)

**Legacy analysis:** Technical debt inventory (outdated deps, deprecated APIs, security vulns, performance bottlenecks, anti-patterns). Component complexity scores (1-10), dependency mapping, DB coupling analysis. Identify quick wins vs complex refactoring targets.

**Dependency mapping:** Internal module deps, external services, shared DB schemas, cross-system data flows. Identify integration points requiring facades/adapters. Highlight circular deps and tight coupling.

**Risk assessment:** Business criticality (revenue impact), user traffic, data sensitivity, regulatory requirements, fallback complexity. Priority scoring: (Business Value × 0.4) + (Technical Risk × 0.3) + (Quick Win × 0.3). Define rollback strategies.

## Phase 2: Test Coverage (Parallel Execution)

> **Orchestration Note**: Execute coverage analysis, contract testing, and data preparation concurrently.

**Coverage analysis:** Existing test coverage. Use coverage tools for untested paths, missing integration/E2E tests. For <40% coverage, generate characterization tests capturing current behavior. Create test harness for safe refactoring.

**Contract testing:** Consumer-driven contracts for APIs, message queues, DB schemas. Contract verification in CI/CD. Performance baselines for response times/throughput to validate SLAs.

**Test data:** Data generation for edge cases, data masking for sensitive info, DB refresh procedures. Monitoring for data consistency during migration.

## Phase 3: Incremental Migration (Iterative/Parallel)

> **Orchestration Note**: Infrastructure setup can proceed in parallel with logic extraction.

**Infrastructure:** API gateway for traffic routing. Feature flags for gradual rollout (env vars or feature service). Proxy layer with routing rules (URL patterns, headers, user segments). Circuit breakers and fallbacks. Observability dashboard for dual-system monitoring.

**Modernization:** Extract business logic from legacy. Implement using modern patterns (DI, SOLID). Ensure backward compatibility through adapters. Maintain data consistency with event sourcing or dual writes. Follow 12-factor app principles.

**Security:** OAuth 2.0/JWT auth, RBAC, input validation/sanitization, SQL injection prevention, XSS protection, secrets management. OWASP top 10 compliance. Security headers, rate limiting.

## Phase 4: Performance & Rollout (Sequential)

**Performance:** Load tests simulating production traffic. Measure response times, throughput, resource utilization. Identify regressions. Optimize: DB queries with indexing, caching (Redis/Memcached), connection pooling, async processing. Validate SLAs.

**Progressive rollout:** 5% → 25% → 50% → 100% with 24h observation periods. Auto-rollback triggers: error >1%, latency >2x baseline, business metric degradation.

## Phase 5: Completion (Sequential)

**Decommissioning:** Verify no deps via traffic analysis (30d at 0% traffic). Archive legacy code with functionality docs. Update CI/CD to remove legacy builds. Clean unused DB tables, remove deprecated APIs. Document retained legacy with sunset timeline.

**Documentation:** Architectural diagrams (before/after), API docs with migration guides, runbooks for dual-system operation, troubleshooting guides, lessons learned, developer onboarding guide, technical decisions and trade-offs.

## Options

- `--parallel-systems`: Keep both running indefinitely
- `--big-bang`: Full cutover after validation
- `--by-feature`: Migrate features vs technical components
- `--database-first`: DB modernization before app layer
- `--api-first`: Modernize API while maintaining legacy backend

## Success

All high-priority components modernized with >80% test coverage, zero unplanned downtime, performance maintained/improved (P95 ≤110% baseline), security vulns reduced >90%, technical debt improved >60%, successful 30d operation post-migration, complete docs enabling <1w onboarding
