---
version: 1.0.3
description: Safe dependency upgrade orchestration with breaking change management and security-first prioritization
argument-hint: [--security-only] [--mode quick|standard|deep] [--strategy incremental|batch]
category: framework-migration
purpose: Upgrade dependencies safely with automated testing, compatibility validation, and rollback procedures
execution_time:
  quick: "15-25 minutes - Security patches only"
  standard: "30-60 minutes - Safe minor/patch upgrades"
  deep: "1-3 hours - Major version upgrades with migration guides"
color: green
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task
external_docs:
  - dependency-strategies-guide.md
  - testing-strategies.md
  - rollback-procedures.md
agents:
  primary:
    - framework-migration:legacy-modernizer
  conditional:
    - agent: comprehensive-review:security-auditor
      trigger: argument "--security-only" OR pattern "security|vulnerability|CVE"
    - agent: unit-testing:test-automator
      trigger: pattern "test|coverage"
  orchestrated: true
tags: [dependency-management, security-patches, version-upgrades, breaking-changes, compatibility, npm, pip]
---

# Dependency Upgrade Orchestrator

**Safe, incremental dependency upgrades with security-first prioritization, breaking change management, and comprehensive compatibility validation**

## Execution Modes

Parse `$ARGUMENTS` to determine mode (default: standard):

### Quick Mode (15-25 min)
**Scope**: Security patches only
- Vulnerability scan
- Critical CVE identification
- Security patch application
- Smoke testing

**Output**: Security vulnerabilities patched

**Use When**: Emergency security updates, CVE remediation

### Standard Mode (30-60 min) - RECOMMENDED
**Scope**: Safe minor and patch upgrades
- All Quick mode deliverables
- Outdated dependency analysis
- Minor/patch version upgrades
- Compatibility validation
- Full test suite execution

**Output**: Dependencies updated, tests passing

**Use When**: Regular maintenance, dependency freshness, non-breaking updates

### Deep Mode (1-3 hours)
**Scope**: Major version upgrades with migration
- All Standard mode deliverables
- Major version upgrade planning
- Breaking change analysis
- Migration guide application
- Performance benchmarking
- Automated codemod execution

**Output**: Major versions upgraded with migration documentation

**Use When**: Framework major upgrades, modernization efforts, technical debt reduction

---

## Configuration Options

- `--security-only`: Only upgrade packages with known vulnerabilities
- `--mode <quick|standard|deep>`: Execution depth (default: standard)
- `--strategy <incremental|batch>`: One-at-a-time or grouped upgrades
- `--dry-run`: Show upgrade plan without applying changes
- `--interactive`: Prompt for confirmation on each upgrade
- `--skip-tests`: Skip test execution (not recommended)

---

## Your Task

**Target**: $ARGUMENTS
**Mode**: [Auto-detected or specified]

---

## Phase 1: Dependency Analysis & Audit

**Objective**: Identify outdated dependencies and security vulnerabilities

### Step 1A: Vulnerability Scanning

**Scan for security vulnerabilities**:

```bash
# Node.js/npm
npm audit --json > audit-report.json

# Python/pip
pip-audit --format json > audit-report.json

# Ruby
bundle audit check --format json > audit-report.json
```

**Use Task tool** with `subagent_type="comprehensive-review:security-auditor"`:

```
Analyze security vulnerabilities in dependencies for: $ARGUMENTS

Scan for:
- Known CVEs (Common Vulnerabilities and Exposures)
- Severity levels (Critical/High/Medium/Low)
- Affected packages and versions
- Remediation paths
- Transitive dependency vulnerabilities

Generate prioritized security fix list.

Reference: docs/framework-migration/dependency-strategies-guide.md (security-first strategy)
```

**Expected Output**: Vulnerability report with CVSS scores and fix priorities

### Step 1B: Outdated Dependency Detection

**Identify outdated packages**:

```bash
# Node.js/npm
npm outdated --json > outdated-packages.json

# Python/pip
pip list --outdated --format=json > outdated-packages.json

# Ruby
bundle outdated --parseable > outdated-packages.txt
```

**Categorize by update type**:
- **Patch** (1.2.3 → 1.2.4): Bug fixes only, safe to upgrade
- **Minor** (1.2.4 → 1.3.0): New features, backward compatible
- **Major** (1.3.0 → 2.0.0): Breaking changes, requires migration

### Step 1C: Compatibility Matrix Generation

**Check peer dependencies and compatibility**:

```
Use Task tool with subagent_type="framework-migration:legacy-modernizer"

Generate compatibility matrix for: $ARGUMENTS

Analyze:
- Peer dependency conflicts
- Version range compatibility
- Framework ecosystem compatibility (e.g., React 18 + Router 6)
- Transitive dependency resolution

Identify incompatible combinations.

Reference: docs/framework-migration/dependency-strategies-guide.md (compatibility matrix)
```

**Expected Output**: Compatibility matrix highlighting conflicts

**Success Criteria for Phase 1**:
- ✅ All vulnerabilities identified with CVSS scores
- ✅ Outdated packages categorized by type (patch/minor/major)
- ✅ Compatibility conflicts documented
- ✅ Upgrade priority order established

**🚨 Quick Mode Exits Here** - Deliver security vulnerability report

---

## Phase 2: Upgrade Strategy & Prioritization

**Objective**: Select upgrade strategy and prioritize updates

### Step 2A: Strategy Selection

**Decision Tree**:

```
Are there critical security vulnerabilities (CVSS > 7.0)?
├─ Yes → SECURITY-FIRST STRATEGY (immediate upgrade)
└─ No → Number of major version upgrades > 5?
    ├─ Yes → INCREMENTAL STRATEGY (one major at a time)
    └─ No → BATCH STRATEGY (group similar updates)
```

**Upgrade Strategies**:

**1. Security-First** (Critical CVEs present):
- Priority: Patch vulnerabilities immediately
- Approach: Security packages first, rest later
- Risk: Low (targeted fixes)
- Timeline: Same day for critical, 1 week for high

**2. Incremental** (Many major versions):
- Priority: Minimize risk through small steps
- Approach: One major version at a time
- Risk: Low (thorough testing between steps)
- Timeline: 1-2 weeks per major version

**3. Batch** (Low-risk updates):
- Priority: Efficiency for minor/patch updates
- Approach: Group similar packages
- Risk: Low (backward compatible)
- Timeline: 1 sprint cycle

**📚 See**: [Dependency Strategies Guide](../docs/framework-migration/dependency-strategies-guide.md) for detailed strategy descriptions

### Step 2B: Prioritization

**Upgrade Priority Order**:

1. **P0 - Critical Security** (CVSS > 7.0):
   - Immediate upgrade required
   - Examples: RCE, authentication bypass, SQL injection

2. **P1 - High Security** (CVSS 4.0-7.0):
   - Upgrade within 1 week
   - Examples: XSS, CSRF, information disclosure

3. **P2 - Core Framework** (React, Vue, Angular, Django):
   - Framework drives ecosystem
   - Upgrade before dependent packages

4. **P3 - Direct Dependencies**:
   - Packages used directly in code
   - Higher impact than transitive deps

5. **P4 - Minor/Patch Updates**:
   - Low risk, backward compatible
   - Can be batched

6. **P5 - Dev Dependencies**:
   - Testing/build tools
   - Lower priority than production deps

### Step 2C: Breaking Change Analysis

**For major version upgrades, identify breaking changes**:

```
Use Task tool with subagent_type="framework-migration:legacy-modernizer"

Analyze breaking changes for major upgrades in: $ARGUMENTS

For each major version upgrade:
- Fetch changelog and migration guide
- Identify deprecated APIs
- Find removed features
- Document renamed exports
- Estimate migration effort (hours)

Generate breaking change summary.

Reference: docs/framework-migration/dependency-strategies-guide.md (breaking change catalog)
```

**Expected Output**: Breaking change report with migration effort estimates

**Success Criteria for Phase 2**:
- ✅ Upgrade strategy selected and justified
- ✅ Priority order established (P0-P5)
- ✅ Breaking changes documented for major upgrades
- ✅ Timeline estimate provided

---

## Phase 3: Upgrade Execution

**Objective**: Apply upgrades safely with validation

### Step 3A: Backup & Checkpoint

**Create rollback point**:

```bash
# Git checkpoint
git add package.json package-lock.json
git commit -m "checkpoint: pre-upgrade dependency versions"
git tag pre-upgrade-$(date +%Y%m%d-%H%M%S)

# Backup lock files
cp package-lock.json package-lock.json.backup
cp yarn.lock yarn.lock.backup 2>/dev/null || true
```

### Step 3B: Apply Upgrades (Incremental Strategy)

**Upgrade one major version at a time**:

```bash
# Example: React 16 → 17 → 18 (not 16 → 18 directly)

# Step 1: React 16 → 17
npm install react@17 react-dom@17
npm test
git commit -m "upgrade: React 16 → 17"

# Step 2: React 17 → 18
npm install react@18 react-dom@18
npm test
git commit -m "upgrade: React 17 → 18"
```

**Between each major version**:
1. Run full test suite
2. Check for deprecation warnings
3. Fix breaking changes
4. Commit before next upgrade

### Step 3C: Apply Upgrades (Batch Strategy)

**Group similar packages**:

```bash
# Batch 1: Security patches
npm upgrade @package/a@^1.2.3 @package/b@^2.3.4

# Batch 2: Minor updates
npm upgrade @package/c @package/d --latest

# Run tests after each batch
npm test
```

### Step 3D: Codemod Application (for major upgrades)

**Apply automated transformations**:

**React 17 → 18**:
```bash
# Update to new root API
npx react-codemod update-react-imports
```

**Vue 2 → 3**:
```bash
# Migrate to Composition API
npx @vue/compat-migration
```

**Python 2 → 3**:
```bash
# Automated syntax updates
2to3 -w src/
```

**📚 See**: [Migration Patterns Library](../docs/framework-migration/migration-patterns-library.md) for codemod examples

### Step 3E: Manual Breaking Change Fixes

**For changes requiring manual intervention**:

1. **Read migration guide** for package
2. **Identify affected code locations** (grep for deprecated APIs)
3. **Apply fixes** following guide recommendations
4. **Update tests** if assertions change
5. **Verify functionality** with E2E tests

**Example - React 17 → 18 Event Handling**:
```javascript
// React 17: Event delegation at document level
document.addEventListener('click', handleClick);

// React 18: Event delegation at root container
// No code change needed - automatic migration
// But remove manual document event listeners if any
```

**Success Criteria for Phase 3**:
- ✅ All upgrades applied successfully
- ✅ package.json and lock files updated
- ✅ Build completes without errors
- ✅ No deprecation warnings (or documented)

---

## Phase 4: Testing & Validation

**Objective**: Ensure upgrades don't break functionality

### Step 4A: Test Suite Execution

**Run comprehensive tests**:

```bash
# Unit tests
npm test

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# Type checking (TypeScript)
npx tsc --noEmit
```

**Use Task tool** with `subagent_type="unit-testing:test-automator"`:

```
Run full test suite and fix failures for: $ARGUMENTS

Execute:
- Unit tests (jest, pytest, rspec)
- Integration tests
- End-to-end tests (Cypress, Playwright)
- Visual regression tests (if available)

For test failures:
- Analyze root cause
- Fix code or update test expectations
- Re-run until 100% pass rate

Reference: docs/framework-migration/testing-strategies.md
```

**Expected Output**: 100% test pass rate

### Step 4B: Performance Benchmarking

**Compare before/after performance**:

```bash
# Run benchmarks
npm run benchmark > upgraded-performance.json

# Compare with baseline
diff baseline-performance.json upgraded-performance.json
```

**Performance Criteria**:
- Bundle size: < +10% increase
- Build time: < +20% increase
- Runtime performance: < +10% latency increase
- Memory usage: < +15% increase

**If performance regressed unacceptably**:
```
Use Task tool with subagent_type="full-stack-orchestration:performance-engineer"

Analyze and optimize performance regression in: $ARGUMENTS

Profile:
- Bundle analysis (webpack-bundle-analyzer)
- Runtime profiling (Chrome DevTools)
- Memory profiling (heap snapshots)

Optimize:
- Tree-shaking configuration
- Code splitting
- Lazy loading
- Remove unused dependencies
```

### Step 4C: Smoke Testing

**Test critical user flows**:
- [ ] Application starts without errors
- [ ] User authentication works
- [ ] Core features functional
- [ ] API integrations operational
- [ ] No console errors or warnings

**Success Criteria for Phase 4**:
- ✅ All tests passing (100% pass rate)
- ✅ Performance within acceptable limits
- ✅ Smoke tests successful
- ✅ No new errors or warnings

**🚨 Standard Mode Complete** - Dependencies upgraded and validated

---

## Phase 5: Deployment & Monitoring (Deep Mode Only)

**Objective**: Deploy upgrades safely with monitoring

### Step 5A: Canary Deployment

**Progressive rollout**:

1. **5% traffic** (Day 1):
   - Deploy to canary environment
   - Monitor error rates, latency
   - If stable, continue

2. **25% traffic** (Day 2):
   - Increase canary traffic
   - Monitor for 24 hours
   - Check business metrics

3. **50% → 100%** (Days 3-4):
   - Continue progressive rollout
   - Full deployment if stable

**Rollback Triggers**:
- Error rate > 5% (vs baseline < 1%)
- p95 latency > 2x baseline
- Any critical functionality broken

**📚 See**: [Rollback Procedures](../docs/framework-migration/rollback-procedures.md)

### Step 5B: Monitoring Dashboard

**Key metrics**:
- **Error Rate**: Track overall and per-endpoint errors
- **Latency**: p50, p95, p99 response times
- **Throughput**: Requests per second
- **Dependencies**: External API health
- **Business KPIs**: Conversion rate, revenue

**Alert Configuration**:
```yaml
# Example - Prometheus alerts
- alert: DependencyUpgradeErrorRateHigh
  expr: (rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) > 0.05
  for: 5m
  annotations:
    summary: "Error rate >5% after dependency upgrade"
    description: "Consider rollback"
```

### Step 5C: Documentation Updates

**Update project documentation**:
- `package.json`: Updated versions
- `CHANGELOG.md`: List upgraded dependencies
- `README.md`: Update installation/setup if needed
- `UPGRADING.md`: Migration notes for team
- Runbooks: Update for new dependency behavior

**Example CHANGELOG entry**:
```markdown
## [2.1.0] - 2025-11-07

### Dependencies
- Upgraded React 17.0.2 → 18.2.0
  - Migration: Updated to new root API
  - Breaking: Event delegation changes (automatic)
- Upgraded webpack 5.75.0 → 5.88.2
  - Security: Patched CVE-2023-xxxx
- Upgraded eslint 8.45.0 → 8.52.0
  - New rules enabled: no-unused-private-class-members
```

**Success Criteria for Phase 5**:
- ✅ Deployed to production successfully
- ✅ Monitoring shows stable metrics
- ✅ No incidents requiring rollback
- ✅ Documentation updated

---

## Phase 6: Continuous Dependency Management (Deep Mode Only)

**Objective**: Establish automated dependency maintenance

### Step 6A: Automated Dependency Updates

**Setup Dependabot or Renovate**:

**Dependabot Configuration** (`.github/dependabot.yml`):
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    groups:
      production-dependencies:
        patterns:
          - "*"
        exclude-patterns:
          - "dev-*"
      security-updates:
        patterns:
          - "*"
        update-types:
          - "security"
    ignore:
      # Major versions need manual review
      - dependency-name: "react"
        update-types: ["version-update:semver-major"]
```

**Renovate Configuration** (`.github/renovate.json`):
```json
{
  "extends": ["config:base"],
  "schedule": ["after 10pm every weekday", "before 5am every weekday"],
  "packageRules": [
    {
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true,
      "automergeType": "pr",
      "requiredStatusChecks": ["test", "build"]
    },
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "labels": ["major-upgrade", "needs-review"]
    }
  ]
}
```

### Step 6B: Dependency Health Dashboard

**Track dependency metrics**:
- Outdated package count
- Security vulnerability count (by severity)
- Average dependency age
- Maintenance status (deprecated packages)

### Step 6C: Upgrade Cadence Policy

**Establish regular upgrade schedule**:

**Weekly**: Security patches (automated via Dependabot)
**Monthly**: Minor version updates (batch upgrade)
**Quarterly**: Major version upgrades (planned with testing)
**As-Needed**: Critical CVEs (immediate response)

**Success Criteria for Phase 6**:
- ✅ Automated dependency bot configured
- ✅ Regular upgrade schedule established
- ✅ Dependency health monitored
- ✅ Team trained on dependency management

**🎯 Deep Mode Complete** - Automated dependency management established

---

## Safety Guarantees

**This command will**:
- ✅ Create backup before upgrades (git tag)
- ✅ Upgrade incrementally (one major at a time)
- ✅ Run full test suite after each upgrade
- ✅ Validate performance against baseline
- ✅ Provide instant rollback capability
- ✅ Document all changes in CHANGELOG

**This command will NEVER**:
- ❌ Skip security vulnerability patches
- ❌ Upgrade multiple major versions simultaneously
- ❌ Deploy without test validation
- ❌ Ignore performance regressions
- ❌ Lose previous dependency state
- ❌ Apply untested upgrades to production

---

## Usage Examples

### Security-Only Updates
```bash
# Patch only packages with known vulnerabilities
/deps-upgrade --security-only

# Dry run to see security fixes
/deps-upgrade --security-only --dry-run
```

### Standard Maintenance
```bash
# Safe minor and patch updates
/deps-upgrade

# With test coverage enforcement
/deps-upgrade --mode standard

# Interactive mode (confirm each upgrade)
/deps-upgrade --interactive
```

### Major Version Upgrades
```bash
# Comprehensive major version upgrade
/deps-upgrade --mode deep

# Incremental strategy (one major at a time)
/deps-upgrade --mode deep --strategy incremental

# Batch strategy (group compatible updates)
/deps-upgrade --mode deep --strategy batch
```

---

**Execute safe dependency upgrades with security-first prioritization, comprehensive testing, and instant rollback capability**
