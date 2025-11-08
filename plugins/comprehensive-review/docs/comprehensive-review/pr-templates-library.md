# Pull Request Templates Library

Comprehensive collection of PR templates for different change types, ensuring consistent, high-quality pull request descriptions across your team.

## Standard PR Template Structure

Every PR should include:
1. **Summary**: What and why in 1-2 sentences
2. **Change Description**: Detailed what changed
3. **Motivation**: Why this change is needed
4. **Testing**: How it was tested
5. **Checklist**: Review items
6. **Additional Context**: Screenshots, links, notes

---

## Template 1: Feature Addition

```markdown
## Feature: [Feature Name]

### Summary
[1-2 sentence description of the feature and its value]

### User Story
**As a** [user type]
**I want** [feature/capability]
**So that** [benefit/value]

### Acceptance Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]
- [ ] [Criterion 4]

### What Changed
[Detailed description of technical changes]

**New Components/Modules**:
- `ComponentA`: [Description]
- `ServiceB`: [Description]

**Modified Components**:
- `ExistingComponent`: [What changed and why]

**Database Changes**:
- [ ] No database changes
- [ ] New tables: [list]
- [ ] Schema changes: [list]
- [ ] Migration script: [link]

### Technical Implementation

**Architecture**:
[High-level architecture description or diagram]

**Key Design Decisions**:
1. [Decision 1]: [Rationale]
2. [Decision 2]: [Rationale]

**Dependencies Added**:
- `library-name@version`: [Why needed]

### How Has This Been Tested?

**Test Coverage**: [X]% (lines), [Y]% (branches)

**Test Types**:
- [ ] Unit tests: [Description]
- [ ] Integration tests: [Description]
- [ ] E2E tests: [Description]
- [ ] Manual testing: [Steps]

**Test Scenarios Covered**:
1. [Scenario 1]: ✅ Pass
2. [Scenario 2]: ✅ Pass
3. [Edge case 1]: ✅ Pass

### Performance Impact

**Metrics**:
- Load time: [Before] → [After]
- API response time: [Before] → [After]
- Memory usage: [Before] → [After]

**Optimization Notes**:
[Any performance considerations or optimizations made]

### UI/UX Changes

**Screenshots**:
| Before | After |
|--------|-------|
| [Screenshot] | [Screenshot] |

**Responsive Design**:
- [ ] Tested on desktop
- [ ] Tested on tablet
- [ ] Tested on mobile

**Accessibility**:
- [ ] Keyboard navigation works
- [ ] Screen reader compatible
- [ ] Color contrast meets WCAG AA
- [ ] ARIA labels added

### Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes documented below:

[If breaking changes, describe impact and migration path]

### Deployment Notes
[Any special deployment considerations, configuration changes, or migration steps]

### Documentation
- [ ] README updated
- [ ] API documentation updated
- [ ] User guide updated
- [ ] Code comments added for complex logic

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Documentation updated
- [ ] No console.log or debug code
- [ ] No hardcoded values
- [ ] Error handling implemented
- [ ] Security considerations addressed

### Related Issues
Closes #[issue-number]
Related to #[issue-number]

### Demo
[Link to demo, video, or live preview if applicable]

---
**Estimated Review Time**: [X] minutes
**Risk Level**: 🟢 Low / 🟡 Medium / 🟠 High
**Priority**: P0 / P1 / P2 / P3
```

---

## Template 2: Bug Fix

```markdown
## Bug Fix: [Brief Description]

### Issue
**Reported in**: #[issue-number]
**Severity**: 🔴 Critical / 🟠 High / 🟡 Medium / 🟢 Low
**Affected Versions**: [version range]
**User Impact**: [Description of impact on users]

### Summary
[1-2 sentence description of the bug and fix]

### Root Cause Analysis

**What Went Wrong**:
[Detailed explanation of the underlying cause]

**How It Manifested**:
[Observable symptoms and error messages]

**Why It Wasn't Caught**:
- [ ] Edge case not covered in tests
- [ ] Integration issue between components
- [ ] Timing/race condition
- [ ] Environment-specific issue
- [ ] Other: [explanation]

### Steps to Reproduce (Before Fix)
1. [Step 1]
2. [Step 2]
3. [Step 3]
**Expected**: [Expected behavior]
**Actual**: [Actual buggy behavior]

### Solution Implemented

**Approach**:
[Description of the fix]

**Code Changes**:
- Modified: `file1.js`: [What changed]
- Modified: `file2.py`: [What changed]

**Why This Solution**:
[Rationale for chosen approach vs alternatives]

**Alternatives Considered**:
1. [Alternative 1]: Rejected because [reason]
2. [Alternative 2]: Rejected because [reason]

### Testing & Verification

**Reproduction Verified**:
- [ ] Bug reproduced on [environment/version]
- [ ] Bug confirmed with original steps

**Fix Verification**:
- [ ] Original issue no longer occurs
- [ ] Edge cases tested
- [ ] No regressions introduced

**Tests Added**:
- [ ] Unit test for root cause
- [ ] Integration test for symptom
- [ ] Regression test to prevent recurrence

**Test Coverage**: [Before: X%] → [After: Y%]

### Regression Testing
[List of areas tested to ensure no side effects]
- [ ] [Area 1]: No regressions
- [ ] [Area 2]: No regressions

### Performance Impact
- [ ] No performance change
- [ ] Performance improved: [metrics]
- [ ] Negligible performance cost: [metrics]

### Deployment Strategy
- [ ] Can deploy immediately
- [ ] Requires database migration
- [ ] Requires configuration change
- [ ] Requires coordinated deployment
- [ ] Requires feature flag

**Rollback Plan**:
[How to rollback if issues arise]

### Monitoring & Alerting
[Any new monitoring or alerts added to prevent recurrence]

### Post-Deployment Verification Steps
1. [Step 1]
2. [Step 2]
3. [Success criteria]

### Documentation Updates
- [ ] Known issues page updated
- [ ] Release notes include fix
- [ ] Troubleshooting guide updated

---
**Urgency**: 🔴 Critical (ASAP) / 🟠 High (This Sprint) / 🟡 Medium (Next Sprint)
**Estimated Review Time**: [X] minutes
```

---

## Template 3: Refactoring

```markdown
## Refactoring: [Component/Module Name]

### Summary
[1-2 sentence description of what was refactored and why]

### Motivation

**Problems Addressed**:
- [Problem 1]: [Description]
- [Problem 2]: [Description]

**Benefits**:
- ✅ Improved [metric]: [Before → After]
- ✅ Reduced [metric]: [Before → After]
- ✅ Enhanced [aspect]: [Description]

### Changes Made

**Structural Changes**:
- Extracted: [Component A] → [New location/structure]
- Renamed: [Old name] → [New name]
- Merged: [Components X, Y] → [Unified component]
- Deleted: [Removed component] (no longer needed)

**Code Quality Improvements**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cyclomatic Complexity | X | Y | Z% |
| Code Duplication | X% | Y% | Z% |
| Test Coverage | X% | Y% | +Z% |
| Lines of Code | X | Y | Z fewer |

### Compatibility & Breaking Changes

- [x] No breaking changes
- [ ] Public API unchanged
- [ ] Backward compatible
- [ ] No performance regression

**If breaking changes**:
[List changes and migration guide]

### Testing Strategy

**Tests Updated**:
- [X] All existing tests passing
- [X] Tests refactored to match new structure
- [ ] New tests added for extracted components

**Test Coverage**: [Before: X%] → [After: Y%]

**Verification**:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] End-to-end tests pass
- [ ] Performance tests pass

### Performance Impact

**Benchmarks**:
| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| [Operation 1] | Xms | Yms | ±Z% |
| [Operation 2] | Xms | Yms | ±Z% |

- [ ] No performance change
- [ ] Performance improved
- [ ] Negligible performance cost (<5%)

### Deployment Plan
- [ ] Deploy as normal (no special considerations)
- [ ] Requires configuration update
- [ ] Recommend deploying with [related PR]

---
**Code Complexity**: Before: [X] → After: [Y]
**Estimated Review Time**: [X] minutes
```

---

## Template 4: Performance Optimization

```markdown
## Performance Optimization: [Area/Component]

### Summary
[1-2 sentence description of performance improvement]

### Performance Problem

**Symptoms**:
- [Metric 1]: [Current value] (Target: [desired value])
- [Metric 2]: [Current value] (Target: [desired value])

**Impact**:
- User experience: [Description]
- System load: [Description]
- Cost: [Description if applicable]

### Root Cause Analysis

**Profiling Results**:
[Attach or link to profiling data]

**Bottleneck Identified**:
[Detailed description of where time/resources are spent]

### Optimization Approach

**Changes Made**:
1. [Optimization 1]: [Description and rationale]
2. [Optimization 2]: [Description and rationale]

**Techniques Used**:
- [ ] Caching (type: [in-memory/Redis/CDN])
- [ ] Database query optimization
- [ ] Algorithm improvement (O(n²) → O(n log n))
- [ ] Lazy loading
- [ ] Resource pooling
- [ ] Parallel processing
- [ ] Code-level optimization
- [ ] Infrastructure scaling
- [ ] Other: [description]

### Performance Results

**Benchmarks** (averaged over [N] runs):
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Response Time (p50) | Xms | Yms | Z% faster |
| Response Time (p95) | Xms | Yms | Z% faster |
| Response Time (p99) | Xms | Yms | Z% faster |
| Throughput | X req/s | Y req/s | +Z% |
| CPU Usage | X% | Y% | -Z% |
| Memory Usage | X MB | Y MB | -Z% |
| Database Queries | X | Y | -Z queries |

**Load Testing Results**:
[Results from load testing showing performance under stress]

### Trade-offs

**Benefits**:
- ✅ [Benefit 1]
- ✅ [Benefit 2]

**Costs/Trade-offs**:
- ⚠️ Increased code complexity: [Mitigation]
- ⚠️ Additional infrastructure: [Cost/benefit]
- ⚠️ Other: [Description and mitigation]

### Verification

**Testing**:
- [ ] Benchmarks run and documented
- [ ] Load tests pass
- [ ] All functional tests pass
- [ ] No regressions in other areas

**Monitoring**:
[What metrics to watch post-deployment]

---
**Performance Improvement**: [X]% faster
**Estimated Review Time**: [X] minutes
```

---

## Template 5: Security Fix

```markdown
## Security Fix: [Brief Description]

⚠️ **SECURITY SENSITIVE - Review with care**

### Security Issue

**Severity**: 🔴 Critical / 🟠 High / 🟡 Medium / 🟢 Low
**CVSS Score**: [Score] ([Vector string])
**CVE ID**: CVE-YYYY-XXXXX (if assigned)

**Vulnerability Type**:
- [ ] SQL Injection
- [ ] Cross-Site Scripting (XSS)
- [ ] Authentication Bypass
- [ ] Authorization Issue
- [ ] Information Disclosure
- [ ] Cryptographic Weakness
- [ ] Dependency Vulnerability
- [ ] Other: [description]

**Attack Vector**:
[Description of how vulnerability could be exploited]

**Impact**:
[What could an attacker achieve]

**Affected Components**:
- [Component 1]
- [Component 2]

**Disclosure Timeline**:
- Discovered: [Date]
- Fix developed: [Date]
- Deployment planned: [Date]

### Solution Implemented

**Fix Description**:
[Detailed description of security fix without revealing exploit details]

**Security Controls Added**:
- [Control 1]: [Description]
- [Control 2]: [Description]

**Defense-in-Depth Layers**:
1. [Layer 1]
2. [Layer 2]

### Testing

**Security Testing**:
- [ ] Exploit attempt fails
- [ ] Automated security scans pass
- [ ] Penetration test passed
- [ ] Code security review completed

**Regression Testing**:
- [ ] All functional tests pass
- [ ] No new security issues introduced

### Deployment

**Urgency**: Deploy [immediately/within 24h/this week]

**Deployment Strategy**:
- [ ] Standard deployment
- [ ] Hotfix process
- [ ] Coordinated with security team

**Post-Deployment**:
- [ ] Monitor for exploit attempts
- [ ] Verify fix in production
- [ ] Customer notification (if needed)

---
**Exploitability**: Easy / Medium / Hard
**Recommendation**: Prioritize for immediate deployment
```

---

## Template 6: Documentation Update

```markdown
## Documentation: [Topic/Area]

### Summary
[What documentation is being added/updated and why]

### Changes Made

**New Documentation**:
- [ ] README
- [ ] API documentation
- [ ] User guide
- [ ] Architecture decision record (ADR)
- [ ] Runbook
- [ ] Other: [description]

**Updated Documentation**:
- [File 1]: [What changed]
- [File 2]: [What changed]

### Motivation
[Why this documentation is needed]

### Target Audience
- [ ] End users
- [ ] Developers
- [ ] Operations team
- [ ] Product team
- [ ] Other: [description]

### Review Checklist
- [ ] Accurate and up-to-date
- [ ] Clear and concise
- [ ] Examples provided
- [ ] Links valid
- [ ] Formatting correct
- [ ] Spelling/grammar checked

---
**Estimated Review Time**: 5-10 minutes
```

---

## Template 7: Dependency Update

```markdown
## Dependency Update: [Package Name]

### Summary
Update [package] from [old-version] to [new-version]

### Motivation
- [ ] Security vulnerability (CVE-YYYY-XXXXX)
- [ ] Bug fixes
- [ ] New features needed
- [ ] Performance improvements
- [ ] Maintenance (staying current)
- [ ] Required by other dependency

**Security Vulnerabilities Fixed** (if applicable):
- CVE-YYYY-XXXXX: [Description] (CVSS: X.X)

### Changes in New Version

**Breaking Changes**:
- [ ] No breaking changes
- [ ] Breaking changes addressed:
  - [Change 1]: [How we adapted]
  - [Change 2]: [How we adapted]

**New Features**:
- [Feature 1]: [Will we use this?]

**Bug Fixes**:
- [Fix 1]: [Relevant to us?]

**Full Changelog**: [Link]

### Code Changes Required
- [ ] No code changes needed
- [ ] Minor adjustments: [description]
- [ ] Significant refactoring: [description]

### Testing
- [ ] All existing tests pass
- [ ] Manual testing performed
- [ ] No regressions found

**Areas Tested**:
- [Area 1]: ✅ Pass
- [Area 2]: ✅ Pass

### Deployment
- [ ] Standard deployment
- [ ] No special considerations

---
**Risk Level**: 🟢 Low / 🟡 Medium / 🟠 High
**Recommended Action**: Approve and merge
```

---

## Template 8: Configuration Change

```markdown
## Configuration: [What's Being Configured]

### Summary
[Brief description of configuration change]

### Changes Made

**Configuration Files Modified**:
- `config.yml`: [Changes]
- `.env.example`: [Changes]

**New Environment Variables**:
- `VARIABLE_NAME`: [Purpose, format, required/optional]

**Modified Environment Variables**:
- `EXISTING_VAR`: [What changed and why]

**Removed Environment Variables**:
- `OLD_VAR`: [Why it's no longer needed]

### Deployment Instructions

**Development**:
```bash
# Add to your .env file
VARIABLE_NAME=value
```

**Staging**:
[Specific instructions for staging]

**Production**:
[Specific instructions for production]

### Validation
[How to verify configuration is correct]

### Rollback
[How to rollback if issues occur]

### Security Considerations
- [ ] No sensitive data in version control
- [ ] Secrets properly encrypted
- [ ] Access controls verified

---
**Deployment Coordination**: [Any coordination needed with ops/platform team]
```

---

## Template Selection Guide

| PR Type | Template to Use |
|---------|----------------|
| New feature | Feature Addition |
| Bug fix | Bug Fix |
| Code cleanup | Refactoring |
| Speed improvement | Performance Optimization |
| Security issue | Security Fix |
| Docs only | Documentation Update |
| Package update | Dependency Update |
| Environment changes | Configuration Change |

## Quick PR Description Generator

For small PRs, use this minimal format:

```markdown
## [Type]: [Brief Description]

**What**: [One sentence]
**Why**: [One sentence]
**How**: [One sentence]

**Testing**: [How tested]

- [ ] Tests pass
- [ ] Self-review complete
```

---

Use these templates as starting points. Customize based on your team's needs and project requirements. The goal is consistency and completeness, not bureaucracy.
