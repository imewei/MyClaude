---
description: Automatically analyzes GitHub Actions failures, identifies root causes through advanced pattern matching, correlates errors across workflows, learns from successful fixes, applies intelligent solutions, validates changes, and reruns workflows—all with adaptive automation that improves over time.
allowed-tools: Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(yarn:*), Bash(pip:*), Bash(cargo:*), Bash(go:*), Bash(make:*), Bash(grep:*), Bash(find:*)
argument-hint: [workflow-id|commit-sha|pr-number] [--auto-fix] [--learn]
color: red
agents:
  primary:
    - devops-security-engineer
  conditional:
    - agent: code-quality
      trigger: pattern "test.*fail|lint.*error|quality"
    - agent: fullstack-developer
      trigger: pattern "npm|yarn|webpack|build.*error" OR files "package.json"
    - agent: ai-systems-architect
      trigger: pattern "model|inference|ml.*pipeline" OR files "*.h5|*.pkl"
  orchestrated: false

# MCP Integration (ENHANCED)
mcp-integration:
  profile: github-workflow

  mcps:
    - name: github
      priority: critical
      preload: true
      config:
        use_conditional_requests: true  # ETags for rate limiting
        batch_api_calls: true

    - name: serena
      priority: medium
      preload: true
      config:
        analyze_yaml: true
        local_log_analysis: true

    - name: memory-bank
      priority: critical  # UPGRADED from low (2/5) to critical (5/5)
      operations: [read, write]
      cache_patterns:
        - "ci_failure:{workflow}:{job}"
        - "ci_failure:{workflow}:{job}:analysis"
        - "ci_fix_history:{pattern_hash}"
        - "ci_fix_history:{pattern_hash}:solution"
        - "workflow_patterns:{workflow}"
        - "failure_correlation:{error_signature}"
      ttl:
        ci_failures: 15552000  # 180 days (6 months)
        ci_fixes: 31536000  # 365 days (1 year)
        workflow_patterns: 15552000  # 180 days

  learning:
    enabled: true
    pattern_matching: true
    success_tracking: true
    confidence_scoring: true
    auto_fix_threshold: 0.9  # Only auto-fix if >90% confidence
    correlation_analysis: true  # Analyze failure patterns across workflows
---

# Intelligent GitHub Actions Failure Resolution System

## Phase 0: GitHub Actions Discovery & Authentication

### Authentication Check
- GitHub CLI status: !`gh auth status 2>&1 | head -5`
- Current repository: !`gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "Not in a GitHub repository"`
- User permissions: !`gh api user -q .login 2>/dev/null || echo "Not authenticated"`

### Repository Context
- Current branch: !`git branch --show-current 2>/dev/null`
- Last commit: !`git log -1 --oneline 2>/dev/null`
- Remote URL: !`git remote get-url origin 2>/dev/null`
- Uncommitted changes: !`git status --short 2>/dev/null`

---

## Phase 1: Failure Detection & Data Collection

### Recent Workflow Runs Analysis
- All recent runs: !`gh run list --limit 20 2>/dev/null`
- Failed runs: !`gh run list --status failure --limit 10 2>/dev/null`
- Latest run status: !`gh run list --limit 1 2>/dev/null`

### Targeted Run Identification
**Arguments Received**: `$ARGUMENTS`

**Resolution Strategy**:
- If `$ARGUMENTS` is workflow run ID → analyze that specific run
- If `$ARGUMENTS` is commit SHA → find runs for that commit
- If `$ARGUMENTS` is PR number → analyze PR check runs
- If `$ARGUMENTS` is empty → analyze latest failed run
- If `--auto-fix` flag → apply fixes automatically
- If `--learn` flag → update knowledge base with successful fixes

### Workflow Metadata Collection
For each failed run, collect:
- Run ID, workflow name, trigger event
- Commit SHA, branch, author
- Run duration, conclusion, status
- Job names and their statuses
- Artifact availability
- Re-run history (detect flaky tests)

---

## Phase 2: Multi-Agent Error Analysis System

### Agent 1: Log Fetcher & Parser
**Mission**: Retrieve and structure all error logs

**Tasks**:
1. Fetch complete logs for all failed jobs
   ```bash
   gh run view $RUN_ID --log-failed
   ```

2. Parse logs into structured format:
   - Timestamp of each error
   - Error type/category
   - File location (if applicable)
   - Stack traces
   - Context lines (before/after error)

3. Extract key error indicators:
   - Exit codes
   - Error messages
   - Warning messages
   - Failed commands
   - Assertion failures

### Agent 2: Pattern Matcher & Categorizer
**Mission**: Classify errors using advanced pattern matching

**Error Category Detection**:

#### Dependency Errors
- **NPM/Yarn**: !`echo "Checking for npm patterns..."`
  - `npm ERR!` → npm install failures
  - `ERESOLVE` → dependency conflicts
  - `404 Not Found` → package not found
  - `peer dependency` → version mismatches
  - `gyp ERR!` → native module build failures

- **Python/Pip**: !`echo "Checking for pip patterns..."`
  - `ERROR: Could not find` → missing package
  - `VersionConflict` → dependency version issues
  - `No module named` → import errors
  - `pip._vendor` → pip internal errors

- **Rust/Cargo**: !`echo "Checking for cargo patterns..."`
  - `error[E0425]` → unresolved name
  - `error[E0277]` → trait not implemented
  - `cargo:warning` → compilation warnings
  - `unresolved import` → use statement errors

- **Go**: !`echo "Checking for go patterns..."`
  - `undefined:` → missing imports
  - `cannot find package` → dependency issues
  - `module not found` → missing go.mod entry

#### Build & Compilation Errors
- TypeScript: `TS[0-9]+:`, `error TS`, type mismatches
- ESLint: `ESLint.*error`, rule violations
- Webpack: `Module not found`, `Can't resolve`
- Babel: `SyntaxError: Unexpected token`
- C/C++: `error:.*undefined reference`, linker errors

#### Test Failures
- Jest/Vitest: `FAIL`, `● Test suite failed`, assertion errors
- Pytest: `FAILED`, `AssertionError`, fixture failures
- Go Test: `FAIL:`, `panic:`, test timeouts
- RSpec: `Failure/Error:`, expectation failures

#### Runtime Errors
- Memory: `OOM`, `out of memory`, `heap exhausted`
- Timeout: `ETIMEDOUT`, `timeout of.*exceeded`
- Network: `ECONNREFUSED`, `getaddrinfo ENOTFOUND`
- Permission: `EACCES`, `permission denied`
- File System: `ENOENT`, `no such file or directory`

#### CI-Specific Errors
- Cache failures: `Failed to restore cache`, `cache miss`
- Setup failures: `setup-node`, `setup-python` errors
- Checkout failures: `git checkout failed`
- Artifact failures: `upload-artifact failed`

### Agent 3: Root Cause Analyzer
**Mission**: Determine underlying causes using UltraThink intelligence

**Multi-Dimensional Analysis**:

#### Technical Analysis
For each error, ask:
1. **What failed?** (immediate cause)
2. **Why did it fail?** (root cause)
3. **When did it start failing?** (regression analysis)
4. **Where is the failure?** (file, line, component)
5. **How does it propagate?** (cascading failures)

#### Historical Analysis
```bash
# Compare with successful runs
gh run list --status success --limit 5
# Find when it last worked
gh run list --branch main --status success --limit 1
```

Questions:
- Did this work in a previous commit?
- What changed between last success and first failure?
- Is this a new test or existing test?
- Are there related failures in other workflows?

#### Correlation Analysis
- Multiple jobs failing with same error → systemic issue
- Only one job failing → job-specific configuration
- Intermittent failures → flaky test or race condition
- Consistent failures → deterministic bug

#### Environmental Analysis
- OS-specific: Fails on ubuntu but not macos?
- Version-specific: Fails on Node 18 but not Node 20?
- Timing-specific: Fails during peak hours?
- Resource-specific: Fails with memory/CPU constraints?

### Agent 4: Knowledge Base Consultant
**Mission**: Learn from past fixes and apply proven solutions

**Knowledge Base Structure** (stored in `.github/fix-commit-errors/knowledge.json`):
```json
{
  "error_patterns": [
    {
      "id": "npm-eresolve-001",
      "pattern": "ERESOLVE.*peer dependency",
      "category": "dependency_conflict",
      "root_cause": "Peer dependency version mismatch",
      "solutions": [
        {
          "action": "npm_install_legacy_peer_deps",
          "command": "npm install --legacy-peer-deps",
          "success_rate": 0.85,
          "applications": 12
        },
        {
          "action": "update_package_json",
          "pattern": "Update peer dependency constraints",
          "success_rate": 0.95,
          "applications": 8
        }
      ],
      "last_seen": "2025-10-01T12:00:00Z",
      "occurrences": 20
    }
  ],
  "successful_fixes": [
    {
      "run_id": "12345",
      "error_pattern": "npm-eresolve-001",
      "solution_applied": "npm_install_legacy_peer_deps",
      "commit_sha": "abc123",
      "rerun_successful": true,
      "timestamp": "2025-10-01T12:30:00Z"
    }
  ],
  "statistics": {
    "total_errors_analyzed": 150,
    "auto_fixed": 85,
    "manual_intervention_required": 45,
    "success_rate": 0.65
  }
}
```

**Learning Mechanism**:
1. Query knowledge base for matching error patterns
2. Retrieve solutions ranked by success rate
3. Apply highest-confidence solution first
4. Track outcome for future learning
5. Update success rates based on actual results

### Agent 5: Solution Generator
**Mission**: Generate fix strategies using UltraThink reasoning

**Solution Framework**:

For each identified error, generate solutions with confidence scores:

```
Error: ERESOLVE unable to resolve dependency tree
Confidence: 95% (seen 20 times, fixed 17 times)

Solutions (ranked by success probability):
1. [90%] Install with --legacy-peer-deps flag
2. [75%] Update conflicting package to compatible version
3. [60%] Use npm overrides in package.json
4. [40%] Switch to yarn with resolutions
5. [20%] Manual dependency tree resolution
```

**Solution Categories**:

#### Category 1: Dependency Fixes (High Confidence)
- Install missing dependencies
- Update dependency versions
- Fix peer dependency conflicts
- Clear cache and reinstall
- Update lock files

#### Category 2: Configuration Fixes (Medium Confidence)
- Update workflow YAML configurations
- Fix environment variables
- Correct setup-* action versions
- Update Node/Python/Go versions
- Fix cache configurations

#### Category 3: Code Fixes (Medium Confidence)
- Fix import statements
- Update API calls to match new versions
- Fix type errors
- Update deprecated syntax
- Fix test assertions

#### Category 4: Build Fixes (Medium Confidence)
- Update build scripts
- Fix TypeScript configurations
- Update webpack/vite configs
- Fix babel transformations
- Update compiler flags

#### Category 5: Test Fixes (Lower Confidence)
- Increase timeouts
- Fix flaky test timing
- Update test snapshots
- Mock external dependencies
- Fix test setup/teardown

---

## Phase 3: UltraThink Intelligence Layer

### Deep Reasoning Framework

**1. Problem Space Analysis**

**Current State Assessment**:
- Workflow failing on: `[job_names]`
- Error type: `[category]`
- Failure frequency: `[first_seen → now]`
- Impact radius: `[affected_branches/PRs]`

**Constraint Analysis**:
- Time pressure: Is this blocking deployments?
- Risk tolerance: Production vs dev branch?
- Resource availability: Can we run expensive fixes?
- Dependency chain: What depends on this passing?

**2. Multi-Perspective Solution Design**

#### Engineering Perspective
- **Quick Fix**: What's the fastest way to green?
- **Root Fix**: What's the proper long-term solution?
- **Risk Assessment**: What could this break?
- **Testing Strategy**: How do we verify the fix?

#### DevOps Perspective
- **Pipeline Health**: How does this affect CI/CD?
- **Cost Impact**: Compute time and resource usage
- **Workflow Optimization**: Can we prevent future occurrences?
- **Monitoring**: Should we add alerts?

#### Team Perspective
- **Developer Experience**: Will this block others?
- **Technical Debt**: Are we patching or fixing?
- **Knowledge Sharing**: Document for team learning
- **Process Improvement**: Update development practices?

**3. Solution Synthesis**

For each candidate solution, evaluate:

```
Solution: Install with --legacy-peer-deps

PROS:
✅ Immediate fix (high success rate: 90%)
✅ Low risk (doesn't modify package.json)
✅ Preserves existing dependency versions
✅ Quick to test and validate

CONS:
❌ Doesn't fix root cause
❌ May hide compatibility issues
❌ Not a long-term solution
❌ Needs workflow YAML modification

RISK ASSESSMENT:
- Breaking change probability: 5%
- Reversion difficulty: Easy (git revert)
- Side effects: Minimal
- Long-term maintainability: Medium

RECOMMENDATION: Apply as immediate fix, follow up with root cause analysis
```

**4. Adaptive Learning Integration**

**Pattern Recognition**:
- This error pattern seen `N` times before
- Previous solutions had `X%` success rate
- Similar errors in `[related_projects]`
- Common in `[framework/language]` ecosystem

**Context-Aware Decision Making**:
```python
if error_seen_before and success_rate > 0.8:
    confidence = "HIGH"
    recommendation = "Auto-apply with monitoring"
elif error_seen_before and success_rate > 0.5:
    confidence = "MEDIUM"
    recommendation = "Apply in test branch first"
else:
    confidence = "LOW"
    recommendation = "Manual review required"
```

**5. Second-Order Consequences**

Consider ripple effects:
- Will fixing dependency X break tests that expect old behavior?
- Will updating Node version affect local dev environments?
- Will changing workflow affect deployment pipeline?
- Will this fix apply to all branches or just this one?

---

## Phase 4: Automated Fix Application

### Pre-Fix Safety Protocol

**Safety Checklist**:
- [ ] Create backup branch: `fix/ci-error-${RUN_ID}-backup`
- [ ] Run local tests if available
- [ ] Check for uncommitted changes
- [ ] Verify write permissions
- [ ] Confirm auto-fix mode enabled

### Fix Execution Strategy

#### Level 1: Configuration Fixes (Safest)
✅ Apply automatically with high confidence

**Workflow YAML Updates**:
```bash
# Example: Update Node version
sed -i 's/node-version: 16/node-version: 18/' .github/workflows/*.yml

# Example: Add --legacy-peer-deps flag
sed -i 's/npm ci/npm ci --legacy-peer-deps/' .github/workflows/*.yml

# Example: Update cache action version
sed -i 's/actions\/cache@v2/actions\/cache@v3/' .github/workflows/*.yml
```

**Package Manager Fixes**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Update specific package
npm update [package]@[version]

# Add missing dependency
npm install [package] --save-dev

# Python: Update requirements
pip install --upgrade [package]

# Go: Tidy dependencies
go mod tidy
```

#### Level 2: Code Fixes (Moderate Risk)
⚠️ Apply with validation

**Import Fixes**:
```bash
# Fix missing imports (detected from error messages)
# Example: Add missing React import
find src -name "*.tsx" -exec grep -l "useState" {} \; | \
  xargs sed -i '1i import React, { useState } from "react";'
```

**Type Fixes**:
```bash
# Update TypeScript types based on error messages
# Apply known type fixes from knowledge base
```

**Test Fixes**:
```bash
# Update snapshots
npm test -- -u

# Increase timeout for slow tests
sed -i 's/timeout: 5000/timeout: 10000/' test/*.test.js
```

#### Level 3: Complex Fixes (Manual Review)
🔴 Generate PR with proposed changes

**Breaking Changes**:
- Major version updates
- API signature changes
- Database migrations
- Infrastructure changes

**Uncertain Fixes**:
- First-time error patterns
- Low confidence solutions (<50%)
- Multiple possible solutions
- Conflicting fix strategies

### Validation Loop

After applying each fix:

```bash
# 1. Run affected tests locally if possible
npm test 2>&1 | tee test-output.log

# 2. Check for new errors introduced
grep -i "error\|fail" test-output.log

# 3. Verify build still works
npm run build 2>&1 | tee build-output.log

# 4. Lint check
npm run lint 2>&1 | tee lint-output.log

# If all pass:
#   → Commit and push
# If any fail:
#   → Rollback and try next solution
```

### Commit Strategy

**Semantic Commits**:
```bash
# For dependency fixes
git commit -m "fix(deps): resolve npm peer dependency conflict

- Add --legacy-peer-deps to CI workflow
- Resolves GitHub Actions run #${RUN_ID}
- Error pattern: ERESOLVE dependency tree

Auto-fixed by fix-commit-errors command"

# For configuration fixes
git commit -m "ci: update Node version to 18 in workflow

- Upgrade from Node 16 to 18
- Fixes deprecated action warnings
- Related to run #${RUN_ID}

Auto-fixed by fix-commit-errors command"

# For code fixes
git commit -m "fix: add missing type imports

- Add missing React import in components
- Fixes TypeScript compilation errors
- Resolves run #${RUN_ID}

Auto-fixed by fix-commit-errors command"
```

---

## Phase 5: Workflow Re-execution & Monitoring

### Trigger Workflow Rerun

```bash
# Push fixes to trigger new workflow run
git push origin $(git branch --show-current)

# Alternative: Manually rerun existing workflow
gh run rerun $RUN_ID

# Watch the new run
gh run watch
```

### Real-Time Monitoring

**Watch Strategy**:
```bash
# Start watching immediately
NEW_RUN_ID=$(gh run list --limit 1 --json databaseId -q .[0].databaseId)
echo "Monitoring run: $NEW_RUN_ID"

# Poll every 30 seconds
while true; do
  STATUS=$(gh run view $NEW_RUN_ID --json status,conclusion -q .status)
  CONCLUSION=$(gh run view $NEW_RUN_ID --json conclusion -q .conclusion)

  if [ "$STATUS" = "completed" ]; then
    if [ "$CONCLUSION" = "success" ]; then
      echo "✅ SUCCESS! Fix worked!"
      # Update knowledge base with successful fix
      break
    else
      echo "❌ Still failing. Analyzing new errors..."
      # Fetch new errors and try next solution
      break
    fi
  fi

  sleep 30
done
```

### Adaptive Response

**Success Path**:
1. ✅ Workflow succeeds → Update knowledge base
2. 📊 Record solution success → Increase confidence score
3. 🎯 Learn patterns → Apply to future similar errors
4. 📝 Generate success report

**Failure Path**:
1. ❌ Workflow still fails → Analyze new errors
2. 🔄 Try next solution in ranked list
3. 📊 Record solution failure → Decrease confidence score
4. 🚨 If all solutions exhausted → Alert for manual intervention

---

## Phase 6: Knowledge Base Learning System

### Success Recording

When a fix succeeds, update knowledge base:

```json
{
  "fix_event": {
    "timestamp": "2025-10-03T14:30:00Z",
    "run_id": "67890",
    "error_pattern_id": "npm-eresolve-001",
    "solution_applied": {
      "action": "npm_install_legacy_peer_deps",
      "files_changed": [".github/workflows/ci.yml"],
      "commit_sha": "def456"
    },
    "outcome": "success",
    "rerun_id": "67891",
    "time_to_fix": 180,
    "iterations": 1
  }
}
```

### Pattern Learning

**Automatic Pattern Extraction**:
1. Analyze successful fixes to identify common patterns
2. Extract regex patterns from error messages
3. Correlate solutions with error signatures
4. Build decision trees for solution selection

**Example Pattern Evolution**:
```json
{
  "pattern": "ERESOLVE.*peer dependency.*react@",
  "initial_success_rate": 0.5,
  "current_success_rate": 0.85,
  "learning_iterations": 12,
  "refinements": [
    {
      "date": "2025-09-01",
      "change": "Added version range detection",
      "impact": "+15% success rate"
    },
    {
      "date": "2025-09-15",
      "change": "Prioritized package.json update over flag",
      "impact": "+10% success rate"
    }
  ]
}
```

### Solution Optimization

**A/B Testing for Solutions**:
- When multiple solutions have similar confidence
- Randomly select solution to gather more data
- Track success rates independently
- Promote better-performing solutions

**Confidence Calibration**:
```python
def update_confidence(pattern_id, solution_id, outcome):
    """Bayesian update of solution confidence"""
    prior = get_current_confidence(pattern_id, solution_id)

    if outcome == "success":
        # Increase confidence
        posterior = prior + (1 - prior) * 0.1
    else:
        # Decrease confidence
        posterior = prior * 0.9

    save_confidence(pattern_id, solution_id, posterior)
```

### Cross-Repository Learning

**Optional: Share anonymized patterns**:
```json
{
  "community_patterns": {
    "enable": false,
    "share_anonymized": false,
    "import_public_patterns": true,
    "pattern_sources": [
      "https://github.com/fix-commit-errors/patterns/main.json"
    ]
  }
}
```

---

## Phase 7: Comprehensive Reporting

### Real-Time Progress Updates

```markdown
🔍 Analyzing GitHub Actions failure...

Run ID: 12345
Workflow: CI/CD Pipeline
Branch: feature/new-api
Commit: abc123 (John Doe)
Failed at: 2025-10-03 14:15:30 UTC

📊 Error Analysis:
- Category: Dependency Resolution
- Pattern: npm-eresolve-001 (95% confidence)
- Root Cause: Peer dependency conflict (react@17 vs react@18)
- Impact: Blocks PR merge

🧠 Knowledge Base Lookup:
✅ Pattern found (seen 20 times)
✅ 3 solutions available
✅ Best solution: 90% success rate

🎯 Applying Solution #1:
- Action: Update workflow with --legacy-peer-deps
- Files: .github/workflows/ci.yml
- Risk: Low
- Reversibility: High

⚙️ Executing fix...
✅ Workflow YAML updated
✅ Changes committed (sha: def456)
✅ Pushed to feature/new-api

🔄 Triggering workflow rerun...
⏱️ Waiting for run #12346 to start...
📡 Monitoring progress...

✅ SUCCESS! Workflow passed!

📈 Updating knowledge base...
✅ Solution success recorded
✅ Confidence score: 90% → 91%

📝 Summary Report generated: .github/fix-commit-errors/reports/run-12345.md
```

### Detailed Fix Report

Generate: `.github/fix-commit-errors/reports/run-${RUN_ID}.md`

```markdown
# Fix Report: Run #12345

## Summary
- **Status**: ✅ Successfully Fixed
- **Time to Resolution**: 3 minutes
- **Solution Iterations**: 1
- **Confidence**: High (95%)

## Error Details
**Original Error**:
```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! Found: react@17.0.2
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^18.0.0" from react-dom@18.2.0
```

**Classification**:
- Category: Dependency Resolution
- Subcategory: Peer Dependency Conflict
- Pattern ID: npm-eresolve-001
- Severity: Medium

## Root Cause Analysis
The workflow failed because:
1. Project uses React 17.0.2
2. A dependency requires React 18 as peer dependency
3. npm strict dependency resolution rejects the tree

## Solution Applied
**Solution ID**: npm_install_legacy_peer_deps
**Confidence**: 90%
**Risk Level**: Low

**Changes Made**:
- File: `.github/workflows/ci.yml`
- Line 25: `npm ci` → `npm ci --legacy-peer-deps`

**Rationale**:
The `--legacy-peer-deps` flag tells npm to ignore peer dependency
conflicts and install packages using the npm v6 algorithm. This is
safe because:
- React 17 and 18 have high backward compatibility
- No breaking changes in the components we use
- Allows time for proper React 18 migration

## Validation Results
✅ Build succeeded
✅ Tests passed (245/245)
✅ Linting passed
✅ Type checking passed

## Follow-Up Recommendations
1. 🔔 **High Priority**: Plan React 18 migration
   - Current: react@17.0.2
   - Target: react@18.2.0
   - Estimated effort: 2-4 hours

2. 📚 **Documentation**: Update dependency upgrade guide
3. 🧪 **Testing**: Add peer dependency tests to prevent regressions

## Knowledge Base Impact
- Pattern confidence: 90% → 91% (+1%)
- Total applications: 12 → 13
- Success rate: 11/12 → 12/13

## Related Issues
- Similar to run #11234 (2 weeks ago)
- Related to PR #456 (React 18 upgrade attempt)

## Rollback Instructions
If this fix causes issues:
```bash
git revert def456
git push origin feature/new-api
```
```

---

## Your Task: Intelligent CI Error Resolution

**Arguments Received**: `$ARGUMENTS`

**Execution Plan**:

### Step 1: Identify Target Run
```bash
if [ -z "$ARGUMENTS" ]; then
  # No arguments: get latest failed run
  RUN_ID=$(gh run list --status failure --limit 1 --json databaseId -q .[0].databaseId)
elif [[ "$ARGUMENTS" =~ ^[0-9]+$ ]]; then
  # Numeric argument: treat as run ID or PR number
  if gh run view "$ARGUMENTS" &>/dev/null; then
    RUN_ID="$ARGUMENTS"
  else
    # Might be PR number
    RUN_ID=$(gh pr checks "$ARGUMENTS" --json databaseId -q .[0].databaseId)
  fi
else
  # Could be commit SHA
  RUN_ID=$(gh run list --commit "$ARGUMENTS" --status failure --limit 1 --json databaseId -q .[0].databaseId)
fi
```

### Step 2: Fetch and Analyze Errors
```bash
# Get complete error logs
gh run view $RUN_ID --log-failed > error_logs.txt

# Extract error patterns
grep -E "ERR|ERROR|FAIL|Error|error:" error_logs.txt > errors_only.txt

# Categorize errors using pattern matching
./categorize_errors.sh errors_only.txt
```

### Step 3: Consult Knowledge Base
```bash
# Load knowledge base
KB_FILE=".github/fix-commit-errors/knowledge.json"

# Find matching patterns
./match_patterns.sh errors_only.txt $KB_FILE

# Rank solutions by confidence
./rank_solutions.sh
```

### Step 4: Apply UltraThink Analysis
- Deep reasoning about error context
- Multi-perspective solution evaluation
- Risk assessment for each solution
- Second-order consequence analysis
- Confidence score calculation

### Step 5: Execute Fix (if --auto-fix)
```bash
if [[ "$ARGUMENTS" == *"--auto-fix"* ]]; then
  # Apply top-ranked solution
  ./apply_solution.sh $SOLUTION_ID

  # Validate locally
  npm test || (git reset --hard && exit 1)

  # Commit and push
  git commit -am "fix(ci): auto-fix for run #$RUN_ID"
  git push

  # Watch rerun
  gh run watch
fi
```

### Step 6: Learn (if --learn or fix succeeded)
```bash
if [[ "$ARGUMENTS" == *"--learn"* ]] || [ "$FIX_SUCCEEDED" = true ]; then
  # Update knowledge base
  ./update_knowledge_base.sh $RUN_ID $SOLUTION_ID $OUTCOME

  # Recalculate confidence scores
  ./recalibrate_confidence.sh
fi
```

### Step 7: Generate Report
```bash
# Create detailed report
./generate_report.sh $RUN_ID > ".github/fix-commit-errors/reports/run-${RUN_ID}.md"

# Update statistics
./update_stats.sh
```

---

## Execution Modes

### 1. Analysis Mode (Default)
```bash
/fix-commit-errors
# Analyzes latest failed run, provides recommendations, no automatic fixes
```

### 2. Auto-Fix Mode
```bash
/fix-commit-errors --auto-fix
# Applies high-confidence fixes automatically
```

### 3. Targeted Fix Mode
```bash
/fix-commit-errors 12345 --auto-fix
# Fixes specific run ID
```

### 4. PR Mode
```bash
/fix-commit-errors PR#123
# Analyzes and fixes all failed checks on PR
```

### 5. Learning Mode
```bash
/fix-commit-errors --learn
# Updates knowledge base from recent fixes
```

---

## Success Metrics

Track and report:
- **Resolution Rate**: % of errors fixed successfully
- **Time to Fix**: Average time from error to fix
- **Automation Level**: % of fixes applied automatically
- **Knowledge Growth**: Patterns learned over time
- **Confidence Accuracy**: How well confidence predicts success

---

## Safety Guarantees

✅ **Never breaks working code**
- All fixes validated locally first
- Rollback available for all changes
- Confidence threshold for auto-apply

✅ **Full transparency**
- Every action logged and reported
- All changes committed with clear messages
- Detailed reasoning for each decision

✅ **Learning from mistakes**
- Failed fixes decrease solution confidence
- Patterns refined based on outcomes
- Alternative solutions promoted on failure

---

## Now Execute

Begin intelligent analysis of GitHub Actions failures. Use all agents in parallel, apply ultrathink reasoning, consult knowledge base, and generate comprehensive fix strategy.

**Remember**:
- Safety first: validate before applying
- Learn from outcomes: update knowledge base
- Improve over time: adaptive confidence scoring
- Full transparency: detailed reporting
- Automate intelligently: human oversight for risky changes

Let's fix those failing workflows! 🚀
