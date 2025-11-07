---
version: "1.0.3"
category: "cicd-automation"
command: "/fix-commit-errors"

execution-modes:
  quick-fix:
    description: "Rapid error resolution for urgent CI failures"
    time: "5-10 minutes"
    phases: "Discovery + Fix Application only"
    steps: "Phase 1, 4"
    use-case: "Hot fix production CI, urgent deployments, simple errors"
    auto-fix: "Always enabled"

  standard:
    description: "Full intelligent error resolution with learning"
    time: "15-30 minutes"
    phases: "All 7 phases with multi-agent analysis"
    use-case: "Typical CI failure investigation (default)"
    auto-fix: "Optional with --auto-fix"

  comprehensive:
    description: "Deep analysis with cross-workflow correlation"
    time: "30-60 minutes"
    phases: "All 7 phases + cross-workflow analysis + knowledge base deep dive"
    use-case: "Recurring failures, pattern investigation, knowledge base building"
    learning: "Always enabled with --learn"
    reports: "Detailed post-mortem generated"

documentation:
  multi-agent-system: "../docs/cicd-automation/multi-agent-error-analysis.md"
  error-patterns: "../docs/cicd-automation/error-pattern-library.md"
  fix-strategies: "../docs/cicd-automation/fix-strategies.md"
  knowledge-base: "../docs/cicd-automation/knowledge-base-system.md"
  examples: "../docs/cicd-automation/fix-examples.md"

description: Automatically analyzes GitHub Actions failures, identifies root causes through advanced pattern matching, correlates errors across workflows, learns from successful fixes, applies intelligent solutions, validates changes, and reruns workflows—all with adaptive automation that improves over time.
allowed-tools: Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(yarn:*), Bash(pip:*), Bash(cargo:*), Bash(go:*), Bash(make:*), Bash(grep:*), Bash(find:*)
argument-hint: [workflow-id|commit-sha|pr-number] [--auto-fix] [--learn] [--mode=quick-fix|standard|comprehensive]
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
---

# Intelligent GitHub Actions Failure Resolution System

Automatically resolve CI/CD failures through multi-agent error analysis, UltraThink reasoning, and adaptive learning from successful fixes.

## Available Skills

**iterative-error-resolution**: Complete framework for analyzing GitHub Actions failures, applying intelligent fixes, and iterating until zero errors. Automatically detects error categories (dependency, build, test, runtime, CI-specific), applies proven fixes from knowledge base, executes iterative validation loops, and learns from outcomes. Use with `--auto-fix` flag for continuous error resolution.

## Multi-Agent Error Analysis System

| Agent | Primary Role | Key Techniques | Output |
|-------|--------------|----------------|--------|
| Log Fetcher & Parser | Retrieve and structure error logs | GitHub API, log parsing, error extraction | Structured error data with timestamps, locations, stack traces |
| Pattern Matcher & Categorizer | Classify errors by type | Regex patterns, ML classification, error taxonomy | Error categories (dependency/build/test/runtime/CI), severity ratings |
| Root Cause Analyzer | Determine underlying causes | Dependency graph analysis, UltraThink reasoning, historical correlation | Root cause identification, regression analysis, cascading failure detection |
| Knowledge Base Consultant | Apply proven solutions | Historical fix lookup, success rate analysis, Bayesian confidence | Recommended solutions ranked by success probability |
| Solution Generator | Generate fix strategies | UltraThink reasoning, code generation, risk assessment | Executable fix code with confidence scores and rollback plans |

[→ Detailed Agent Implementation](../docs/cicd-automation/multi-agent-error-analysis.md)

---

## Phase 1: Failure Detection & Data Collection

### Repository Context
```bash
# GitHub authentication and repository validation
gh auth status && gh repo view --json nameWithOwner -q .nameWithOwner
git branch --show-current && git log -1 --oneline
```

### Workflow Run Identification

**Arguments**: `$ARGUMENTS` → Resolve to target run:
- Workflow run ID → Analyze specific run
- Commit SHA → Find runs for commit
- PR number → Analyze PR check runs
- Empty → Latest failed run

**Flags**:
- `--auto-fix`: Apply fixes automatically with iterative resolution
- `--learn`: Update knowledge base with outcomes
- `--mode`: Execution mode (quick-fix|standard|comprehensive)

### Metadata Collection
```bash
# For each failed run, fetch:
gh run list --status failure --limit 10
gh run view $RUN_ID --json workflowName,headSha,headBranch,event,conclusion
gh run view $RUN_ID --log-failed > error_logs.txt
```

Collect: Run ID, workflow name, commit SHA, branch, author, duration, job statuses, re-run history (flaky test detection)

---

## Phase 2: Multi-Agent Error Analysis

### Error Pattern Detection

**Agent 2** categorizes errors using advanced pattern matching [→ Complete Error Taxonomy](../docs/cicd-automation/error-pattern-library.md):

#### Dependency Errors
- **NPM/Yarn**: `npm ERR!`, `ERESOLVE`, `404 Not Found`, `peer dependency`, `gyp ERR!`
- **Python/Pip**: `ERROR: Could not find`, `VersionConflict`, `No module named`
- **Rust/Cargo**: `error[E0425]`, `error[E0277]`, `unresolved import`
- **Go**: `undefined:`, `cannot find package`, `module not found`

#### Build & Compilation Errors
- TypeScript: `TS[0-9]+:`, type mismatches
- Webpack: `Module not found`, `Can't resolve`
- C/C++: `undefined reference`, linker errors

#### Test Failures
- Jest/Vitest: `FAIL`, test timeouts, assertion errors
- Pytest: `FAILED`, `AssertionError`
- Go Test: `FAIL:`, `panic:`

#### Runtime & CI Errors
- Memory: `OOM`, `out of memory`
- Network: `ECONNREFUSED`, `ETIMEDOUT`
- Cache: `Failed to restore cache`
- Setup: `setup-node`, `setup-python` failures

[→ Complete Pattern Library with 100+ Patterns](../docs/cicd-automation/error-pattern-library.md)

### Root Cause Analysis

**Agent 3** performs multi-dimensional analysis:

1. **Technical Analysis**: What failed? Why? When did it start? Where? How does it propagate?
2. **Historical Analysis**: Compare with successful runs, find regression point
3. **Correlation Analysis**: Systemic vs job-specific, intermittent vs consistent
4. **Environmental Analysis**: OS/version/timing/resource-specific failures

```bash
# Compare with successful runs
gh run list --status success --limit 5
gh run list --branch main --status success --limit 1
```

[→ Root Cause Methodology](../docs/cicd-automation/multi-agent-error-analysis.md#root-cause-analyzer)

---

## Phase 3: UltraThink Intelligence Layer

### Deep Reasoning Framework

**1. Problem Space Analysis**
- Current state: Workflow failing on `[jobs]`, error type `[category]`, failure frequency, impact radius
- Constraints: Time pressure, risk tolerance, resource availability, dependency chain

**2. Multi-Perspective Solution Design**
- **Engineering**: Quick fix vs root fix, risk assessment, testing strategy
- **DevOps**: Pipeline health, cost impact, workflow optimization, monitoring
- **Team**: Developer experience, technical debt, knowledge sharing, process improvement

**3. Solution Synthesis** - For each candidate solution, evaluate:
```
Solution: Install with --legacy-peer-deps

PROS: ✅ Immediate fix (90% success), low risk, preserves versions
CONS: ❌ Doesn't fix root cause, hides compatibility issues
RISK: 5% breaking change, easy reversion
RECOMMENDATION: Apply as immediate fix, follow up with root cause
```

**4. Adaptive Learning**: Pattern recognition, context-aware decision making, confidence calibration
```python
if error_seen_before and success_rate > 0.8:
    confidence = "HIGH" → "Auto-apply with monitoring"
elif success_rate > 0.5:
    confidence = "MEDIUM" → "Apply in test branch first"
else:
    confidence = "LOW" → "Manual review required"
```

[→ Complete UltraThink Methodology](../docs/cicd-automation/multi-agent-error-analysis.md#ultrathink-reasoning)

---

## Phase 4: Automated Fix Application

### Fix Execution Strategy by Risk Level

#### Level 1: Configuration Fixes (Safest - Auto-apply)
```bash
# Workflow YAML updates
sed -i 's/node-version: 16/node-version: 18/' .github/workflows/*.yml
sed -i 's/npm ci/npm ci --legacy-peer-deps/' .github/workflows/*.yml

# Package manager fixes
rm -rf node_modules package-lock.json && npm install
npm update [package]@[version]
go mod tidy
```

#### Level 2: Code Fixes (Moderate Risk - Validate)
```bash
# Fix missing imports, update types, update snapshots
npm test -- -u
sed -i 's/timeout: 5000/timeout: 10000/' test/*.test.js
```

#### Level 3: Complex Fixes (Manual Review)
- Major version updates, API signature changes, database migrations → Generate PR with proposed changes

[→ Complete Fix Strategy Library](../docs/cicd-automation/fix-strategies.md)

### Validation Loop
```bash
# After each fix:
npm test 2>&1 | tee test-output.log
npm run build 2>&1 | tee build-output.log
npm run lint 2>&1 | tee lint-output.log

# If all pass → commit and push
# If any fail → rollback and try next solution
```

### Semantic Commits
```bash
git commit -m "fix(deps): resolve npm peer dependency conflict

- Add --legacy-peer-deps to CI workflow
- Resolves GitHub Actions run #${RUN_ID}
- Error pattern: ERESOLVE dependency tree

Auto-fixed by fix-commit-errors command"
```

---

## Phase 5: Workflow Re-execution & Monitoring

### Trigger Rerun & Watch
```bash
git push origin $(git branch --show-current)  # Trigger new run
gh run watch  # Real-time monitoring

# Adaptive response:
# ✅ Success → Update knowledge base, increase confidence
# ❌ Failure → Try next solution, decrease confidence
```

### Iterative Resolution Loop

When `--auto-fix` is enabled, the iterative-error-resolution engine:
1. Analyzes all errors in failed run
2. Categorizes and prioritizes fixes by confidence score
3. Applies fixes and commits changes
4. Triggers new workflow run and monitors
5. **Repeats until zero errors or max iterations (default: 5)**
6. Updates knowledge base with outcomes

[→ Iterative Resolution Algorithm](../docs/cicd-automation/fix-strategies.md#iterative-approach)

---

## Phase 6: Knowledge Base Learning System

### Knowledge Base Structure

Stored in `.github/fix-commit-errors/knowledge.json`:
```json
{
  "error_patterns": [{
    "id": "npm-eresolve-001",
    "pattern": "ERESOLVE.*peer dependency",
    "category": "dependency_conflict",
    "solutions": [{"action": "npm_install_legacy_peer_deps", "success_rate": 0.85}],
    "occurrences": 20
  }],
  "statistics": {"total_errors_analyzed": 150, "auto_fixed": 85, "success_rate": 0.65}
}
```

### Learning Mechanism
1. Query knowledge base for matching patterns
2. Retrieve solutions ranked by success rate
3. Apply highest-confidence solution first
4. Track outcome and update success rates
5. Extract new patterns from successful fixes

[→ Complete Knowledge Base System](../docs/cicd-automation/knowledge-base-system.md)

---

## Phase 7: Comprehensive Reporting

### Real-Time Progress
```markdown
🔍 Analyzing GitHub Actions failure...
Run ID: 12345 | Workflow: CI/CD | Branch: feature/new-api

📊 Error Analysis:
- Category: Dependency Resolution (npm-eresolve-001, 95% confidence)
- Root Cause: Peer dependency conflict (react@17 vs react@18)

🧠 Knowledge Base: ✅ Pattern found (20 times), Best solution: 90% success

🎯 Applying Solution: Update workflow with --legacy-peer-deps

✅ SUCCESS! Workflow passed!
📈 Knowledge base updated | Confidence: 90% → 91%
```

### Detailed Fix Report

Generate `.github/fix-commit-errors/reports/run-${RUN_ID}.md` with:
- Summary (status, time to resolution, iterations, confidence)
- Error details (classification, pattern ID, severity)
- Root cause analysis
- Solution applied (ID, confidence, risk, changes made, rationale)
- Validation results
- Follow-up recommendations
- Knowledge base impact
- Rollback instructions

[→ Report Templates and Examples](../docs/cicd-automation/fix-examples.md)

---

## Execution Parameters

### Required
- **Target**: `[workflow-id|commit-sha|pr-number]` (optional - defaults to latest failed run)

### Optional Flags
- `--auto-fix`: Enable automatic fix application with iterative resolution
- `--learn`: Update knowledge base from successful fixes
- `--mode`: Execution mode (quick-fix|standard|comprehensive) - default: standard

### Examples
```bash
# Analysis only (default)
/fix-commit-errors

# Auto-fix latest failed run
/fix-commit-errors --auto-fix

# Fix specific run with learning
/fix-commit-errors 12345 --auto-fix --learn

# Fix PR checks in comprehensive mode
/fix-commit-errors PR#123 --auto-fix --mode=comprehensive
```

---

## Success Criteria

### Phase-Specific Outcomes
- **Phase 1**: Target run identified, metadata collected, authentication verified
- **Phase 2**: Errors categorized with >90% pattern match confidence
- **Phase 3**: Root cause identified, solution strategies ranked
- **Phase 4**: Fix applied with validation passing, changes committed
- **Phase 5**: Workflow rerun successful or next solution attempted
- **Phase 6**: Knowledge base updated with outcome (success/failure)
- **Phase 7**: Detailed report generated with recommendations

### Overall Success
- ✅ Resolution rate >65% (auto-fixed errors / total errors)
- ✅ Average time to fix <15 minutes
- ✅ Knowledge base growing (patterns learned over time)
- ✅ Confidence accuracy >80% (predictions match actual outcomes)
- ✅ Zero regressions (fixes don't break working code)

---

## Safety Guarantees

✅ **Never breaks working code**: All fixes validated locally first, rollback available, confidence threshold for auto-apply

✅ **Full transparency**: Every action logged, all changes committed with clear messages, detailed reasoning provided

✅ **Learning from mistakes**: Failed fixes decrease confidence, patterns refined based on outcomes, alternative solutions promoted

---

## Now Execute

**Arguments Received**: `$ARGUMENTS`

Begin intelligent analysis of GitHub Actions failures. Use all 5 agents in parallel, apply UltraThink reasoning, consult knowledge base, and generate comprehensive fix strategy.

**Mode Selection**:
- **quick-fix**: Urgent production CI failures (5-10 min)
- **standard**: Typical CI investigation with learning (15-30 min) - DEFAULT
- **comprehensive**: Deep analysis for recurring issues (30-60 min)

**Remember**:
- Safety first: validate before applying
- Learn from outcomes: update knowledge base
- Improve over time: adaptive confidence scoring
- Full transparency: detailed reporting
- Automate intelligently: human oversight for risky changes

Let's fix those failing workflows! 🚀
