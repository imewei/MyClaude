---
description: Advanced codebase cleanup with AST-based unused import removal, dead code elimination, and multi-agent analysis with ultrathink intelligence
allowed-tools: Bash(find:*), Bash(grep:*), Bash(wc:*), Bash(ls:*), Bash(du:*), Bash(git:*), Bash(head:*), Bash(tail:*)
argument-hint: [path] [--auto-fix] [--aggressive]
color: green
---

# Advanced Codebase Cleanup & Dead Code Elimination

## Phase 0: Pre-Flight Checks & Safety

### Safety Validation
- Git status: !`git status --short 2>/dev/null || echo "Not a git repository - BACKUP RECOMMENDED"`
- Uncommitted changes: !`git diff --stat 2>/dev/null || echo "Git not available"`
- Current branch: !`git branch --show-current 2>/dev/null || echo "N/A"`
- Stash available: !`git stash list 2>/dev/null | head -3 || echo "N/A"`

⚠️ **SAFETY PROTOCOL ACTIVATED**
- Backup strategy will be created before any modifications
- All changes will be reversible via git or backup files
- Dry-run analysis performed first, fixes applied only with explicit confirmation

---

## Phase 1: Codebase Discovery & Language Detection

### Project Structure Analysis
- Total files: !`find ${ARGUMENTS:-.} -type f 2>/dev/null | wc -l`
- Code files by language: !`find ${ARGUMENTS:-.} -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.java" -o -name "*.go" -o -name "*.rs" -o -name "*.rb" -o -name "*.php" -o -name "*.cs" -o -name "*.cpp" -o -name "*.c" \) 2>/dev/null | wc -l`
- Project size: !`du -sh ${ARGUMENTS:-.} 2>/dev/null`

### Language Distribution
- Python: !`find ${ARGUMENTS:-.} -name "*.py" 2>/dev/null | wc -l` files
- JavaScript: !`find ${ARGUMENTS:-.} -name "*.js" 2>/dev/null | wc -l` files
- TypeScript: !`find ${ARGUMENTS:-.} -name "*.ts" -o -name "*.tsx" 2>/dev/null | wc -l` files
- Java: !`find ${ARGUMENTS:-.} -name "*.java" 2>/dev/null | wc -l` files
- Go: !`find ${ARGUMENTS:-.} -name "*.go" 2>/dev/null | wc -l` files
- Rust: !`find ${ARGUMENTS:-.} -name "*.rs" 2>/dev/null | wc -l` files
- C/C++: !`find ${ARGUMENTS:-.} -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" 2>/dev/null | wc -l` files

### Build & Config Files
- Package configs: !`find ${ARGUMENTS:-.} -maxdepth 3 \( -name "package.json" -o -name "pyproject.toml" -o -name "Cargo.toml" -o -name "go.mod" -o -name "pom.xml" -o -name "build.gradle" \) 2>/dev/null`
- Linter configs: !`find ${ARGUMENTS:-.} -maxdepth 2 \( -name ".eslintrc*" -o -name "pylintrc" -o -name ".flake8" -o -name "tslint.json" \) 2>/dev/null`
- Type configs: !`find ${ARGUMENTS:-.} -maxdepth 2 \( -name "tsconfig.json" -o -name "mypy.ini" -o -name ".pyre_configuration" \) 2>/dev/null`

---

## Phase 2: Multi-Agent Analysis Deployment

### Agent 1: AST-Based Import Analyzer
**Mission**: Detect unused imports using Abstract Syntax Tree analysis

**Python Analysis**:
- Imports to analyze: !`grep -r "^import \|^from .* import" ${ARGUMENTS:-.} --include="*.py" 2>/dev/null | wc -l`
- Common patterns: !`grep -r "^import \|^from .* import" ${ARGUMENTS:-.} --include="*.py" 2>/dev/null | head -20`

**JavaScript/TypeScript Analysis**:
- Import statements: !`grep -r "^import \|^const .* = require" ${ARGUMENTS:-.} --include="*.{js,ts,jsx,tsx}" 2>/dev/null | wc -l`
- ES6 imports: !`grep -r "^import.*from" ${ARGUMENTS:-.} --include="*.{js,ts,jsx,tsx}" 2>/dev/null | head -20`

**Tasks**:
1. Parse all source files to build AST
2. Extract all import/require statements
3. Track symbol usage across entire codebase
4. Identify imports with zero references
5. Detect redundant imports (duplicate or re-exported)
6. Find wildcard imports that could be specific

### Agent 2: Dead Code Detector
**Mission**: Identify unreachable and unused code

**Function/Method Analysis**:
- Function definitions (Python): !`grep -r "^def \|^async def " ${ARGUMENTS:-.} --include="*.py" 2>/dev/null | wc -l`
- Function definitions (JS/TS): !`grep -r "function \|const .* = .*=> \|async.*=>" ${ARGUMENTS:-.} --include="*.{js,ts}" 2>/dev/null | wc -l`
- Class definitions: !`grep -r "^class \|^export class \|^interface " ${ARGUMENTS:-.} --include="*.{py,js,ts,java,go,rs}" 2>/dev/null | wc -l`

**Tasks**:
1. Identify all function/method/class definitions
2. Build call graph of entire codebase
3. Detect unreferenced functions (0 callers)
4. Find unreachable code paths (after return/break/throw)
5. Identify unused parameters
6. Detect constants/variables that are never read
7. Find empty functions/classes
8. Detect redundant else blocks after return
9. Find unused private methods

### Agent 3: Dependency Cleaner
**Mission**: Clean up unused dependencies and imports

**Package Analysis**:
- Package.json: @${ARGUMENTS:-.}/package.json
- Requirements.txt: @${ARGUMENTS:-.}/requirements.txt
- Cargo.toml: @${ARGUMENTS:-.}/Cargo.toml
- Go.mod: @${ARGUMENTS:-.}/go.mod

**Tasks**:
1. Parse dependency manifests
2. Scan codebase for actual import/require usage
3. Identify unused dependencies (declared but never imported)
4. Find devDependencies used in production
5. Detect duplicate dependencies (different versions)
6. Identify deprecated packages
7. Recommend lighter alternatives

### Agent 4: Code Duplication Hunter
**Mission**: Find and eliminate duplicate code

**Duplication Patterns**:
- Exact duplicates (copy-paste)
- Structural duplicates (same logic, different names)
- Partial duplicates (common code blocks)

**Tasks**:
1. Calculate code similarity hashes
2. Find exact duplicate functions/blocks
3. Detect near-duplicate code (>80% similar)
4. Identify refactoring opportunities
5. Suggest common utility extraction
6. Find duplicate type definitions

### Agent 5: Comment & Documentation Cleaner
**Mission**: Remove obsolete comments and improve documentation

**Comment Analysis**:
- Comment lines: !`grep -r "^\s*#\|^\s*//\|^\s*/\*" ${ARGUMENTS:-.} --include="*.{py,js,ts,java,go,rs}" 2>/dev/null | wc -l`
- TODO comments: !`grep -ri "TODO\|FIXME\|HACK\|XXX" ${ARGUMENTS:-.} --include="*.{py,js,ts,java,go,rs}" 2>/dev/null | wc -l`

**Tasks**:
1. Identify commented-out code
2. Find outdated comments (referring to removed code)
3. Detect redundant comments (restating obvious code)
4. Find TODO/FIXME older than 6 months
5. Remove auto-generated boilerplate comments
6. Identify missing docstrings for public APIs

---

## Phase 3: UltraThink Intelligence Layer

### Deep Analysis Framework

**1. Problem Space Understanding**
- **Current State**: Analyze the complexity and maintainability of existing codebase
- **Pain Points**: Identify areas causing the most maintenance burden
- **Risk Assessment**: Evaluate what code is safe to remove vs risky
- **Business Context**: Consider feature usage patterns and deprecation timelines

**2. Multi-Dimensional Cleanup Strategy**

#### Technical Perspective
- **AST Accuracy**: Ensure static analysis doesn't create false positives
- **Runtime Analysis**: Consider dynamic imports, reflection, meta-programming
- **Build System**: Account for tree-shaking, code splitting, bundler behavior
- **Type Safety**: Preserve type checking capabilities during cleanup

#### Safety Perspective
- **Blast Radius**: Assess impact of each cleanup operation
- **Rollback Strategy**: Ensure every change is reversible
- **Testing Coverage**: Verify test suites still pass after cleanup
- **Gradual Migration**: Phase cleanup to minimize disruption

#### Performance Perspective
- **Build Time**: How cleanup affects compilation/bundling speed
- **Bundle Size**: Impact on final artifact size
- **Runtime Performance**: Ensure no accidental runtime overhead
- **Developer Experience**: Improve IDE performance and code navigation

**3. Intelligent Decision Making**

For each cleanup candidate, apply decision framework:

```
IF unused_code_confidence > 95% AND test_coverage > 80%:
    → AUTO-REMOVE (safe)

ELIF unused_code_confidence > 80% AND has_usage_analytics:
    → CONDITIONAL-REMOVE (verify with usage data)

ELIF unused_code_confidence > 60% AND marked_deprecated:
    → DEPRECATION-WARNING (schedule for removal)

ELSE:
    → MANUAL-REVIEW (flag for human decision)
```

**4. Pattern Recognition & Learning**

- Learn from codebase patterns (framework conventions, team style)
- Adapt to project-specific idioms (dependency injection, factories)
- Recognize framework-specific patterns (React hooks, Django models)
- Account for meta-programming (decorators, macros, code generation)

**5. Second-Order Consequences**

Consider ripple effects:
- Will removing import X break import Y in another file?
- Does dead code serve as documentation/examples?
- Are "unused" functions actually called via reflection/eval?
- Will cleanup break external consumers (if library)?

---

## Phase 4: Automated Cleanup Execution

### Pre-Cleanup Checklist
- [ ] Create git branch: `cleanup/automated-${DATE}`
- [ ] Run full test suite baseline
- [ ] Create backup: `.backup-${TIMESTAMP}/`
- [ ] Document all changes in `CLEANUP_REPORT.md`

### Cleanup Operations (by safety level)

#### Level 1: SAFE (Auto-apply)
✅ Remove trailing whitespace
✅ Remove unused imports (100% confidence)
✅ Remove empty files
✅ Remove commented-out code (older than 3 months)
✅ Remove duplicate imports
✅ Sort imports
✅ Remove console.log/print debugging statements
✅ Remove unreachable code after return

#### Level 2: MODERATE (Apply with review)
⚠️ Remove unused private functions (not exported)
⚠️ Remove unused variables/constants
⚠️ Remove unused parameters
⚠️ Remove empty classes/interfaces
⚠️ Remove unused type definitions
⚠️ Consolidate duplicate code

#### Level 3: RISKY (Manual review required)
🔴 Remove public API functions (breaking change)
🔴 Remove entire files
🔴 Remove dependencies from package.json
🔴 Remove functions with 0 callers but exported
🔴 Remove code used only in comments/docs

### Execution Strategy

```bash
# Stage 1: Safe cleanup (auto-apply)
for each SAFE operation:
    - Apply transformation
    - Run affected tests
    - If tests fail: rollback, flag for manual review
    - Commit with detailed message

# Stage 2: Moderate cleanup (with confirmation)
for each MODERATE operation:
    - Show diff preview
    - Request confirmation (if not --auto-fix)
    - Apply transformation
    - Run full test suite
    - Commit separately for easy revert

# Stage 3: Risky cleanup (report only)
for each RISKY operation:
    - Add to manual review report
    - Suggest migration path
    - Document breaking change implications
```

---

## Phase 5: Validation & Quality Assurance

### Automated Testing
1. **Run test suites**:
   - Unit tests
   - Integration tests
   - E2E tests
   - Type checking
   - Linting

2. **Build verification**:
   - Development build succeeds
   - Production build succeeds
   - Bundle size comparison (before/after)

3. **Runtime checks**:
   - No new runtime errors
   - Import resolution works
   - All entry points functional

### Metric Tracking

**Code Reduction**:
- Lines of code removed: `X lines (-Y%)`
- Files removed: `X files`
- Dependencies removed: `X packages`
- Import statements cleaned: `X imports`

**Quality Improvement**:
- Cyclomatic complexity reduction
- Maintainability index increase
- Test coverage unchanged or improved
- Build time improvement

**Safety Metrics**:
- Test pass rate: `100%` (required)
- Type errors: `0` (required)
- Linting errors: `≤ baseline`
- Breaking changes: `documented`

---

## Phase 6: Comprehensive Reporting

### Cleanup Summary Report

Generate detailed report: `CLEANUP_REPORT.md`

```markdown
# Codebase Cleanup Report
Generated: ${TIMESTAMP}

## Executive Summary
- **Total changes**: X files modified, Y files removed
- **Code reduction**: Z lines removed (-W%)
- **Dependencies cleaned**: N packages removed
- **Safety level**: All tests passing ✅

## Detailed Changes

### Unused Imports Removed (N instances)
| File | Import | Reason | Confidence |
|------|--------|--------|------------|
| src/foo.ts | lodash | No usage found | 99% |

### Dead Code Eliminated (N functions)
| File | Function | Lines Saved | Risk Level |
|------|----------|-------------|------------|
| src/bar.py | old_helper() | 45 | Low |

### Dependencies Removed (N packages)
| Package | Version | Reason | Migration Notes |
|---------|---------|--------|-----------------|
| moment | 2.29.1 | Unused | Use date-fns instead |

### Code Duplication Eliminated (N instances)
| Location | Duplicate | Solution | Lines Saved |
|----------|-----------|----------|-------------|
| src/a.ts, src/b.ts | Data validation | Extracted to utils | 30 |

### Manual Review Required (N items)
| File | Issue | Recommendation | Risk |
|------|-------|----------------|------|
| src/api.ts | Exported but unused | Verify with consumers | High |

## Quality Metrics

**Before Cleanup**:
- Total lines: 50,000
- Cyclomatic complexity: 25 (avg)
- Dependencies: 120
- Import statements: 3,500
- Test coverage: 78%

**After Cleanup**:
- Total lines: 42,000 (-16%)
- Cyclomatic complexity: 18 (avg)
- Dependencies: 95 (-21%)
- Import statements: 2,800 (-20%)
- Test coverage: 80% (+2%)

## Validation Results
✅ All unit tests passing (1,234/1,234)
✅ All integration tests passing (89/89)
✅ Type checking successful (0 errors)
✅ Linting passed with 0 errors
✅ Build time improved: 45s → 38s (-15%)
✅ Bundle size reduced: 2.3MB → 1.9MB (-17%)

## Recommendations for Next Steps
1. Review manual-review items (3 items flagged)
2. Consider removing deprecated functions in next release
3. Update documentation to reflect removed APIs
4. Run performance benchmarks to verify improvements
5. Schedule dependency updates for remaining packages

## Rollback Instructions
```bash
# If issues arise, rollback with:
git checkout main
git branch -D cleanup/automated-${DATE}
# Or restore from backup:
cp -r .backup-${TIMESTAMP}/* .
```
```

---

## Your Task: Execute Advanced Codebase Cleanup

**Arguments Received**: `$ARGUMENTS`

**Execution Mode**:
- `--auto-fix` flag: Apply SAFE changes automatically
- `--aggressive` flag: Apply MODERATE changes with confirmation
- Default: Analysis only, no automatic fixes

### Step-by-Step Execution Plan

1. **Initialize Safety Protocol**
   - Create git branch: `cleanup/automated-$(date +%Y%m%d-%H%M%S)`
   - Run baseline tests to ensure starting from green state
   - Create backup directory with timestamp

2. **Deploy Multi-Agent Analysis**
   - Launch 5 specialized agents in parallel
   - Each agent analyzes their domain independently
   - Aggregate findings with confidence scores

3. **Apply UltraThink Intelligence**
   - Deep analysis of cleanup candidates
   - Multi-dimensional risk assessment
   - Pattern recognition and learning
   - Second-order consequence evaluation
   - Generate intelligent cleanup strategy

4. **Execute Cleanup Operations**
   - Level 1 (SAFE): Auto-apply if --auto-fix
   - Level 2 (MODERATE): Apply if --aggressive
   - Level 3 (RISKY): Report only, no automatic changes

5. **Validation & Testing**
   - Run complete test suite
   - Verify build process
   - Check for runtime issues
   - Compare metrics (before/after)

6. **Generate Comprehensive Report**
   - Create detailed `CLEANUP_REPORT.md`
   - Include all metrics and changes
   - Provide rollback instructions
   - List manual review items

7. **Create Git Commits**
   - Separate commits for each cleanup category
   - Detailed commit messages
   - Easy to review and revert if needed

### Special Considerations

**Framework-Specific Intelligence**:
- **React/Vue**: Recognize unused hooks, components, context
- **Django/Flask**: Identify unused views, models, serializers
- **Express/Fastify**: Detect unused middleware, routes
- **Rust**: Understand unused derive macros, trait implementations
- **Go**: Account for blank imports, init() side effects

**Meta-Programming Awareness**:
- Don't remove code accessed via reflection
- Preserve decorator/macro targets
- Keep code generation templates
- Maintain plugin system hooks

**Library Mode**:
- If package.json has "main" or "exports" → library mode
- Preserve all exported symbols (even if unused internally)
- Warn about public API changes
- Check for consumers in monorepo

### Success Criteria

✅ Zero test failures after cleanup
✅ Zero type errors after cleanup
✅ Build succeeds for all targets
✅ No new runtime warnings/errors
✅ Code metrics improved or maintained
✅ All changes are reversible
✅ Comprehensive report generated

### Failure Recovery

If any stage fails:
1. Immediately halt cleanup process
2. Rollback all changes
3. Report exact failure point
4. Provide debugging information
5. Suggest manual intervention steps

---

## Now Execute

Begin the advanced codebase cleanup process with full ultrathink intelligence.

**Remember**:
- Safety first: Never break working code
- Confidence threshold: Only remove code when confidence > 95%
- Test always: Verify after every change
- Report everything: Full transparency in report
- Rollback ready: Every change must be reversible

Start the multi-agent analysis and apply ultrathink reasoning to create the cleanest, most maintainable codebase possible while preserving all functionality.
