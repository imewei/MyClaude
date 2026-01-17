---
description: Systematically fix broken imports across the codebase with session continuity
triggers:
- /fix-imports
- workflow for fix imports
version: 1.0.7
argument-hint: '[path-or-pattern] [resume|status|new]'
category: codebase-cleanup
purpose: Resolve broken imports with intelligent strategies and session management
execution_time:
  quick: 3-8 minutes
  standard: 10-20 minutes
  comprehensive: 20-45 minutes
color: purple
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task
external_docs:
- import-resolution-strategies.md
- session-management-guide.md
- refactoring-patterns.md
agents:
  primary:
  - code-quality
  - fullstack-developer
  conditional:
  - agent: legacy-modernizer
    trigger: pattern "migration|refactor|modernize"
  - agent: systems-architect
    trigger: pattern "architecture|barrel.*export|module.*system"
  orchestrated: false
---


# Fix Broken Imports

Systematically fix import statements broken by file moves, renames, or refactoring with session continuity.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| `--quick` | 3-8 min | High-confidence fixes only, skip ambiguous |
| standard (default) | 10-20 min | Full scan with confidence scoring, session management |
| `--comprehensive` | 20-45 min | + Barrel export optimization, circular dependency detection, path alias standardization |

**Commands:** `resume` (continue session), `status` (show progress), `new` (fresh scan)

---

## Session Management

**Session Directory:** `fix-imports/` at project root

| File | Purpose |
|------|---------|
| `plan.md` | All broken imports with resolution strategies |
| `state.json` | Progress tracking (total, fixed, remaining) |
| `decisions.json` | Consistency tracking for ambiguous resolutions |

**Auto-detection:** If session exists → resume; otherwise → fresh scan

---

## Phase 1: Import Detection

### Check for Existing Session
1. Check `fix-imports/state.json` existence
2. If exists: Show progress summary, ask to continue or start fresh
3. If not: Proceed to fresh scan

### Detection Methods

| Language | Build System | Direct Scan |
|----------|--------------|-------------|
| TypeScript/JS | `tsc --noEmit 2>&1 \| grep "TS2307\|TS2305"` | `grep -r "from ['\"]"` |
| Python | `mypy . 2>&1 \| grep "Cannot find"` | `grep -r "^import \|^from "` |
| Rust | `cargo check 2>&1 \| grep "unresolved import"` | `grep -r "^use "` |
| Go | `go build 2>&1 \| grep "cannot find package"` | `grep -r "^import "` |

**Context Understanding:** Detect path aliases (tsconfig/webpack/vite), barrel exports, external vs internal imports, monorepo boundaries

---

## Phase 2: Resolution Planning

### Resolution Strategies (priority order)

| Strategy | Description |
|----------|-------------|
| 1. Exact filename match | Search for exact filename, calculate new relative path |
| 2. Similar name suggestions | Fuzzy matching for typos, case differences |
| 3. Export symbol search | Find exported symbol across all files |
| 4. Path recalculation | Calculate new path if file moved, check alias usage |
| 5. Dependency analysis | Check if from deleted package, suggest alternatives |

### Confidence Scoring

| Level | Range | Action |
|-------|-------|--------|
| High | 90-100% | Auto-fix with verification |
| Medium | 60-89% | Fix with context, track decision |
| Low | <60% | **Require user decision** |

### Create Fix Plan
Write `fix-imports/plan.md` with all broken imports categorized by confidence, including:
- File:line location
- Broken import statement
- Issue description
- Proposed resolution
- Confidence percentage

---

## Phase 3: Systematic Fixing

### Workflow

1. **Create Git Checkpoint:** `git stash push -m "checkpoint: before import fixes"`
2. **Fix by Confidence:** High → Medium → Low (ask user for ambiguous)
3. **Verify Each Fix:** Run syntax/type check after each
4. **Update Progress:** Mark fixed in plan.md, update state.json
5. **Track Decisions:** Store in decisions.json for consistency

### Ambiguity Handling
- **Multiple matches:** Show all options with context
- **Uncertain:** Ask user to choose
- **Apply consistently:** Same decision for similar imports
- **Never guess:** Always ask when uncertain

### Commit Strategy
- Per-file for large changes: `git commit -m "fix(imports): resolve in Dashboard.tsx"`
- Batch for related fixes: `git commit -m "fix(imports): resolve in components/"`

---

## Phase 4: Verification

```bash
npm run type-check || tsc --noEmit    # Full type check
npx madge --circular src/             # Circular deps (comprehensive)
npm run build                         # Build verification
```

### Final Status Output
```
Total Fixed:         47/47 (100%)
High Confidence:     28
Medium Confidence:   15
Required User Input: 4
Build Status:        ✅ Passing
```

---

## Import Style Preservation

Maintains: quote style, import grouping order, spacing/formatting, comments, multiline formatting

**Path Alias Support:** tsconfig.json, jsconfig.json, webpack/vite aliases, Next.js @/ aliases

---

## Safety Guarantees

**Will:**
- ✅ Create git checkpoint before fixes
- ✅ Verify after each fix
- ✅ Ask for ambiguous cases
- ✅ Track all decisions
- ✅ Preserve import style

**Never:**
- ❌ Guess ambiguous imports
- ❌ Break working imports
- ❌ Create circular dependencies
- ❌ Skip verification
- ❌ Lose session progress
