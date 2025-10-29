---
description: Systematically fix broken imports across the codebase with session continuity
argument-hint: [path-or-pattern] [resume|status|new]
color: purple
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task
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

**Systematically fix import statements broken by file moves, renames, or refactoring with full session continuity**

## Your Task: $ARGUMENTS

## Session Intelligence & Continuity

**Session Files (stored in current project directory):**
- `fix-imports/plan.md` - Comprehensive list of all broken imports and resolution strategies
- `fix-imports/state.json` - Progress tracking and resolution decisions
- `fix-imports/decisions.json` - Consistency tracking for ambiguous resolutions

**Auto-Detection Behavior:**
- If session exists: Resume from last import automatically
- If no session: Perform fresh scan and create new session
- Commands: `resume`, `status`, `new`, or specific path/pattern

**IMPORTANT:** Session files are always stored in `fix-imports/` folder at the current project root, never in parent directories.

---

## Phase 1: Import Analysis & Detection

### Mandatory First Steps

1. **Check for Existing Session**
   ```bash
   # Check if fix-imports directory exists in current working directory
   ls fix-imports/state.json fix-imports/plan.md 2>/dev/null
   ```

2. **Session Decision Logic**
   - If session files exist:
     - Parse `state.json` for progress statistics
     - Load `plan.md` for broken imports list
     - Show resume summary
     - Ask user: Continue from last position or start fresh?

   - If no session exists:
     - Proceed to fresh import scan
     - Create session directory and files
     - Initialize progress tracking

3. **Display Session Status** (if resuming)
   ```
   RESUMING IMPORT FIX SESSION
   ═══════════════════════════════════════
   Total Broken Imports: 47
   Fixed:               28 (60%)
   Remaining:           19 (40%)
   Current File:        src/components/Dashboard.tsx
   Last Resolution:     Updated relative path
   ═══════════════════════════════════════
   ```

### Fresh Import Scan

**Detection Strategy:**

1. **Language-Specific Import Scanning**
   ```bash
   # TypeScript/JavaScript
   grep -r "from ['\"]" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.jsx"

   # Python
   grep -r "^import \|^from " --include="*.py"

   # Rust
   grep -r "^use " --include="*.rs"

   # Go
   grep -r "^import " --include="*.go"
   ```

2. **Broken Import Patterns**
   - File not found errors from linter/compiler output
   - Module resolution failures
   - Non-existent path references
   - Moved or renamed files
   - Deleted dependencies
   - Circular references

3. **Build System Integration**
   ```bash
   # Try to run type checker or build to find import errors
   npm run type-check 2>&1 | grep -i "cannot find\|module not found"
   tsc --noEmit 2>&1 | grep "TS2307\|TS2305"
   mypy . 2>&1 | grep "Cannot find\|No module"
   cargo check 2>&1 | grep "unresolved import"
   ```

4. **Smart Context Understanding**
   - Detect path aliases (tsconfig.json, webpack, vite config)
   - Recognize barrel exports (index.ts files)
   - Distinguish external vs internal imports
   - Understand monorepo package boundaries

---

## Phase 2: Resolution Planning & Strategy

### Create Comprehensive Fix Plan

**For each broken import, determine:**

1. **Import Context**
   - Importing file location
   - Imported module/file
   - Import type (default, named, namespace)
   - Import statement syntax

2. **Resolution Strategies** (in priority order)

   **Strategy 1: Exact Filename Match**
   ```bash
   # Find files with exact name match
   find . -name "ComponentName.tsx" -o -name "ComponentName.ts"
   ```

   **Strategy 2: Similar Name Suggestions**
   ```bash
   # Find files with similar names (typos, case differences)
   find . -iname "*component*" | grep -i "name"
   ```

   **Strategy 3: Export Symbol Search**
   ```bash
   # Search for the exported symbol in the codebase
   grep -r "export.*ComponentName" --include="*.ts" --include="*.tsx"
   ```

   **Strategy 4: Path Recalculation**
   - If file moved, calculate new relative path
   - Check if path alias should be used instead
   - Verify import style matches project conventions

   **Strategy 5: Dependency Analysis**
   - Check if import was from deleted package
   - Search package.json for alternatives
   - Suggest modern replacement if deprecated

3. **Confidence Scoring**
   - **High (90-100%)**: Single exact match, clear resolution
   - **Medium (60-89%)**: Multiple matches, context helps narrow
   - **Low (<60%)**: Ambiguous, requires user decision

### Write Fix Plan

Create `fix-imports/plan.md`:
```markdown
# Import Fix Plan

Generated: 2025-10-20 14:35:22
Total Broken Imports: 47

## High Confidence (28 imports)

### 1. src/components/Dashboard.tsx:12
**Broken:** `import { UserProfile } from './UserProfile'`
**Issue:** File moved to src/components/user/UserProfile.tsx
**Resolution:** Update to `import { UserProfile } from './user/UserProfile'`
**Confidence:** 95%

### 2. src/utils/helpers.ts:5
**Broken:** `import { formatDate } from '@/lib/dates'`
**Issue:** Package moved to utils/dates
**Resolution:** Update to `import { formatDate } from '@/utils/dates'`
**Confidence:** 100%

## Medium Confidence (15 imports)

### 29. src/pages/Home.tsx:8
**Broken:** `import { Button } from 'components/Button'`
**Issue:** Multiple Button components found
**Matches:**
  - src/components/ui/Button.tsx (UI library)
  - src/components/legacy/Button.tsx (deprecated)
**Resolution:** Requires user choice - likely UI library version
**Confidence:** 70%

## Low Confidence (4 imports)

### 44. src/api/client.ts:3
**Broken:** `import { API } from 'api-client'`
**Issue:** Package not found, may be deleted
**Suggestions:**
  - Install `api-client` package
  - Use native fetch
  - Use axios or similar
**Resolution:** Requires user decision
**Confidence:** 30%
```

Create `fix-imports/state.json`:
```json
{
  "version": "1.0",
  "created": "2025-10-20T14:35:22Z",
  "updated": "2025-10-20T14:35:22Z",
  "totalBroken": 47,
  "fixed": 0,
  "remaining": 47,
  "currentIndex": 0,
  "decisions": {},
  "gitCheckpoint": "abc123def"
}
```

---

## Phase 3: Intelligent Fixing Process

### Fix Execution Workflow

**For each import (in order of confidence):**

1. **Create Git Checkpoint** (before first fix)
   ```bash
   git add -A
   git stash push -m "checkpoint: before import fixes"
   # Store stash hash in state.json
   ```

2. **Apply Fix with Verification**

   **Fix Pattern Matching:**
   - Preserve import style (single quotes vs double quotes)
   - Maintain import grouping (external → internal → local)
   - Follow project sorting conventions
   - Keep multiline formatting consistent

   **Example Fix:**
   ```typescript
   // Before
   import { UserProfile } from './UserProfile'

   // After (calculated new relative path)
   import { UserProfile } from './user/UserProfile'
   ```

3. **Post-Fix Verification**
   ```bash
   # Quick syntax check
   npm run type-check --noEmit
   # or
   tsc --noEmit path/to/fixed-file.ts
   ```

4. **Update Progress**
   - Mark import as fixed in plan.md
   - Update state.json with new counts
   - Record resolution decision in decisions.json
   - Save timestamp

5. **Ambiguity Handling**
   - **Multiple Matches**: Show user all options with context
   - **Uncertain Resolution**: Ask user to choose
   - **Track Decisions**: Apply same choice to similar imports
   - **Never Guess**: Always ask when uncertain

### Incremental Commit Strategy

**Option 1: Commit per file** (recommended for large changes)
```bash
git add path/to/fixed-file.ts
git commit -m "fix(imports): resolve broken imports in Dashboard.tsx"
```

**Option 2: Batch commit** (for small, related fixes)
```bash
# After fixing 5-10 related imports
git add .
git commit -m "fix(imports): resolve imports in components/ directory"
```

---

## Phase 4: Verification & Validation

### Post-Fix Checks

1. **Syntax Validation**
   ```bash
   # Full type check
   npm run type-check || tsc --noEmit

   # Linting
   npm run lint
   ```

2. **No New Broken Imports**
   ```bash
   # Re-scan for import errors
   tsc --noEmit 2>&1 | grep "TS2307\|TS2305" | wc -l
   # Should be 0 or less than before
   ```

3. **Circular Dependency Check**
   ```bash
   # Use madge or similar
   npx madge --circular src/
   ```

4. **Build Verification**
   ```bash
   # Attempt production build
   npm run build
   # or
   cargo build --release
   # or
   python -m build
   ```

5. **Update Final Status**
   ```
   IMPORT FIX COMPLETE
   ═══════════════════════════════════════
   Total Fixed:         47/47 (100%)
   High Confidence:     28
   Medium Confidence:   15
   Required User Input: 4
   Build Status:        ✅ Passing
   Time Elapsed:        8m 42s
   ═══════════════════════════════════════

   All imports successfully resolved!
   Session files saved in: fix-imports/
   ```

---

## Session Continuity & Resume

### Resume Capability

**When you run `/fix-imports` or `/fix-imports resume`:**

1. **Load Session State**
   ```bash
   # Read state.json
   cat fix-imports/state.json | jq '.totalBroken, .fixed, .remaining'
   ```

2. **Display Progress**
   ```
   RESUMING IMPORT FIX SESSION
   ═══════════════════════════════════════
   Session Created:     2025-10-20 14:35
   Last Updated:        2025-10-20 15:12
   Total Broken:        47
   Fixed:               28 (60%)
   Remaining:           19 (40%)

   Current Import:
   File: src/components/Dashboard.tsx:12
   Broken: import { UserProfile } from './UserProfile'
   Resolution: Update to './user/UserProfile'
   Confidence: 95%
   ═══════════════════════════════════════

   Continue fixing? [Y/n]
   ```

3. **Apply Consistent Resolution Patterns**
   - Load `decisions.json` for previous user choices
   - Apply same patterns to similar imports
   - Example: If user chose UI Button over legacy, always pick UI version

### Status Command

**`/fix-imports status`**
```
IMPORT FIX STATUS
═══════════════════════════════════════
Progress:           28/47 (60%)
High Confidence:    20/28 fixed
Medium Confidence:  7/15 fixed
Low Confidence:     1/4 fixed

Recently Fixed:
  ✅ src/components/Dashboard.tsx
  ✅ src/components/Header.tsx
  ✅ src/utils/helpers.ts

Next Up:
  ⏭  src/components/Sidebar.tsx
  ⏭  src/pages/Home.tsx
  ⏭  src/api/client.ts
═══════════════════════════════════════
```

### New Scan Command

**`/fix-imports new`**
- Archive old session to `fix-imports/archive/YYYY-MM-DD-HHMMSS/`
- Perform fresh import scan
- Create new session files
- Start from beginning

---

## Usage Examples

### Basic Usage

```bash
# Fix all broken imports in the project
/fix-imports

# Fix imports in specific directory
/fix-imports src/components

# Fix imports matching pattern
/fix-imports "UserProfile"

# Focus on specific file type
/fix-imports "*.tsx"
```

### Session Management

```bash
# Resume from last session
/fix-imports resume

# Check current progress
/fix-imports status

# Start fresh scan (archive old session)
/fix-imports new

# Fix with verbose output
/fix-imports --verbose
```

### Advanced Scenarios

```bash
# Fix imports after large refactor
/fix-imports --deep-scan

# Fix and update path aliases
/fix-imports --update-aliases

# Dry run (show what would be fixed)
/fix-imports --dry-run
```

---

## Safety Guarantees & Best Practices

### Protection Measures

1. **Git Integration**
   - Create stash checkpoint before any fixes
   - Incremental commits for traceability
   - Easy rollback if issues found

2. **Verification at Every Step**
   - Syntax check after each fix
   - Type check before moving to next import
   - Build verification at the end

3. **User Confirmation**
   - Always ask when multiple resolutions possible
   - Show context for decision making
   - Never guess ambiguous imports

4. **Audit Trail**
   - Record all fixes in plan.md
   - Track decisions in decisions.json
   - Timestamp all changes in state.json

### What This Command Will NEVER Do

- ❌ Guess ambiguous imports without user input
- ❌ Break working imports
- ❌ Add AI attribution comments
- ❌ Create circular dependencies
- ❌ Modify unrelated code
- ❌ Skip verification steps
- ❌ Lose session progress

---

## Implementation Workflow

**When this command is invoked, I will:**

### Step 1: Session Check (2 minutes)
1. Check for `fix-imports/` directory
2. Load existing session or create new
3. Display status summary
4. Ask user to confirm continuation

### Step 2: Import Analysis (5-10 minutes)
1. Scan codebase for broken imports
2. Parse build/type check errors
3. Categorize by confidence level
4. Create comprehensive fix plan

### Step 3: Resolution Planning (3-5 minutes)
1. For each import, find possible resolutions
2. Score confidence for each resolution
3. Write detailed plan to plan.md
4. Initialize state.json tracking

### Step 4: Systematic Fixing (varies)
1. Create git checkpoint
2. Fix high-confidence imports first
3. Ask user for ambiguous cases
4. Verify after each fix
5. Update progress continuously

### Step 5: Final Verification (2-3 minutes)
1. Run full type check
2. Check for circular dependencies
3. Attempt build
4. Generate completion report

### Step 6: Session Cleanup
1. Mark session as complete
2. Archive or keep session files
3. Provide summary statistics
4. Suggest next steps if needed

---

## Technical Details

### Supported Languages

- **TypeScript/JavaScript**: ES6 imports, require(), dynamic imports
- **Python**: import statements, from...import
- **Rust**: use statements
- **Go**: import statements
- **Java**: import statements (basic support)

### Path Alias Support

Automatically detects and respects:
- `tsconfig.json` paths
- `jsconfig.json` paths
- Webpack aliases
- Vite aliases
- Next.js aliases (@/)
- Custom project aliases

### Import Style Preservation

Maintains:
- Quote style (single vs double)
- Import grouping order
- Spacing and formatting
- Comment preservation
- Multiline import formatting

---

**Execute systematic import fixing with perfect session continuity, intelligent resolution strategies, and comprehensive verification at every step**
