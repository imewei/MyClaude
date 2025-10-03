# Your First Commands

A hands-on guide to running your first commands with the Claude Code Command Executor.

## Before You Start

Ensure you have:
- Installed the framework ([Installation Guide](installation.md))
- A code project to work with
- Basic command line familiarity

## Command 1: Check Code Quality

Let's start with the most useful command - checking code quality.

### Step 1: Preview Mode (Dry-Run)

```bash
# Navigate to your project
cd /path/to/your/project

# Run quality check in preview mode
/check-code-quality --dry-run
```

**What happened:**
- Framework analyzed your codebase
- Selected optimal agents automatically (`--agents=auto` is default)
- Generated a quality report
- **Made NO changes** (dry-run mode)

### Step 2: Review the Output

You'll see output like:

```
🔍 Analyzing codebase...
✅ Analysis complete

📊 Code Quality Report:
  Files analyzed: 42
  Issues found: 23

🤖 Agents used: 5 (auto-selected)
  - code-quality-master
  - systems-architect
  - scientific-computing-master
  - documentation-architect
  - multi-agent-orchestrator

📋 Issues by Severity:
  HIGH (5):
    - 3 unused imports in src/utils.py
    - 2 functions missing docstrings

  MEDIUM (12):
    - 8 code complexity warnings
    - 4 naming convention violations

  LOW (6):
    - 6 comment improvements suggested

⏱️  Duration: 2.3s
```

### Step 3: Apply Fixes

Now let's actually fix the issues:

```bash
# Apply automatic fixes
/check-code-quality --auto-fix
```

**What happened:**
- Framework created a backup automatically
- Applied safe fixes (unused imports, formatting)
- Left complex changes for manual review
- Generated a summary of changes

### Step 4: Verify Changes

```bash
# Check what changed
git diff

# Run tests to ensure nothing broke
/run-all-tests
```

## Command 2: Optimize Performance

Now let's optimize your code for better performance.

### Step 1: Profile and Analyze

```bash
# Analyze performance with profiling
/optimize --profile src/
```

**Output:**
```
🔍 Profiling code performance...
📊 Performance Analysis:

Hot spots identified:
  1. src/algorithm.py:45 - O(n²) complexity (HIGH)
  2. src/utils.py:120 - Inefficient string concatenation (MEDIUM)
  3. src/data.py:78 - Redundant computations (MEDIUM)

🎯 Optimization Opportunities:
  [HIGH] Algorithm complexity reduction - Est. 10x speedup
  [MEDIUM] String operation optimization - Est. 2x speedup
  [MEDIUM] Computation caching - Est. 1.5x speedup

💡 Recommendations:
  1. Replace nested loops with numpy operations
  2. Use join() instead of += for strings
  3. Add @lru_cache decorator for pure functions

⏱️  Duration: 4.1s
```

### Step 2: Preview Optimizations

```bash
# See proposed changes without applying
/optimize --implement --dry-run src/
```

### Step 3: Apply Optimizations

```bash
# Apply the optimizations
/optimize --implement src/
```

### Step 4: Validate Results

```bash
# Run tests to ensure correctness
/run-all-tests --benchmark

# Compare performance
# Before: 2.3s average
# After: 0.4s average (5.75x faster!)
```

## Command 3: Clean Codebase

Remove unused code and imports.

### Step 1: Analyze What Can Be Cleaned

```bash
# Preview cleanup operations
/clean-codebase --imports --dead-code --dry-run
```

**Output:**
```
🧹 Analyzing codebase for cleanup...

📊 Cleanup Analysis:

Unused Imports: 47 found
  src/main.py: 8 unused imports
  src/utils.py: 12 unused imports
  src/models.py: 6 unused imports
  ... (27 more)

Dead Code: 12 blocks found
  src/old_algorithm.py: Entire file unused (323 lines)
  src/utils.py: Function 'deprecated_helper' (45 lines)
  ... (10 more)

💾 Space Savings: ~1,234 lines (4.2% of codebase)

✅ Safety: All removals verified safe through AST analysis

⏱️  Duration: 3.7s
```

### Step 2: Apply Cleanup

```bash
# Clean unused imports
/clean-codebase --imports --backup
```

**Safety features enabled:**
- `--backup` creates automatic backup
- AST analysis ensures safe removal
- Can rollback if needed

### Step 3: Verify No Breakage

```bash
# Ensure code still works
/run-all-tests

# If something broke (unlikely):
# Framework provides rollback instructions
```

## Command 4: Generate Tests

Create test suites for your code.

### Step 1: Generate Unit Tests

```bash
# Generate tests for a module
/generate-tests src/algorithm.py --type=unit
```

**Output:**
```
🧪 Generating unit tests...

📝 Test Generation Plan:
  Target: src/algorithm.py
  Functions to test: 8
  Test framework: pytest (auto-detected)

✅ Generated Tests:
  tests/test_algorithm.py - 24 test cases
    - test_quick_sort_empty
    - test_quick_sort_single
    - test_quick_sort_sorted
    ... (21 more)

📊 Coverage: 95% (8/8 functions)

⏱️  Duration: 2.8s
```

### Step 2: Run the Generated Tests

```bash
# Run new tests
/run-all-tests tests/test_algorithm.py
```

### Step 3: Generate More Test Types

```bash
# Add integration tests
/generate-tests src/api/ --type=integration

# Add performance tests
/generate-tests src/algorithm.py --type=performance
```

## Command 5: Generate Documentation

Create comprehensive documentation.

### Step 1: Generate README

```bash
# Generate or update README
/update-docs --type=readme
```

**What gets generated:**
- Project overview
- Installation instructions
- Usage examples
- API reference
- Contributing guidelines

### Step 2: Generate API Documentation

```bash
# Generate complete API docs
/update-docs --type=api --format=markdown
```

### Step 3: Generate All Documentation

```bash
# Complete documentation suite
/update-docs --type=all
```

## Command 6: Run All Tests

Execute your test suite with coverage.

### Step 1: Run with Coverage

```bash
# Run all tests with coverage report
/run-all-tests --coverage
```

**Output:**
```
🧪 Running test suite...

📊 Test Results:
  Total tests: 156
  Passed: 154 (98.7%)
  Failed: 2 (1.3%)
  Skipped: 0

⏱️  Duration: 12.3s

📈 Coverage Report:
  Lines covered: 1,847 / 2,103 (87.8%)
  Branches covered: 342 / 401 (85.3%)

❌ Failed Tests:
  1. test_algorithm.py::test_edge_case_negative
  2. test_utils.py::test_deprecated_function

💡 Suggestion: Fix failing tests or use --auto-fix
```

### Step 2: Auto-Fix Failing Tests

```bash
# Attempt to fix failing tests
/run-all-tests --auto-fix
```

### Step 3: Benchmark Performance

```bash
# Run performance benchmarks
/run-all-tests --benchmark --profile
```

## Putting It All Together

Here's a complete workflow combining all commands:

```bash
# Step 1: Check quality and fix issues
/check-code-quality --auto-fix

# Step 2: Clean up codebase
/clean-codebase --imports --dead-code --backup

# Step 3: Optimize performance
/optimize --implement src/

# Step 4: Ensure tests exist
/generate-tests src/ --type=all --coverage=95

# Step 5: Run tests to verify
/run-all-tests --coverage --auto-fix

# Step 6: Generate documentation
/update-docs --type=all

# Step 7: Commit changes
/commit --template=refactor --ai-message
```

## Understanding the Output

### Status Indicators

- ✅ Success - Operation completed successfully
- ⚠️  Warning - Completed with warnings
- ❌ Error - Operation failed
- 🔍 Analyzing - Analysis in progress
- 🤖 Agents - Agent information
- 📊 Results - Analysis results
- 💡 Suggestion - Recommendations
- ⏱️  Duration - Execution time

### Agent Selection

Most commands show which agents were used:

```
🤖 Agents used: 5 (auto-selected)
  - code-quality-master
  - systems-architect
  - scientific-computing-master
  - documentation-architect
  - multi-agent-orchestrator
```

You can control this with `--agents`:
- `--agents=auto` - Smart selection (default)
- `--agents=core` - 5 essential agents
- `--agents=scientific` - 8 scientific specialists
- `--agents=all` - All 23 agents

## Common Patterns

### Pattern 1: Quick Quality Check

```bash
# Fast check with core agents
/check-code-quality --agents=core --quick
```

### Pattern 2: Deep Analysis

```bash
# Comprehensive analysis with all agents
/check-code-quality --agents=all --orchestrate --detailed
```

### Pattern 3: Safe Modification

```bash
# Always use these flags for safety
/refactor-clean --patterns=modern --dry-run --backup --interactive
```

### Pattern 4: Scientific Code Workflow

```bash
# Specialized for scientific computing
/optimize --agents=scientific --language=python --category=algorithm
/generate-tests --type=scientific --framework=pytest
/update-docs --type=research --format=latex
```

## Next Steps

Now that you've run your first commands, explore:

- **[Understanding Agents](understanding-agents.md)** - Learn the 23-agent system
- **[Common Workflows](common-workflows.md)** - Typical development patterns
- **[Command Reference](../guides/command-reference.md)** - Complete command documentation
- **[Tutorials](../tutorials/)** - Step-by-step guides

## Quick Reference

### Most Important Flags

| Flag | Purpose | Always Use When |
|------|---------|-----------------|
| `--dry-run` | Preview without changes | First time trying |
| `--agents=auto` | Smart agent selection | Most of the time |
| `--backup` | Create backup | Making changes |
| `--auto-fix` | Apply automatic fixes | Trust the framework |
| `--interactive` | Confirm each change | Critical files |
| `--orchestrate` | Coordinate agents | Using --agents=all |

### Safety Checklist

Before running any command that modifies code:

- [ ] Run with `--dry-run` first
- [ ] Review the proposed changes
- [ ] Enable `--backup` flag
- [ ] Ensure tests pass beforehand
- [ ] Have version control committed

---

**Great job!** You've mastered the basics. → [Understanding Agents](understanding-agents.md)