# Tutorial 01: Introduction to Claude Code Command Executor Framework

> Get started with the AI-powered development automation system

**Duration**: 30 minutes | **Level**: Beginner | **Prerequisites**: Basic command-line knowledge

---

## Learning Objectives

By the end of this tutorial, you will:
- Understand what the framework is and how it works
- Run your first commands
- Understand the 23-agent system
- Use basic workflows
- Know where to find help

---

## Part 1: Understanding the System (5 minutes)

### What is This System?

The Claude Code Command Executor Framework provides:
- **14 Commands** - Specialized automation for different tasks
- **23 AI Agents** - Expert agents for different domains
- **Workflows** - Multi-step automation
- **Plugins** - Extensibility

### System Architecture

```
You → Commands → Agents → Tasks → Results
         ↓
    Workflows & Plugins
```

### The 23 Agents

**Core** (3 agents):
- Orchestrator - Coordinates all agents
- Quality Assurance - Code quality
- DevOps - CI/CD

**Specialized** (20 agents):
- Scientific Computing (4 agents)
- AI/ML (3 agents)
- Engineering (5 agents)
- Domain-specific (8 agents)

---

## Part 2: Your First Commands (10 minutes)

### Setup

Navigate to a project directory:
```bash
cd /path/to/your/project

# Or create a sample project
mkdir tutorial-project
cd tutorial-project

# Create sample Python file
cat > example.py << 'EOF'
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

def main():
    nums = [1, 2, 3, 4, 5]
    result = calculate_sum(nums)
    print(result)

if __name__ == "__main__":
    main()
EOF
```

### Command 1: Check Code Quality

```bash
/check-code-quality --language=python --auto-fix .
```

**What happens:**
1. Analyzes your Python code
2. Identifies quality issues
3. Automatically fixes them
4. Shows quality score

**Expected output:**
```
✓ Analyzing Python code...
✓ Found 1 file
✓ Identified 3 quality issues:
  - Missing docstrings
  - Non-Pythonic loop
  - Missing type hints
✓ Auto-fixing...
✓ Fixed 2 issues
✓ 1 issue requires review

Quality Score: 65/100 → 85/100
```

**Observe:**
- File was automatically improved
- Quality score increased
- Some issues auto-fixed, others flagged

### Command 2: Generate Tests

```bash
/generate-tests --type=unit --coverage=90 example.py
```

**What happens:**
1. Analyzes code structure
2. Generates comprehensive tests
3. Creates test file with 90% coverage

**Expected output:**
```
✓ Analyzing example.py...
✓ Generating unit tests...
✓ Created test_example.py
✓ Test coverage: 92%

Generated tests:
- test_calculate_sum_basic
- test_calculate_sum_empty_list
- test_calculate_sum_negative_numbers
- test_main_function
```

**Check result:**
```bash
ls -la
# You'll see test_example.py was created

cat test_example.py
# Review generated tests
```

### Command 3: Run Tests

```bash
/run-all-tests --auto-fix --coverage
```

**What happens:**
1. Runs all tests
2. Auto-fixes failures if possible
3. Shows coverage report

**Expected output:**
```
✓ Running tests...
✓ 4 tests passed
✓ 0 tests failed
✓ Coverage: 92%

Test Results:
  test_example.py ✓✓✓✓

Coverage Report:
  example.py: 92% (target: 90%) ✓
```

### Command 4: Verify Completeness

```bash
/double-check "code quality is good and tests pass"
```

**What happens:**
1. Verifies code quality
2. Checks test coverage
3. Identifies any gaps
4. Auto-completes if needed

**Expected output:**
```
✓ Verification complete

Criteria Checked:
  ✓ Code quality score ≥ 80
  ✓ Test coverage ≥ 90%
  ✓ All tests passing
  ✓ No critical issues

Status: COMPLETE ✓
```

---

## Part 3: Understanding Agents (5 minutes)

### Automatic Agent Selection

Commands automatically select appropriate agents:

```bash
# Quality check uses: Quality Assurance + Python Expert
/check-code-quality --language=python

# Optimization uses: Performance Engineer + Scientific Computing (if applicable)
/optimize --category=performance

# Testing uses: Testing + Quality Assurance
/generate-tests --type=unit
```

### Explicit Agent Selection

You can specify which agents to use:

```bash
# Use only core agents
/multi-agent-optimize --agents=core

# Use scientific agents
/optimize --agents=scientific

# Use all agents with orchestration
/multi-agent-optimize --agents=all --orchestrate
```

### Try It

```bash
# Compare different agent selections
/optimize --agents=auto example.py

# vs.

/optimize --agents=scientific example.py

# Observe different recommendations
```

---

## Part 4: Basic Workflows (5 minutes)

### What Are Workflows?

Workflows combine multiple commands into automated sequences.

### Quality Improvement Workflow

```bash
# Complete quality workflow
/multi-agent-optimize --mode=review --focus=quality --implement
```

**This executes:**
1. Check code quality
2. Auto-fix issues
3. Generate tests
4. Run tests
5. Optimize code
6. Verify completeness

**Watch the progress:**
```
Step 1/6: Analyzing code quality... ✓
Step 2/6: Fixing issues... ✓
Step 3/6: Generating tests... ✓
Step 4/6: Running tests... ✓
Step 5/6: Optimizing code... ✓
Step 6/6: Verifying... ✓

Workflow Complete!
Quality Score: 65 → 92
Test Coverage: 0% → 95%
Performance: +15%
```

### Performance Workflow

```bash
/multi-agent-optimize --mode=optimize --focus=performance --implement
```

**This executes:**
1. Profile code
2. Identify bottlenecks
3. Optimize algorithms
4. Optimize data structures
5. Verify improvements

---

## Part 5: Exploring Commands (3 minutes)

### Analysis & Planning

```bash
# Deep analysis of a problem
/think-ultra --depth=ultra "How to improve this code's performance?"

# Reflect on your work
/reflection --type=session
```

### Code Quality

```bash
# Refactor code
/refactor-clean --patterns=modern --implement

# Clean codebase
/clean-codebase --imports --dead-code
```

### Testing & Debugging

```bash
# Debug issues
/debug --issue=performance --profile

# Run all tests
/run-all-tests --scope=all
```

### Development Workflow

```bash
# Smart git commits
/commit --ai-message --validate

# Setup CI/CD
/ci-setup --platform=github
```

---

## Part 6: Getting Help (2 minutes)

### Documentation

```
final/docs/
├── MASTER_INDEX.md      # Start here
├── GETTING_STARTED.md   # Quick start
├── USER_GUIDE.md        # Complete guide
├── TROUBLESHOOTING.md   # Problem solving
└── FAQ.md               # Common questions
```

### Command Help

```bash
# Get help for any command
/command-name --help

# Examples
/check-code-quality --help
/generate-tests --help
```

### Diagnostics

```bash
# Check system status
claude-commands status

# Run diagnostics
claude-commands diagnostics
```

---

## Practice Exercise (5 minutes)

Create a complete development workflow:

```bash
# 1. Create new Python file
cat > calculator.py << 'EOF'
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    result = a
    for i in range(b - 1):
        result = result + a
    return result

def divide(a, b):
    return a / b
EOF

# 2. Check quality and fix
/check-code-quality --auto-fix calculator.py

# 3. Generate tests
/generate-tests --coverage=95 calculator.py

# 4. Run tests
/run-all-tests --auto-fix

# 5. Optimize
/optimize --implement calculator.py

# 6. Verify
/double-check "calculator module is production-ready"

# 7. Commit
/commit --ai-message --validate
```

### Expected Results

- Quality score ≥ 85
- Test coverage ≥ 95%
- All tests passing
- Code optimized (multiply function improved)
- Ready to commit

---

## Key Takeaways

✓ **Commands are easy** - Just use `/command-name` format
✓ **Agents work automatically** - System selects appropriate agents
✓ **Workflows automate** - Combine commands for complex tasks
✓ **Auto-fix is powerful** - Many issues fixed automatically
✓ **Documentation helps** - Comprehensive docs available

---

## Next Steps

1. **[Tutorial 02: Code Quality](tutorial-02-code-quality.md)** - Deep dive into quality improvements
2. **[Tutorial 03: Performance](tutorial-03-performance.md)** - Learn optimization techniques
3. **[User Guide](../docs/USER_GUIDE.md)** - Explore all features

---

## Quick Reference

```bash
# Quality
/check-code-quality --auto-fix
/refactor-clean --implement
/clean-codebase

# Testing
/generate-tests --coverage=90
/run-all-tests --auto-fix
/debug --profile

# Development
/commit --ai-message
/ci-setup --platform=github

# Analysis
/think-ultra --depth=ultra
/double-check "task description"

# Workflows
/multi-agent-optimize --focus=quality --implement
/multi-agent-optimize --focus=performance --implement
```

---

**Congratulations!** You've completed Tutorial 01. You now understand the basics of the Claude Code Command Executor Framework.

**Continue learning:** [Tutorial 02: Code Quality →](tutorial-02-code-quality.md)

---

**Tutorial Info**
- **Version**: 1.0.0
- **Last Updated**: September 2025
- **Difficulty**: Beginner
- **Time**: 30 minutes