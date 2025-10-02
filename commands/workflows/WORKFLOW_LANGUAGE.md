# Workflow Definition Language Reference

## Complete YAML Syntax Reference

### Workflow Section

```yaml
workflow:
  name: string                    # Required: Workflow identifier
  description: string             # Required: Human-readable description
  version: string                 # Required: Semantic version (e.g., "1.0")
  author: string                  # Optional: Workflow author
  tags: [string]                  # Optional: Tags for categorization
  rollback_on_error: boolean      # Optional: Auto-rollback on failure (default: false)
  max_parallel: integer           # Optional: Max parallel steps (default: 5)
  timeout: integer                # Optional: Workflow timeout in seconds
```

### Variables Section

```yaml
variables:
  variable_name: value            # Define workflow variables
  target_path: "."
  language: "auto"
  coverage_target: 90
  custom_config:
    nested: "value"
```

**Variable Usage:**
- Reference with `${variable_name}` syntax
- Available in all step fields
- Can be overridden at runtime with CLI `--var` flag
- Support nested variables: `${config.nested}`

### Steps Section

```yaml
steps:
  - id: string                    # Required: Unique step identifier
    command: string               # Required: Command name
    description: string           # Optional: Step description
    flags: [string]               # Optional: Command flags
    input: any                    # Optional: Input data/path
    depends_on: [string]          # Optional: Step ID dependencies
    condition: string             # Optional: Execution condition
    parallel: boolean             # Optional: Enable parallel execution
    on_error: string              # Optional: Error handling (continue|stop|rollback)
    retry: object                 # Optional: Retry configuration
    rollback_command: string      # Optional: Rollback command
    rollback_flags: [string]      # Optional: Rollback flags
    timeout: integer              # Optional: Step timeout in seconds
```

## Step Field Details

### Required Fields

#### id
Unique identifier for the step. Used for dependency references.

```yaml
id: check_quality
```

**Rules:**
- Must be unique within workflow
- Use snake_case or kebab-case
- Should be descriptive

#### command
Command to execute. Must be a valid command from the command system.

```yaml
command: check-code-quality
```

**Available Commands:**
- check-code-quality
- optimize
- run-all-tests
- generate-tests
- refactor-clean
- update-docs
- commit
- ci-setup
- debug
- explain-code
- fix-commit-errors
- fix-github-issue
- multi-agent-optimize
- double-check
- adopt-code
- clean-codebase
- reflection
- think-ultra

### Optional Fields

#### description
Human-readable description of what the step does.

```yaml
description: "Analyze code quality and identify issues"
```

#### flags
List of command-line flags to pass to the command.

```yaml
flags:
  - --language=python
  - --analysis=thorough
  - --auto-fix
```

**Flag Format:**
- Single dash for short flags: `-v`
- Double dash for long flags: `--verbose`
- Equals for values: `--language=python`
- Space-separated for multi-word: `--output file.txt`

#### input
Input data or path for the command.

```yaml
# Path input
input: "src/"

# Variable reference
input: ${target_path}

# Previous step output
input: "step1.output"

# Complex input
input:
  path: "src/"
  config: "config.yaml"
```

#### depends_on
List of step IDs that must complete before this step runs.

```yaml
depends_on: [step1, step2, step3]
```

**Dependency Rules:**
- All dependencies must exist
- No circular dependencies allowed
- Steps with same dependencies can run in parallel
- Empty list means no dependencies (runs first)

#### condition
Execution condition expression. Step only runs if condition evaluates to true.

```yaml
# Check step success
condition: "step1.success"

# Check variable value
condition: "quality_score > 80"

# Check variable existence
condition: "has_performance_issues"

# Complex condition (simplified evaluation)
condition: "step1.success and quality_score > 80"
```

**Supported Conditions:**
- `stepX.success` - Step completed successfully
- `variable_name` - Variable exists and is truthy
- Comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Logical operators: `and`, `or`, `not`

#### parallel
Enable parallel execution with other steps at same level.

```yaml
parallel: true
```

**Parallel Execution:**
- Steps with same dependencies can run in parallel
- Use explicit `parallel: true` flag
- Or group steps in parallel block:

```yaml
- id: parallel_group
  parallel: true
  steps:
    - id: task1
      command: check-code-quality
    - id: task2
      command: run-all-tests
```

#### on_error
Error handling strategy when step fails.

```yaml
on_error: continue  # Options: continue, stop, rollback
```

**Strategies:**
- `continue`: Log error and continue with next step
- `stop`: Stop workflow immediately (default)
- `rollback`: Execute rollback actions and stop

#### retry
Retry configuration for failed steps.

```yaml
retry:
  max_attempts: 3              # Maximum retry attempts
  backoff: exponential         # Backoff strategy (linear|exponential)
  initial_delay: 1             # Initial delay in seconds (default: 1)
```

**Backoff Strategies:**
- `linear`: Delay increases linearly (1s, 2s, 3s, ...)
- `exponential`: Delay doubles each time (1s, 2s, 4s, 8s, ...)

#### rollback_command
Command to execute for rolling back this step's changes.

```yaml
rollback_command: git
rollback_flags: ["reset", "--hard", "HEAD~1"]
```

#### timeout
Step timeout in seconds. Step fails if exceeds timeout.

```yaml
timeout: 300  # 5 minutes
```

## Advanced Features

### Nested Steps (Parallel Groups)

Execute multiple steps in parallel:

```yaml
steps:
  - id: parallel_optimizations
    parallel: true
    depends_on: [initial_analysis]
    steps:
      - id: optimize_memory
        command: optimize
        flags: [--category=memory, --implement]

      - id: optimize_io
        command: optimize
        flags: [--category=io, --implement]

      - id: optimize_concurrency
        command: optimize
        flags: [--category=concurrency, --implement]
```

### Variable Substitution

Use variables throughout workflow:

```yaml
variables:
  base_path: "src/"
  language: "python"
  config:
    coverage: 90
    strict: true

steps:
  - id: test
    command: generate-tests
    flags:
      - --language=${language}
      - --coverage=${config.coverage}
    input: ${base_path}
```

### Conditional Workflows

Create dynamic workflows based on conditions:

```yaml
steps:
  - id: check_quality
    command: check-code-quality

  - id: fix_minor_issues
    command: check-code-quality
    flags: [--auto-fix]
    depends_on: [check_quality]
    condition: "quality_score < 80 and quality_score >= 60"

  - id: major_refactoring
    command: refactor-clean
    flags: [--implement, --patterns=modern]
    depends_on: [check_quality]
    condition: "quality_score < 60"
```

### Error Handling Patterns

Comprehensive error handling:

```yaml
steps:
  - id: risky_operation
    command: refactor-clean
    flags: [--implement]
    on_error: rollback
    retry:
      max_attempts: 3
      backoff: exponential
    rollback_command: git
    rollback_flags: ["checkout", "backup-branch"]

  - id: alternative_approach
    command: refactor-clean
    flags: [--scope=file]
    depends_on: [risky_operation]
    condition: "not risky_operation.success"
```

## Best Practices

### Naming Conventions

**Step IDs:**
- Use snake_case: `check_code_quality`
- Be descriptive: `run_integration_tests` not `step3`
- Group related steps: `optimize_memory`, `optimize_io`

**Variable Names:**
- Use snake_case: `target_path`, `coverage_target`
- Be specific: `python_version` not `version`
- Group related: `config.coverage`, `config.parallel`

### Dependency Organization

**Good:**
```yaml
steps:
  - id: prepare
    command: check-code-quality

  - id: optimize
    command: optimize
    depends_on: [prepare]

  - id: test
    command: run-all-tests
    depends_on: [optimize]
```

**Better (with parallelization):**
```yaml
steps:
  - id: prepare
    command: check-code-quality

  - id: optimize
    command: optimize
    depends_on: [prepare]

  - id: refactor
    command: refactor-clean
    depends_on: [prepare]
    # Can run in parallel with optimize

  - id: test
    command: run-all-tests
    depends_on: [optimize, refactor]
```

### Error Handling

Always add error handling to critical steps:

```yaml
steps:
  - id: critical_step
    command: refactor-clean
    on_error: rollback
    retry:
      max_attempts: 2
      backoff: linear
    rollback_command: git
    rollback_flags: ["reset", "--hard", "HEAD"]
```

### Documentation

Add descriptions to complex steps:

```yaml
steps:
  - id: complex_optimization
    command: optimize
    description: |
      Performs comprehensive optimization including:
      - Algorithm complexity reduction
      - Memory allocation optimization
      - I/O operation batching
    flags: [--category=all, --implement]
```

## Examples

### Simple Sequential Workflow

```yaml
workflow:
  name: "simple-quality-check"
  description: "Basic quality check and fix"
  version: "1.0"

steps:
  - id: check
    command: check-code-quality
    flags: [--language=python]

  - id: fix
    command: check-code-quality
    flags: [--auto-fix]
    depends_on: [check]

  - id: verify
    command: run-all-tests
    depends_on: [fix]
```

### Parallel Workflow

```yaml
workflow:
  name: "parallel-checks"
  description: "Run multiple checks in parallel"
  version: "1.0"

steps:
  - id: quality_check
    command: check-code-quality

  - id: test_check
    command: run-all-tests

  - id: security_check
    command: debug
    flags: [--issue=security]

  # All three run in parallel (no dependencies)

  - id: commit
    command: commit
    depends_on: [quality_check, test_check, security_check]
```

### Conditional Workflow

```yaml
workflow:
  name: "conditional-optimization"
  description: "Optimize only if needed"
  version: "1.0"

variables:
  quality_threshold: 80

steps:
  - id: assess
    command: check-code-quality

  - id: optimize
    command: optimize
    flags: [--implement]
    depends_on: [assess]
    condition: "quality_score < ${quality_threshold}"

  - id: skip_optimize
    command: update-docs
    depends_on: [assess]
    condition: "quality_score >= ${quality_threshold}"
```

### Error Recovery Workflow

```yaml
workflow:
  name: "safe-refactoring"
  description: "Refactor with automatic rollback"
  version: "1.0"
  rollback_on_error: true

steps:
  - id: backup
    command: commit
    flags: [--staged]
    rollback_command: git
    rollback_flags: ["reset", "--hard", "HEAD~1"]

  - id: refactor
    command: refactor-clean
    flags: [--implement]
    depends_on: [backup]
    on_error: rollback
    retry:
      max_attempts: 2
      backoff: linear

  - id: validate
    command: run-all-tests
    depends_on: [refactor]
    on_error: rollback

  - id: commit
    command: commit
    flags: [--all, --ai-message]
    depends_on: [validate]
```

## Validation Rules

Workflows are validated for:

1. **Structure**: Required sections and fields present
2. **Syntax**: Valid YAML and field types
3. **Semantics**: Logical consistency
4. **Dependencies**: No circular dependencies, all references exist
5. **Commands**: All commands are valid
6. **Flags**: Compatible flag combinations
7. **Best Practices**: Recommended patterns followed

Use `python workflows/cli.py validate workflow.yaml` to check your workflow.