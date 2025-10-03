# Workflow Framework - Claude Code Command Executor

## Overview

The Workflow Framework provides a comprehensive system for orchestrating command execution with dependency resolution, error handling, progress tracking, and result aggregation. It enables you to define complex multi-step workflows in YAML and execute them with automatic dependency management.

## Features

- **YAML-based workflow definitions** - Simple, declarative workflow language
- **Dependency resolution** - Automatic execution order with DAG-based topological sorting
- **Parallel execution** - Run independent steps in parallel for better performance
- **Error handling** - Configurable error strategies (continue, stop, rollback)
- **Retry logic** - Automatic retry with exponential/linear backoff
- **Progress tracking** - Real-time workflow execution monitoring
- **Variable substitution** - Dynamic workflow parameterization
- **Conditional execution** - Execute steps based on conditions
- **Rollback support** - Automatic rollback on failures
- **Template library** - Pre-built workflow templates for common scenarios

## Quick Start

### 1. List Available Workflows

```bash
python workflows/cli.py list
```

### 2. Run a Workflow

```bash
# Run quality improvement workflow
python workflows/cli.py run quality-improvement

# Run with custom variables
python workflows/cli.py run quality-improvement --var target_path=src/ --var language=python

# Dry run (simulate without executing)
python workflows/cli.py run quality-improvement --dry-run
```

### 3. Create Custom Workflow

```bash
# Create from template
python workflows/cli.py create my-workflow --template quality-improvement \
  --description "My custom workflow" --register
```

### 4. Validate Workflow

```bash
python workflows/cli.py validate my-workflow.yaml
```

## Workflow Definition Language

### Basic Structure

```yaml
workflow:
  name: "my-workflow"
  description: "Workflow description"
  version: "1.0"
  author: "Your Name"
  tags: [quality, testing]
  rollback_on_error: false

variables:
  target_path: "."
  language: "auto"
  coverage_target: 90

steps:
  - id: step1
    command: check-code-quality
    description: "Check code quality"
    flags:
      - --language=${language}
      - --analysis=basic
    input: ${target_path}

  - id: step2
    command: optimize
    flags:
      - --implement
      - --category=all
    depends_on: [step1]
    condition: "step1.success"
    on_error: continue
    retry:
      max_attempts: 3
      backoff: exponential
```

### Step Fields

- **id** (required): Unique step identifier
- **command** (required): Command to execute
- **description**: Human-readable description
- **flags**: List of command flags
- **input**: Input data for command
- **depends_on**: List of step IDs this step depends on
- **condition**: Execution condition expression
- **parallel**: Run in parallel with other steps at same level
- **on_error**: Error handling strategy (continue, stop, rollback)
- **retry**: Retry configuration
- **rollback_command**: Command to execute for rollback
- **rollback_flags**: Flags for rollback command
- **timeout**: Step timeout in seconds

### Dependency Resolution

Dependencies are automatically resolved into execution levels:

```yaml
steps:
  - id: A
    command: check-code-quality
    # No dependencies - runs first

  - id: B
    command: optimize
    depends_on: [A]
    # Runs after A

  - id: C
    command: generate-tests
    depends_on: [A]
    # Runs in parallel with B (same dependencies)

  - id: D
    command: run-all-tests
    depends_on: [B, C]
    # Runs after both B and C complete
```

Execution order: `[A] -> [B, C] -> [D]`

### Parallel Execution

Execute steps in parallel explicitly:

```yaml
steps:
  - id: parallel_checks
    parallel: true
    steps:
      - id: quality_check
        command: check-code-quality

      - id: security_check
        command: check-code-quality
        flags: [--security]

      - id: test_check
        command: run-all-tests
```

### Conditional Execution

Execute steps based on conditions:

```yaml
steps:
  - id: check_quality
    command: check-code-quality

  - id: fix_issues
    command: check-code-quality
    flags: [--auto-fix]
    depends_on: [check_quality]
    condition: "check_quality.success"
    # Only runs if check_quality succeeded
```

### Error Handling

Configure error handling per step:

```yaml
steps:
  - id: risky_step
    command: refactor-clean
    on_error: rollback  # Options: continue, stop, rollback
    retry:
      max_attempts: 3
      backoff: exponential  # Options: linear, exponential
    rollback_command: git
    rollback_flags: ["reset", "--hard", "HEAD~1"]
```

### Variable Substitution

Use variables with `${variable_name}` syntax:

```yaml
variables:
  language: "python"
  coverage: 90

steps:
  - id: test
    command: generate-tests
    flags:
      - --language=${language}
      - --coverage=${coverage}
```

Override variables at runtime:

```bash
python workflows/cli.py run my-workflow --var language=julia --var coverage=95
```

## Pre-built Workflow Templates

### quality-improvement.yaml
Complete code quality improvement workflow:
- Check code quality
- Auto-fix issues
- Clean codebase
- Refactor code
- Generate tests
- Run tests
- Commit changes

### performance-optimization.yaml
Performance optimization workflow:
- Profile baseline
- Identify bottlenecks
- Optimize code (algorithm, memory, I/O)
- Validate optimizations
- Benchmark improvements
- Update documentation

### refactoring-workflow.yaml
Safe refactoring workflow with backup:
- Create backup
- Run baseline tests
- Analyze complexity
- Apply refactoring
- Clean after refactor
- Validate with tests
- Check performance
- Commit or rollback

### documentation-generation.yaml
Documentation generation workflow:
- Analyze code structure
- Generate API docs
- Create README
- Generate examples
- Update changelog
- Commit documentation

### ci-cd-setup.yaml
CI/CD pipeline setup:
- Analyze project
- Setup CI pipeline
- Configure testing, security, monitoring
- Generate test suite
- Validate CI config
- Commit CI setup

### complete-development-cycle.yaml
Full development cycle:
- Quality check
- Clean codebase
- Optimize performance
- Refactor code
- Generate tests
- Run all tests
- Generate documentation
- Double-check work
- Create commit

### research-workflow.yaml
Scientific computing workflow:
- Analyze scientific code
- Debug scientific issues
- Optimize algorithms
- Parallel optimization (memory, I/O, concurrency)
- Generate scientific tests
- Run reproducible tests
- Benchmark performance
- Generate research docs

### migration-workflow.yaml
Code migration/modernization:
- Analyze legacy code
- Plan migration
- Backup codebase
- Adopt code
- Apply modern patterns
- Clean migrated code
- Validate functionality
- Performance comparison

## CLI Commands

### List Workflows

```bash
# List all workflows
python workflows/cli.py list

# Filter by category
python workflows/cli.py list --category template

# Filter by tag
python workflows/cli.py list --tag quality
```

### Run Workflow

```bash
# Run workflow by name
python workflows/cli.py run quality-improvement

# Run workflow from file
python workflows/cli.py run /path/to/workflow.yaml

# Dry run
python workflows/cli.py run quality-improvement --dry-run

# With variables
python workflows/cli.py run quality-improvement \
  --var target_path=src/ \
  --var language=python \
  --var coverage_target=95

# Save result to file
python workflows/cli.py run quality-improvement --output result.json

# Verbose logging
python workflows/cli.py run quality-improvement --verbose

# Log to file
python workflows/cli.py run quality-improvement --log workflow.log
```

### Validate Workflow

```bash
# Validate workflow
python workflows/cli.py validate my-workflow.yaml

# Strict validation (fail on warnings)
python workflows/cli.py validate my-workflow.yaml --strict
```

### Create Workflow

```bash
# Create from template
python workflows/cli.py create my-workflow \
  --template quality-improvement \
  --description "My custom quality workflow"

# Create with variables
python workflows/cli.py create my-workflow \
  --template quality-improvement \
  --var language=julia \
  --var coverage_target=95

# Create and register
python workflows/cli.py create my-workflow \
  --template quality-improvement \
  --register

# Specify output path
python workflows/cli.py create my-workflow \
  --template quality-improvement \
  --output custom/my-workflow.yaml
```

### Search Workflows

```bash
# Search by name, description, or tags
python workflows/cli.py search quality

python workflows/cli.py search optimization

python workflows/cli.py search testing
```

### Workflow Info

```bash
# Show detailed workflow information
python workflows/cli.py info quality-improvement
```

### Registry Statistics

```bash
# Show registry statistics
python workflows/cli.py stats
```

## Python API

### Execute Workflow

```python
import asyncio
from pathlib import Path
from workflows import WorkflowExecutor

async def main():
    executor = WorkflowExecutor(dry_run=False, verbose=True)

    result = await executor.execute(
        workflow_path=Path("workflows/templates/quality-improvement.yaml"),
        variables={'target_path': 'src/', 'language': 'python'}
    )

    print(f"Status: {result.status.value}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Successful steps: {result.metadata['successful_steps']}")

asyncio.run(main())
```

### Workflow Registry

```python
from workflows import WorkflowRegistry

registry = WorkflowRegistry()

# List workflows
workflows = registry.list_workflows()

# Search workflows
results = registry.search_workflows("quality")

# Get workflow path
path = registry.get_workflow_path("quality-improvement")

# Get workflows by command
workflows = registry.get_workflows_by_command("optimize")

# Get statistics
stats = registry.get_workflow_stats()
```

### Workflow Validator

```python
from pathlib import Path
from workflows import WorkflowValidator

validator = WorkflowValidator()

result = validator.validate_workflow(
    workflow_path=Path("my-workflow.yaml"),
    strict=False
)

if result['valid']:
    print("Workflow is valid")
else:
    print("Errors:", result['errors'])
    print("Warnings:", result['warnings'])
```

## Architecture

### Core Components

- **WorkflowEngine**: Main orchestrator for workflow execution
- **WorkflowParser**: YAML parser and structure validator
- **DependencyResolver**: Resolves dependencies and determines execution order
- **CommandComposer**: Chains and composes commands together

### Library Components

- **WorkflowRegistry**: Manages available workflows and templates
- **WorkflowValidator**: Comprehensive workflow validation
- **WorkflowExecutor**: High-level execution interface

## Best Practices

1. **Always validate workflows** before running in production
2. **Use dry-run** to test workflows before actual execution
3. **Add error handling** (on_error, retry) to critical steps
4. **Include rollback commands** for destructive operations
5. **Use descriptive step IDs** and descriptions
6. **Tag workflows** for easy discovery
7. **Version workflows** and track changes
8. **Test workflows** with different input parameters
9. **Monitor workflow execution** with verbose logging
10. **Save results** for audit and debugging

## Troubleshooting

### Workflow validation fails

```bash
# Check validation errors
python workflows/cli.py validate my-workflow.yaml

# Common issues:
# - Circular dependencies
# - Unknown commands
# - Missing required fields
# - Invalid flag combinations
```

### Workflow execution hangs

```bash
# Use verbose mode to see progress
python workflows/cli.py run my-workflow --verbose

# Check for:
# - Deadlocks in dependencies
# - Long-running commands without timeout
# - Missing step dependencies
```

### Steps executed in wrong order

```bash
# Validate dependencies
python workflows/cli.py info my-workflow

# Check:
# - Step dependencies are correct
# - No circular dependencies
# - Parallel steps have same dependencies
```

## Contributing

To add new workflow templates:

1. Create YAML file in `workflows/templates/`
2. Follow the workflow definition language specification
3. Validate the workflow: `python workflows/cli.py validate my-template.yaml`
4. Test the workflow: `python workflows/cli.py run my-template --dry-run`
5. Document the workflow purpose and usage

## Support

For issues and questions:
- Check validation output for errors
- Run with `--verbose` flag for detailed logging
- Review workflow execution results
- Consult WORKFLOW_LANGUAGE.md for syntax reference