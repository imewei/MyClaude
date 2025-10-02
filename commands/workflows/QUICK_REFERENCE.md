# Workflow Framework - Quick Reference

## CLI Commands Cheat Sheet

### List Workflows
```bash
python workflows/cli.py list                    # List all
python workflows/cli.py list --category template # Templates only
python workflows/cli.py list --tag quality      # By tag
```

### Run Workflow
```bash
python workflows/cli.py run WORKFLOW            # Run by name
python workflows/cli.py run WORKFLOW --dry-run  # Simulate
python workflows/cli.py run WORKFLOW --var key=val --var key2=val2
python workflows/cli.py run WORKFLOW -o result.json --verbose
```

### Validate
```bash
python workflows/cli.py validate workflow.yaml
python workflows/cli.py validate workflow.yaml --strict
```

### Create
```bash
python workflows/cli.py create NAME -t TEMPLATE
python workflows/cli.py create NAME -t TEMPLATE --var key=val --register
```

### Search & Info
```bash
python workflows/cli.py search QUERY
python workflows/cli.py info WORKFLOW
python workflows/cli.py stats
```

## YAML Syntax Quick Reference

### Minimal Workflow
```yaml
workflow:
  name: "my-workflow"
  description: "Description"
  version: "1.0"

steps:
  - id: step1
    command: check-code-quality

  - id: step2
    command: run-all-tests
    depends_on: [step1]
```

### With Variables
```yaml
variables:
  target: "src/"
  language: "python"

steps:
  - id: step1
    command: check-code-quality
    flags:
      - --language=${language}
    input: ${target}
```

### With Error Handling
```yaml
steps:
  - id: step1
    command: refactor-clean
    on_error: rollback
    retry:
      max_attempts: 3
      backoff: exponential
    rollback_command: git
    rollback_flags: ["reset", "--hard", "HEAD"]
```

### Parallel Execution
```yaml
steps:
  - id: parallel_tasks
    parallel: true
    steps:
      - id: task1
        command: check-code-quality
      - id: task2
        command: run-all-tests
```

### Conditional Execution
```yaml
steps:
  - id: step1
    command: check-code-quality

  - id: step2
    command: optimize
    depends_on: [step1]
    condition: "step1.success"
```

## Python API Quick Reference

### Execute Workflow
```python
import asyncio
from pathlib import Path
from workflows import WorkflowExecutor

async def main():
    executor = WorkflowExecutor(dry_run=False, verbose=True)
    result = await executor.execute(
        workflow_path=Path("workflow.yaml"),
        variables={'key': 'value'}
    )
    print(f"Status: {result.status.value}")

asyncio.run(main())
```

### Registry
```python
from workflows import WorkflowRegistry

registry = WorkflowRegistry()
workflows = registry.list_workflows()
path = registry.get_workflow_path("quality-improvement")
```

### Validator
```python
from workflows import WorkflowValidator

validator = WorkflowValidator()
result = validator.validate_workflow(Path("workflow.yaml"))
print(result['errors'])
```

## Template Quick Reference

| Template | Use Case | Duration |
|----------|----------|----------|
| quality-improvement | Code quality enhancement | 5-15 min |
| performance-optimization | Performance tuning | 10-30 min |
| refactoring-workflow | Safe refactoring | 10-25 min |
| documentation-generation | Auto-generate docs | 5-10 min |
| ci-cd-setup | CI/CD pipeline | 5-15 min |
| complete-development-cycle | Full workflow | 20-45 min |
| research-workflow | Scientific computing | 15-40 min |
| migration-workflow | Code migration | 30-60 min |

## Common Patterns

### Sequential Workflow
```yaml
steps:
  - id: A
    command: cmd1
  - id: B
    command: cmd2
    depends_on: [A]
  - id: C
    command: cmd3
    depends_on: [B]
# Execution: A → B → C
```

### Parallel Workflow
```yaml
steps:
  - id: A
    command: cmd1
  - id: B
    command: cmd2
  - id: C
    command: cmd3
  # All run in parallel
```

### Diamond Pattern
```yaml
steps:
  - id: A
    command: cmd1
  - id: B
    command: cmd2
    depends_on: [A]
  - id: C
    command: cmd3
    depends_on: [A]
  - id: D
    command: cmd4
    depends_on: [B, C]
# Execution: A → [B, C] → D
```

## Error Handling Strategies

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| continue | Log error, continue | Non-critical steps |
| stop | Stop workflow | Default, critical steps |
| rollback | Execute rollback, stop | Destructive operations |

## Validation Checklist

- [ ] workflow section with name, description, version
- [ ] steps is a list with at least one step
- [ ] Each step has unique id
- [ ] Each step has valid command
- [ ] Dependencies reference existing steps
- [ ] No circular dependencies
- [ ] Valid flag combinations
- [ ] Error handling on critical steps

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Validation fails | Check `python workflows/cli.py validate` output |
| Execution hangs | Use `--verbose` flag, check dependencies |
| Wrong order | Verify `depends_on` fields |
| Command not found | Check command name spelling |
| Variable not substituted | Use `${var}` syntax, check variable name |

## File Locations

```
workflows/
├── cli.py                    # CLI interface
├── core/
│   ├── workflow_engine.py    # Main engine
│   ├── workflow_parser.py    # YAML parser
│   ├── dependency_resolver.py # DAG resolver
│   └── command_composer.py   # Command chaining
├── library/
│   ├── workflow_registry.py  # Template registry
│   ├── workflow_validator.py # Validation
│   └── workflow_executor.py  # Execution
└── templates/               # Pre-built workflows
    ├── quality-improvement.yaml
    ├── performance-optimization.yaml
    └── ...
```

## Useful Commands

```bash
# Test workflow without execution
python workflows/cli.py run WORKFLOW --dry-run --verbose

# Create custom workflow and test
python workflows/cli.py create my-workflow -t quality-improvement
python workflows/cli.py validate custom/my-workflow.yaml
python workflows/cli.py run my-workflow --dry-run

# Find workflows for specific command
python workflows/cli.py list | grep optimize

# Get help
python workflows/cli.py --help
python workflows/cli.py run --help
```

## Best Practices

1. Always test with `--dry-run` first
2. Add descriptions to complex steps
3. Use error handling on critical steps
4. Version your custom workflows
5. Validate before running
6. Use meaningful step IDs
7. Add rollback commands for destructive operations
8. Tag workflows for easy discovery
9. Document custom workflows
10. Save execution results for audit