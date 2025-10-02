# Workflow Framework - Complete Index

## Documentation

### Getting Started
- [README.md](README.md) - Complete user guide with examples
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference and cheat sheet

### Reference Documentation
- [WORKFLOW_LANGUAGE.md](WORKFLOW_LANGUAGE.md) - Complete YAML syntax reference
- [TEMPLATE_GUIDE.md](TEMPLATE_GUIDE.md) - Template documentation and guide

### Implementation
- [PHASE_5_IMPLEMENTATION_SUMMARY.md](PHASE_5_IMPLEMENTATION_SUMMARY.md) - Implementation details

## Source Code

### Core Engine (1,845 lines)
- [core/workflow_engine.py](core/workflow_engine.py) - Main orchestrator (513 lines)
- [core/workflow_parser.py](core/workflow_parser.py) - YAML parser (419 lines)
- [core/dependency_resolver.py](core/dependency_resolver.py) - DAG resolver (434 lines)
- [core/command_composer.py](core/command_composer.py) - Command chaining (465 lines)
- [core/__init__.py](core/__init__.py) - Core exports (14 lines)

### Library Components (1,129 lines)
- [library/workflow_registry.py](library/workflow_registry.py) - Template registry (337 lines)
- [library/workflow_validator.py](library/workflow_validator.py) - Validation (419 lines)
- [library/workflow_executor.py](library/workflow_executor.py) - Execution interface (363 lines)
- [library/__init__.py](library/__init__.py) - Library exports (10 lines)

### CLI & Interface (357 lines)
- [cli.py](cli.py) - Command-line interface (357 lines)

### Configuration
- [requirements.txt](requirements.txt) - Python dependencies

## Workflow Templates

### Production Templates
1. [templates/quality-improvement.yaml](templates/quality-improvement.yaml)
   - Code quality enhancement workflow
   - 8 steps: check → fix → clean → refactor → test → commit

2. [templates/performance-optimization.yaml](templates/performance-optimization.yaml)
   - Performance optimization workflow
   - 8 steps with parallel optimizations

3. [templates/refactoring-workflow.yaml](templates/refactoring-workflow.yaml)
   - Safe refactoring with rollback
   - 9 steps: backup → analyze → refactor → validate → commit

4. [templates/documentation-generation.yaml](templates/documentation-generation.yaml)
   - Documentation generation workflow
   - 7 steps with parallel doc generation

5. [templates/ci-cd-setup.yaml](templates/ci-cd-setup.yaml)
   - CI/CD pipeline setup workflow
   - 7 steps: analyze → setup → configure → validate → commit

6. [templates/complete-development-cycle.yaml](templates/complete-development-cycle.yaml)
   - Full development cycle workflow
   - 11 steps: quality → optimize → refactor → test → docs → commit

7. [templates/research-workflow.yaml](templates/research-workflow.yaml)
   - Scientific computing workflow
   - 10 steps: debug → optimize → test → benchmark → docs → commit

8. [templates/migration-workflow.yaml](templates/migration-workflow.yaml)
   - Code migration and modernization workflow
   - 13 steps: analyze → plan → migrate → validate → optimize → commit

### Example Templates
9. [templates/simple-example.yaml](templates/simple-example.yaml)
   - Simple example demonstrating basic features
   - 3 steps: check → test → commit

## Architecture

### Component Hierarchy
```
WorkflowExecutor (High-level interface)
    └── WorkflowEngine (Core orchestrator)
        ├── WorkflowParser (YAML parsing)
        ├── DependencyResolver (DAG resolution)
        └── CommandComposer (Command execution)

WorkflowRegistry (Template management)
WorkflowValidator (Validation)
```

### Data Flow
```
YAML File → Parser → Validator → Dependency Resolver → Engine → Executor
                                        ↓
                                  CommandComposer
                                        ↓
                                  Command Execution
```

## Statistics

### Code Metrics
- **Total Lines**: 3,331 lines of Python code
- **Core Components**: 1,845 lines (55%)
- **Library Components**: 1,129 lines (34%)
- **CLI Interface**: 357 lines (11%)
- **Templates**: 9 YAML workflows
- **Documentation**: 5 comprehensive guides

### File Count
- **Python Modules**: 11 files
- **YAML Templates**: 9 files
- **Documentation**: 5 files
- **Configuration**: 1 file
- **Total**: 26 files

## Key Features by Component

### WorkflowEngine
- Async workflow execution
- State management
- Error handling and rollback
- Progress tracking
- Retry logic with backoff
- Variable substitution
- Condition evaluation

### WorkflowParser
- YAML parsing
- Structure validation
- Circular dependency detection
- Command verification
- Flag validation
- Metadata extraction

### DependencyResolver
- DAG construction
- Topological sorting
- Parallel step grouping
- Critical path calculation
- Cycle detection
- Execution optimization

### CommandComposer
- Command execution
- Command chaining
- Parallel execution
- Conditional execution
- Output transformation
- Subprocess management

### WorkflowRegistry
- Template discovery
- Workflow registration
- Search and filtering
- Metadata management
- Statistics

### WorkflowValidator
- Comprehensive validation
- Command compatibility
- Flag validation
- Performance analysis
- Best practices checking

### WorkflowExecutor
- High-level interface
- Progress tracking
- Result persistence
- Batch execution
- Reporting

## CLI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| list | List workflows | `python workflows/cli.py list` |
| run | Execute workflow | `python workflows/cli.py run WORKFLOW` |
| validate | Validate workflow | `python workflows/cli.py validate FILE` |
| create | Create from template | `python workflows/cli.py create NAME -t TEMPLATE` |
| search | Search workflows | `python workflows/cli.py search QUERY` |
| info | Show workflow info | `python workflows/cli.py info WORKFLOW` |
| stats | Registry statistics | `python workflows/cli.py stats` |

## API Classes

### Core Classes
- `WorkflowEngine` - Main orchestration engine
- `WorkflowParser` - YAML parser
- `DependencyResolver` - Dependency resolution
- `CommandComposer` - Command composition

### Library Classes
- `WorkflowRegistry` - Template registry
- `WorkflowValidator` - Validation
- `WorkflowExecutor` - High-level executor

### Data Classes
- `WorkflowContext` - Execution context
- `WorkflowResult` - Execution result
- `StepResult` - Step result

### Enums
- `WorkflowStatus` - Workflow states
- `StepStatus` - Step states

## Usage Examples

### CLI
```bash
# List all workflows
python workflows/cli.py list

# Run workflow with dry-run
python workflows/cli.py run quality-improvement --dry-run

# Create custom workflow
python workflows/cli.py create my-workflow -t quality-improvement

# Validate workflow
python workflows/cli.py validate my-workflow.yaml
```

### Python API
```python
from workflows import WorkflowExecutor
import asyncio

async def main():
    executor = WorkflowExecutor()
    result = await executor.execute(
        workflow_path=Path("workflow.yaml"),
        variables={'language': 'python'}
    )

asyncio.run(main())
```

## Dependencies

- PyYAML >= 6.0.1 (YAML parsing)
- networkx >= 3.0 (Graph operations)
- rich >= 13.0.0 (CLI output, optional)
- asyncio >= 3.4.3 (Async operations)
- typing-extensions >= 4.5.0 (Type hints)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. List Available Workflows
```bash
python workflows/cli.py list
```

### 3. Run a Workflow
```bash
python workflows/cli.py run quality-improvement --dry-run
```

### 4. Create Custom Workflow
```bash
python workflows/cli.py create my-workflow -t quality-improvement
```

## Testing

### Dry Run
```bash
python workflows/cli.py run WORKFLOW --dry-run
```

### Validation
```bash
python workflows/cli.py validate workflow.yaml --strict
```

### Verbose Output
```bash
python workflows/cli.py run WORKFLOW --verbose
```

## Contributing

### Adding New Templates
1. Create YAML file in `templates/`
2. Follow workflow language specification
3. Validate: `python workflows/cli.py validate template.yaml`
4. Test: `python workflows/cli.py run template --dry-run`
5. Document in TEMPLATE_GUIDE.md

### Extending Core
1. Add functionality to appropriate module
2. Maintain type hints and docstrings
3. Follow existing patterns
4. Update documentation

## Support & Documentation

- **User Guide**: README.md
- **Quick Reference**: QUICK_REFERENCE.md
- **Language Reference**: WORKFLOW_LANGUAGE.md
- **Template Guide**: TEMPLATE_GUIDE.md
- **Implementation**: PHASE_5_IMPLEMENTATION_SUMMARY.md

## License

Part of the Claude Code Command Executor Framework

---

**Framework Version**: 1.0.0
**Phase**: 5 (Workflow Framework)
**Status**: Complete
**Last Updated**: 2025-09-29