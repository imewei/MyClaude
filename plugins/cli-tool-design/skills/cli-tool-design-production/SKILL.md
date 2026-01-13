---
name: cli-tool-design-production
version: "1.0.7"
maturity: "5-Expert"
specialization: CLI Tool Development
description: Build production-ready CLI tools with Python Click/Typer, Node.js Commander/Inquirer, Go Cobra, and shell scripting. Use when creating command-line applications with interactive prompts, implementing developer workflow automation, designing project scaffolding tools, building TUIs with progress bars and spinners, or creating deployment and release automation tools.
---

# CLI Tool Design and Production

Production-ready command-line tools with modern frameworks and workflow automation.

---

## Framework Selection

| Framework | Language | Best For | Features |
|-----------|----------|----------|----------|
| Click | Python | Complex CLIs | Groups, callbacks, decorators |
| Typer | Python | Type-hinted CLIs | Auto-completion, less boilerplate |
| Commander | Node.js | npm packages | Subcommands, options |
| Inquirer | Node.js | Interactive prompts | Wizards, menus |
| Cobra | Go | Compiled CLIs | Fast, flags, config |

---

## Python Click CLI

```python
import click
from rich.console import Console
from rich.progress import Progress

console = Console()

@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', count=True)
@click.pass_context
def cli(ctx, verbose: int):
    """Production CLI tool."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('name')
@click.option('--template', default='default')
@click.option('--force', is_flag=True)
def init(name: str, template: str, force: bool):
    """Initialize new project."""
    with Progress() as progress:
        task = progress.add_task(f"Creating '{name}'...", total=None)
        # Create project structure
        Path(name).mkdir(exist_ok=True)
        progress.update(task, completed=True)
    console.print(f"[green]✓[/green] Project '{name}' created!")

@cli.command()
@click.option('--env', type=click.Choice(['dev', 'staging', 'production']))
@click.option('--dry-run', is_flag=True)
def deploy(env: str, dry_run: bool):
    """Deploy application."""
    if env == 'production' and not dry_run:
        if not click.confirm('Deploy to production?'):
            return
    console.print(f"[cyan]Deploying to {env}...[/cyan]")
```

### pyproject.toml Entry Point

```toml
[tool.poetry.scripts]
cli-tool = "cli_tool.cli:cli"

[tool.poetry.dependencies]
click = "^8.1"
rich = "^13.0"
```

---

## Node.js Interactive CLI

```javascript
const { Command } = require('commander');
const inquirer = require('inquirer');
const chalk = require('chalk');
const ora = require('ora');

const program = new Command();

program
  .name('dev-tool')
  .version('1.0.0');

program
  .command('init')
  .description('Initialize project')
  .action(async () => {
    const answers = await inquirer.prompt([
      {
        type: 'input',
        name: 'projectName',
        message: 'Project name:',
        validate: input => /^[a-z0-9-]+$/.test(input) || 'Lowercase alphanumeric'
      },
      {
        type: 'list',
        name: 'framework',
        message: 'Framework:',
        choices: ['React', 'Vue', 'Svelte']
      },
      {
        type: 'checkbox',
        name: 'features',
        message: 'Features:',
        choices: [
          { name: 'TypeScript', checked: true },
          { name: 'ESLint', checked: true },
          { name: 'Tailwind' }
        ]
      }
    ]);

    const spinner = ora('Creating project...').start();
    await createProject(answers);
    spinner.succeed('Project created!');
  });

program.parse();
```

### package.json

```json
{
  "bin": { "dev-tool": "./cli.js" },
  "dependencies": {
    "commander": "^11.0", "inquirer": "^9.2",
    "chalk": "^5.3", "ora": "^7.0"
  }
}
```

---

## Workflow Automation Engine

```python
from dataclasses import dataclass
from typing import List, Callable, Dict

@dataclass
class Task:
    name: str
    command: Callable
    depends_on: List[str] = None

class WorkflowEngine:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, Any] = {}

    def task(self, name: str, depends_on: List[str] = None):
        def decorator(func):
            self.tasks[name] = Task(name, func, depends_on or [])
            return func
        return decorator

    def run(self, task_name: str):
        if task_name in self.results:
            return
        task = self.tasks[task_name]
        for dep in task.depends_on:
            self.run(dep)
        self.results[task_name] = task.command()

workflow = WorkflowEngine()

@workflow.task("lint")
def lint():
    subprocess.run(['ruff', 'check', 'src/'], check=True)

@workflow.task("test", depends_on=["lint"])
def test():
    subprocess.run(['pytest', 'tests/'], check=True)

@workflow.task("build", depends_on=["test"])
def build():
    subprocess.run(['python', '-m', 'build'], check=True)
```

---

## Shell Script Automation

```bash
#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

task_deploy() {
    local env=${1:-staging}
    log_info "Deploying to $env..."

    if [ "$env" = "production" ]; then
        read -p "Deploy to PRODUCTION? (yes/no): " confirm
        [ "$confirm" != "yes" ] && exit 0
    fi

    # Run tests, build, deploy
    pytest tests/ && python -m build
    log_info "✓ Deployed to $env!"
}

main() {
    case "${1:-help}" in
        deploy) task_deploy "${@:2}" ;;
        *) echo "Usage: $0 {deploy}" ;;
    esac
}

main "$@"
```

---

## UX Patterns

### Progress Indicators

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
    task = progress.add_task("Processing...", total=100)
    for i in range(100):
        progress.update(task, advance=1)
```

### Confirmation for Destructive Actions

```python
if click.confirm('This will delete all data. Continue?', default=False):
    perform_destructive_action()
```

### Dry-Run Mode

```python
@click.option('--dry-run', is_flag=True, help='Preview without executing')
def deploy(dry_run: bool):
    if dry_run:
        console.print("[yellow]DRY RUN[/yellow] - no changes made")
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| **Help text** | Use --help, --version, subcommand help |
| **Progress** | Spinners/bars for operations > 1s |
| **Confirmation** | Prompt before destructive actions |
| **Exit codes** | 0=success, non-zero=error |
| **Dry-run** | Allow preview of changes |
| **Colors** | Use rich/chalk for readability |
| **Config files** | Support YAML/TOML configuration |
| **Idempotent** | Safe to run multiple times |

---

## Distribution

| Platform | Method |
|----------|--------|
| Python | `pip install`, entry_points in pyproject.toml |
| Node.js | `npm link`, bin in package.json |
| Go | `go install`, single binary |
| Shell | `chmod +x`, add to PATH |

---

## Checklist

- [ ] Clear --help text with examples
- [ ] Progress indicators for long operations
- [ ] Confirmation prompts for destructive actions
- [ ] Dry-run mode for safe testing
- [ ] Appropriate exit codes
- [ ] Shell completion support
- [ ] Non-interactive mode for CI/CD
- [ ] Configuration file support

---

**Version**: 1.0.5
