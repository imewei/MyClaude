# CLI Tool Design and Production Implementation

Expert guidance on building production-ready command-line tools with modern frameworks, interactive prompts, and developer automation workflows. Use when creating CLI applications, automation scripts, or developer productivity tools.

## Overview

This skill provides complete production-ready examples for CLI tool development across Python, Node.js, and Go, with emphasis on user experience, interactive prompts, and workflow automation.

## Core Topics

### 1. Production CLI Tools with Python Click/Typer

#### Complete CLI Application with Click

```python
# cli_tool/cli.py
import click
import sys
from pathlib import Path
from typing import Optional
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--verbose', '-v', count=True, help='Verbosity level')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: int):
    """
    Production CLI tool for project management.

    Examples:
        cli_tool init my-project
        cli_tool build --watch
        cli_tool deploy --env production
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    # Load config
    if config:
        with open(config) as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        ctx.obj['config'] = {}

@cli.command()
@click.argument('name')
@click.option('--template', default='default', help='Project template')
@click.option('--force', is_flag=True, help='Overwrite existing directory')
@click.pass_context
def init(ctx, name: str, template: str, force: bool):
    """Initialize a new project."""
    project_path = Path(name)

    if project_path.exists() and not force:
        console.print(f"[red]Error:[/red] Directory '{name}' already exists. Use --force to overwrite.")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Creating project '{name}'...", total=None)

        # Create project structure
        project_path.mkdir(exist_ok=True)
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()

        # Create config files
        (project_path / "pyproject.toml").write_text(f"""
[tool.poetry]
name = "{name}"
version = "0.1.0"
description = "Project created with cli-tool"

[tool.poetry.dependencies]
python = "^3.12"
""")

        progress.update(task, completed=True)

    console.print(f"[green]✓[/green] Project '{name}' created successfully!")
    console.print(f"\nNext steps:")
    console.print(f"  cd {name}")
    console.print(f"  poetry install")

@cli.command()
@click.option('--watch', is_flag=True, help='Watch for changes')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.pass_context
def build(ctx, watch: bool, output: Optional[str]):
    """Build the project."""
    if ctx.obj['verbose']:
        console.print("[dim]Running build with verbose logging[/dim]")

    with Progress() as progress:
        task = progress.add_task("[cyan]Building...", total=100)

        # Simulate build steps
        import time
        steps = [
            ("Compiling sources", 30),
            ("Running tests", 40),
            ("Creating bundle", 30)
        ]

        for step_name, step_weight in steps:
            progress.console.print(f"  {step_name}...")
            time.sleep(0.5)  # Simulate work
            progress.update(task, advance=step_weight)

    console.print("[green]✓[/green] Build completed successfully!")

    if watch:
        console.print("\n[yellow]Watching for changes...[/yellow]")
        console.print("Press Ctrl+C to stop")

@cli.command()
@click.option('--env', type=click.Choice(['dev', 'staging', 'production']), default='dev')
@click.option('--dry-run', is_flag=True, help='Preview without executing')
@click.pass_context
def deploy(ctx, env: str, dry_run: bool):
    """Deploy the project."""
    if env == 'production' and not dry_run:
        if not click.confirm('Deploy to production?', default=False):
            console.print("[yellow]Deployment cancelled[/yellow]")
            return

    console.print(f"[cyan]Deploying to {env}...[/cyan]")

    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    # Show deployment summary
    table = Table(title=f"Deployment to {env}")
    table.add_column("Step", style="cyan")
    table.add_column("Status", style="green")

    steps = [
        ("Build artifacts", "✓ Complete"),
        ("Run tests", "✓ Complete"),
        ("Upload to server", "✓ Complete" if not dry_run else "○ Skipped"),
        ("Health check", "✓ Complete" if not dry_run else "○ Skipped")
    ]

    for step, status in steps:
        table.add_row(step, status)

    console.print(table)

    if not dry_run:
        console.print(f"\n[green]✓[/green] Deployed successfully to {env}!")
    else:
        console.print(f"\n[yellow]Dry run complete. Use without --dry-run to deploy.[/yellow]")

@cli.command()
def status():
    """Show project status."""
    table = Table(title="Project Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version")

    table.add_row("Build", "✓ Ready", "1.2.3")
    table.add_row("Tests", "✓ Passing", "145/145")
    table.add_row("Coverage", "✓ 94%", "")
    table.add_row("Deployment", "✓ production", "v1.2.3")

    console.print(table)

if __name__ == '__main__':
    cli(obj={})
```

**Setup and Installation**
```python
# setup.py or pyproject.toml
[tool.poetry]
name = "cli-tool"
version = "1.0.0"
description = "Production CLI tool"

[tool.poetry.dependencies]
python = "^3.12"
click = "^8.1.0"
rich = "^13.0.0"
pyyaml = "^6.0"

[tool.poetry.scripts]
cli-tool = "cli_tool.cli:cli"
```

### 2. Interactive Prompts and TUI

#### Node.js CLI with Inquirer

```javascript
// cli.js
#!/usr/bin/env node

const { Command } = require('commander');
const inquirer = require('inquirer');
const chalk = require('chalk');
const ora = require('ora');
const boxen = require('boxen');
const fs = require('fs').promises;
const path = require('path');

const program = new Command();

program
  .name('dev-tool')
  .description('Interactive CLI for developer workflows')
  .version('1.0.0');

program
  .command('init')
  .description('Initialize a new project interactively')
  .action(async () => {
    console.log(boxen(
      chalk.bold.cyan('Project Setup Wizard'),
      { padding: 1, borderStyle: 'round' }
    ));

    const answers = await inquirer.prompt([
      {
        type: 'input',
        name: 'projectName',
        message: 'What is your project name?',
        default: 'my-project',
        validate: (input) => {
          if (/^[a-z0-9-]+$/.test(input)) return true;
          return 'Project name must be lowercase alphanumeric with dashes';
        }
      },
      {
        type: 'list',
        name: 'framework',
        message: 'Choose a framework:',
        choices: [
          { name: 'React', value: 'react' },
          { name: 'Vue', value: 'vue' },
          { name: 'Angular', value: 'angular' },
          { name: 'Svelte', value: 'svelte' }
        ]
      },
      {
        type: 'checkbox',
        name: 'features',
        message: 'Select features:',
        choices: [
          { name: 'TypeScript', value: 'typescript', checked: true },
          { name: 'ESLint', value: 'eslint', checked: true },
          { name: 'Prettier', value: 'prettier', checked: true },
          { name: 'Jest', value: 'jest' },
          { name: 'Tailwind CSS', value: 'tailwind' },
          { name: 'Docker', value: 'docker' }
        ]
      },
      {
        type: 'confirm',
        name: 'gitInit',
        message: 'Initialize git repository?',
        default: true
      },
      {
        type: 'list',
        name: 'packageManager',
        message: 'Package manager:',
        choices: ['npm', 'yarn', 'pnpm'],
        default: 'npm'
      }
    ]);

    // Create project
    const spinner = ora('Creating project...').start();

    try {
      const projectPath = path.join(process.cwd(), answers.projectName);

      // Create directory structure
      await fs.mkdir(projectPath, { recursive: true });
      await fs.mkdir(path.join(projectPath, 'src'));
      await fs.mkdir(path.join(projectPath, 'public'));

      // Create package.json
      const packageJson = {
        name: answers.projectName,
        version: '0.1.0',
        description: `${answers.framework} project`,
        scripts: {
          start: 'react-scripts start',
          build: 'react-scripts build',
          test: 'jest'
        },
        dependencies: {},
        devDependencies: {}
      };

      await fs.writeFile(
        path.join(projectPath, 'package.json'),
        JSON.stringify(packageJson, null, 2)
      );

      // Create basic files
      if (answers.features.includes('typescript')) {
        await fs.writeFile(
          path.join(projectPath, 'tsconfig.json'),
          JSON.stringify({
            compilerOptions: {
              target: 'ES2020',
              module: 'ESNext',
              strict: true
            }
          }, null, 2)
        );
      }

      if (answers.gitInit) {
        await fs.writeFile(
          path.join(projectPath, '.gitignore'),
          'node_modules/\ndist/\n.env\n'
        );
      }

      spinner.succeed('Project created successfully!');

      console.log('\n' + boxen(
        chalk.green('Next steps:\n\n') +
        chalk.cyan(`  cd ${answers.projectName}\n`) +
        chalk.cyan(`  ${answers.packageManager} install\n`) +
        chalk.cyan(`  ${answers.packageManager} start`),
        { padding: 1, borderStyle: 'round' }
      ));

    } catch (error) {
      spinner.fail('Project creation failed');
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

program
  .command('deploy')
  .description('Deploy application')
  .action(async () => {
    const deployAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'environment',
        message: 'Select deployment environment:',
        choices: [
          { name: 'Development', value: 'dev' },
          { name: 'Staging', value: 'staging' },
          { name: 'Production', value: 'production' }
        ]
      },
      {
        type: 'confirm',
        name: 'runTests',
        message: 'Run tests before deployment?',
        default: true,
        when: (answers) => answers.environment === 'production'
      },
      {
        type: 'password',
        name: 'apiKey',
        message: 'Enter deployment API key:',
        mask: '*',
        validate: (input) => input.length > 0 || 'API key is required'
      }
    ]);

    const spinner = ora('Deploying...').start();

    // Simulate deployment
    await new Promise(resolve => setTimeout(resolve, 2000));

    spinner.succeed(`Deployed to ${deployAnswers.environment}!`);

    console.log(chalk.green('\n✓ Deployment successful!'));
    console.log(chalk.cyan(`Environment: ${deployAnswers.environment}`));
  });

program
  .command('config')
  .description('Interactive configuration')
  .action(async () => {
    const configAnswers = await inquirer.prompt([
      {
        type: 'editor',
        name: 'description',
        message: 'Enter project description (opens editor):'
      },
      {
        type: 'number',
        name: 'port',
        message: 'Server port:',
        default: 3000,
        validate: (input) => {
          if (input >= 1000 && input <= 65535) return true;
          return 'Port must be between 1000 and 65535';
        }
      },
      {
        type: 'autocomplete',
        name: 'region',
        message: 'AWS region:',
        source: async (answersSoFar, input) => {
          const regions = [
            'us-east-1', 'us-west-2', 'eu-west-1',
            'ap-southeast-1', 'ap-northeast-1'
          ];
          if (!input) return regions;
          return regions.filter(r => r.includes(input));
        }
      }
    ]);

    console.log(chalk.green('\nConfiguration saved!'));
    console.log(JSON.stringify(configAnswers, null, 2));
  });

program.parse(process.argv);
```

**package.json**
```json
{
  "name": "dev-tool",
  "version": "1.0.0",
  "bin": {
    "dev-tool": "./cli.js"
  },
  "dependencies": {
    "commander": "^11.0.0",
    "inquirer": "^9.2.0",
    "chalk": "^5.3.0",
    "ora": "^7.0.0",
    "boxen": "^7.1.0"
  }
}
```

### 3. Workflow Automation Scripts

#### Python Automation Framework

```python
# automation/workflows.py
from typing import List, Dict, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from rich.console import Console
from rich.progress import track
import yaml

console = Console()

@dataclass
class Task:
    """Workflow task definition."""
    name: str
    command: Callable
    depends_on: List[str] = None

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []

class WorkflowEngine:
    """Workflow automation engine."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, Any] = {}

    def task(self, name: str, depends_on: List[str] = None):
        """Decorator to register tasks."""
        def decorator(func: Callable):
            task = Task(name=name, command=func, depends_on=depends_on or [])
            self.tasks[name] = task
            return func
        return decorator

    def run(self, task_name: str) -> None:
        """Run a task and its dependencies."""
        if task_name in self.results:
            return  # Already executed

        task = self.tasks.get(task_name)
        if not task:
            console.print(f"[red]Error:[/red] Task '{task_name}' not found")
            sys.exit(1)

        # Run dependencies first
        for dep in task.depends_on:
            self.run(dep)

        # Run task
        console.print(f"\n[cyan]Running:[/cyan] {task.name}")
        try:
            result = task.command()
            self.results[task_name] = result
            console.print(f"[green]✓[/green] {task.name} completed")
        except Exception as e:
            console.print(f"[red]✗[/red] {task.name} failed: {e}")
            sys.exit(1)

# Create workflow instance
workflow = WorkflowEngine()

@workflow.task("clean")
def clean():
    """Clean build artifacts."""
    import shutil
    dirs_to_clean = ['dist', 'build', '__pycache__', '.pytest_cache']
    for dir_name in dirs_to_clean:
        path = Path(dir_name)
        if path.exists():
            shutil.rmtree(path)
            console.print(f"  Removed {dir_name}/")
    return {"cleaned": len(dirs_to_clean)}

@workflow.task("lint", depends_on=["clean"])
def lint():
    """Run code linting."""
    result = subprocess.run(
        ['ruff', 'check', 'src/'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        console.print(result.stdout)
        raise Exception("Linting failed")
    return {"status": "passed"}

@workflow.task("test", depends_on=["lint"])
def test():
    """Run tests."""
    result = subprocess.run(
        ['pytest', 'tests/', '-v', '--cov=src'],
        capture_output=True,
        text=True
    )
    console.print(result.stdout)
    if result.returncode != 0:
        raise Exception("Tests failed")
    return {"status": "passed"}

@workflow.task("build", depends_on=["test"])
def build():
    """Build project."""
    subprocess.run(['python', '-m', 'build'], check=True)
    return {"status": "built", "artifact": "dist/"}

@workflow.task("deploy", depends_on=["build"])
def deploy():
    """Deploy to production."""
    # Simulate deployment
    console.print("  Uploading artifacts...")
    console.print("  Running migrations...")
    console.print("  Restarting services...")
    return {"status": "deployed", "version": "1.0.0"}

# CLI interface
if __name__ == '__main__':
    import click

    @click.command()
    @click.argument('task', type=click.Choice(list(workflow.tasks.keys())))
    @click.option('--list', is_flag=True, help='List available tasks')
    def cli(task: str, list: bool):
        """Workflow automation tool."""
        if list:
            console.print("[cyan]Available tasks:[/cyan]")
            for name, task_obj in workflow.tasks.items():
                deps = f" (depends on: {', '.join(task_obj.depends_on)})" if task_obj.depends_on else ""
                console.print(f"  • {name}{deps}")
            return

        workflow.run(task)
        console.print(f"\n[green]✓ Workflow completed successfully![/green]")

    cli()
```

#### Shell Script Automation

```bash
#!/usr/bin/env bash
# dev-workflow.sh - Developer workflow automation

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Task functions
task_setup() {
    log_info "Setting up development environment..."

    # Check dependencies
    command -v python3 >/dev/null 2>&1 || { log_error "Python 3 not found"; exit 1; }
    command -v node >/dev/null 2>&1 || { log_error "Node.js not found"; exit 1; }

    # Install dependencies
    log_info "Installing Python dependencies..."
    python3 -m pip install -r requirements.txt --quiet

    log_info "Installing Node dependencies..."
    npm install --silent

    # Setup pre-commit hooks
    if [ -f .git/hooks/pre-commit ]; then
        log_warn "Pre-commit hook already exists"
    else
        log_info "Installing pre-commit hook..."
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
npm run lint
npm run test
EOF
        chmod +x .git/hooks/pre-commit
    fi

    log_info "✓ Setup complete!"
}

task_lint() {
    log_info "Running linters..."

    # Python linting
    log_info "Linting Python code..."
    ruff check src/ || { log_error "Python linting failed"; exit 1; }

    # JavaScript linting
    log_info "Linting JavaScript code..."
    npm run lint || { log_error "JavaScript linting failed"; exit 1; }

    log_info "✓ Linting passed!"
}

task_test() {
    log_info "Running tests..."

    # Python tests
    log_info "Running Python tests..."
    pytest tests/ --cov=src --cov-report=term-missing || {
        log_error "Python tests failed"
        exit 1
    }

    # JavaScript tests
    log_info "Running JavaScript tests..."
    npm test || { log_error "JavaScript tests failed"; exit 1; }

    log_info "✓ All tests passed!"
}

task_build() {
    log_info "Building project..."

    # Build Python package
    python3 -m build

    # Build frontend
    npm run build

    log_info "✓ Build complete!"
}

task_deploy() {
    local env=${1:-staging}

    log_info "Deploying to $env..."

    # Pre-deployment checks
    if [ "$env" = "production" ]; then
        read -p "Deploy to PRODUCTION? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_warn "Deployment cancelled"
            exit 0
        fi
    fi

    # Run full test suite
    task_test

    # Build for deployment
    task_build

    # Deploy (example with rsync)
    log_info "Uploading artifacts..."
    # rsync -avz dist/ user@server:/var/www/app/

    # Run migrations
    log_info "Running database migrations..."
    # ssh user@server 'cd /var/www/app && ./migrate.sh'

    # Restart services
    log_info "Restarting services..."
    # ssh user@server 'systemctl restart app'

    log_info "✓ Deployed to $env successfully!"
}

task_release() {
    local version=${1:-}

    if [ -z "$version" ]; then
        log_error "Version required: ./dev-workflow.sh release 1.0.0"
        exit 1
    fi

    log_info "Creating release $version..."

    # Update version in files
    sed -i "s/version = \".*\"/version = \"$version\"/" pyproject.toml
    sed -i "s/\"version\": \".*\"/\"version\": \"$version\"/" package.json

    # Commit version bump
    git add pyproject.toml package.json
    git commit -m "chore: bump version to $version"

    # Create tag
    git tag -a "v$version" -m "Release v$version"

    # Build
    task_build

    log_info "✓ Release $version ready!"
    log_info "Next: git push && git push --tags"
}

# Help menu
show_help() {
    cat << EOF
Developer Workflow Automation

Usage: $0 <command> [options]

Commands:
  setup           Setup development environment
  lint            Run code linters
  test            Run test suite
  build           Build project
  deploy [env]    Deploy to environment (default: staging)
  release <ver>   Create release with version
  help            Show this help message

Examples:
  $0 setup
  $0 deploy production
  $0 release 1.0.0

EOF
}

# Main command dispatcher
main() {
    local command=${1:-help}
    shift || true

    case "$command" in
        setup)
            task_setup
            ;;
        lint)
            task_lint
            ;;
        test)
            task_test
            ;;
        build)
            task_build
            ;;
        deploy)
            task_deploy "$@"
            ;;
        release)
            task_release "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
```

## Best Practices Summary

### CLI Design
1. Use standard conventions (--help, --version, subcommands)
2. Provide clear error messages with actionable guidance
3. Show progress for long-running operations
4. Confirm destructive actions
5. Support both interactive and non-interactive modes

### User Experience
1. Use colors and formatting for readability
2. Provide comprehensive help text with examples
3. Implement tab completion for shells
4. Show progress indicators for operations >1s
5. Return appropriate exit codes (0 for success, non-zero for errors)

### Automation
1. Make scripts idempotent (safe to run multiple times)
2. Implement proper error handling and rollback
3. Log operations for debugging
4. Support dry-run mode for testing
5. Use configuration files for repeatability

### Distribution
1. Package as installable tool (pip, npm, cargo)
2. Support multiple platforms (Linux, macOS, Windows)
3. Include shell completion scripts
4. Provide clear installation instructions
5. Version your CLI tool semantically

## Quick Reference

### Python CLI
```bash
# Install dependencies
pip install click rich pyyaml

# Run CLI
python -m cli_tool init my-project

# Install as command
pip install -e .
cli-tool --help
```

### Node.js CLI
```bash
# Install dependencies
npm install commander inquirer chalk ora

# Make executable
chmod +x cli.js

# Link globally
npm link
dev-tool --help
```

### Shell Scripts
```bash
# Make executable
chmod +x dev-workflow.sh

# Run command
./dev-workflow.sh setup
./dev-workflow.sh deploy production
```

## When to Use This Skill

Use this skill when you need to:
- Build production-ready CLI tools with Python, Node.js, or Go
- Create interactive command-line interfaces with prompts
- Implement developer workflow automation
- Build project scaffolding and code generators
- Create deployment and release automation
- Implement build tools and task runners
- Design terminal user interfaces (TUI)
- Automate repetitive development tasks

This skill provides battle-tested patterns for professional CLI tool development.
