---
name: programming-scripting-languages
description: Master programming languages and frameworks for building robust CLI tools across Python (Click, Typer, argparse), Shell scripting (Bash, Zsh, PowerShell), Go (Cobra, Viper), Rust (Clap), and cross-language patterns. Use when implementing argument parsing and validation with Click decorators or Typer type hints, creating Bash automation scripts with error handling and signal trapping, writing PowerShell cmdlets for cross-platform scripting, building Go CLIs with Cobra command structures and Viper configuration, implementing Rust CLIs with Clap derive macros, designing interactive prompts with Inquirer.js or Python's prompt_toolkit, handling command-line arguments and flags with proper validation, implementing subcommand hierarchies and command composition, managing configuration files (YAML, TOML, JSON) in CLI applications, creating cross-platform shell scripts with environment detection, implementing proper exit codes and error handling patterns, building CLI tools with logging and debugging capabilities, creating progress indicators and spinners with terminal libraries, implementing secure credential handling in command-line tools, or designing help systems and command documentation. Use this skill when working with CLI framework files (Python Click/Typer, Node Commander, Go Cobra, Rust Clap), shell scripts (.sh, .bash, .ps1), command entry points (main.py, main.go, cli.js), argument parser implementations, interactive prompt code, or cross-platform compatibility layers.
---

# Programming and Scripting Languages for CLI Development

Expert guidance on programming languages and frameworks for building robust CLI tools. Covers Python (Click, Typer, argparse), Shell scripting (Bash/Zsh/PowerShell), Go (Cobra), Rust (Clap), and cross-language best practices.

## When to use this skill

- When implementing Python CLI tools with Click framework and decorators
- When building type-safe Python CLIs with Typer and type hints
- When creating Bash automation scripts with advanced error handling
- When writing PowerShell cmdlets for Windows and cross-platform use
- When implementing Go command-line applications with Cobra and Viper
- When building Rust CLIs with Clap derive macros for type safety
- When designing argument parsing with proper validation and error messages
- When implementing interactive prompts with Inquirer.js or prompt_toolkit
- When creating subcommand hierarchies and command composition patterns
- When managing configuration files (YAML, TOML, JSON) in CLI applications
- When building cross-platform shell scripts with environment detection
- When implementing proper exit codes (0 for success, non-zero for errors)
- When creating comprehensive error handling and signal trapping
- When building CLI tools with structured logging and debugging
- When implementing progress indicators, spinners, and terminal animations
- When handling secure credentials and secrets in command-line tools
- When designing help systems with usage examples and documentation
- When working with CLI framework files (click.py, cobra.go, clap.rs)
- When creating shell scripts (.sh, .bash, .zsh, .ps1, .fish)
- When implementing command entry points (__main__.py, main.go, cli.js)
- When building argument parsers with required/optional parameters
- When creating flag and option handlers with short/long forms (-v, --verbose)
- When implementing environment variable handling and defaults
- When building CLI tools that read from stdin/stdout/stderr
- When creating shell completion scripts for bash, zsh, or fish

## Overview

Master multiple programming languages and their CLI-specific libraries to build efficient, maintainable command-line tools for any use case.

## 1. Python CLI Development

### Click Framework - Production CLI

```python
# mycli/cli.py
import click
from pathlib import Path
import sys
from typing import Optional

@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version')
@click.pass_context
def cli(ctx, version):
    """Modern CLI tool with Click."""
    if version:
        click.echo('mycli version 1.0.0')
        ctx.exit()

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='Process recursively')
@click.option('--pattern', '-p', default='*.py', help='File pattern')
@click.option('--exclude', multiple=True, help='Exclude patterns')
def process(path: str, recursive: bool, pattern: str, exclude: tuple):
    """Process files matching pattern."""
    click.echo(f"Processing {path}")

    # Implementation
    from glob import glob
    import os

    if recursive:
        pattern = f"**/{pattern}"

    files = glob(os.path.join(path, pattern), recursive=recursive)

    # Apply exclusions
    for exc in exclude:
        files = [f for f in files if exc not in f]

    with click.progressbar(files, label='Processing files') as bar:
        for file in bar:
            # Process file
            pass

    click.secho(f"✓ Processed {len(files)} files", fg='green')

@cli.command()
@click.option('--name', prompt='Your name', help='Your name')
@click.option('--email', prompt='Your email', help='Your email')
@click.password_option(help='Your password')
def login(name: str, email: str, password: str):
    """Interactive login with prompts."""
    click.echo(f"Logging in as {name} ({email})")

    if click.confirm('Save credentials?'):
        # Save to config
        config_path = Path.home() / '.mycli' / 'config'
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(f"name={name}\nemail={email}")
        click.secho('✓ Credentials saved', fg='green')

if __name__ == '__main__':
    cli()
```

### Typer - Modern Type-Hint Based CLI

```python
# typer_cli/main.py
import typer
from typing import Optional, List
from pathlib import Path
from enum import Enum

app = typer.Typer(help="Modern CLI with Typer and type hints")

class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"

@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option("default", help="Project template"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
    log_level: LogLevel = typer.Option(LogLevel.info, help="Log level")
):
    """Initialize a new project."""
    project_path = Path(name)

    if project_path.exists() and not force:
        typer.secho(
            f"Error: Directory '{name}' already exists. Use --force to overwrite.",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(1)

    typer.echo(f"Creating project '{name}' with template '{template}'")
    project_path.mkdir(exist_ok=True)

    typer.secho("✓ Project created successfully!", fg=typer.colors.GREEN)

@app.command()
def build(
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    exclude: List[str] = typer.Option([], "--exclude", "-e", help="Exclude patterns")
):
    """Build the project."""
    typer.echo(f"Building project...")

    if watch:
        typer.echo("Watching for changes... (Ctrl+C to stop)")

    if output:
        typer.echo(f"Output directory: {output}")

    if exclude:
        typer.echo(f"Excluding: {', '.join(exclude)}")

@app.command()
def config(
    key: Optional[str] = typer.Argument(None),
    value: Optional[str] = typer.Argument(None),
    list_all: bool = typer.Option(False, "--list", help="List all config")
):
    """Manage configuration."""
    if list_all:
        typer.echo("Configuration:")
        typer.echo("  key1 = value1")
        typer.echo("  key2 = value2")
    elif key and value:
        typer.echo(f"Setting {key} = {value}")
    elif key:
        typer.echo(f"Getting {key}: value")
    else:
        typer.echo("Use --list to show all config")

if __name__ == "__main__":
    app()
```

## 2. Shell Scripting

### Bash - Advanced CLI Script

```bash
#!/usr/bin/env bash
# advanced-cli.sh - Production Bash CLI

set -euo pipefail
IFS=$'\n\t'

# Script metadata
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration
CONFIG_FILE="${HOME}/.config/mycli/config"
VERBOSE=false
DRY_RUN=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "[DEBUG] $*" >&2
    fi
}

# Error handling
error_exit() {
    log_error "$1"
    exit "${2:-1}"
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    log_debug "Cleaning up..."
    # Cleanup code here
    exit "$exit_code"
}

trap cleanup EXIT
trap 'error_exit "Script interrupted" 130' INT

# Utility functions
check_dependencies() {
    local deps=("$@")
    local missing=()

    for cmd in "${deps[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        error_exit "Missing dependencies: ${missing[*]}"
    fi
}

confirm() {
    local prompt="$1"
    local default="${2:-n}"

    if [[ "$default" == "y" ]]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi

    read -rp "$prompt" response
    response=${response:-$default}

    [[ "$response" =~ ^[Yy]$ ]]
}

# Command: init
cmd_init() {
    local name=""
    local template="default"
    local force=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--template)
                template="$2"
                shift 2
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -*)
                error_exit "Unknown option: $1"
                ;;
            *)
                name="$1"
                shift
                ;;
        esac
    done

    [[ -z "$name" ]] && error_exit "Project name required"

    if [[ -d "$name" ]] && [[ "$force" != true ]]; then
        error_exit "Directory '$name' already exists. Use --force to overwrite."
    fi

    log_info "Creating project '$name' with template '$template'"

    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN: Would create project directory"
        return 0
    fi

    mkdir -p "$name"/{src,tests,docs}

    cat > "$name/README.md" <<EOF
# $name

Project created with $SCRIPT_NAME

## Getting Started

\`\`\`bash
cd $name
# Your commands here
\`\`\`
EOF

    log_success "Project '$name' created successfully!"
}

# Command: build
cmd_build() {
    local watch=false
    local output="dist"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -w|--watch)
                watch=true
                shift
                ;;
            -o|--output)
                output="$2"
                shift 2
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done

    log_info "Building project..."

    # Simulate build steps
    local steps=("Compiling sources" "Running tests" "Creating bundle")
    for step in "${steps[@]}"; do
        log_info "$step..."
        sleep 0.5
    done

    log_success "Build complete! Output: $output"

    if [[ "$watch" == true ]]; then
        log_info "Watching for changes... (Ctrl+C to stop)"
        while true; do
            sleep 1
        done
    fi
}

# Command: deploy
cmd_deploy() {
    local env="dev"
    local skip_tests=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                env="$2"
                shift 2
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done

    if [[ "$env" == "production" ]]; then
        confirm "Deploy to PRODUCTION?" "n" || {
            log_warning "Deployment cancelled"
            exit 0
        }
    fi

    log_info "Deploying to $env..."

    if [[ "$skip_tests" != true ]]; then
        log_info "Running tests..."
        # Run tests
    fi

    log_info "Uploading artifacts..."
    log_info "Health check..."

    log_success "Deployed to $env successfully!"
}

# Help text
show_help() {
    cat <<EOF
$SCRIPT_NAME v$SCRIPT_VERSION

Usage: $SCRIPT_NAME [OPTIONS] COMMAND [ARGS]

Commands:
  init [NAME]           Initialize new project
  build                 Build project
  deploy                Deploy project
  help                  Show this help

Options:
  -v, --verbose        Verbose output
  -d, --dry-run        Dry run mode
  -h, --help           Show help

Examples:
  $SCRIPT_NAME init my-project --template react
  $SCRIPT_NAME build --watch
  $SCRIPT_NAME deploy --env production

EOF
}

# Main command dispatcher
main() {
    # Check dependencies
    check_dependencies "mkdir" "cat"

    # Parse global options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                error_exit "Unknown global option: $1"
                ;;
            *)
                break
                ;;
        esac
    done

    # Get command
    local command="${1:-help}"
    shift || true

    # Dispatch to command
    case "$command" in
        init)
            cmd_init "$@"
            ;;
        build)
            cmd_build "$@"
            ;;
        deploy)
            cmd_deploy "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error_exit "Unknown command: $command"
            ;;
    esac
}

main "$@"
```

### PowerShell - Cross-Platform CLI

```powershell
# mycli.ps1
[CmdletBinding()]
param(
    [Parameter(Position=0, Mandatory=$false)]
    [ValidateSet('init', 'build', 'deploy', 'help')]
    [string]$Command = 'help',

    [switch]$Verbose,
    [switch]$DryRun
)

# Script configuration
$Script:Version = "1.0.0"
$Script:ConfigPath = Join-Path $env:USERPROFILE ".mycli\config.json"

# Color output functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Command: Initialize project
function Invoke-Init {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [string]$Template = "default",
        [switch]$Force
    )

    $projectPath = Join-Path (Get-Location) $Name

    if ((Test-Path $projectPath) -and -not $Force) {
        Write-Error "Directory '$Name' already exists. Use -Force to overwrite."
        exit 1
    }

    Write-Info "Creating project '$Name' with template '$Template'"

    if ($DryRun) {
        Write-Warning "DRY RUN: Would create project"
        return
    }

    New-Item -ItemType Directory -Path $projectPath -Force | Out-Null
    @('src', 'tests', 'docs') | ForEach-Object {
        New-Item -ItemType Directory -Path (Join-Path $projectPath $_) -Force | Out-Null
    }

    $readme = @"
# $Name

Project created with mycli

## Getting Started

``````powershell
cd $Name
# Your commands here
``````
"@

    Set-Content -Path (Join-Path $projectPath "README.md") -Value $readme

    Write-Success "Project '$Name' created successfully!"
}

# Command: Build project
function Invoke-Build {
    [CmdletBinding()]
    param(
        [switch]$Watch,
        [string]$Output = "dist"
    )

    Write-Info "Building project..."

    $steps = @("Compiling sources", "Running tests", "Creating bundle")
    foreach ($step in $steps) {
        Write-Info $step
        Start-Sleep -Milliseconds 500
    }

    Write-Success "Build complete! Output: $Output"

    if ($Watch) {
        Write-Info "Watching for changes... (Ctrl+C to stop)"
        while ($true) {
            Start-Sleep -Seconds 1
        }
    }
}

# Command: Deploy project
function Invoke-Deploy {
    [CmdletBinding()]
    param(
        [ValidateSet('dev', 'staging', 'production')]
        [string]$Environment = 'dev',

        [switch]$SkipTests
    )

    if ($Environment -eq 'production') {
        $confirmation = Read-Host "Deploy to PRODUCTION? (yes/no)"
        if ($confirmation -ne 'yes') {
            Write-Warning "Deployment cancelled"
            return
        }
    }

    Write-Info "Deploying to $Environment..."

    if (-not $SkipTests) {
        Write-Info "Running tests..."
    }

    Write-Info "Uploading artifacts..."
    Write-Info "Health check..."

    Write-Success "Deployed to $Environment successfully!"
}

# Show help
function Show-Help {
    Write-Host @"
mycli v$($Script:Version)

Usage: .\mycli.ps1 [Command] [Options]

Commands:
  init [Name]          Initialize new project
  build                Build project
  deploy               Deploy project
  help                 Show this help

Options:
  -Verbose             Verbose output
  -DryRun              Dry run mode

Examples:
  .\mycli.ps1 init MyProject -Template react
  .\mycli.ps1 build -Watch
  .\mycli.ps1 deploy -Environment production

"@
}

# Main execution
switch ($Command) {
    'init' {
        $name = Read-Host "Project name"
        Invoke-Init -Name $name
    }
    'build' {
        Invoke-Build
    }
    'deploy' {
        Invoke-Deploy
    }
    default {
        Show-Help
    }
}
```

## 3. Go CLI with Cobra

```go
// main.go
package main

import (
    "fmt"
    "os"
    "github.com/spf13/cobra"
    "github.com/spf13/viper"
)

var (
    cfgFile string
    verbose bool
)

var rootCmd = &cobra.Command{
    Use:   "mycli",
    Short: "Modern CLI tool built with Cobra",
    Long:  `A production-ready CLI tool demonstrating Cobra framework.`,
    Version: "1.0.0",
}

var initCmd = &cobra.Command{
    Use:   "init [name]",
    Short: "Initialize a new project",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        name := args[0]
        template, _ := cmd.Flags().GetString("template")
        force, _ := cmd.Flags().GetBool("force")

        fmt.Printf("Creating project '%s' with template '%s'\n", name, template)

        if force {
            fmt.Println("Force mode enabled")
        }

        // Create project structure
        os.MkdirAll(name+"/src", 0755)
        os.MkdirAll(name+"/tests", 0755)

        fmt.Println("✓ Project created successfully!")
    },
}

var buildCmd = &cobra.Command{
    Use:   "build",
    Short: "Build the project",
    Run: func(cmd *cobra.Command, args []string) {
        watch, _ := cmd.Flags().GetBool("watch")
        output, _ := cmd.Flags().GetString("output")

        fmt.Println("Building project...")

        if verbose {
            fmt.Println("Verbose mode enabled")
        }

        if watch {
            fmt.Println("Watching for changes...")
        }

        fmt.Printf("Output: %s\n", output)
        fmt.Println("✓ Build complete!")
    },
}

func init() {
    cobra.OnInitialize(initConfig)

    // Global flags
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.mycli.yaml)")
    rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

    // Init command flags
    initCmd.Flags().StringP("template", "t", "default", "Project template")
    initCmd.Flags().BoolP("force", "f", false, "Force overwrite")

    // Build command flags
    buildCmd.Flags().BoolP("watch", "w", false, "Watch mode")
    buildCmd.Flags().StringP("output", "o", "dist", "Output directory")

    rootCmd.AddCommand(initCmd)
    rootCmd.AddCommand(buildCmd)
}

func initConfig() {
    if cfgFile != "" {
        viper.SetConfigFile(cfgFile)
    } else {
        home, err := os.UserHomeDir()
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }

        viper.AddConfigPath(home)
        viper.SetConfigName(".mycli")
    }

    viper.AutomaticEnv()

    if err := viper.ReadInConfig(); err == nil && verbose {
        fmt.Println("Using config file:", viper.ConfigFileUsed())
    }
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
}
```

## Best Practices

### Language Selection Guide
- **Python**: Rapid prototyping, data processing, API integration
- **Bash**: System automation, CI/CD scripts, Unix workflows
- **PowerShell**: Windows automation, cross-platform scripting
- **Go**: High-performance tools, concurrent operations
- **Rust**: Safe, memory-efficient tools with strict guarantees

### CLI Development Principles
1. Follow POSIX conventions (stdin/stdout/stderr)
2. Use meaningful exit codes (0=success, 1=error)
3. Support --help and --version flags
4. Implement proper signal handling
5. Validate inputs early and clearly

## Quick Reference

```bash
# Python
pip install click typer rich
python -m mycli --help

# Go
go get github.com/spf13/cobra
go build -o mycli

# PowerShell
pwsh mycli.ps1 help
```

## When to Use This Skill

Use when building CLI tools requiring:
- Argument parsing and validation
- Interactive prompts
- Cross-platform compatibility
- Performance optimization
- Type safety and robustness
