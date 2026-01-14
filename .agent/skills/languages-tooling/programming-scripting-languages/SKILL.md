---
name: programming-scripting-languages
version: "1.0.7"
maturity: "5-Expert"
specialization: Multi-Language CLI Development
description: Master CLI development across Python (Click, Typer), Bash, PowerShell, and Go (Cobra). Use when implementing argument parsing, creating shell automation scripts, building Go CLIs with Cobra, designing interactive prompts, handling subcommands, or creating cross-platform CLI tools.
---

# Programming and Scripting Languages for CLI

Multi-language patterns for building robust command-line tools.

---

## Framework Selection

| Language | Framework | Best For |
|----------|-----------|----------|
| Python | Click | Complex CLIs, plugins |
| Python | Typer | Type-safe, auto-complete |
| Bash | Native | Unix automation, scripts |
| PowerShell | Native | Windows, cross-platform |
| Go | Cobra | Fast, single binary |
| Rust | Clap | Memory-safe, performance |

---

## Python Click

```python
import click
from pathlib import Path

@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    """CLI tool with Click."""
    ctx.obj = {'verbose': verbose}

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True)
@click.option('--pattern', '-p', default='*.py')
def process(path, recursive, pattern):
    """Process files matching pattern."""
    click.echo(f"Processing {path}")
    # Implementation

if __name__ == '__main__':
    cli()
```

---

## Python Typer

```python
import typer
from pathlib import Path
from enum import Enum

app = typer.Typer()

class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    error = "error"

@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option("default", "--template", "-t"),
    force: bool = typer.Option(False, "--force", "-f"),
    log_level: LogLevel = typer.Option(LogLevel.info)
):
    """Initialize a new project."""
    if Path(name).exists() and not force:
        typer.secho(f"Error: '{name}' exists", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"Creating {name} with {template}")
    Path(name).mkdir(exist_ok=True)
    typer.secho("✓ Done!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
```

---

## Bash Script

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly VERSION="1.0.0"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

cmd_init() {
    local name="" force=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--force) force=true; shift ;;
            -*) log_error "Unknown option: $1" ;;
            *) name="$1"; shift ;;
        esac
    done

    [[ -z "$name" ]] && log_error "Name required"
    [[ -d "$name" && "$force" != true ]] && log_error "'$name' exists"

    mkdir -p "$name"/{src,tests}
    log_info "Created $name"
}

show_help() {
    cat <<EOF
$SCRIPT_NAME v$VERSION

Usage: $SCRIPT_NAME COMMAND [OPTIONS]

Commands:
  init NAME    Initialize project
  help         Show this help
EOF
}

main() {
    case "${1:-help}" in
        init) shift; cmd_init "$@" ;;
        help|--help|-h) show_help ;;
        *) log_error "Unknown: $1" ;;
    esac
}

main "$@"
```

---

## PowerShell

```powershell
[CmdletBinding()]
param(
    [ValidateSet('init', 'build', 'help')]
    [string]$Command = 'help'
)

function Write-Success { param([string]$Msg); Write-Host "[✓] $Msg" -ForegroundColor Green }
function Write-Error { param([string]$Msg); Write-Host "[ERROR] $Msg" -ForegroundColor Red }

function Invoke-Init {
    param([Parameter(Mandatory)][string]$Name, [switch]$Force)

    if ((Test-Path $Name) -and -not $Force) {
        Write-Error "'$Name' exists. Use -Force."
        exit 1
    }

    New-Item -ItemType Directory -Path $Name -Force | Out-Null
    'src', 'tests' | ForEach-Object { New-Item -ItemType Directory -Path "$Name/$_" -Force | Out-Null }
    Write-Success "Created $Name"
}

switch ($Command) {
    'init' { $n = Read-Host "Name"; Invoke-Init -Name $n }
    default { Write-Host "Usage: .\cli.ps1 [init|build|help]" }
}
```

---

## Go Cobra

```go
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
    Use:     "mycli",
    Short:   "CLI tool with Cobra",
    Version: "1.0.0",
}

var initCmd = &cobra.Command{
    Use:   "init [name]",
    Short: "Initialize project",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        name := args[0]
        template, _ := cmd.Flags().GetString("template")
        force, _ := cmd.Flags().GetBool("force")

        fmt.Printf("Creating '%s' with '%s'\n", name, template)
        if force { fmt.Println("Force mode") }

        os.MkdirAll(name+"/src", 0755)
        fmt.Println("✓ Created!")
    },
}

func init() {
    rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file")
    rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose")

    initCmd.Flags().StringP("template", "t", "default", "template")
    initCmd.Flags().BoolP("force", "f", false, "force overwrite")

    rootCmd.AddCommand(initCmd)
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        os.Exit(1)
    }
}
```

---

## Language Selection Guide

| Use Case | Recommended |
|----------|-------------|
| Rapid prototyping | Python (Typer) |
| Data processing | Python (Click) |
| System automation | Bash |
| Windows automation | PowerShell |
| High performance | Go (Cobra) |
| Memory safety | Rust (Clap) |

---

## CLI Design Principles

| Principle | Implementation |
|-----------|----------------|
| POSIX conventions | Use stdin/stdout/stderr |
| Exit codes | 0=success, non-zero=error |
| Help flags | --help, --version |
| Signal handling | Trap INT, TERM |
| Input validation | Fail early with clear errors |

---

## Checklist

- [ ] --help and --version flags
- [ ] Meaningful exit codes
- [ ] Colored output (with --no-color option)
- [ ] Signal handling (Ctrl+C)
- [ ] Input validation with clear errors
- [ ] Subcommand structure for complex CLIs
- [ ] Shell completion support
- [ ] Cross-platform testing

---

**Version**: 1.0.5
