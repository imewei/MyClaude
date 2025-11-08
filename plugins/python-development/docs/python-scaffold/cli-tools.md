# CLI Tool Development

Build modern command-line interfaces with Typer, Rich console output, and professional user experience.

## Directory Structure

```
cli-tool/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── cli_tool/
│       ├── __init__.py
│       ├── cli.py
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── process.py
│       │   └── utils.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── validators.py
│       └── utils/
│           ├── __init__.py
│           ├── formatting.py
│           └── helpers.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_cli.py
```

## pyproject.toml

```toml
[project]
name = "cli-tool"
version = "0.1.0"
description = "Modern CLI tool with Typer and Rich"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.7.0",
    "pydantic>=2.6.0",
    "click>=8.1.0",  # Required by Typer
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
]

[project.scripts]
cli-tool = "cli_tool.cli:app"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## src/cli_tool/cli.py

```python
"""Main CLI application with Typer"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="cli-tool",
    help="Modern CLI tool built with Typer and Rich",
    add_completion=True,
)
console = Console()


@app.command()
def hello(
    name: str = typer.Option(..., "--name", "-n", help="Your name"),
    greeting: str = typer.Option("Hello", "--greeting", "-g", help="Greeting word"),
    loud: bool = typer.Option(False, "--loud", "-l", help="Uppercase output"),
) -> None:
    """
    Greet someone with a personalized message.

    Example:
        $ cli-tool hello --name "Alice" --loud
    """
    message = f"{greeting} {name}!"

    if loud:
        message = message.upper()

    console.print(f"[bold green]{message}[/bold green]")


@app.command()
def process(
    input_file: Path = typer.Argument(
        ...,
        help="Input file to process",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: stdout)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Process an input file and optionally save results.

    Example:
        $ cli-tool process data.txt --output results.txt --verbose
    """
    if verbose:
        console.print(f"[blue]Processing {input_file}...[/blue]")

    # Simulated processing with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=100)

        # Your processing logic here
        content = input_file.read_text()
        lines = content.splitlines()

        for i in range(100):
            progress.update(task, advance=1)

        result = f"Processed {len(lines)} lines"

    if output_file:
        output_file.write_text(result)
        console.print(f"[green]✓[/green] Results saved to {output_file}")
    else:
        console.print(result)


@app.command()
def list_items(
    filter_by: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter items"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max items to display"),
) -> None:
    """
    Display items in a formatted table.

    Example:
        $ cli-tool list-items --filter "active" --limit 5
    """
    # Create rich table
    table = Table(title="Items", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Count", justify="right", style="blue")

    # Sample data
    items = [
        ("1", "Item A", "active", "42"),
        ("2", "Item B", "inactive", "17"),
        ("3", "Item C", "active", "93"),
        ("4", "Item D", "pending", "5"),
        ("5", "Item E", "active", "28"),
    ]

    # Filter and add rows
    count = 0
    for item_id, name, status, value in items:
        if filter_by and filter_by not in status:
            continue

        if count >= limit:
            break

        table.add_row(item_id, name, status, value)
        count += 1

    console.print(table)


@app.command()
def config(
    set_key: Optional[str] = typer.Option(None, "--set", help="Config key to set"),
    value: Optional[str] = typer.Option(None, "--value", help="Config value"),
    get_key: Optional[str] = typer.Option(None, "--get", help="Config key to get"),
    list_all: bool = typer.Option(False, "--list", "-l", help="List all config"),
) -> None:
    """
    Manage configuration settings.

    Examples:
        $ cli-tool config --set api_key --value "abc123"
        $ cli-tool config --get api_key
        $ cli-tool config --list
    """
    from .core.config import Config

    cfg = Config()

    if set_key and value:
        cfg.set(set_key, value)
        console.print(f"[green]✓[/green] Set {set_key} = {value}")

    elif get_key:
        val = cfg.get(get_key)
        if val:
            console.print(f"{get_key} = {val}")
        else:
            console.print(f"[red]✗[/red] Key '{get_key}' not found", style="bold red")
            raise typer.Exit(code=1)

    elif list_all:
        table = Table(title="Configuration", show_header=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        for key, val in cfg.all().items():
            table.add_row(key, str(val))

        console.print(table)

    else:
        console.print("[yellow]Specify --set, --get, or --list[/yellow]")
        raise typer.Exit(code=1)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """
    CLI Tool - Modern command-line interface
    """
    if version:
        from . import __version__
        console.print(f"cli-tool version {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
```

## src/cli_tool/core/config.py

```python
"""Configuration management"""

import json
from pathlib import Path
from typing import Any


class Config:
    """Simple JSON-based configuration"""

    def __init__(self, config_file: Path | None = None):
        if config_file is None:
            config_file = Path.home() / ".cli-tool" / "config.json"

        self.config_file = config_file
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self.config_file.write_text("{}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        data = self._load()
        return data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        data = self._load()
        data[key] = value
        self._save(data)

    def all(self) -> dict[str, Any]:
        """Get all configuration"""
        return self._load()

    def _load(self) -> dict[str, Any]:
        """Load configuration from file"""
        return json.loads(self.config_file.read_text())

    def _save(self, data: dict[str, Any]) -> None:
        """Save configuration to file"""
        self.config_file.write_text(json.dumps(data, indent=2))
```

## tests/test_cli.py

```python
from typer.testing import CliRunner
from cli_tool.cli import app

runner = CliRunner()


def test_hello_command():
    """Test hello command"""
    result = runner.invoke(app, ["hello", "--name", "Alice"])
    assert result.exit_code == 0
    assert "Hello Alice" in result.stdout


def test_hello_loud():
    """Test hello command with loud flag"""
    result = runner.invoke(app, ["hello", "--name", "Bob", "--loud"])
    assert result.exit_code == 0
    assert "HELLO BOB" in result.stdout


def test_version():
    """Test version flag"""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()


def test_list_items():
    """Test list items command"""
    result = runner.invoke(app, ["list-items", "--limit", "3"])
    assert result.exit_code == 0
    # Check table is rendered
    assert "Items" in result.stdout
```

## Advanced Features

### 1. Subcommands with App Groups

```python
# Create subcommand group
db_app = typer.Typer(help="Database commands")
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init():
    """Initialize database"""
    console.print("[green]Database initialized[/green]")


@db_app.command("migrate")
def db_migrate():
    """Run database migrations"""
    console.print("[blue]Running migrations...[/blue]")


# Usage: cli-tool db init
```

### 2. Interactive Prompts

```python
import typer


@app.command()
def interactive():
    """Interactive mode with prompts"""
    name = typer.prompt("What's your name?")
    age = typer.prompt("What's your age?", type=int)
    password = typer.prompt("Enter password", hide_input=True)

    confirm = typer.confirm("Are you sure?")
    if not confirm:
        console.print("[yellow]Aborted[/yellow]")
        raise typer.Abort()

    console.print(f"Hello {name}, age {age}")
```

### 3. Progress Bars

```python
from rich.progress import track
import time


@app.command()
def download():
    """Simulate download with progress"""
    for i in track(range(100), description="Downloading..."):
        time.sleep(0.01)  # Simulate work

    console.print("[green]Download complete![/green]")
```

### 4. Rich Panels and Markdown

```python
from rich.panel import Panel
from rich.markdown import Markdown


@app.command()
def show_help():
    """Display rich help"""
    help_text = """
    # CLI Tool Help

    ## Commands
    - `hello`: Greet someone
    - `process`: Process files
    - `list-items`: Display items

    ## Examples
    ```bash
    cli-tool hello --name Alice
    cli-tool process data.txt -o output.txt
    ```
    """

    md = Markdown(help_text)
    panel = Panel(md, title="Help", border_style="blue")
    console.print(panel)
```

### 5. Error Handling

```python
from rich.console import Console
from rich.traceback import install

# Install rich tracebacks
install(show_locals=True)


@app.command()
def risky_operation(
    fail: bool = typer.Option(False, "--fail", help="Trigger error"),
):
    """Operation that might fail"""
    try:
        if fail:
            raise ValueError("Something went wrong!")

        console.print("[green]Success![/green]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}", style="bold red")
        raise typer.Exit(code=1)
```

## Best Practices

### 1. Documentation
- Add help text to all commands and options
- Include usage examples in docstrings
- Generate docs from CLI: `typer utils docs`

### 2. User Experience
- Use Rich for colorful, readable output
- Show progress for long operations
- Provide clear error messages
- Support `--help` on all commands

### 3. Testing
- Use `CliRunner` for testing commands
- Test both success and error cases
- Mock file I/O and external dependencies

### 4. Distribution
```bash
# Build standalone executable with PyInstaller
uv add --dev pyinstaller
pyinstaller --onefile --name cli-tool src/cli_tool/cli.py

# Or use shiv for zipapp
uv add --dev shiv
shiv -c cli-tool -o cli-tool.pyz .
```

### 5. Shell Completion
```bash
# Generate completion script
cli-tool --install-completion

# For bash
cli-tool --show-completion bash > ~/.bash_completion.d/cli-tool

# For zsh
cli-tool --show-completion zsh > ~/.zsh/completion/_cli-tool
```
