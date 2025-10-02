"""
Beautiful error message formatter.

Transforms error messages into helpful, actionable information with
context, suggestions, and visual formatting.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import traceback
import sys
import re

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ErrorCategory(Enum):
    """Error category for classification."""
    SYNTAX = "Syntax Error"
    RUNTIME = "Runtime Error"
    CONFIGURATION = "Configuration Error"
    VALIDATION = "Validation Error"
    NETWORK = "Network Error"
    FILESYSTEM = "File System Error"
    PERMISSION = "Permission Error"
    DEPENDENCY = "Dependency Error"
    AGENT = "Agent Error"
    CACHE = "Cache Error"
    UNKNOWN = "Unknown Error"


class ErrorSeverity(Enum):
    """Error severity level."""
    CRITICAL = ("🔴", "red", "CRITICAL")
    ERROR = ("🔴", "red", "ERROR")
    WARNING = ("⚠️", "yellow", "WARNING")
    INFO = ("ℹ️", "cyan", "INFO")


@dataclass
class ErrorContext:
    """Context information for an error."""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    function_name: Optional[str] = None
    command: Optional[str] = None
    agent: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorSuggestion:
    """Suggested fix for an error."""
    title: str
    description: str
    action: Optional[str] = None
    command: Optional[str] = None
    confidence: float = 1.0


@dataclass
class FormattedError:
    """Formatted error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    title: str
    message: str
    context: Optional[ErrorContext] = None
    suggestions: List[ErrorSuggestion] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    stack_trace: Optional[str] = None
    documentation_url: Optional[str] = None


class ErrorFormatter:
    """
    Beautiful error message formatter.

    Features:
    - Error categorization (Syntax, Runtime, Config, etc.)
    - Context display with code snippets
    - Actionable suggestions to fix errors
    - Related error references
    - Pretty-printed stack traces
    - Unique error IDs for documentation
    - Color coding by severity
    - Icons and visual indicators

    Example:
        formatter = ErrorFormatter()

        try:
            # Some code that fails
            result = process_data()
        except Exception as e:
            formatted = formatter.format_exception(
                e,
                category=ErrorCategory.RUNTIME,
                context={"command": "optimize"}
            )
            formatter.print_error(formatted)
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        enabled: bool = True,
        show_stack_trace: bool = True,
        show_suggestions: bool = True,
        max_context_lines: int = 5
    ):
        """
        Initialize error formatter.

        Args:
            console: Rich console instance
            enabled: Whether formatting is enabled
            show_stack_trace: Show stack traces
            show_suggestions: Show fix suggestions
            max_context_lines: Max lines of context to show
        """
        self.console = console or (Console(stderr=True) if RICH_AVAILABLE else None)
        self.enabled = enabled and RICH_AVAILABLE
        self.show_stack_trace = show_stack_trace
        self.show_suggestions = show_suggestions
        self.max_context_lines = max_context_lines

        self._error_counter = 0

    def _generate_error_id(self, category: ErrorCategory) -> str:
        """Generate unique error ID."""
        self._error_counter += 1
        prefix = category.name[:3].upper()
        return f"{prefix}-{self._error_counter:04d}"

    def format_exception(
        self,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[ErrorSuggestion]] = None
    ) -> FormattedError:
        """
        Format an exception into structured error information.

        Args:
            exception: The exception to format
            category: Error category (auto-detected if None)
            severity: Error severity
            context: Additional context information
            suggestions: List of fix suggestions

        Returns:
            Formatted error information
        """
        # Auto-detect category if not provided
        if not category:
            category = self._categorize_exception(exception)

        # Extract context from exception
        error_context = self._extract_context(exception, context or {})

        # Generate suggestions if not provided
        if suggestions is None:
            suggestions = self._generate_suggestions(exception, category)

        # Get stack trace
        stack_trace = None
        if self.show_stack_trace:
            stack_trace = "".join(traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__
            ))

        # Create formatted error
        formatted = FormattedError(
            error_id=self._generate_error_id(category),
            category=category,
            severity=severity,
            title=f"{category.value}: {type(exception).__name__}",
            message=str(exception),
            context=error_context,
            suggestions=suggestions,
            stack_trace=stack_trace,
            documentation_url=self._get_documentation_url(category)
        )

        return formatted

    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """Auto-detect error category from exception."""
        exc_type = type(exception).__name__

        if exc_type in ["SyntaxError", "IndentationError", "TabError"]:
            return ErrorCategory.SYNTAX

        elif exc_type in ["FileNotFoundError", "IsADirectoryError", "NotADirectoryError"]:
            return ErrorCategory.FILESYSTEM

        elif exc_type in ["PermissionError", "OSError"]:
            return ErrorCategory.PERMISSION

        elif exc_type in ["ImportError", "ModuleNotFoundError"]:
            return ErrorCategory.DEPENDENCY

        elif exc_type in ["ValueError", "TypeError", "KeyError", "AttributeError"]:
            return ErrorCategory.RUNTIME

        elif exc_type in ["ConnectionError", "TimeoutError", "HTTPError"]:
            return ErrorCategory.NETWORK

        elif "Config" in exc_type or "Setting" in exc_type:
            return ErrorCategory.CONFIGURATION

        elif "Validation" in exc_type:
            return ErrorCategory.VALIDATION

        return ErrorCategory.UNKNOWN

    def _extract_context(self, exception: Exception, extra_context: Dict[str, Any]) -> ErrorContext:
        """Extract context from exception and traceback."""
        context = ErrorContext()

        # Extract from extra context
        context.command = extra_context.get("command")
        context.agent = extra_context.get("agent")
        context.additional_info = {k: v for k, v in extra_context.items() if k not in ["command", "agent"]}

        # Extract from traceback
        if exception.__traceback__:
            tb = exception.__traceback__
            frame = tb.tb_frame
            context.file_path = frame.f_code.co_filename
            context.line_number = tb.tb_lineno
            context.function_name = frame.f_code.co_name

            # Try to extract code snippet
            try:
                with open(context.file_path, 'r') as f:
                    lines = f.readlines()
                    start = max(0, context.line_number - 3)
                    end = min(len(lines), context.line_number + 2)
                    context.code_snippet = "".join(lines[start:end])
            except Exception:
                pass

        return context

    def _generate_suggestions(self, exception: Exception, category: ErrorCategory) -> List[ErrorSuggestion]:
        """Generate fix suggestions based on exception and category."""
        suggestions = []
        exc_type = type(exception).__name__
        message = str(exception).lower()

        # File not found
        if exc_type == "FileNotFoundError":
            suggestions.append(ErrorSuggestion(
                title="Check file path",
                description="Verify that the file path is correct and the file exists.",
                action="Check file path and permissions",
                confidence=0.9
            ))

        # Import errors
        elif exc_type in ["ImportError", "ModuleNotFoundError"]:
            module_name = self._extract_module_name(message)
            if module_name:
                suggestions.append(ErrorSuggestion(
                    title="Install missing dependency",
                    description=f"The module '{module_name}' is not installed.",
                    command=f"pip install {module_name}",
                    confidence=0.95
                ))

        # Permission errors
        elif exc_type == "PermissionError":
            suggestions.append(ErrorSuggestion(
                title="Check permissions",
                description="You may need elevated permissions to perform this operation.",
                action="Run with appropriate permissions or change file ownership",
                confidence=0.8
            ))

        # Type errors
        elif exc_type == "TypeError":
            suggestions.append(ErrorSuggestion(
                title="Check argument types",
                description="Verify that you're passing the correct types to the function.",
                action="Review function signature and argument types",
                confidence=0.7
            ))

        # Value errors
        elif exc_type == "ValueError":
            suggestions.append(ErrorSuggestion(
                title="Validate input values",
                description="The input value is not valid for this operation.",
                action="Check input validation and value ranges",
                confidence=0.7
            ))

        # Configuration errors
        elif category == ErrorCategory.CONFIGURATION:
            suggestions.append(ErrorSuggestion(
                title="Check configuration",
                description="Review your configuration file for errors or missing values.",
                action="Validate configuration against schema",
                confidence=0.8
            ))

        # Add generic suggestion
        if not suggestions:
            suggestions.append(ErrorSuggestion(
                title="Review error details",
                description="Check the error message and stack trace for more information.",
                action="Debug the code at the error location",
                confidence=0.5
            ))

        return suggestions

    def _extract_module_name(self, message: str) -> Optional[str]:
        """Extract module name from import error message."""
        patterns = [
            r"No module named '([^']+)'",
            r"No module named ([^\s]+)",
            r"cannot import name '([^']+)'"
        ]

        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)

        return None

    def _get_documentation_url(self, category: ErrorCategory) -> str:
        """Get documentation URL for error category."""
        base_url = "https://docs.claude-commands.dev/errors"
        return f"{base_url}/{category.name.lower()}"

    def print_error(self, error: FormattedError):
        """
        Print formatted error to console.

        Args:
            error: Formatted error to display
        """
        if not self.enabled or not self.console:
            # Fallback to simple print
            print(f"Error: {error.title}", file=sys.stderr)
            print(f"Message: {error.message}", file=sys.stderr)
            return

        self.console.print()

        # Main error panel
        icon, color, severity_text = error.severity.value
        title = f"{icon} {error.title} [dim]({error.error_id})[/dim]"

        content = f"[bold {color}]{error.message}[/bold {color}]"

        panel = Panel(
            content,
            title=title,
            border_style=color,
            expand=False
        )
        self.console.print(panel)

        # Context information
        if error.context:
            self._print_context(error.context)

        # Suggestions
        if self.show_suggestions and error.suggestions:
            self._print_suggestions(error.suggestions)

        # Stack trace
        if self.show_stack_trace and error.stack_trace:
            self._print_stack_trace(error.stack_trace)

        # Documentation link
        if error.documentation_url:
            self.console.print(
                f"\n[dim]📚 Documentation: {error.documentation_url}[/dim]"
            )

        self.console.print()

    def _print_context(self, context: ErrorContext):
        """Print error context."""
        if not self.console:
            return

        parts = []

        if context.file_path:
            location = f"[cyan]{context.file_path}[/cyan]"
            if context.line_number:
                location += f":[yellow]{context.line_number}[/yellow]"
                if context.column_number:
                    location += f":[yellow]{context.column_number}[/yellow]"
            parts.append(f"[bold]Location:[/bold] {location}")

        if context.function_name:
            parts.append(f"[bold]Function:[/bold] [cyan]{context.function_name}[/cyan]")

        if context.command:
            parts.append(f"[bold]Command:[/bold] [magenta]{context.command}[/magenta]")

        if context.agent:
            parts.append(f"[bold]Agent:[/bold] [blue]{context.agent}[/blue]")

        if parts:
            self.console.print(Panel(
                "\n".join(parts),
                title="[bold]Context[/bold]",
                border_style="dim",
                expand=False
            ))

        # Code snippet
        if context.code_snippet and context.line_number:
            syntax = Syntax(
                context.code_snippet,
                "python",
                theme="monokai",
                line_numbers=True,
                start_line=context.line_number - 2,
                highlight_lines={context.line_number}
            )
            self.console.print(Panel(
                syntax,
                title="[bold]Code[/bold]",
                border_style="dim",
                expand=False
            ))

    def _print_suggestions(self, suggestions: List[ErrorSuggestion]):
        """Print fix suggestions."""
        if not self.console or not suggestions:
            return

        table = Table(show_header=True, header_style="bold green", box=None)
        table.add_column("", width=2)
        table.add_column("Suggestion", style="cyan")
        table.add_column("Action")

        for idx, suggestion in enumerate(suggestions, 1):
            # Format confidence indicator
            confidence_icon = "⭐" if suggestion.confidence > 0.8 else "○"

            # Format action
            action_text = suggestion.command or suggestion.action or suggestion.description
            if suggestion.command:
                action_text = f"[yellow]$ {suggestion.command}[/yellow]"

            table.add_row(
                confidence_icon,
                f"[bold]{suggestion.title}[/bold]\n{suggestion.description}",
                action_text
            )

        self.console.print(Panel(
            table,
            title="[bold green]💡 Suggestions[/bold green]",
            border_style="green",
            expand=False
        ))

    def _print_stack_trace(self, stack_trace: str):
        """Print formatted stack trace."""
        if not self.console:
            return

        # Syntax highlight stack trace
        syntax = Syntax(
            stack_trace,
            "pytb",
            theme="monokai",
            word_wrap=True
        )

        self.console.print(Panel(
            syntax,
            title="[bold red]Stack Trace[/bold red]",
            border_style="red",
            expand=False,
            padding=(1, 2)
        ))


# Convenience function
def format_error(
    exception: Exception,
    **kwargs
) -> FormattedError:
    """Format an exception with default formatter."""
    formatter = ErrorFormatter()
    return formatter.format_exception(exception, **kwargs)


def print_error(exception: Exception, **kwargs):
    """Format and print an exception."""
    formatter = ErrorFormatter()
    formatted = formatter.format_exception(exception, **kwargs)
    formatter.print_error(formatted)