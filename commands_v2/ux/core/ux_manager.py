"""
Central UX coordinator.

Manages themes, layouts, animations, verbosity, and accessibility features.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

try:
    from rich.console import Console
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ThemeMode(Enum):
    """Theme modes."""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class VerbosityLevel(Enum):
    """Verbosity levels."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class OutputFormat(Enum):
    """Output formats."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class UXConfig:
    """UX configuration."""
    theme: ThemeMode = ThemeMode.DARK
    animations_enabled: bool = True
    progress_style: str = "bar"
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    confirmation_prompts: bool = True
    error_suggestions: bool = True
    command_recommendations: bool = True
    tutorial_mode: bool = False
    accessibility_mode: bool = False
    color_enabled: bool = True
    output_format: OutputFormat = OutputFormat.TEXT


class UXManager:
    """
    Central UX coordinator.

    Features:
    - Theme management (dark/light)
    - Layout configuration
    - Animation settings
    - Verbosity levels (quiet, normal, verbose, debug)
    - Output formatting (JSON, text, markdown)
    - Accessibility support

    Example:
        ux = UXManager()

        # Configure UX
        ux.config.theme = ThemeMode.DARK
        ux.config.animations_enabled = True
        ux.config.verbosity = VerbosityLevel.VERBOSE

        # Get console with theme
        console = ux.get_console()
        console.print("[green]Success![/green]")
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize UX manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or (Path.home() / ".claude" / "ux_config.json")
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        self.config = UXConfig()
        self._console: Optional[Console] = None

        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                    self.config.theme = ThemeMode(data.get("theme", "dark"))
                    self.config.animations_enabled = data.get("animations", True)
                    self.config.progress_style = data.get("progress_style", "bar")
                    self.config.verbosity = VerbosityLevel(data.get("verbosity", 1))
                    self.config.confirmation_prompts = data.get("confirmation_prompts", True)
                    self.config.error_suggestions = data.get("error_suggestions", True)
                    self.config.command_recommendations = data.get("command_recommendations", True)
                    self.config.tutorial_mode = data.get("tutorial_mode", False)
                    self.config.accessibility_mode = data.get("accessibility_mode", False)
                    self.config.color_enabled = data.get("color_enabled", True)
                    self.config.output_format = OutputFormat(data.get("output_format", "text"))
            except Exception:
                pass

    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({
                    "theme": self.config.theme.value,
                    "animations": self.config.animations_enabled,
                    "progress_style": self.config.progress_style,
                    "verbosity": self.config.verbosity.value,
                    "confirmation_prompts": self.config.confirmation_prompts,
                    "error_suggestions": self.config.error_suggestions,
                    "command_recommendations": self.config.command_recommendations,
                    "tutorial_mode": self.config.tutorial_mode,
                    "accessibility_mode": self.config.accessibility_mode,
                    "color_enabled": self.config.color_enabled,
                    "output_format": self.config.output_format.value
                }, f, indent=2)
        except Exception:
            pass

    def get_console(self) -> Console:
        """
        Get configured console instance.

        Returns:
            Rich Console with theme
        """
        if not RICH_AVAILABLE:
            return None

        if self._console is None:
            theme = self._get_theme()
            self._console = Console(
                theme=theme,
                color_system="auto" if self.config.color_enabled else None,
                force_terminal=not self.config.accessibility_mode
            )

        return self._console

    def _get_theme(self) -> Theme:
        """Get Rich theme based on configuration."""
        if self.config.theme == ThemeMode.DARK:
            return Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "red bold",
                "success": "green",
                "highlight": "magenta"
            })
        else:
            return Theme({
                "info": "blue",
                "warning": "yellow",
                "error": "red bold",
                "success": "green",
                "highlight": "purple"
            })

    def should_show_progress(self) -> bool:
        """Check if progress indicators should be shown."""
        return (
            self.config.verbosity >= VerbosityLevel.NORMAL
            and not self.config.accessibility_mode
        )

    def should_show_animations(self) -> bool:
        """Check if animations should be shown."""
        return (
            self.config.animations_enabled
            and not self.config.accessibility_mode
        )

    def should_show_suggestions(self) -> bool:
        """Check if suggestions should be shown."""
        return (
            self.config.error_suggestions
            or self.config.command_recommendations
        )

    def is_quiet_mode(self) -> bool:
        """Check if in quiet mode."""
        return self.config.verbosity == VerbosityLevel.QUIET

    def is_verbose_mode(self) -> bool:
        """Check if in verbose mode."""
        return self.config.verbosity >= VerbosityLevel.VERBOSE

    def is_debug_mode(self) -> bool:
        """Check if in debug mode."""
        return self.config.verbosity == VerbosityLevel.DEBUG

    def format_output(self, data: Any, format_type: Optional[OutputFormat] = None) -> str:
        """
        Format output according to configuration.

        Args:
            data: Data to format
            format_type: Override output format

        Returns:
            Formatted string
        """
        format_type = format_type or self.config.output_format

        if format_type == OutputFormat.JSON:
            return json.dumps(data, indent=2)

        elif format_type == OutputFormat.MARKDOWN:
            return self._format_as_markdown(data)

        elif format_type == OutputFormat.HTML:
            return self._format_as_html(data)

        else:  # TEXT
            return str(data)

    def _format_as_markdown(self, data: Any) -> str:
        """Format data as markdown."""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"**{key}**: {value}")
            return "\n".join(lines)
        return str(data)

    def _format_as_html(self, data: Any) -> str:
        """Format data as HTML."""
        if isinstance(data, dict):
            lines = ["<ul>"]
            for key, value in data.items():
                lines.append(f"<li><strong>{key}</strong>: {value}</li>")
            lines.append("</ul>")
            return "\n".join(lines)
        return f"<p>{data}</p>"


# Global UX manager instance
_global_ux_manager: Optional[UXManager] = None


def get_ux_manager() -> UXManager:
    """Get or create global UX manager."""
    global _global_ux_manager
    if _global_ux_manager is None:
        _global_ux_manager = UXManager()
    return _global_ux_manager