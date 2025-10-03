"""
Intelligent error suggestion system.

Provides AI-powered error resolution with common fixes, pattern matching,
and context-aware suggestions.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json
from pathlib import Path

from .error_formatter import ErrorCategory, ErrorSuggestion


@dataclass
class ErrorPattern:
    """Pattern for matching errors."""
    pattern: str
    category: ErrorCategory
    suggestions: List[Dict[str, Any]]
    confidence: float = 0.8


class ErrorSuggestionEngine:
    """
    Intelligent error suggestion engine.

    Features:
    - Database of common errors and fixes
    - Pattern matching for error messages
    - Context-aware suggestions
    - Ranked suggestions (most likely first)
    - Copy-paste ready commands
    - Documentation links
    - Similar issue references

    Example:
        engine = ErrorSuggestionEngine()

        # Get suggestions for an error
        suggestions = engine.suggest_fixes(
            error_message="No module named 'numpy'",
            category=ErrorCategory.DEPENDENCY,
            context={"command": "optimize"}
        )

        for suggestion in suggestions:
            print(f"{suggestion.title}: {suggestion.command}")
    """

    def __init__(self):
        """Initialize suggestion engine."""
        self.patterns: List[ErrorPattern] = []
        self._load_patterns()

    def _load_patterns(self):
        """Load error patterns from database."""
        # Python import errors
        self.patterns.append(ErrorPattern(
            pattern=r"No module named ['\"]([^'\"]+)['\"]",
            category=ErrorCategory.DEPENDENCY,
            suggestions=[
                {
                    "title": "Install missing package",
                    "description": "Install the missing Python package",
                    "command": "pip install {0}",
                    "confidence": 0.95
                },
                {
                    "title": "Check package name",
                    "description": "Verify the package name is correct",
                    "action": "Search PyPI for correct package name",
                    "confidence": 0.7
                }
            ]
        ))

        # File not found errors
        self.patterns.append(ErrorPattern(
            pattern=r"No such file or directory: ['\"]([^'\"]+)['\"]",
            category=ErrorCategory.FILESYSTEM,
            suggestions=[
                {
                    "title": "Check file path",
                    "description": "Verify the file path is correct",
                    "action": "Check if file exists at: {0}",
                    "confidence": 0.9
                },
                {
                    "title": "Create missing file",
                    "description": "Create the file if it should exist",
                    "command": "touch {0}",
                    "confidence": 0.6
                }
            ]
        ))

        # Permission errors
        self.patterns.append(ErrorPattern(
            pattern=r"Permission denied",
            category=ErrorCategory.PERMISSION,
            suggestions=[
                {
                    "title": "Check file permissions",
                    "description": "Verify you have the necessary permissions",
                    "command": "ls -la {0}",
                    "confidence": 0.85
                },
                {
                    "title": "Change permissions",
                    "description": "Grant appropriate permissions",
                    "command": "chmod +x {0}",
                    "confidence": 0.7
                }
            ]
        ))

        # Configuration errors
        self.patterns.append(ErrorPattern(
            pattern=r"(?:missing|invalid) (?:config|configuration)",
            category=ErrorCategory.CONFIGURATION,
            suggestions=[
                {
                    "title": "Check configuration file",
                    "description": "Verify configuration file exists and is valid",
                    "action": "Review configuration file syntax",
                    "confidence": 0.85
                },
                {
                    "title": "Create default configuration",
                    "description": "Generate a default configuration file",
                    "action": "Run setup command to create config",
                    "confidence": 0.7
                }
            ]
        ))

        # Syntax errors
        self.patterns.append(ErrorPattern(
            pattern=r"invalid syntax",
            category=ErrorCategory.SYNTAX,
            suggestions=[
                {
                    "title": "Check syntax",
                    "description": "Review the code for syntax errors",
                    "action": "Check for missing colons, brackets, or quotes",
                    "confidence": 0.8
                },
                {
                    "title": "Run linter",
                    "description": "Use a linter to identify syntax issues",
                    "command": "pylint {0}",
                    "confidence": 0.75
                }
            ]
        ))

        # Network errors
        self.patterns.append(ErrorPattern(
            pattern=r"(?:connection|network|timeout) (?:error|failed)",
            category=ErrorCategory.NETWORK,
            suggestions=[
                {
                    "title": "Check network connection",
                    "description": "Verify you have internet connectivity",
                    "command": "ping google.com",
                    "confidence": 0.85
                },
                {
                    "title": "Check firewall settings",
                    "description": "Ensure firewall isn't blocking the connection",
                    "action": "Review firewall rules",
                    "confidence": 0.7
                },
                {
                    "title": "Retry with timeout",
                    "description": "Increase timeout duration and retry",
                    "action": "Add --timeout flag with higher value",
                    "confidence": 0.65
                }
            ]
        ))

        # Memory errors
        self.patterns.append(ErrorPattern(
            pattern=r"(?:memory|out of memory|oom)",
            category=ErrorCategory.RUNTIME,
            suggestions=[
                {
                    "title": "Reduce memory usage",
                    "description": "Process data in smaller chunks",
                    "action": "Implement batch processing",
                    "confidence": 0.8
                },
                {
                    "title": "Increase available memory",
                    "description": "Close other applications or increase system memory",
                    "action": "Free up system resources",
                    "confidence": 0.7
                }
            ]
        ))

        # Agent errors
        self.patterns.append(ErrorPattern(
            pattern=r"agent (?:failed|error|timeout)",
            category=ErrorCategory.AGENT,
            suggestions=[
                {
                    "title": "Retry with different agent",
                    "description": "Try running with a different agent",
                    "action": "Use --agents flag to specify alternative agents",
                    "confidence": 0.75
                },
                {
                    "title": "Check agent configuration",
                    "description": "Verify agent is properly configured",
                    "action": "Review agent settings",
                    "confidence": 0.7
                }
            ]
        ))

    def suggest_fixes(
        self,
        error_message: str,
        category: Optional[ErrorCategory] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorSuggestion]:
        """
        Generate fix suggestions for an error.

        Args:
            error_message: Error message text
            category: Error category (optional)
            context: Additional context information

        Returns:
            List of suggestions, ranked by confidence
        """
        suggestions = []
        context = context or {}

        # Find matching patterns
        for pattern in self.patterns:
            # Skip if category doesn't match
            if category and pattern.category != category:
                continue

            # Check if pattern matches
            match = re.search(pattern.pattern, error_message, re.IGNORECASE)
            if match:
                # Create suggestions from pattern
                for sugg_data in pattern.suggestions:
                    # Format with regex groups
                    title = sugg_data["title"]
                    description = sugg_data["description"]
                    action = sugg_data.get("action", "")
                    command = sugg_data.get("command", "")

                    # Replace placeholders with matched groups
                    if match.groups():
                        action = action.format(*match.groups()) if action else None
                        command = command.format(*match.groups()) if command else None

                    # Adjust confidence based on context
                    confidence = sugg_data.get("confidence", 0.7) * pattern.confidence

                    suggestion = ErrorSuggestion(
                        title=title,
                        description=description,
                        action=action if action else None,
                        command=command if command else None,
                        confidence=confidence
                    )

                    suggestions.append(suggestion)

        # Add context-aware suggestions
        suggestions.extend(self._get_context_suggestions(error_message, context))

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        # Remove duplicates
        seen = set()
        unique_suggestions = []
        for sugg in suggestions:
            key = (sugg.title, sugg.command)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(sugg)

        return unique_suggestions[:5]  # Return top 5

    def _get_context_suggestions(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[ErrorSuggestion]:
        """Get context-aware suggestions."""
        suggestions = []

        # Command-specific suggestions
        command = context.get("command", "")

        if command == "optimize":
            if "memory" in error_message.lower():
                suggestions.append(ErrorSuggestion(
                    title="Use memory-efficient optimization",
                    description="Try optimization with reduced memory usage",
                    action="Add --memory-efficient flag",
                    confidence=0.75
                ))

        elif command == "test":
            if "failed" in error_message.lower():
                suggestions.append(ErrorSuggestion(
                    title="Run tests in verbose mode",
                    description="Get more details about test failures",
                    command="pytest -v",
                    confidence=0.7
                ))

        # File path in context
        if "file_path" in context:
            file_path = context["file_path"]
            suggestions.append(ErrorSuggestion(
                title="Check file",
                description=f"Verify file at {file_path}",
                command=f"cat {file_path}",
                confidence=0.6
            ))

        return suggestions

    def get_similar_issues(
        self,
        error_message: str,
        limit: int = 3
    ) -> List[Dict[str, str]]:
        """
        Find similar issues and solutions.

        Args:
            error_message: Error message to match
            limit: Maximum number of similar issues

        Returns:
            List of similar issues with links
        """
        # This would typically query a database or API
        # For now, return mock data
        similar = []

        if "module" in error_message.lower():
            similar.append({
                "title": "ImportError: No module named 'package'",
                "url": "https://github.com/org/repo/issues/123",
                "solution": "Install the package with pip"
            })

        return similar[:limit]

    def search_documentation(
        self,
        error_message: str,
        category: ErrorCategory
    ) -> List[str]:
        """
        Search documentation for relevant help.

        Args:
            error_message: Error message
            category: Error category

        Returns:
            List of documentation URLs
        """
        base_url = "https://docs.claude-commands.dev"

        docs = [
            f"{base_url}/errors/{category.name.lower()}",
            f"{base_url}/troubleshooting"
        ]

        # Add specific docs based on error
        if "import" in error_message.lower():
            docs.append(f"{base_url}/dependencies")

        if "config" in error_message.lower():
            docs.append(f"{base_url}/configuration")

        return docs


# Convenience functions
def suggest_fixes(error_message: str, **kwargs) -> List[ErrorSuggestion]:
    """Get fix suggestions for an error message."""
    engine = ErrorSuggestionEngine()
    return engine.suggest_fixes(error_message, **kwargs)


def get_similar_issues(error_message: str, **kwargs) -> List[Dict[str, str]]:
    """Find similar issues and solutions."""
    engine = ErrorSuggestionEngine()
    return engine.get_similar_issues(error_message, **kwargs)