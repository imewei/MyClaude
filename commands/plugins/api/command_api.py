#!/usr/bin/env python3
"""
Command Plugin API
==================

API for creating custom command plugins.

Provides helper functions and utilities for command development.

Author: Claude Code Framework
Version: 1.0
Last Updated: 2025-09-29
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Import from parent package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.plugin_base import CommandPlugin, PluginMetadata, PluginContext, PluginResult, PluginType


class CommandAPI:
    """
    Helper API for command plugin development.

    Provides utilities for:
    - Argument parsing
    - File operations
    - Result formatting
    - Error handling
    """

    @staticmethod
    def create_metadata(
        name: str,
        version: str,
        description: str,
        author: str,
        **kwargs
    ) -> PluginMetadata:
        """
        Create command plugin metadata.

        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            author: Plugin author
            **kwargs: Additional metadata fields

        Returns:
            PluginMetadata instance
        """
        return PluginMetadata(
            name=name,
            version=version,
            plugin_type=PluginType.COMMAND,
            description=description,
            author=author,
            **kwargs
        )

    @staticmethod
    def success_result(
        plugin_name: str,
        data: Dict[str, Any] = None,
        message: str = ""
    ) -> PluginResult:
        """
        Create a success result.

        Args:
            plugin_name: Plugin name
            data: Result data
            message: Success message

        Returns:
            PluginResult
        """
        result_data = data or {}
        if message:
            result_data['message'] = message

        return PluginResult(
            success=True,
            plugin_name=plugin_name,
            data=result_data
        )

    @staticmethod
    def error_result(
        plugin_name: str,
        error: str,
        data: Dict[str, Any] = None
    ) -> PluginResult:
        """
        Create an error result.

        Args:
            plugin_name: Plugin name
            error: Error message
            data: Additional data

        Returns:
            PluginResult
        """
        return PluginResult(
            success=False,
            plugin_name=plugin_name,
            data=data or {},
            errors=[error]
        )

    @staticmethod
    def parse_flags(args: List[str]) -> Dict[str, Any]:
        """
        Parse command line flags.

        Args:
            args: Argument list

        Returns:
            Dictionary of parsed flags
        """
        flags = {}
        positional = []

        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith('--'):
                # Long flag
                flag_name = arg[2:]

                # Check for value
                if '=' in flag_name:
                    key, value = flag_name.split('=', 1)
                    flags[key] = value
                elif i + 1 < len(args) and not args[i + 1].startswith('--'):
                    flags[flag_name] = args[i + 1]
                    i += 1
                else:
                    flags[flag_name] = True

            elif arg.startswith('-'):
                # Short flag
                flag_name = arg[1:]
                flags[flag_name] = True

            else:
                # Positional argument
                positional.append(arg)

            i += 1

        flags['_positional'] = positional
        return flags

    @staticmethod
    def read_file(file_path: Path) -> Optional[str]:
        """
        Read file contents.

        Args:
            file_path: Path to file

        Returns:
            File contents or None if error
        """
        try:
            return file_path.read_text()
        except Exception:
            return None

    @staticmethod
    def write_file(file_path: Path, content: str) -> bool:
        """
        Write file contents.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            True if successful
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return True
        except Exception:
            return False

    @staticmethod
    def read_json(file_path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file"""
        try:
            return json.loads(file_path.read_text())
        except Exception:
            return None

    @staticmethod
    def write_json(file_path: Path, data: Dict[str, Any]) -> bool:
        """Write JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(json.dumps(data, indent=2))
            return True
        except Exception:
            return False

    @staticmethod
    def format_output(result: PluginResult) -> str:
        """
        Format plugin result for display.

        Args:
            result: Plugin result

        Returns:
            Formatted string
        """
        lines = []

        if result.success:
            lines.append(f"✅ {result.plugin_name} completed successfully")
        else:
            lines.append(f"❌ {result.plugin_name} failed")

        # Data
        if result.data:
            if 'message' in result.data:
                lines.append(f"\n{result.data['message']}")

            for key, value in result.data.items():
                if key != 'message':
                    lines.append(f"  {key}: {value}")

        # Warnings
        if result.warnings:
            lines.append("\n⚠️  Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        # Errors
        if result.errors:
            lines.append("\n❌ Errors:")
            for error in result.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)


# Export for convenience
def create_command_plugin(
    name: str,
    version: str,
    description: str,
    author: str,
    execute_func: callable
) -> type:
    """
    Create a simple command plugin class.

    Args:
        name: Plugin name
        version: Plugin version
        description: Description
        author: Author
        execute_func: Execution function

    Returns:
        Command plugin class

    Example:
        def my_execute(self, context):
            return CommandAPI.success_result(
                self.metadata.name,
                {"result": "success"}
            )

        MyCommand = create_command_plugin(
            "my-command",
            "1.0.0",
            "My custom command",
            "Author Name",
            my_execute
        )
    """
    metadata = CommandAPI.create_metadata(
        name=name,
        version=version,
        description=description,
        author=author
    )

    class GeneratedCommandPlugin(CommandPlugin):
        def __init__(self, metadata=metadata, config=None):
            super().__init__(metadata, config)

        def load(self):
            return True

        def execute(self, context):
            return execute_func(self, context)

        def get_command_info(self):
            return {
                "name": name,
                "description": description,
                "version": version,
                "author": author
            }

    GeneratedCommandPlugin.__name__ = f"{name.replace('-', '_').title()}Plugin"
    return GeneratedCommandPlugin


def main():
    """Test command API"""
    print("Command Plugin API")
    print("==================\n")
    print("Utilities for command plugin development")
    return 0


if __name__ == "__main__":
    sys.exit(main())