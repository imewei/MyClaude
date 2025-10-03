#!/usr/bin/env python3
"""
Hello World Plugin
==================

Simple example command plugin that demonstrates basic plugin structure.

Usage:
    /hello-world [name]
"""

import sys
from pathlib import Path

# Add plugin system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import CommandPlugin, PluginContext, PluginResult
from api.command_api import CommandAPI


class HelloWorldPlugin(CommandPlugin):
    """Simple hello world command plugin"""

    def load(self) -> bool:
        """Load plugin"""
        self.logger.info(f"Loading {self.metadata.name} plugin")
        return True

    def execute(self, context: PluginContext) -> PluginResult:
        """
        Execute hello world command.

        Args:
            context: Plugin execution context

        Returns:
            Plugin execution result
        """
        # Get configuration
        greeting = self.get_config('greeting', 'Hello')

        # Parse arguments
        args = context.config.get('args', [])
        name = args[0] if args else "World"

        # Create message
        message = f"{greeting}, {name}!"

        # Log execution
        self.logger.info(f"Greeting: {message}")

        # Return success result
        return CommandAPI.success_result(
            plugin_name=self.metadata.name,
            data={
                "message": message,
                "greeted": name
            }
        )

    def get_command_info(self) -> dict:
        """Get command information"""
        return {
            "name": "hello-world",
            "description": "Greet the world or a specific person",
            "usage": "/hello-world [name]",
            "arguments": [
                {
                    "name": "name",
                    "description": "Name to greet (optional)",
                    "required": False,
                    "default": "World"
                }
            ],
            "examples": [
                {
                    "command": "/hello-world",
                    "description": "Greet the world"
                },
                {
                    "command": "/hello-world Alice",
                    "description": "Greet Alice"
                }
            ]
        }


# Plugin metadata (used by entry point loading)
METADATA = {
    "name": "hello-world",
    "version": "1.0.0",
    "type": "command",
    "description": "Simple hello world command",
    "author": "Claude Code Framework"
}


def main():
    """Test plugin"""
    from core.plugin_base import PluginMetadata, PluginType

    metadata = PluginMetadata(
        name="hello-world",
        version="1.0.0",
        plugin_type=PluginType.COMMAND,
        description="Hello world plugin",
        author="Test"
    )

    plugin = HelloWorldPlugin(metadata)
    plugin.load()

    context = PluginContext(
        plugin_name="hello-world",
        command_name="hello-world",
        work_dir=Path.cwd(),
        config={"args": ["Claude"]},
        framework_version="2.0.0"
    )

    result = plugin.execute(context)
    print(CommandAPI.format_output(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())