#!/usr/bin/env python3
"""
Plugin Hook System
==================

Hook system for extending framework behavior with plugins.

Hook Types:
- Execution hooks: Pre/post command execution
- Validation hooks: Custom validation logic
- Agent hooks: Agent selection and execution
- Cache hooks: Cache operations
- Report hooks: Report generation
- Error hooks: Error handling

Author: Claude Code Framework
Version: 1.0
Last Updated: 2025-09-29
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .plugin_base import HookType, PluginContext


# ============================================================================
# Hook Registry
# ============================================================================

@dataclass
class HookHandler:
    """Hook handler registration"""
    plugin_name: str
    hook_type: HookType
    handler: Callable
    priority: int = 5  # 1-10, higher executes first
    enabled: bool = True


class HookRegistry:
    """
    Central registry for all plugin hooks.

    Features:
    - Hook registration and management
    - Priority-based execution order
    - Hook enable/disable
    - Hook execution with error handling
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hooks: Dict[HookType, List[HookHandler]] = {}

        # Initialize hook lists
        for hook_type in HookType:
            self.hooks[hook_type] = []

    def register_hook(
        self,
        plugin_name: str,
        hook_type: HookType,
        handler: Callable,
        priority: int = 5
    ):
        """
        Register a hook handler.

        Args:
            plugin_name: Name of plugin registering the hook
            hook_type: Type of hook
            handler: Hook handler function
            priority: Priority (1-10, higher executes first)
        """
        hook_handler = HookHandler(
            plugin_name=plugin_name,
            hook_type=hook_type,
            handler=handler,
            priority=priority
        )

        self.hooks[hook_type].append(hook_handler)

        # Sort by priority (descending)
        self.hooks[hook_type].sort(key=lambda h: h.priority, reverse=True)

        self.logger.info(f"Registered hook: {plugin_name} -> {hook_type.value} (priority {priority})")

    def unregister_hook(self, plugin_name: str, hook_type: HookType):
        """
        Unregister a hook handler.

        Args:
            plugin_name: Plugin name
            hook_type: Hook type
        """
        self.hooks[hook_type] = [
            h for h in self.hooks[hook_type]
            if h.plugin_name != plugin_name
        ]

        self.logger.info(f"Unregistered hook: {plugin_name} -> {hook_type.value}")

    def unregister_all_hooks(self, plugin_name: str):
        """
        Unregister all hooks for a plugin.

        Args:
            plugin_name: Plugin name
        """
        for hook_type in HookType:
            self.unregister_hook(plugin_name, hook_type)

    def execute_hooks(
        self,
        hook_type: HookType,
        context: PluginContext,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute all registered hooks for a type.

        Args:
            hook_type: Type of hook to execute
            context: Plugin context
            data: Hook data (passed through chain)

        Returns:
            Modified data after all hooks
        """
        handlers = [h for h in self.hooks[hook_type] if h.enabled]

        if not handlers:
            return data or {}

        self.logger.debug(f"Executing {len(handlers)} hooks for {hook_type.value}")

        result_data = data or {}

        for handler in handlers:
            try:
                # Execute hook
                result_data = handler.handler(context, result_data)

                self.logger.debug(f"  ✓ {handler.plugin_name}")

            except Exception as e:
                self.logger.error(f"Hook execution failed: {handler.plugin_name} - {e}")

                # Add error to result
                if 'hook_errors' not in result_data:
                    result_data['hook_errors'] = []

                result_data['hook_errors'].append({
                    'plugin': handler.plugin_name,
                    'hook_type': hook_type.value,
                    'error': str(e)
                })

        return result_data

    def enable_hook(self, plugin_name: str, hook_type: HookType):
        """Enable a hook"""
        for handler in self.hooks[hook_type]:
            if handler.plugin_name == plugin_name:
                handler.enabled = True

    def disable_hook(self, plugin_name: str, hook_type: HookType):
        """Disable a hook"""
        for handler in self.hooks[hook_type]:
            if handler.plugin_name == plugin_name:
                handler.enabled = False

    def get_hooks(self, hook_type: HookType) -> List[HookHandler]:
        """Get all hooks for a type"""
        return self.hooks[hook_type]

    def get_hook_count(self, hook_type: HookType) -> int:
        """Get count of hooks for a type"""
        return len([h for h in self.hooks[hook_type] if h.enabled])

    def get_stats(self) -> Dict[str, Any]:
        """Get hook statistics"""
        return {
            "total_hooks": sum(len(handlers) for handlers in self.hooks.values()),
            "enabled_hooks": sum(
                len([h for h in handlers if h.enabled])
                for handlers in self.hooks.values()
            ),
            "hooks_by_type": {
                hook_type.value: {
                    "total": len(handlers),
                    "enabled": len([h for h in handlers if h.enabled]),
                    "plugins": [h.plugin_name for h in handlers if h.enabled]
                }
                for hook_type, handlers in self.hooks.items()
            }
        }


# ============================================================================
# Hook Manager
# ============================================================================

class HookManager:
    """
    High-level hook management.

    Integrates with PluginManager to automatically register/unregister
    hooks when plugins are loaded/unloaded.
    """

    def __init__(self, registry: HookRegistry = None):
        """
        Initialize hook manager.

        Args:
            registry: Hook registry (creates new if not provided)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = registry or HookRegistry()

    def register_plugin_hooks(self, plugin):
        """
        Register all hooks from a plugin.

        Args:
            plugin: Plugin instance
        """
        from .plugin_base import BasePlugin

        if not isinstance(plugin, BasePlugin):
            return

        # Register hooks from plugin.hooks dict
        for hook_type, handlers in plugin.hooks.items():
            for handler in handlers:
                self.registry.register_hook(
                    plugin_name=plugin.metadata.name,
                    hook_type=hook_type,
                    handler=handler,
                    priority=plugin.metadata.priority if hasattr(plugin.metadata, 'priority') else 5
                )

    def unregister_plugin_hooks(self, plugin):
        """
        Unregister all hooks from a plugin.

        Args:
            plugin: Plugin instance
        """
        from .plugin_base import BasePlugin

        if not isinstance(plugin, BasePlugin):
            return

        self.registry.unregister_all_hooks(plugin.metadata.name)

    def execute_hooks(
        self,
        hook_type: HookType,
        context: PluginContext,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute hooks through registry.

        Args:
            hook_type: Hook type
            context: Plugin context
            data: Hook data

        Returns:
            Modified data
        """
        return self.registry.execute_hooks(hook_type, context, data)


# ============================================================================
# Hook Decorators
# ============================================================================

def hook(hook_type: HookType, priority: int = 5):
    """
    Decorator to mark a method as a hook handler.

    Usage:
        class MyPlugin(BasePlugin):
            @hook(HookType.PRE_EXECUTION, priority=8)
            def my_pre_execution_hook(self, context, data):
                # Hook logic
                return data

    Args:
        hook_type: Type of hook
        priority: Priority (1-10)
    """
    def decorator(func):
        func._hook_type = hook_type
        func._hook_priority = priority
        return func
    return decorator


def discover_hooks(plugin) -> List[tuple]:
    """
    Discover hooks in a plugin using decorators.

    Args:
        plugin: Plugin instance

    Returns:
        List of (hook_type, handler, priority) tuples
    """
    hooks = []

    for name in dir(plugin):
        if name.startswith('_'):
            continue

        attr = getattr(plugin, name)

        if hasattr(attr, '_hook_type'):
            hooks.append((
                attr._hook_type,
                attr,
                attr._hook_priority
            ))

    return hooks


# ============================================================================
# Common Hook Implementations
# ============================================================================

def create_logging_hook(hook_type: HookType) -> Callable:
    """Create a logging hook for debugging"""
    def logging_hook(context: PluginContext, data: Dict[str, Any]) -> Dict[str, Any]:
        logger = logging.getLogger("plugin.hooks")
        logger.info(f"Hook executed: {hook_type.value} for {context.command_name}")
        return data
    return logging_hook


def create_timing_hook(hook_type: HookType) -> Callable:
    """Create a timing hook for performance monitoring"""
    import time

    def timing_hook(context: PluginContext, data: Dict[str, Any]) -> Dict[str, Any]:
        if hook_type == HookType.PRE_EXECUTION:
            data['_hook_start_time'] = time.time()
        elif hook_type == HookType.POST_EXECUTION:
            if '_hook_start_time' in data:
                duration = time.time() - data['_hook_start_time']
                data['_hook_duration'] = duration
                del data['_hook_start_time']
        return data

    return timing_hook


def create_validation_hook(validator: Callable) -> Callable:
    """Create a validation hook"""
    def validation_hook(context: PluginContext, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            is_valid, errors = validator(data)
            if not is_valid:
                data['validation_errors'] = errors
        except Exception as e:
            data['validation_errors'] = [str(e)]
        return data

    return validation_hook


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Test hook system"""
    logging.basicConfig(level=logging.INFO)

    print("Plugin Hook System")
    print("==================\n")

    # Create registry
    registry = HookRegistry()

    # Create test context
    from pathlib import Path
    context = PluginContext(
        plugin_name="test",
        command_name="test-command",
        work_dir=Path.cwd(),
        config={},
        framework_version="2.0.0"
    )

    # Register test hooks
    def test_hook_1(ctx, data):
        print(f"  Hook 1 executed")
        data['hook1'] = True
        return data

    def test_hook_2(ctx, data):
        print(f"  Hook 2 executed")
        data['hook2'] = True
        return data

    registry.register_hook("plugin1", HookType.PRE_EXECUTION, test_hook_1, priority=8)
    registry.register_hook("plugin2", HookType.PRE_EXECUTION, test_hook_2, priority=6)

    # Execute hooks
    print("Executing pre-execution hooks:")
    result = registry.execute_hooks(HookType.PRE_EXECUTION, context, {})
    print(f"\nResult: {result}")

    # Show stats
    print(f"\nHook Statistics:")
    stats = registry.get_stats()
    print(f"Total hooks: {stats['total_hooks']}")
    print(f"Enabled hooks: {stats['enabled_hooks']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())