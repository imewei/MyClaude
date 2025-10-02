#!/usr/bin/env python3
"""
Plugin Base Classes
===================

Base classes for all plugin types in the command executor framework.

Plugin Types:
- CommandPlugin: Custom slash commands
- AgentPlugin: Custom agent implementations
- ValidatorPlugin: Custom validation logic
- CacheProviderPlugin: Custom cache backends
- ReporterPlugin: Custom report formats
- IntegrationPlugin: External service integrations

Author: Claude Code Framework
Version: 1.0
Last Updated: 2025-09-29
"""

import sys
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ============================================================================
# Plugin Types and Enums
# ============================================================================

class PluginType(Enum):
    """Types of plugins supported"""
    COMMAND = "command"
    AGENT = "agent"
    VALIDATOR = "validator"
    CACHE_PROVIDER = "cache_provider"
    REPORTER = "reporter"
    INTEGRATION = "integration"
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"


class PluginStatus(Enum):
    """Plugin lifecycle status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class HookType(Enum):
    """Plugin hook types"""
    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"
    PRE_VALIDATION = "pre_validation"
    POST_VALIDATION = "post_validation"
    PRE_AGENT = "pre_agent"
    POST_AGENT = "post_agent"
    PRE_CACHE = "pre_cache"
    POST_CACHE = "post_cache"
    PRE_REPORT = "pre_report"
    POST_REPORT = "post_report"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str

    # Requirements
    framework_version: str = ">=2.0.0"
    python_version: str = ">=3.8"
    dependencies: List[str] = field(default_factory=list)

    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    supported_commands: List[str] = field(default_factory=list)

    # Configuration
    config_schema: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)

    # Security
    permissions: List[str] = field(default_factory=list)
    sandbox: bool = True

    # Metadata
    homepage: str = ""
    repository: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)


@dataclass
class PluginContext:
    """Context passed to plugin execution"""
    plugin_name: str
    command_name: str
    work_dir: Path
    config: Dict[str, Any]
    framework_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginResult:
    """Result from plugin execution"""
    success: bool
    plugin_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Base Plugin Class
# ============================================================================

class BasePlugin(ABC):
    """
    Base class for all plugins.

    All plugins must inherit from this class and implement required methods.

    Lifecycle:
    1. __init__ - Plugin instantiation
    2. load() - Plugin loading and initialization
    3. validate() - Plugin validation
    4. execute() - Plugin execution
    5. cleanup() - Plugin cleanup

    Features:
    - Configuration management
    - Hook registration
    - Error handling
    - Logging
    - State management
    """

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        """
        Initialize plugin.

        Args:
            metadata: Plugin metadata
            config: Plugin configuration
        """
        self.metadata = metadata
        self.config = config or metadata.default_config.copy()
        self.status = PluginStatus.UNLOADED
        self.logger = self._setup_logging()
        self.hooks: Dict[HookType, List[callable]] = {}
        self.state: Dict[str, Any] = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup plugin logging"""
        logger = logging.getLogger(f"plugin.{self.metadata.name}")
        logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def load(self) -> bool:
        """
        Load and initialize the plugin.

        Called when plugin is loaded by the system.
        Perform any initialization, validation, or setup here.

        Returns:
            True if load successful, False otherwise
        """
        pass

    @abstractmethod
    def execute(self, context: PluginContext) -> PluginResult:
        """
        Execute plugin logic.

        Args:
            context: Plugin execution context

        Returns:
            Plugin execution result
        """
        pass

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate plugin configuration and requirements.

        Returns:
            Tuple of (success, error_messages)
        """
        errors = []

        # Validate required fields
        if not self.metadata.name:
            errors.append("Plugin name is required")

        if not self.metadata.version:
            errors.append("Plugin version is required")

        # Validate dependencies
        for dep in self.metadata.dependencies:
            if not self._check_dependency(dep):
                errors.append(f"Missing dependency: {dep}")

        return len(errors) == 0, errors

    def cleanup(self):
        """
        Cleanup plugin resources.

        Called when plugin is unloaded or system shuts down.
        """
        self.logger.info(f"Cleaning up plugin: {self.metadata.name}")
        self.status = PluginStatus.UNLOADED

    def register_hook(self, hook_type: HookType, handler: callable):
        """
        Register a hook handler.

        Args:
            hook_type: Type of hook
            handler: Hook handler function
        """
        if hook_type not in self.hooks:
            self.hooks[hook_type] = []
        self.hooks[hook_type].append(handler)
        self.logger.debug(f"Registered hook: {hook_type.value}")

    def execute_hooks(
        self,
        hook_type: HookType,
        context: PluginContext,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute registered hooks.

        Args:
            hook_type: Type of hook to execute
            context: Plugin context
            data: Hook data

        Returns:
            Modified data after hooks
        """
        if hook_type not in self.hooks:
            return data or {}

        result_data = data or {}

        for handler in self.hooks[hook_type]:
            try:
                result_data = handler(context, result_data)
            except Exception as e:
                self.logger.error(f"Hook execution failed: {e}")

        return result_data

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get plugin state value"""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any):
        """Set plugin state value"""
        self.state[key] = value

    def _check_dependency(self, dependency: str) -> bool:
        """Check if dependency is available"""
        try:
            parts = dependency.split("==")
            module_name = parts[0]
            __import__(module_name)
            return True
        except ImportError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin to dictionary"""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "type": self.metadata.plugin_type.value,
            "status": self.status.value,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "capabilities": self.metadata.capabilities,
            "config": self.config,
        }


# ============================================================================
# Command Plugin Base
# ============================================================================

class CommandPlugin(BasePlugin):
    """
    Base class for command plugins.

    Command plugins add new slash commands to the framework.

    Example:
        class MyCommand(CommandPlugin):
            def execute(self, context):
                return PluginResult(
                    success=True,
                    plugin_name=self.metadata.name,
                    data={"message": "Hello!"}
                )
    """

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        super().__init__(metadata, config)
        if metadata.plugin_type != PluginType.COMMAND:
            metadata.plugin_type = PluginType.COMMAND

    @abstractmethod
    def get_command_info(self) -> Dict[str, Any]:
        """
        Get command information.

        Returns:
            Dictionary with command metadata:
            - name: Command name
            - description: Command description
            - arguments: List of command arguments
            - examples: Usage examples
        """
        pass

    def parse_arguments(self, args: List[str]) -> Dict[str, Any]:
        """
        Parse command arguments.

        Args:
            args: Raw argument list

        Returns:
            Parsed arguments dictionary
        """
        # Default implementation - override for custom parsing
        return {"args": args}


# ============================================================================
# Agent Plugin Base
# ============================================================================

class AgentPlugin(BasePlugin):
    """
    Base class for agent plugins.

    Agent plugins add new agents to the multi-agent system.
    """

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        super().__init__(metadata, config)
        if metadata.plugin_type != PluginType.AGENT:
            metadata.plugin_type = PluginType.AGENT

    @abstractmethod
    def get_agent_profile(self) -> Dict[str, Any]:
        """
        Get agent profile information.

        Returns:
            Dictionary with agent profile:
            - capabilities: List of capabilities
            - specializations: List of specializations
            - languages: Supported languages
            - frameworks: Supported frameworks
        """
        pass

    @abstractmethod
    def analyze(self, context: PluginContext) -> Dict[str, Any]:
        """
        Perform agent analysis.

        Args:
            context: Analysis context

        Returns:
            Analysis results
        """
        pass


# ============================================================================
# Validator Plugin Base
# ============================================================================

class ValidatorPlugin(BasePlugin):
    """Base class for validator plugins"""

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        super().__init__(metadata, config)
        if metadata.plugin_type != PluginType.VALIDATOR:
            metadata.plugin_type = PluginType.VALIDATOR

    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate input data.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass


# ============================================================================
# Cache Provider Plugin Base
# ============================================================================

class CacheProviderPlugin(BasePlugin):
    """Base class for cache provider plugins"""

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        super().__init__(metadata, config)
        if metadata.plugin_type != PluginType.CACHE_PROVIDER:
            metadata.plugin_type = PluginType.CACHE_PROVIDER

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value"""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete cached value"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all cached values"""
        pass


# ============================================================================
# Reporter Plugin Base
# ============================================================================

class ReporterPlugin(BasePlugin):
    """Base class for reporter plugins"""

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        super().__init__(metadata, config)
        if metadata.plugin_type != PluginType.REPORTER:
            metadata.plugin_type = PluginType.REPORTER

    @abstractmethod
    def generate_report(
        self,
        data: Dict[str, Any],
        format: str = "text"
    ) -> str:
        """
        Generate report from data.

        Args:
            data: Report data
            format: Report format (text, json, html, etc.)

        Returns:
            Formatted report string
        """
        pass


# ============================================================================
# Integration Plugin Base
# ============================================================================

class IntegrationPlugin(BasePlugin):
    """Base class for integration plugins"""

    def __init__(self, metadata: PluginMetadata, config: Dict[str, Any] = None):
        super().__init__(metadata, config)
        if metadata.plugin_type != PluginType.INTEGRATION:
            metadata.plugin_type = PluginType.INTEGRATION

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to external service.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def send(self, data: Dict[str, Any]) -> bool:
        """
        Send data to external service.

        Args:
            data: Data to send

        Returns:
            True if send successful
        """
        pass

    def disconnect(self):
        """Disconnect from external service"""
        pass


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Test plugin base classes"""
    print("Plugin Base Classes")
    print("===================\n")
    print("Base classes for plugin development:")
    print("  - BasePlugin: Core plugin functionality")
    print("  - CommandPlugin: Custom commands")
    print("  - AgentPlugin: Custom agents")
    print("  - ValidatorPlugin: Custom validation")
    print("  - CacheProviderPlugin: Custom cache backends")
    print("  - ReporterPlugin: Custom report formats")
    print("  - IntegrationPlugin: External integrations")
    return 0


if __name__ == "__main__":
    sys.exit(main())