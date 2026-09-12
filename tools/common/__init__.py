"""
Shared utilities for MyClaude plugin validation tools.

This module provides common functionality used across multiple tools:
- PluginLoader: Unified plugin.json loading and caching
- ValidationResult: Standardized validation result dataclass
"""

from tools.common.loader import PluginLoader
from tools.common.models import (
    PluginMetadata,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    "PluginLoader",
    "PluginMetadata",
    "ValidationIssue",
    "ValidationResult",
]
