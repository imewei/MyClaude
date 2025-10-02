#!/usr/bin/env python3
"""
Agent Plugin API
================

API for creating custom agent plugins.

Author: Claude Code Framework
Version: 1.0
Last Updated: 2025-09-29
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.plugin_base import AgentPlugin, PluginMetadata, PluginContext, PluginResult, PluginType


class AgentAPI:
    """Helper API for agent plugin development"""

    @staticmethod
    def create_metadata(
        name: str,
        version: str,
        description: str,
        author: str,
        capabilities: List[str] = None,
        **kwargs
    ) -> PluginMetadata:
        """Create agent plugin metadata"""
        return PluginMetadata(
            name=name,
            version=version,
            plugin_type=PluginType.AGENT,
            description=description,
            author=author,
            capabilities=capabilities or [],
            **kwargs
        )

    @staticmethod
    def create_agent_profile(
        capabilities: List[str],
        specializations: List[str],
        languages: List[str],
        frameworks: List[str],
        priority: int = 5
    ) -> Dict[str, Any]:
        """
        Create agent profile.

        Args:
            capabilities: List of capabilities
            specializations: List of specializations
            languages: Supported languages
            frameworks: Supported frameworks
            priority: Agent priority (1-10)

        Returns:
            Agent profile dictionary
        """
        return {
            "capabilities": capabilities,
            "specializations": specializations,
            "languages": languages,
            "frameworks": frameworks,
            "priority": priority
        }

    @staticmethod
    def success_result(
        plugin_name: str,
        findings: List[str] = None,
        recommendations: List[str] = None,
        data: Dict[str, Any] = None
    ) -> PluginResult:
        """Create success result for agent"""
        result_data = data or {}
        result_data['findings'] = findings or []
        result_data['recommendations'] = recommendations or []

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
        """Create error result"""
        return PluginResult(
            success=False,
            plugin_name=plugin_name,
            data=data or {},
            errors=[error]
        )


def main():
    """Test agent API"""
    print("Agent Plugin API")
    print("================\n")
    print("Utilities for agent plugin development")
    return 0


if __name__ == "__main__":
    sys.exit(main())