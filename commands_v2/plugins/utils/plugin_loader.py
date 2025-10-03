#!/usr/bin/env python3
"""
Plugin Loader Utilities
=======================

Advanced plugin loading utilities.
"""

import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Type


class DynamicPluginLoader:
    """Dynamic plugin loading utility"""

    @staticmethod
    def load_plugin_from_file(
        file_path: Path,
        class_name: Optional[str] = None
    ) -> Optional[Type]:
        """
        Load plugin class from Python file.

        Args:
            file_path: Path to plugin .py file
            class_name: Name of plugin class (optional, auto-detected if not provided)

        Returns:
            Plugin class or None
        """
        if not file_path.exists():
            return None

        # Create module name
        module_name = f"dynamic_plugin_{file_path.stem}"

        # Load module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            return None

        # Find plugin class
        if class_name:
            return getattr(module, class_name, None)

        # Auto-detect plugin class
        from core.plugin_base import BasePlugin

        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and
                issubclass(obj, BasePlugin) and
                obj is not BasePlugin):
                return obj

        return None

    @staticmethod
    def reload_plugin_module(module_name: str) -> bool:
        """
        Reload a plugin module.

        Args:
            module_name: Module name

        Returns:
            True if successful
        """
        if module_name not in sys.modules:
            return False

        try:
            importlib.reload(sys.modules[module_name])
            return True
        except Exception:
            return False