#!/usr/bin/env python3
"""
Enable all plugins from the scientific-computing-workflows marketplace in Claude Code.

Usage:
    python3 tools/maintenance/enable_all_plugins.py
"""

import json
from pathlib import Path

def enable_all_plugins():
    """Enable all plugins from the marketplace in Claude Code settings."""

    # Paths
    settings_path = Path.home() / '.claude' / 'settings.json'
    marketplace_path = Path(__file__).parent.parent / '.claude-plugin' / 'marketplace.json'

    # Read current settings
    with open(settings_path, 'r') as f:
        settings = json.load(f)

    # Read marketplace.json to get all plugin names
    with open(marketplace_path, 'r') as f:
        marketplace = json.load(f)

    # Get all plugin names from the marketplace
    plugins = marketplace.get('plugins', [])
    marketplace_name = marketplace.get('name', 'scientific-computing-workflows')

    # Add all plugins to enabledPlugins
    enabled_plugins = settings.get('enabledPlugins', {})

    print(f"Enabling {len(plugins)} plugins from {marketplace_name} marketplace...\n")

    newly_enabled = 0
    already_enabled = 0

    for plugin in plugins:
        plugin_name = plugin['name']
        plugin_key = f"{plugin_name}@{marketplace_name}"

        if plugin_key in enabled_plugins and enabled_plugins[plugin_key]:
            print(f"  ✓ {plugin_name} (already enabled)")
            already_enabled += 1
        else:
            enabled_plugins[plugin_key] = True
            print(f"  + {plugin_name} (newly enabled)")
            newly_enabled += 1

    # Update settings
    settings['enabledPlugins'] = enabled_plugins

    # Write back
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)

    print(f"\n{'='*60}")
    print("✅ Configuration updated!")
    print(f"   Newly enabled: {newly_enabled}")
    print(f"   Already enabled: {already_enabled}")
    print(f"   Total plugins in marketplace: {len(plugins)}")
    print(f"   Total enabled plugins: {len(enabled_plugins)}")
    print("\n🔄 Please restart Claude Code for changes to take effect.")
    print(f"{'='*60}")

if __name__ == '__main__':
    enable_all_plugins()
