#!/usr/bin/env python3
"""
README Regenerator

Regenerates plugin README.md files from plugin.json metadata to ensure consistency.
Follows standardized template from spec.md lines 218-246.

Features:
- Parses plugin.json using metadata-validator patterns
- Generates standardized README.md with consistent sections
- Supports dry-run mode for preview
- Can regenerate all plugin READMEs at once
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class ReadmeRegenerator:
    """Regenerates README files from plugin.json"""

    def __init__(self, docs_base_url: str = "https://docs.example.com"):
        self.docs_base_url = docs_base_url

    def parse_plugin_json(self, plugin_path: Path) -> Optional[Dict[str, Any]]:
        """Parse plugin.json (using metadata-validator patterns)"""
        plugin_json_path = plugin_path / "plugin.json"

        if not plugin_json_path.exists():
            print(f"⚠️  Warning: No plugin.json found in {plugin_path}")
            return None

        try:
            with open(plugin_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {plugin_json_path}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error reading {plugin_json_path}: {e}")
            return None

    def generate_readme(self, plugin_path: Path) -> str:
        """Generate README.md content from plugin.json"""
        metadata = self.parse_plugin_json(plugin_path)
        if not metadata:
            return ""

        plugin_name = metadata.get("name", plugin_path.name)
        lines = []

        # Title
        title = self._format_title(plugin_name)
        lines.append(f"# {title}")
        lines.append("")

        # Description
        description = metadata.get("description", "")
        lines.append(description)
        lines.append("")

        # Metadata line
        version = metadata.get("version", "unknown")
        category = metadata.get("category", "uncategorized")
        license_type = metadata.get("license", "Unknown")

        lines.append(f"**Version:** {version} | **Category:** {category} | **License:** {license_type}")
        lines.append("")

        # Documentation link
        docs_url = f"{self.docs_base_url}/plugins/{plugin_name}.html"
        lines.append(f"[Full Documentation →]({docs_url})")
        lines.append("")

        # Agents section
        agents = metadata.get("agents", [])
        if agents:
            lines.append(f"## Agents ({len(agents)})")
            lines.append("")
            for agent in agents:
                agent_name = agent.get("name", "unknown")
                agent_desc = agent.get("description", "No description")
                agent_status = agent.get("status", "active")
                lines.append(f"### {agent_name}")
                lines.append("")
                lines.append(f"**Status:** {agent_status}")
                lines.append("")
                lines.append(agent_desc)
                lines.append("")

        # Commands section
        commands = metadata.get("commands", [])
        if commands:
            lines.append(f"## Commands ({len(commands)})")
            lines.append("")
            for command in commands:
                command_name = command.get("name", "unknown")
                command_desc = command.get("description", "No description")
                command_status = command.get("status", "active")
                lines.append(f"### `{command_name}`")
                lines.append("")
                lines.append(f"**Status:** {command_status}")
                lines.append("")
                lines.append(command_desc)
                lines.append("")

        # Skills section
        skills = metadata.get("skills", [])
        if skills:
            lines.append(f"## Skills ({len(skills)})")
            lines.append("")
            for skill in skills:
                skill_name = skill.get("name", "unknown")
                skill_desc = skill.get("description", "No description")
                lines.append(f"### {skill_name}")
                lines.append("")
                lines.append(skill_desc)
                lines.append("")

        # Quick Start section
        lines.append("## Quick Start")
        lines.append("")
        lines.append("To use this plugin:")
        lines.append("")
        lines.append("1. Ensure Claude Code is installed")
        lines.append(f"2. Enable the `{plugin_name}` plugin")

        if agents:
            first_agent = agents[0].get("name", "agent")
            lines.append(f"3. Activate an agent (e.g., `@{first_agent}`)")

        if commands:
            first_command = commands[0].get("name", "/command")
            lines.append(f"4. Try a command (e.g., `{first_command}`)")

        lines.append("")

        # Integration section
        lines.append("## Integration")
        lines.append("")

        integration_points = metadata.get("integration_points", [])
        if integration_points:
            lines.append("This plugin integrates with:")
            lines.append("")
            for point in integration_points:
                lines.append(f"- {point}")
            lines.append("")
        else:
            lines.append("See the full documentation for integration patterns and compatible plugins.")
            lines.append("")

        # Documentation section
        lines.append("## Documentation")
        lines.append("")
        lines.append(f"For comprehensive documentation, see: [Plugin Documentation]({docs_url})")
        lines.append("")
        lines.append("To build documentation locally:")
        lines.append("")
        lines.append("```bash")
        lines.append("cd docs/")
        lines.append("make html")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def _format_title(self, plugin_name: str) -> str:
        """Format plugin name as title"""
        return plugin_name.replace("-", " ").title()

    def regenerate_readme(self, plugin_path: Path, dry_run: bool = False) -> bool:
        """Regenerate README.md for a plugin"""
        readme_content = self.generate_readme(plugin_path)

        if not readme_content:
            return False

        readme_path = plugin_path / "README.md"

        if dry_run:
            print(f"\n{'=' * 60}")
            print(f"Preview: {plugin_path.name}/README.md")
            print('=' * 60)
            print(readme_content)
            print('=' * 60)
            return True

        # Backup existing README if it exists
        if readme_path.exists():
            backup_path = plugin_path / "README.md.backup"
            readme_path.rename(backup_path)
            print(f"📦 Backed up existing README to {backup_path.name}")

        # Write new README
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✅ Generated README for {plugin_path.name}")
        return True

    def regenerate_all(self, plugins_dir: Path, dry_run: bool = False) -> int:
        """Regenerate README.md for all plugins"""
        if not plugins_dir.exists():
            print(f"❌ Error: Plugins directory not found: {plugins_dir}")
            return 0

        plugin_dirs = [d for d in plugins_dir.iterdir() if d.is_dir() and (d / "plugin.json").exists()]

        if not plugin_dirs:
            print("⚠️  No plugins found")
            return 0

        print(f"🔄 Regenerating README.md for {len(plugin_dirs)} plugins...")
        if dry_run:
            print("(DRY RUN - no files will be modified)")

        success_count = 0
        for plugin_dir in sorted(plugin_dirs):
            try:
                if self.regenerate_readme(plugin_dir, dry_run):
                    success_count += 1
            except Exception as e:
                print(f"❌ Error regenerating {plugin_dir.name}: {e}")

        print(f"\n✅ Successfully regenerated {success_count}/{len(plugin_dirs)} README files")
        return success_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Regenerate plugin README.md files from plugin.json metadata"
    )
    parser.add_argument(
        "plugin_path",
        type=Path,
        nargs="?",
        help="Path to specific plugin directory"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate all plugin READMEs"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path.cwd() / "plugins",
        help="Path to plugins directory (used with --all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview output without writing files"
    )
    parser.add_argument(
        "--docs-url",
        type=str,
        default="https://docs.example.com",
        help="Base URL for documentation links"
    )

    args = parser.parse_args()

    regenerator = ReadmeRegenerator(docs_base_url=args.docs_url)

    if args.all:
        # Regenerate all plugins
        success_count = regenerator.regenerate_all(args.plugins_dir, args.dry_run)
        sys.exit(0 if success_count > 0 else 1)
    elif args.plugin_path:
        # Regenerate specific plugin
        if not args.plugin_path.exists():
            print(f"❌ Error: Plugin directory not found: {args.plugin_path}")
            sys.exit(1)

        success = regenerator.regenerate_readme(args.plugin_path, args.dry_run)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
