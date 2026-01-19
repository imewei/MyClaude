#!/usr/bin/env python3
"""
Sphinx Documentation Generator for Plugins

Generates .rst files for each plugin based on metadata in plugin.json.
Used to automate documentation maintenance.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


class SphinxDocGenerator:
    """Generates Sphinx RST documentation for plugins"""

    def __init__(self, plugins_dir: Path, verbose: bool = False):
        self.plugins_dir = plugins_dir
        self.verbose = verbose
        self.plugins: Dict[str, Path] = {}
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.integrations: Dict[str, Set[str]] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}

    def log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(message)

    def discover_plugins(self) -> Dict[str, Path]:
        """Find all valid plugins in the directory"""
        self.plugins = {}
        if not self.plugins_dir.exists():
            return {}

        for item in self.plugins_dir.iterdir():
            if item.is_dir() and (item / "plugin.json").exists():
                self.plugins[item.name] = item
                # Cache metadata
                self.extract_plugin_metadata(item)

        return self.plugins

    def extract_plugin_metadata(self, plugin_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from plugin.json"""
        plugin_json_path = plugin_dir / "plugin.json"
        if not plugin_json_path.exists():
            return None

        try:
            with open(plugin_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Ensure name matches directory if not specified (or use directory as key)
            if 'name' not in data:
                data['name'] = plugin_dir.name

            self.metadata_cache[plugin_dir.name] = data
            return data
        except Exception as e:
            self.log(f"Error reading {plugin_json_path}: {e}")
            return None

    def detect_integrations(self) -> Dict[str, Set[str]]:
        """Detect cross-plugin dependencies from READMEs and metadata"""
        self.integrations = {name: set() for name in self.plugins}

        for name, path in self.plugins.items():
            # Check README for mentions of other plugins
            readme_path = path / "README.md"
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding='utf-8')
                    for other_plugin in self.plugins:
                        if other_plugin != name and other_plugin in content:
                            self.integrations[name].add(other_plugin)
                except Exception:
                    pass

            # Check metadata integration points (if they existed in schema)
            meta = self.metadata_cache.get(name, {})
            # Hypothetical field, but good for completeness
            for integ in meta.get('integration_points', []):
                # This logic assumes integration points might list other plugins
                pass

        return self.integrations

    def build_reverse_dependencies(self) -> Dict[str, Set[str]]:
        """Build reverse dependency map (who depends on me)"""
        self.reverse_deps = {name: set() for name in self.plugins}
        for source, targets in self.integrations.items():
            for target in targets:
                if target not in self.reverse_deps:
                    self.reverse_deps[target] = set()
                self.reverse_deps[target].add(source)
        return self.reverse_deps

    def identify_integration_patterns(self) -> Dict[str, List[str]]:
        """Identify common integration patterns based on categories and keywords"""
        patterns = {}

        # Pattern 1: Category-based grouping
        categories = {}
        for name, data in self.metadata_cache.items():
            cat = data.get('category', 'uncategorized')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)

        # Add sufficiently large categories as patterns
        for cat, plugins in categories.items():
            if len(plugins) >= 2:
                patterns[f"{cat.title()} Workflow"] = plugins

        # Pattern 2: Keyword clustering (simplified)
        keyword_map = {}
        for name, data in self.metadata_cache.items():
            for kw in data.get('keywords', []):
                if kw not in keyword_map:
                    keyword_map[kw] = []
                keyword_map[kw].append(name)

        for kw, plugins in keyword_map.items():
            if len(plugins) >= 2:
                patterns[f"{kw.title()} Integration"] = plugins

        return patterns

    def generate_integration_matrix(self) -> str:
        """Generate an integration matrix as an RST list/table"""
        lines = []
        lines.append("Integration Matrix")
        lines.append("==================")
        lines.append("")

        for source, targets in sorted(self.integrations.items()):
            if not targets:
                continue

            plugin_name = self.metadata_cache.get(source, {}).get('name', source)
            lines.append(f"**{plugin_name}** integrates with:")
            lines.append("")

            for target in sorted(targets):
                target_name = self.metadata_cache.get(target, {}).get('name', target)
                lines.append(f"* :doc:`{target}` ({target_name})")

            lines.append("")

        return "\n".join(lines)

    def generate_plugin_rst(self, plugin_dir: Path, output_dir: Path) -> str:
        """Generate RST content for a single plugin"""
        metadata = self.extract_plugin_metadata(plugin_dir)
        if not metadata:
            return ""

        name = metadata.get('name', plugin_dir.name)

        # Build RST Content
        lines = []

        # Title
        title = name.replace('-', ' ').title()
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

        # Module directive
        lines.append(f".. module:: {name}")
        lines.append("")

        # Description
        lines.append("Description")
        lines.append("-" * 11)
        lines.append("")
        lines.append(metadata.get('description', 'No description available.'))
        lines.append("")

        # Agents
        agents = metadata.get('agents', [])
        if agents:
            lines.append("Agents")
            lines.append("-" * 6)
            lines.append("")
            for agent in agents:
                lines.append(f"* **{agent.get('name', 'Unknown')}**: {agent.get('description', '')}")
            lines.append("")

        # Commands
        commands = metadata.get('commands', [])
        if commands:
            lines.append("Commands")
            lines.append("-" * 8)
            lines.append("")
            for cmd in commands:
                lines.append(f"* ``{cmd.get('name', '')}``: {cmd.get('description', '')}")
            lines.append("")

        # Skills
        skills = metadata.get('skills', [])
        if skills:
            lines.append("Skills")
            lines.append("-" * 6)
            lines.append("")
            for skill in skills:
                lines.append(f"* **{skill.get('name', '')}**: {skill.get('description', '')}")
            lines.append("")

        # Usage Examples (Placeholder extraction logic)
        lines.append("Usage Examples")
        lines.append("-" * 14)
        lines.append("")
        # Here we would normally extract from README, but for this basic implementation:
        lines.append("See the plugin README for usage examples.")
        lines.append("")

        # Integration
        lines.append("Integration")
        lines.append("-" * 11)
        lines.append("")

        # Add integration info if available
        plugin_key = plugin_dir.name
        deps = self.integrations.get(plugin_key, set())
        rev_deps = self.reverse_deps.get(plugin_key, set())

        if deps:
            lines.append("**Dependencies:**")
            for dep in sorted(deps):
                lines.append(f"* :doc:`{dep}`")
            lines.append("")

        if rev_deps:
            lines.append("**Used by:**")
            for rev in sorted(rev_deps):
                lines.append(f"* :doc:`{rev}`")
            lines.append("")

        if not deps and not rev_deps:
            lines.append("This plugin functions as a standalone module.")
            lines.append("")

        # See Also & References
        lines.append("See Also")
        lines.append("-" * 8)
        lines.append("")
        lines.append("* :doc:`/index`")
        lines.append("")

        lines.append("References")
        lines.append("-" * 10)
        lines.append("")
        lines.append(f"* `Source Code <https://github.com/imewei/MyClaude/tree/main/plugins/{plugin_dir.name}>`_")
        lines.append("")

        return "\n".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Sphinx RST documentation for plugins")
    parser.add_argument("--plugins-dir", type=Path, default=Path("plugins"), help="Path to plugins directory")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/plugins"), help="Path to output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    generator = SphinxDocGenerator(args.plugins_dir, verbose=args.verbose)
    generator.discover_plugins()
    generator.detect_integrations()
    generator.build_reverse_dependencies()

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    for name, path in generator.plugins.items():
        rst = generator.generate_plugin_rst(path, args.output_dir)
        if rst:
            out_file = args.output_dir / f"{name}.rst"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(rst)
            if args.verbose:
                print(f"Generated {out_file}")

if __name__ == "__main__":
    main()
