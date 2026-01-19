#!/usr/bin/env python3
"""
Plugin Dependency Mapper

Analyzes cross-plugin relationships by:
- Parsing all plugin.json files
- Extracting agent, command, and skill references
- Building dependency graph
- Identifying integration patterns
- Generating visual and textual dependency maps

Author: Systems Architect / Technical Writer
Part of: Plugin Review and Optimization - Task Group 0.4
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, field
from collections import defaultdict
import sys


@dataclass
class PluginReference:
    """Represents a reference to another plugin"""
    source_plugin: str
    target_plugin: str
    reference_type: str  # 'agent', 'command', 'skill', 'keyword', 'documentation'
    reference_name: str
    context: str
    file_path: str
    line_number: int = 0


@dataclass
class PluginMetadata:
    """Plugin metadata extracted from plugin.json"""
    name: str
    version: str
    description: str
    category: str
    agents: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class DependencyGraph:
    """Complete dependency graph for all plugins"""
    plugins: Dict[str, PluginMetadata] = field(default_factory=dict)
    references: List[PluginReference] = field(default_factory=list)
    dependencies: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_dependencies: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))


class DependencyMapper:
    """Analyzes and maps dependencies between plugins"""

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir
        self.graph = DependencyGraph()
        self.plugin_names = set()

    def analyze_all_plugins(self) -> DependencyGraph:
        """Analyze all plugins and build dependency graph"""
        print("🔍 Analyzing plugin dependencies...")

        # Step 1: Load all plugin metadata
        self._load_plugin_metadata()

        # Step 2: Extract cross-references from documentation
        self._extract_documentation_references()

        # Step 3: Build dependency graph
        self._build_dependency_graph()

        # Step 4: Identify integration patterns
        self._identify_integration_patterns()

        return self.graph

    def _load_plugin_metadata(self):
        """Load metadata from all plugin.json files"""
        plugin_dirs = [d for d in self.plugins_dir.iterdir() if d.is_dir()]

        for plugin_dir in sorted(plugin_dirs):
            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                continue

            try:
                with open(plugin_json) as f:
                    data = json.load(f)

                plugin_name = data.get("name", plugin_dir.name)
                self.plugin_names.add(plugin_name)

                metadata = PluginMetadata(
                    name=plugin_name,
                    version=data.get("version", "unknown"),
                    description=data.get("description", ""),
                    category=data.get("category", "uncategorized"),
                    agents=[a.get("name") for a in data.get("agents", [])],
                    commands=[c.get("name") for c in data.get("commands", [])],
                    skills=[s.get("name") for s in data.get("skills", [])],
                    keywords=data.get("keywords", [])
                )

                self.graph.plugins[plugin_name] = metadata

            except Exception as e:
                print(f"⚠️  Warning: Error loading {plugin_json}: {e}")

    def _extract_documentation_references(self):
        """Extract cross-plugin references from README and markdown files"""
        for plugin_name, metadata in self.graph.plugins.items():
            plugin_dir = self.plugins_dir / plugin_name

            # Check README
            readme_path = plugin_dir / "README.md"
            if readme_path.exists():
                self._scan_file_for_references(readme_path, plugin_name)

            # Check agent documentation
            agents_dir = plugin_dir / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.md"):
                    self._scan_file_for_references(agent_file, plugin_name)

            # Check command documentation
            commands_dir = plugin_dir / "commands"
            if commands_dir.exists():
                for command_file in commands_dir.glob("*.md"):
                    self._scan_file_for_references(command_file, plugin_name)

            # Check skill documentation
            skills_dir = plugin_dir / "skills"
            if skills_dir.exists():
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir():
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.exists():
                            self._scan_file_for_references(skill_file, plugin_name)

    def _scan_file_for_references(self, file_path: Path, source_plugin: str):
        """Scan a file for references to other plugins"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Look for plugin name mentions
                for target_plugin in self.plugin_names:
                    if target_plugin == source_plugin:
                        continue

                    # Check for various reference patterns
                    patterns = [
                        # Plugin name as link or mention
                        rf'\b{re.escape(target_plugin)}\b',
                        # With "plugin" suffix
                        rf'\b{re.escape(target_plugin)}\s+plugin\b',
                        # In markdown link
                        rf'\[.*?{re.escape(target_plugin)}.*?\]',
                    ]

                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Determine reference type
                            ref_type = self._determine_reference_type(line, target_plugin)

                            # Extract context (surrounding text)
                            context = line.strip()
                            if len(context) > 100:
                                context = context[:97] + "..."

                            ref = PluginReference(
                                source_plugin=source_plugin,
                                target_plugin=target_plugin,
                                reference_type=ref_type,
                                reference_name=target_plugin,
                                context=context,
                                file_path=str(file_path.relative_to(self.plugins_dir)),
                                line_number=line_num
                            )

                            self.graph.references.append(ref)
                            break  # Avoid duplicate references from same line

        except Exception as e:
            print(f"⚠️  Warning: Error scanning {file_path}: {e}")

    def _determine_reference_type(self, line: str, target_plugin: str) -> str:
        """Determine the type of reference based on context"""
        line_lower = line.lower()

        # Check for specific contexts
        if any(word in line_lower for word in ['integrate', 'integration', 'combine', 'work with']):
            return 'integration'
        elif any(word in line_lower for word in ['agent', 'expert', 'specialist']):
            return 'agent'
        elif any(word in line_lower for word in ['command', 'slash-command', '/']):
            return 'command'
        elif any(word in line_lower for word in ['skill', 'pattern', 'technique']):
            return 'skill'
        elif any(word in line_lower for word in ['see also', 'related', 'similar']):
            return 'related'
        elif any(word in line_lower for word in ['workflow', 'pipeline', 'process']):
            return 'workflow'
        else:
            return 'documentation'

    def _build_dependency_graph(self):
        """Build forward and reverse dependency maps"""
        for ref in self.graph.references:
            self.graph.dependencies[ref.source_plugin].add(ref.target_plugin)
            self.graph.reverse_dependencies[ref.target_plugin].add(ref.source_plugin)

    def _identify_integration_patterns(self):
        """Identify common integration patterns"""
        # This is done in the reporting phase
        pass

    def generate_report(self, output_path: Path = None) -> str:
        """Generate comprehensive dependency report"""
        lines = []

        # Header
        lines.append("# Plugin Dependency Map")
        lines.append("")
        lines.append("Cross-plugin dependency analysis for Claude Code marketplace")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Plugins:** {len(self.graph.plugins)}")
        lines.append(f"- **Total References:** {len(self.graph.references)}")
        lines.append(f"- **Plugins with Dependencies:** {len([p for p, deps in self.graph.dependencies.items() if deps])}")
        lines.append(f"- **Plugins Referenced:** {len([p for p, refs in self.graph.reverse_dependencies.items() if refs])}")
        lines.append("")

        # Category breakdown
        lines.append("## Plugins by Category")
        lines.append("")
        category_counts = defaultdict(list)
        for plugin_name, metadata in sorted(self.graph.plugins.items()):
            category_counts[metadata.category].append(plugin_name)

        for category in sorted(category_counts.keys()):
            plugins = category_counts[category]
            lines.append(f"### {category.replace('-', ' ').title()} ({len(plugins)} plugins)")
            lines.append("")
            for plugin in sorted(plugins):
                lines.append(f"- `{plugin}`")
            lines.append("")

        # Dependency matrix
        lines.append("## Dependency Matrix")
        lines.append("")
        lines.append("Plugins listed by number of dependencies (references to other plugins):")
        lines.append("")

        plugin_dep_counts = [
            (plugin, len(deps))
            for plugin, deps in self.graph.dependencies.items()
        ]
        plugin_dep_counts.sort(key=lambda x: (-x[1], x[0]))

        for plugin, count in plugin_dep_counts:
            if count > 0:
                deps = sorted(self.graph.dependencies[plugin])
                lines.append(f"**{plugin}** ({count} dependencies):")
                for dep in deps:
                    ref_types = set()
                    for ref in self.graph.references:
                        if ref.source_plugin == plugin and ref.target_plugin == dep:
                            ref_types.add(ref.reference_type)
                    types_str = ", ".join(sorted(ref_types))
                    lines.append(f"  - `{dep}` ({types_str})")
                lines.append("")

        # Reverse dependencies (most referenced plugins)
        lines.append("## Most Referenced Plugins")
        lines.append("")
        lines.append("Plugins referenced by other plugins (sorted by reference count):")
        lines.append("")

        plugin_ref_counts = [
            (plugin, len(refs))
            for plugin, refs in self.graph.reverse_dependencies.items()
        ]
        plugin_ref_counts.sort(key=lambda x: (-x[1], x[0]))

        for plugin, count in plugin_ref_counts[:20]:  # Top 20
            if count > 0:
                refs = sorted(self.graph.reverse_dependencies[plugin])
                lines.append(f"**{plugin}** (referenced {count} times):")
                lines.append(f"  Referenced by: {', '.join(f'`{r}`' for r in refs)}")
                lines.append("")

        # Integration patterns
        lines.append("## Integration Patterns")
        lines.append("")

        # Identify common plugin combinations
        integration_patterns = self._find_integration_patterns()

        if integration_patterns:
            lines.append("Common plugin combinations found in documentation:")
            lines.append("")
            for pattern_name, plugins in integration_patterns.items():
                lines.append(f"### {pattern_name}")
                lines.append("")
                for plugin in plugins:
                    lines.append(f"- `{plugin}`")
                lines.append("")
        else:
            lines.append("No strong integration patterns detected.")
            lines.append("")

        # Reference type breakdown
        lines.append("## Reference Type Distribution")
        lines.append("")

        ref_type_counts = defaultdict(int)
        for ref in self.graph.references:
            ref_type_counts[ref.reference_type] += 1

        for ref_type in sorted(ref_type_counts.keys()):
            count = ref_type_counts[ref_type]
            percentage = (count / len(self.graph.references) * 100) if self.graph.references else 0
            lines.append(f"- **{ref_type}**: {count} ({percentage:.1f}%)")
        lines.append("")

        # Detailed reference listing
        lines.append("## Detailed Reference Listing")
        lines.append("")

        for plugin in sorted(self.graph.plugins.keys()):
            plugin_refs = [r for r in self.graph.references if r.source_plugin == plugin]
            if plugin_refs:
                lines.append(f"### {plugin}")
                lines.append("")

                # Group by target plugin
                by_target = defaultdict(list)
                for ref in plugin_refs:
                    by_target[ref.target_plugin].append(ref)

                for target in sorted(by_target.keys()):
                    refs = by_target[target]
                    lines.append(f"**→ {target}** ({len(refs)} references):")
                    lines.append("")
                    for ref in refs[:5]:  # Limit to first 5
                        lines.append(f"- Type: `{ref.reference_type}`")
                        lines.append(f"  - File: `{ref.file_path}` (line {ref.line_number})")
                        lines.append(f"  - Context: {ref.context}")
                        lines.append("")
                    if len(refs) > 5:
                        lines.append(f"  _(... and {len(refs) - 5} more references)_")
                        lines.append("")
                lines.append("")

        # Isolated plugins
        lines.append("## Isolated Plugins")
        lines.append("")
        lines.append("Plugins with no references to or from other plugins:")
        lines.append("")

        isolated = []
        for plugin in sorted(self.graph.plugins.keys()):
            has_deps = len(self.graph.dependencies.get(plugin, set())) > 0
            has_refs = len(self.graph.reverse_dependencies.get(plugin, set())) > 0
            if not has_deps and not has_refs:
                isolated.append(plugin)

        if isolated:
            for plugin in isolated:
                lines.append(f"- `{plugin}`")
        else:
            lines.append("No isolated plugins found - all plugins are connected!")
        lines.append("")

        # Graph visualization (text-based)
        lines.append("## Dependency Graph Visualization")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph TD")

        # Limit to plugins with dependencies to keep graph readable
        for plugin in sorted(self.graph.plugins.keys()):
            deps = self.graph.dependencies.get(plugin, set())
            if deps:
                plugin_safe = plugin.replace("-", "_")
                for dep in sorted(deps):
                    dep_safe = dep.replace("-", "_")
                    lines.append(f"    {plugin_safe}[\"{plugin}\"] --> {dep_safe}[\"{dep}\"]")

        lines.append("```")
        lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        recommendations = self._generate_recommendations()
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        report = "\n".join(lines)

        # Write to file if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')
            print(f"✅ Dependency map saved to: {output_path}")

        return report

    def _find_integration_patterns(self) -> Dict[str, List[str]]:
        """Identify common plugin combinations"""
        patterns = {}

        # Scientific computing workflows
        sci_plugins = [p for p, m in self.graph.plugins.items()
                      if m.category == 'scientific-computing']
        if len(sci_plugins) >= 3:
            patterns["Scientific Computing Workflow"] = sci_plugins[:5]

        # Development workflows
        dev_plugins = [p for p, m in self.graph.plugins.items()
                      if m.category == 'development']
        if len(dev_plugins) >= 3:
            patterns["Development Workflow"] = dev_plugins[:5]

        # DevOps workflows
        devops_plugins = [p for p, m in self.graph.plugins.items()
                         if m.category in ['devops', 'infrastructure', 'quality']]
        if len(devops_plugins) >= 3:
            patterns["DevOps & Quality Workflow"] = devops_plugins[:5]

        return patterns

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on dependency analysis"""
        recommendations = []

        # Check for isolated plugins
        isolated = [
            p for p in self.graph.plugins.keys()
            if not self.graph.dependencies.get(p) and
               not self.graph.reverse_dependencies.get(p)
        ]
        if isolated:
            recommendations.append(
                f"**Add integration documentation** for {len(isolated)} isolated plugins: "
                f"{', '.join(f'`{p}`' for p in isolated[:5])}"
            )

        # Check for plugins with many dependencies
        high_dep_plugins = [
            p for p, deps in self.graph.dependencies.items()
            if len(deps) >= 5
        ]
        if high_dep_plugins:
            recommendations.append(
                f"**Review integration complexity** for plugins with many dependencies: "
                f"{', '.join(f'`{p}`' for p in high_dep_plugins)}"
            )

        # Check for popular plugins
        popular = [
            p for p, refs in self.graph.reverse_dependencies.items()
            if len(refs) >= 5
        ]
        if popular:
            recommendations.append(
                f"**Create integration guides** for highly-referenced plugins: "
                f"{', '.join(f'`{p}`' for p in popular)}"
            )

        # Suggest workflow documentation
        recommendations.append(
            "**Document common workflows** that combine multiple related plugins "
            "(e.g., Julia + SciML + HPC, Python + Testing + CI/CD)"
        )

        # Suggest terminology standardization
        recommendations.append(
            "**Standardize terminology** across related plugins to improve "
            "cross-referencing and discoverability"
        )

        return recommendations

    def export_json(self, output_path: Path):
        """Export dependency graph as JSON"""
        data = {
            "plugins": {
                name: {
                    "version": meta.version,
                    "description": meta.description,
                    "category": meta.category,
                    "agents": meta.agents,
                    "commands": meta.commands,
                    "skills": meta.skills,
                    "keywords": meta.keywords
                }
                for name, meta in self.graph.plugins.items()
            },
            "references": [
                {
                    "source": ref.source_plugin,
                    "target": ref.target_plugin,
                    "type": ref.reference_type,
                    "name": ref.reference_name,
                    "context": ref.context,
                    "file": ref.file_path,
                    "line": ref.line_number
                }
                for ref in self.graph.references
            ],
            "dependencies": {
                plugin: list(deps)
                for plugin, deps in self.graph.dependencies.items()
            },
            "reverse_dependencies": {
                plugin: list(refs)
                for plugin, refs in self.graph.reverse_dependencies.items()
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Dependency graph exported to: {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze cross-plugin dependencies in Claude Code marketplace"
    )
    parser.add_argument(
        "--plugins-dir",
        type=Path,
        default=Path.cwd() / "plugins",
        help="Path to plugins directory (default: ./plugins)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dependency-map.md"),
        help="Output file for dependency report (default: reports/dependency-map.md)"
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export dependency graph as JSON"
    )

    args = parser.parse_args()

    # Validate plugins directory
    if not args.plugins_dir.exists():
        print(f"❌ Error: Plugins directory not found: {args.plugins_dir}")
        sys.exit(1)

    # Create mapper and analyze
    mapper = DependencyMapper(args.plugins_dir)
    graph = mapper.analyze_all_plugins()

    # Generate report
    print("\n📊 Generating dependency report...")
    _ = mapper.generate_report(args.output)

    # Export JSON if requested
    if args.export_json:
        mapper.export_json(args.export_json)

    # Print summary
    print("\n✅ Analysis complete!")
    print(f"   Plugins analyzed: {len(graph.plugins)}")
    print(f"   References found: {len(graph.references)}")
    print(f"   Report saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
