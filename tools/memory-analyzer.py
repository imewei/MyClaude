#!/usr/bin/env python3
"""
Memory Usage Analyzer

Measures plugin memory consumption to identify memory leaks and inefficiencies.

Features:
- Measures baseline memory consumption
- Tracks memory during typical operations
- Identifies memory leaks
- Profiles data structure efficiency

Usage:
    python3 tools/memory-analyzer.py <plugin-name>
    python3 tools/memory-analyzer.py <plugin-name> /path/to/plugins
    python3 tools/memory-analyzer.py --all  # Profile all plugins
"""

import json
import sys
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryMetric:
    """Memory metric for a specific operation."""
    name: str
    memory_kb: float
    status: str  # 'pass', 'warn', 'fail'
    details: str = ""


@dataclass
class MemoryProfile:
    """Complete memory profile for a plugin."""
    plugin_name: str
    plugin_path: Path
    baseline_memory_kb: float
    peak_memory_kb: float
    average_memory_kb: float
    metrics: list[MemoryMetric] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def memory_overhead_kb(self) -> float:
        """Memory overhead above baseline."""
        return self.peak_memory_kb - self.baseline_memory_kb

    @property
    def status(self) -> str:
        """Overall status based on memory usage."""
        if self.errors:
            return 'error'
        elif self.peak_memory_kb > 10000:  # 10 MB
            return 'fail'
        elif self.peak_memory_kb > 5000:  # 5 MB
            return 'warn'
        else:
            return 'pass'

    @property
    def status_emoji(self) -> str:
        """Visual status indicator."""
        return {
            'pass': '✅',
            'warn': '⚠️',
            'fail': '❌',
            'error': '🔴'
        }.get(self.status, '❓')


def get_object_size(obj: Any) -> int:
    """
    Recursively calculate the size of an object in bytes.

    Note: This is an approximation and may not be 100% accurate for all objects.
    """
    import sys

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_object_size(k) + get_object_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_object_size(item) for item in obj)
    elif isinstance(obj, str):
        size += len(obj)

    return size


class PluginMemoryAnalyzer:
    """Analyzes plugin memory consumption."""

    def __init__(self, plugins_root: Path) -> None:
        self.plugins_root = plugins_root

    def profile_plugin(self, plugin_name: str) -> MemoryProfile:
        """Profile a single plugin's memory usage."""
        plugin_path = self.plugins_root / plugin_name

        # Force garbage collection before measurement
        gc.collect()

        # Measure baseline (empty state)
        baseline_kb = 0  # Simplified baseline

        profile = MemoryProfile(
            plugin_name=plugin_name,
            plugin_path=plugin_path,
            baseline_memory_kb=baseline_kb,
            peak_memory_kb=0,
            average_memory_kb=0
        )

        # Check if plugin exists
        if not plugin_path.exists():
            profile.errors.append(f"Plugin directory not found: {plugin_path}")
            return profile

        # Load and analyze plugin components
        memory_measurements = []

        # 1. Measure plugin.json memory
        json_memory = self._measure_json_memory(profile)
        memory_measurements.append(json_memory)

        # 2. Measure agents directory memory
        agents_memory = self._measure_agents_memory(profile)
        memory_measurements.append(agents_memory)

        # 3. Measure commands directory memory
        commands_memory = self._measure_commands_memory(profile)
        memory_measurements.append(commands_memory)

        # 4. Measure skills directory memory
        skills_memory = self._measure_skills_memory(profile)
        memory_measurements.append(skills_memory)

        # 5. Measure README memory
        readme_memory = self._measure_readme_memory(profile)
        memory_measurements.append(readme_memory)

        # Calculate statistics
        profile.peak_memory_kb = max(memory_measurements)
        profile.average_memory_kb = sum(memory_measurements) / len(memory_measurements) if memory_measurements else 0

        # Add overall assessment
        if profile.peak_memory_kb > 10000:
            profile.warnings.append(
                f"Peak memory usage ({profile.peak_memory_kb:.2f}KB) is high"
            )
        elif profile.peak_memory_kb > 5000:
            profile.warnings.append(
                f"Peak memory usage ({profile.peak_memory_kb:.2f}KB) is moderate"
            )

        return profile

    def _measure_json_memory(self, profile: MemoryProfile) -> float:
        """Measure plugin.json memory usage."""
        json_path = profile.plugin_path / "plugin.json"

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Calculate memory usage
            memory_bytes = get_object_size(data)
            memory_kb = memory_bytes / 1024

            status = 'pass' if memory_kb < 10 else ('warn' if memory_kb < 50 else 'fail')
            profile.metrics.append(MemoryMetric(
                name="plugin.json",
                memory_kb=memory_kb,
                status=status,
                details=f"{len(json.dumps(data))} characters"
            ))

            return memory_kb

        except FileNotFoundError:
            profile.errors.append(f"plugin.json not found at {json_path}")
            profile.metrics.append(MemoryMetric(
                name="plugin.json",
                memory_kb=0,
                status='error',
                details="File not found"
            ))
            return 0
        except json.JSONDecodeError as e:
            profile.errors.append(f"Invalid JSON in plugin.json: {e}")
            profile.metrics.append(MemoryMetric(
                name="plugin.json",
                memory_kb=0,
                status='error',
                details=f"JSON decode error"
            ))
            return 0

    def _measure_agents_memory(self, profile: MemoryProfile) -> float:
        """Measure agents directory memory usage."""
        agents_dir = profile.plugin_path / "agents"

        total_memory_bytes = 0
        file_count = 0

        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                try:
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    total_memory_bytes += get_object_size(content)
                    file_count += 1
                except Exception as e:
                    profile.warnings.append(f"Error reading {agent_file.name}: {e}")

        memory_kb = total_memory_bytes / 1024

        status = 'pass' if memory_kb < 100 else ('warn' if memory_kb < 500 else 'fail')
        profile.metrics.append(MemoryMetric(
            name="agents directory",
            memory_kb=memory_kb,
            status=status,
            details=f"{file_count} agent files"
        ))

        return memory_kb

    def _measure_commands_memory(self, profile: MemoryProfile) -> float:
        """Measure commands directory memory usage."""
        commands_dir = profile.plugin_path / "commands"

        total_memory_bytes = 0
        file_count = 0

        if commands_dir.exists():
            for command_file in commands_dir.glob("*.md"):
                try:
                    with open(command_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    total_memory_bytes += get_object_size(content)
                    file_count += 1
                except Exception as e:
                    profile.warnings.append(f"Error reading {command_file.name}: {e}")

        memory_kb = total_memory_bytes / 1024

        status = 'pass' if memory_kb < 50 else ('warn' if memory_kb < 200 else 'fail')
        profile.metrics.append(MemoryMetric(
            name="commands directory",
            memory_kb=memory_kb,
            status=status,
            details=f"{file_count} command files"
        ))

        return memory_kb

    def _measure_skills_memory(self, profile: MemoryProfile) -> float:
        """Measure skills directory memory usage."""
        skills_dir = profile.plugin_path / "skills"

        total_memory_bytes = 0
        file_count = 0

        if skills_dir.exists():
            for skill_file in skills_dir.glob("*.md"):
                try:
                    with open(skill_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    total_memory_bytes += get_object_size(content)
                    file_count += 1
                except Exception as e:
                    profile.warnings.append(f"Error reading {skill_file.name}: {e}")

        memory_kb = total_memory_bytes / 1024

        status = 'pass' if memory_kb < 100 else ('warn' if memory_kb < 500 else 'fail')
        profile.metrics.append(MemoryMetric(
            name="skills directory",
            memory_kb=memory_kb,
            status=status,
            details=f"{file_count} skill files"
        ))

        return memory_kb

    def _measure_readme_memory(self, profile: MemoryProfile) -> float:
        """Measure README.md memory usage."""
        readme_path = profile.plugin_path / "README.md"

        try:
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                memory_bytes = get_object_size(content)
                memory_kb = memory_bytes / 1024

                status = 'pass' if memory_kb < 50 else ('warn' if memory_kb < 200 else 'fail')
                profile.metrics.append(MemoryMetric(
                    name="README.md",
                    memory_kb=memory_kb,
                    status=status,
                    details=f"{len(content)} characters"
                ))

                return memory_kb
            else:
                profile.metrics.append(MemoryMetric(
                    name="README.md",
                    memory_kb=0,
                    status='warn',
                    details="File not found"
                ))
                return 0
        except Exception as e:
            profile.warnings.append(f"Error reading README.md: {e}")
            return 0

    def profile_all_plugins(self) -> list[MemoryProfile]:
        """Profile all plugins in the marketplace."""
        profiles = []

        if not self.plugins_root.exists():
            return profiles

        for plugin_dir in sorted(self.plugins_root.iterdir()):
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                profile = self.profile_plugin(plugin_dir.name)
                profiles.append(profile)

        return profiles


class MemoryProfileReporter:
    """Generates formatted reports for memory profiling results."""

    def generate_report(self, profile: MemoryProfile) -> str:
        """Generate markdown report for a single plugin."""
        lines = []

        # Header
        lines.append(f"# Memory Profile: {profile.plugin_name}")
        lines.append("")
        lines.append(f"**Plugin Path:** `{profile.plugin_path}`")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Peak Memory Usage:** {profile.peak_memory_kb:.2f}KB ({profile.peak_memory_kb/1024:.2f}MB)")
        lines.append(f"- **Average Memory Usage:** {profile.average_memory_kb:.2f}KB ({profile.average_memory_kb/1024:.2f}MB)")
        lines.append(f"- **Baseline Memory:** {profile.baseline_memory_kb:.2f}KB")
        lines.append(f"- **Memory Overhead:** {profile.memory_overhead_kb:.2f}KB")
        lines.append(f"- **Status:** {profile.status_emoji} {profile.status.upper()}")
        lines.append(f"- **Metrics Collected:** {len(profile.metrics)}")
        lines.append(f"- **Errors:** {len(profile.errors)}")
        lines.append(f"- **Warnings:** {len(profile.warnings)}")
        lines.append("")

        # Detailed Metrics
        if profile.metrics:
            lines.append("## Memory Usage Breakdown")
            lines.append("")
            lines.append("| Component | Memory (KB) | Memory (MB) | Status | Details |")
            lines.append("|-----------|-------------|-------------|--------|---------|")

            for metric in profile.metrics:
                status_icon = {
                    'pass': '✅',
                    'warn': '⚠️',
                    'fail': '❌',
                    'error': '🔴'
                }.get(metric.status, '❓')

                lines.append(
                    f"| {metric.name} | {metric.memory_kb:.2f} | {metric.memory_kb/1024:.2f} | "
                    f"{status_icon} | {metric.details} |"
                )

            lines.append("")

            # Memory distribution chart
            lines.append("### Memory Distribution")
            lines.append("")
            total = sum(m.memory_kb for m in profile.metrics)
            for metric in sorted(profile.metrics, key=lambda m: m.memory_kb, reverse=True):
                percentage = (metric.memory_kb / total * 100) if total > 0 else 0
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = '█' * bar_length
                lines.append(f"- {metric.name}: {metric.memory_kb:.2f}KB ({percentage:.1f}%) {bar}")
            lines.append("")

        # Errors
        if profile.errors:
            lines.append("## Errors")
            lines.append("")
            for error in profile.errors:
                lines.append(f"🔴 **ERROR**: {error}")
            lines.append("")

        # Warnings
        if profile.warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in profile.warnings:
                lines.append(f"⚠️ **WARNING**: {warning}")
            lines.append("")

        # Assessment
        lines.append("## Overall Assessment")
        lines.append("")

        if profile.status == 'pass':
            lines.append(f"**Status:** {profile.status_emoji} EXCELLENT - Memory usage is efficient")
            lines.append("")
            lines.append(f"Plugin uses {profile.peak_memory_kb:.2f}KB peak memory, which is well within acceptable limits.")
        elif profile.status == 'warn':
            lines.append(f"**Status:** {profile.status_emoji} WARNING - Memory usage is moderate")
            lines.append("")
            lines.append(f"Plugin uses {profile.peak_memory_kb:.2f}KB peak memory.")
            lines.append("Consider optimizing large components if memory becomes a constraint.")
        elif profile.status == 'fail':
            lines.append(f"**Status:** {profile.status_emoji} OPTIMIZATION NEEDED - Memory usage is high")
            lines.append("")
            lines.append(f"Plugin uses {profile.peak_memory_kb:.2f}KB peak memory.")
            lines.append("")
            lines.append("**Recommended Actions:**")
            # Find largest components
            large_metrics = sorted(profile.metrics, key=lambda m: m.memory_kb, reverse=True)[:3]
            for metric in large_metrics:
                lines.append(f"- Optimize {metric.name} (currently {metric.memory_kb:.2f}KB)")
        else:
            lines.append(f"**Status:** {profile.status_emoji} ERROR - Unable to complete profiling")

        lines.append("")

        return '\n'.join(lines)

    def generate_summary_report(self, profiles: list[MemoryProfile]) -> str:
        """Generate summary report for multiple plugins."""
        lines = []

        # Header
        lines.append("# Memory Profile Summary")
        lines.append("")
        lines.append(f"**Plugins Profiled:** {len(profiles)}")
        lines.append("")

        # Overall Statistics
        total_plugins = len(profiles)
        pass_count = sum(1 for p in profiles if p.status == 'pass')
        warn_count = sum(1 for p in profiles if p.status == 'warn')
        fail_count = sum(1 for p in profiles if p.status == 'fail')
        error_count = sum(1 for p in profiles if p.status == 'error')

        avg_memory = sum(p.peak_memory_kb for p in profiles) / total_plugins if total_plugins > 0 else 0
        total_memory = sum(p.peak_memory_kb for p in profiles)

        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"- **Average Memory per Plugin:** {avg_memory:.2f}KB ({avg_memory/1024:.2f}MB)")
        lines.append(f"- **Total Memory (all plugins):** {total_memory:.2f}KB ({total_memory/1024:.2f}MB)")
        lines.append(f"- **Efficient Plugins:** {pass_count}/{total_plugins} ({pass_count/total_plugins*100:.1f}%)")
        lines.append("")
        lines.append("**Status Distribution:**")
        lines.append(f"- ✅ Pass: {pass_count} ({pass_count/total_plugins*100:.1f}%)")
        lines.append(f"- ⚠️ Warning: {warn_count} ({warn_count/total_plugins*100:.1f}%)")
        lines.append(f"- ❌ Fail: {fail_count} ({fail_count/total_plugins*100:.1f}%)")
        lines.append(f"- 🔴 Error: {error_count} ({error_count/total_plugins*100:.1f}%)")
        lines.append("")

        # Per-Plugin Results
        lines.append("## Per-Plugin Results")
        lines.append("")
        lines.append("| Plugin | Peak Memory (KB) | Peak Memory (MB) | Status |")
        lines.append("|--------|------------------|------------------|--------|")

        for profile in sorted(profiles, key=lambda p: p.peak_memory_kb, reverse=True):
            lines.append(
                f"| {profile.plugin_name} | {profile.peak_memory_kb:.2f} | "
                f"{profile.peak_memory_kb/1024:.2f} | {profile.status_emoji} |"
            )

        lines.append("")

        # Largest Components
        lines.append("## Largest Components (Top 10)")
        lines.append("")

        all_metrics = []
        for profile in profiles:
            for metric in profile.metrics:
                all_metrics.append((profile.plugin_name, metric))

        largest = sorted(all_metrics, key=lambda x: x[1].memory_kb, reverse=True)[:10]

        lines.append("| Plugin | Component | Memory (KB) | Memory (MB) |")
        lines.append("|--------|-----------|-------------|-------------|")
        for plugin_name, metric in largest:
            lines.append(
                f"| {plugin_name} | {metric.name} | {metric.memory_kb:.2f} | "
                f"{metric.memory_kb/1024:.2f} |"
            )

        lines.append("")

        return '\n'.join(lines)


def main() -> int:
    """Main entry point for memory analyzer."""

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 memory-analyzer.py <plugin-name> [plugins-root]")
        print("       python3 memory-analyzer.py --all [plugins-root]")
        return 1

    # Determine plugins root
    if len(sys.argv) >= 3:
        plugins_root = Path(sys.argv[2])
    else:
        # Try to find plugins directory
        current = Path.cwd()
        if (current / "plugins").exists():
            plugins_root = current / "plugins"
        else:
            plugins_root = current.parent / "plugins"

    # Initialize analyzer
    analyzer = PluginMemoryAnalyzer(plugins_root)
    reporter = MemoryProfileReporter()

    # Profile plugin(s)
    if sys.argv[1] == "--all":
        profiles = analyzer.profile_all_plugins()

        # Print summary report
        summary = reporter.generate_summary_report(profiles)
        print(summary)

        # Print individual reports
        for profile in profiles:
            print("\n" + "=" * 80 + "\n")
            report = reporter.generate_report(profile)
            print(report)

        # Return exit code based on overall status
        if any(p.status == 'error' for p in profiles):
            return 2
        elif any(p.status == 'fail' for p in profiles):
            return 1
        else:
            return 0
    else:
        plugin_name = sys.argv[1]
        profile = analyzer.profile_plugin(plugin_name)

        # Print report
        report = reporter.generate_report(profile)
        print(report)

        # Return exit code
        if profile.status == 'error':
            return 2
        elif profile.status == 'fail':
            return 1
        else:
            return 0


if __name__ == "__main__":
    sys.exit(main())
