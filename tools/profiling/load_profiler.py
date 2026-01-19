#!/usr/bin/env python3
"""
Plugin Load Time Profiler

Measures plugin loading performance to identify bottlenecks and optimize initialization.

Features:
- Measures marketplace initialization time
- Profiles individual plugin loading
- Identifies slow configuration parsing
- Tracks dependency loading overhead
- Target: <100ms load time per plugin

Usage:
    python3 tools/load_profiler.py <plugin-name>
    python3 tools/load_profiler.py <plugin-name> /path/to/plugins
    python3 tools/load_profiler.py --all  # Profile all plugins
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadMetric:
    """Timing metric for a specific loading operation."""
    name: str
    duration_ms: float
    status: str  # 'pass', 'warn', 'fail'
    details: str = ""


@dataclass
class PluginLoadProfile:
    """Complete load performance profile for a plugin."""
    plugin_name: str
    plugin_path: Path
    total_load_time_ms: float
    metrics: list[LoadMetric] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Overall status based on total load time."""
        if self.errors:
            return 'error'
        elif self.total_load_time_ms > 100:
            return 'fail'
        elif self.total_load_time_ms > 75:
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


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class PluginLoadProfiler:
    """Profiles plugin loading performance."""

    TARGET_LOAD_TIME_MS = 100
    WARNING_THRESHOLD_MS = 75

    def __init__(self, plugins_root: Path) -> None:
        self.plugins_root = plugins_root

    def profile_plugin(self, plugin_name: str) -> PluginLoadProfile:
        """Profile a single plugin's load performance."""
        plugin_path = self.plugins_root / plugin_name
        profile = PluginLoadProfile(
            plugin_name=plugin_name,
            plugin_path=plugin_path,
            total_load_time_ms=0
        )

        # Check if plugin exists
        if not plugin_path.exists():
            profile.errors.append(f"Plugin directory not found: {plugin_path}")
            return profile

        # Start total load timer
        total_timer = Timer()
        with total_timer:
            # 1. Measure plugin.json loading
            self._measure_json_load(profile)

            # 2. Measure agents directory scan
            self._measure_agents_load(profile)

            # 3. Measure commands directory scan
            self._measure_commands_load(profile)

            # 4. Measure skills directory scan
            self._measure_skills_load(profile)

            # 5. Measure README loading
            self._measure_readme_load(profile)

        profile.total_load_time_ms = total_timer.duration_ms

        # Add overall assessment
        if profile.total_load_time_ms > self.TARGET_LOAD_TIME_MS:
            profile.warnings.append(
                f"Total load time ({profile.total_load_time_ms:.2f}ms) exceeds target ({self.TARGET_LOAD_TIME_MS}ms)"
            )

        return profile

    def _measure_json_load(self, profile: PluginLoadProfile) -> None:
        """Measure plugin.json parsing time."""
        json_path = profile.plugin_path / "plugin.json"

        timer = Timer()
        try:
            with timer:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Simulate metadata extraction overhead
                _ = data.get('name')
                _ = data.get('version')
                _ = data.get('description')
                _ = data.get('agents', [])
                _ = data.get('commands', [])
                _ = data.get('skills', [])
                _ = data.get('keywords', [])
        except FileNotFoundError:
            profile.errors.append(f"plugin.json not found at {json_path}")
            profile.metrics.append(LoadMetric(
                name="plugin.json parsing",
                duration_ms=0,
                status='error',
                details="File not found"
            ))
            return
        except json.JSONDecodeError as e:
            profile.errors.append(f"Invalid JSON in plugin.json: {e}")
            profile.metrics.append(LoadMetric(
                name="plugin.json parsing",
                duration_ms=timer.duration_ms,
                status='error',
                details=f"JSON decode error: {e}"
            ))
            return

        status = 'pass' if timer.duration_ms < 10 else ('warn' if timer.duration_ms < 25 else 'fail')
        profile.metrics.append(LoadMetric(
            name="plugin.json parsing",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Parsed {len(str(data))} characters"
        ))

    def _measure_agents_load(self, profile: PluginLoadProfile) -> None:
        """Measure agents directory scanning time."""
        agents_dir = profile.plugin_path / "agents"

        timer = Timer()
        with timer:
            if agents_dir.exists():
                agent_files = list(agents_dir.glob("*.md"))
                # Simulate reading file metadata
                for agent_file in agent_files:
                    _ = agent_file.stat()

        status = 'pass' if timer.duration_ms < 5 else ('warn' if timer.duration_ms < 15 else 'fail')
        file_count = len(list(agents_dir.glob("*.md"))) if agents_dir.exists() else 0
        profile.metrics.append(LoadMetric(
            name="agents directory scan",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Scanned {file_count} agent files"
        ))

    def _measure_commands_load(self, profile: PluginLoadProfile) -> None:
        """Measure commands directory scanning time."""
        commands_dir = profile.plugin_path / "commands"

        timer = Timer()
        with timer:
            if commands_dir.exists():
                command_files = list(commands_dir.glob("*.md"))
                # Simulate reading file metadata
                for command_file in command_files:
                    _ = command_file.stat()

        status = 'pass' if timer.duration_ms < 5 else ('warn' if timer.duration_ms < 15 else 'fail')
        file_count = len(list(commands_dir.glob("*.md"))) if commands_dir.exists() else 0
        profile.metrics.append(LoadMetric(
            name="commands directory scan",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Scanned {file_count} command files"
        ))

    def _measure_skills_load(self, profile: PluginLoadProfile) -> None:
        """Measure skills directory scanning time."""
        skills_dir = profile.plugin_path / "skills"

        timer = Timer()
        with timer:
            if skills_dir.exists():
                skill_files = list(skills_dir.glob("*.md"))
                # Simulate reading file metadata
                for skill_file in skill_files:
                    _ = skill_file.stat()

        status = 'pass' if timer.duration_ms < 5 else ('warn' if timer.duration_ms < 15 else 'fail')
        file_count = len(list(skills_dir.glob("*.md"))) if skills_dir.exists() else 0
        profile.metrics.append(LoadMetric(
            name="skills directory scan",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Scanned {file_count} skill files"
        ))

    def _measure_readme_load(self, profile: PluginLoadProfile) -> None:
        """Measure README.md loading time."""
        readme_path = profile.plugin_path / "README.md"

        timer = Timer()
        try:
            with timer:
                if readme_path.exists():
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    _ = len(content)
        except Exception as e:
            profile.warnings.append(f"Error reading README.md: {e}")

        status = 'pass' if timer.duration_ms < 5 else ('warn' if timer.duration_ms < 15 else 'fail')
        profile.metrics.append(LoadMetric(
            name="README.md loading",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Loaded {readme_path.stat().st_size if readme_path.exists() else 0} bytes"
        ))

    def profile_all_plugins(self) -> list[PluginLoadProfile]:
        """Profile all plugins in the marketplace."""
        profiles = []

        if not self.plugins_root.exists():
            return profiles

        for plugin_dir in sorted(self.plugins_root.iterdir()):
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                profile = self.profile_plugin(plugin_dir.name)
                profiles.append(profile)

        return profiles


class LoadProfileReporter:
    """Generates formatted reports for load profiling results."""

    def generate_report(self, profile: PluginLoadProfile) -> str:
        """Generate markdown report for a single plugin."""
        lines = []

        # Header
        lines.append(f"# Load Profile: {profile.plugin_name}")
        lines.append("")
        lines.append(f"**Plugin Path:** `{profile.plugin_path}`")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Load Time:** {profile.total_load_time_ms:.2f}ms")
        lines.append(f"- **Target Load Time:** {PluginLoadProfiler.TARGET_LOAD_TIME_MS}ms")
        lines.append(f"- **Status:** {profile.status_emoji} {profile.status.upper()}")
        lines.append(f"- **Metrics Collected:** {len(profile.metrics)}")
        lines.append(f"- **Errors:** {len(profile.errors)}")
        lines.append(f"- **Warnings:** {len(profile.warnings)}")
        lines.append("")

        # Detailed Metrics
        if profile.metrics:
            lines.append("## Load Time Breakdown")
            lines.append("")
            lines.append("| Component | Time (ms) | Status | Details |")
            lines.append("|-----------|-----------|--------|---------|")

            for metric in profile.metrics:
                status_icon = {
                    'pass': '✅',
                    'warn': '⚠️',
                    'fail': '❌',
                    'error': '🔴'
                }.get(metric.status, '❓')

                lines.append(
                    f"| {metric.name} | {metric.duration_ms:.2f} | {status_icon} | {metric.details} |"
                )

            lines.append("")

            # Performance breakdown chart
            lines.append("### Time Distribution")
            lines.append("")
            total = sum(m.duration_ms for m in profile.metrics)
            for metric in sorted(profile.metrics, key=lambda m: m.duration_ms, reverse=True):
                percentage = (metric.duration_ms / total * 100) if total > 0 else 0
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = '█' * bar_length
                lines.append(f"- {metric.name}: {metric.duration_ms:.2f}ms ({percentage:.1f}%) {bar}")
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
            lines.append(f"**Status:** {profile.status_emoji} EXCELLENT - Load time well below target")
            lines.append("")
            lines.append(f"Plugin loads in {profile.total_load_time_ms:.2f}ms, well under the {PluginLoadProfiler.TARGET_LOAD_TIME_MS}ms target.")
        elif profile.status == 'warn':
            lines.append(f"**Status:** {profile.status_emoji} WARNING - Load time approaching target")
            lines.append("")
            lines.append(f"Plugin loads in {profile.total_load_time_ms:.2f}ms, approaching the {PluginLoadProfiler.TARGET_LOAD_TIME_MS}ms target.")
            lines.append("Consider optimizing slow components.")
        elif profile.status == 'fail':
            lines.append(f"**Status:** {profile.status_emoji} OPTIMIZATION NEEDED - Load time exceeds target")
            lines.append("")
            lines.append(f"Plugin loads in {profile.total_load_time_ms:.2f}ms, exceeding the {PluginLoadProfiler.TARGET_LOAD_TIME_MS}ms target.")
            lines.append("")
            lines.append("**Recommended Actions:**")
            # Find slowest components
            slow_metrics = [m for m in profile.metrics if m.status in ['warn', 'fail']]
            for metric in slow_metrics:
                lines.append(f"- Optimize {metric.name} (currently {metric.duration_ms:.2f}ms)")
        else:
            lines.append(f"**Status:** {profile.status_emoji} ERROR - Unable to complete profiling")

        lines.append("")

        return '\n'.join(lines)

    def generate_summary_report(self, profiles: list[PluginLoadProfile]) -> str:
        """Generate summary report for multiple plugins."""
        lines = []

        # Header
        lines.append("# Load Profile Summary")
        lines.append("")
        lines.append(f"**Plugins Profiled:** {len(profiles)}")
        lines.append("")

        # Overall Statistics
        total_plugins = len(profiles)
        pass_count = sum(1 for p in profiles if p.status == 'pass')
        warn_count = sum(1 for p in profiles if p.status == 'warn')
        fail_count = sum(1 for p in profiles if p.status == 'fail')
        error_count = sum(1 for p in profiles if p.status == 'error')

        avg_load_time = sum(p.total_load_time_ms for p in profiles) / total_plugins if total_plugins > 0 else 0

        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"- **Average Load Time:** {avg_load_time:.2f}ms")
        lines.append(f"- **Target Load Time:** {PluginLoadProfiler.TARGET_LOAD_TIME_MS}ms")
        lines.append(f"- **Pass Rate:** {pass_count}/{total_plugins} ({pass_count/total_plugins*100:.1f}%)")
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
        lines.append("| Plugin | Load Time (ms) | Status | Notes |")
        lines.append("|--------|----------------|--------|-------|")

        for profile in sorted(profiles, key=lambda p: p.total_load_time_ms, reverse=True):
            notes = []
            if profile.errors:
                notes.append(f"{len(profile.errors)} errors")
            if profile.warnings:
                notes.append(f"{len(profile.warnings)} warnings")
            notes_str = ", ".join(notes) if notes else "-"

            lines.append(
                f"| {profile.plugin_name} | {profile.total_load_time_ms:.2f} | "
                f"{profile.status_emoji} | {notes_str} |"
            )

        lines.append("")

        # Slowest Components
        lines.append("## Slowest Components (Top 10)")
        lines.append("")

        all_metrics = []
        for profile in profiles:
            for metric in profile.metrics:
                all_metrics.append((profile.plugin_name, metric))

        slowest = sorted(all_metrics, key=lambda x: x[1].duration_ms, reverse=True)[:10]

        lines.append("| Plugin | Component | Time (ms) |")
        lines.append("|--------|-----------|-----------|")
        for plugin_name, metric in slowest:
            lines.append(f"| {plugin_name} | {metric.name} | {metric.duration_ms:.2f} |")

        lines.append("")

        return '\n'.join(lines)


def main() -> int:
    """Main entry point for load profiler."""

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 load_profiler.py <plugin-name> [plugins-root]")
        print("       python3 load_profiler.py --all [plugins-root]")
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

    # Initialize profiler
    profiler = PluginLoadProfiler(plugins_root)
    reporter = LoadProfileReporter()

    # Profile plugin(s)
    if sys.argv[1] == "--all":
        profiles = profiler.profile_all_plugins()

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
        profile = profiler.profile_plugin(plugin_name)

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
