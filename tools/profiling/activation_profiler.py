#!/usr/bin/env python3
"""
Agent Activation Profiler

Measures agent activation performance to identify bottlenecks in triggering logic.

Features:
- Measures context analysis time
- Profiles triggering condition evaluation
- Tracks pattern matching performance
- Identifies bottlenecks in activation logic
- Target: <50ms activation time

Usage:
    python3 tools/activation_profiler.py <plugin-name>
    python3 tools/activation_profiler.py <plugin-name> /path/to/plugins
    python3 tools/activation_profiler.py --all  # Profile all plugins
"""

import json
import re
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActivationMetric:
    """Timing metric for a specific activation operation."""
    name: str
    duration_ms: float
    status: str  # 'pass', 'warn', 'fail'
    details: str = ""


@dataclass
class ActivationProfile:
    """Complete activation performance profile for a plugin."""
    plugin_name: str
    plugin_path: Path
    total_activation_time_ms: float
    metrics: list[ActivationMetric] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    agent_count: int = 0
    keyword_count: int = 0

    @property
    def status(self) -> str:
        """Overall status based on total activation time."""
        if self.errors:
            return 'error'
        elif self.total_activation_time_ms > 50:
            return 'fail'
        elif self.total_activation_time_ms > 35:
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


class AgentActivationProfiler:
    """Profiles agent activation performance."""

    TARGET_ACTIVATION_TIME_MS = 50
    WARNING_THRESHOLD_MS = 35

    def __init__(self, plugins_root: Path) -> None:
        self.plugins_root = plugins_root

    def profile_plugin(self, plugin_name: str) -> ActivationProfile:
        """Profile a single plugin's activation performance."""
        plugin_path = self.plugins_root / plugin_name
        profile = ActivationProfile(
            plugin_name=plugin_name,
            plugin_path=plugin_path,
            total_activation_time_ms=0
        )

        # Check if plugin exists
        if not plugin_path.exists():
            profile.errors.append(f"Plugin directory not found: {plugin_path}")
            return profile

        # Load plugin metadata
        plugin_data = self._load_plugin_metadata(profile)
        if not plugin_data:
            return profile

        # Start total activation timer
        total_timer = Timer()
        with total_timer:
            # 1. Measure metadata extraction
            self._measure_metadata_extraction(profile, plugin_data)

            # 2. Measure keyword matching (simulated)
            self._measure_keyword_matching(profile, plugin_data)

            # 3. Measure agent selection
            self._measure_agent_selection(profile, plugin_data)

            # 4. Measure agent description parsing
            self._measure_agent_description_parsing(profile, plugin_data)

            # 5. Measure context relevance scoring (simulated)
            self._measure_context_scoring(profile, plugin_data)

        profile.total_activation_time_ms = total_timer.duration_ms

        # Add overall assessment
        if profile.total_activation_time_ms > self.TARGET_ACTIVATION_TIME_MS:
            profile.warnings.append(
                f"Total activation time ({profile.total_activation_time_ms:.2f}ms) exceeds target ({self.TARGET_ACTIVATION_TIME_MS}ms)"
            )

        return profile

    def _load_plugin_metadata(self, profile: ActivationProfile) -> dict[str, Any] | None:
        """Load plugin.json metadata."""
        json_path = profile.plugin_path / "plugin.json"

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            profile.errors.append(f"plugin.json not found at {json_path}")
            return None
        except json.JSONDecodeError as e:
            profile.errors.append(f"Invalid JSON in plugin.json: {e}")
            return None

    def _measure_metadata_extraction(self, profile: ActivationProfile, plugin_data: dict[str, Any]) -> None:
        """Measure metadata extraction time."""
        timer = Timer()

        with timer:
            # Extract key metadata fields
            _ = plugin_data.get('name')
            _ = plugin_data.get('category')
            _ = plugin_data.get('description')
            agents = plugin_data.get('agents', [])
            keywords = plugin_data.get('keywords', [])

            profile.agent_count = len(agents)
            profile.keyword_count = len(keywords)

        status = 'pass' if timer.duration_ms < 1 else ('warn' if timer.duration_ms < 3 else 'fail')
        profile.metrics.append(ActivationMetric(
            name="metadata extraction",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"{profile.agent_count} agents, {profile.keyword_count} keywords"
        ))

    def _measure_keyword_matching(self, profile: ActivationProfile, plugin_data: dict[str, Any]) -> None:
        """Measure keyword matching performance."""
        keywords = plugin_data.get('keywords', [])

        # Simulate context (file content and path)
        simulated_contexts = [
            "import numpy as np\nimport julia\nfrom julia import Main",
            "using DifferentialEquations\nusing Plots",
            "const x = 42",
            "# Regular Python code",
            "System.out.println('Hello');"
        ]

        timer = Timer()

        with timer:
            for context in simulated_contexts:
                context_lower = context.lower()
                matches = 0
                for keyword in keywords:
                    if keyword.lower() in context_lower:
                        matches += 1

        # Average time per context
        avg_time = timer.duration_ms / len(simulated_contexts)

        status = 'pass' if avg_time < 5 else ('warn' if avg_time < 10 else 'fail')
        profile.metrics.append(ActivationMetric(
            name="keyword matching",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"{len(keywords)} keywords across {len(simulated_contexts)} contexts"
        ))

    def _measure_agent_selection(self, profile: ActivationProfile, plugin_data: dict[str, Any]) -> None:
        """Measure agent selection time."""
        agents = plugin_data.get('agents', [])

        timer = Timer()

        with timer:
            # Simulate selecting relevant agents based on status and description
            active_agents = [a for a in agents if a.get('status') == 'active']

            # Simulate description analysis
            for agent in active_agents:
                desc = agent.get('description', '').lower()
                # Check for key terms (simulated pattern matching)
                _ = 'expert' in desc
                _ = 'specialist' in desc
                _ = 'master' in desc

        status = 'pass' if timer.duration_ms < 3 else ('warn' if timer.duration_ms < 8 else 'fail')
        profile.metrics.append(ActivationMetric(
            name="agent selection",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"{len(active_agents)} active agents from {len(agents)} total"
        ))

    def _measure_agent_description_parsing(self, profile: ActivationProfile, plugin_data: dict[str, Any]) -> None:
        """Measure agent description parsing time."""
        agents = plugin_data.get('agents', [])

        timer = Timer()

        with timer:
            for agent in agents:
                description = agent.get('description', '')
                # Simulate NLP-style analysis
                _ = len(description)
                _ = description.split()
                # Extract expertise indicators
                _ = re.findall(r'\b(expert|specialist|master|proficient)\b', description, re.IGNORECASE)

        status = 'pass' if timer.duration_ms < 5 else ('warn' if timer.duration_ms < 12 else 'fail')
        profile.metrics.append(ActivationMetric(
            name="agent description parsing",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Parsed {len(agents)} agent descriptions"
        ))

    def _measure_context_scoring(self, profile: ActivationProfile, plugin_data: dict[str, Any]) -> None:
        """Measure context relevance scoring time."""
        agents = plugin_data.get('agents', [])
        keywords = plugin_data.get('keywords', [])

        # Simulated file contexts
        file_contexts = [
            {"path": "src/main.jl", "extension": ".jl", "content": "using DifferentialEquations"},
            {"path": "test/test_model.py", "extension": ".py", "content": "import pytest"},
            {"path": "README.md", "extension": ".md", "content": "# Documentation"},
        ]

        timer = Timer()

        with timer:
            for context in file_contexts:
                score = 0

                # File extension matching
                ext = context['extension']
                if any(kw in ext for kw in keywords):
                    score += 10

                # Content keyword matching
                content_lower = context['content'].lower()
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        score += 5

                # Agent expertise matching
                for agent in agents:
                    desc = agent.get('description', '').lower()
                    if any(kw.lower() in desc for kw in keywords[:3]):  # Top 3 keywords
                        score += 3

        status = 'pass' if timer.duration_ms < 10 else ('warn' if timer.duration_ms < 20 else 'fail')
        profile.metrics.append(ActivationMetric(
            name="context relevance scoring",
            duration_ms=timer.duration_ms,
            status=status,
            details=f"Scored {len(file_contexts)} contexts"
        ))

    def profile_all_plugins(self) -> list[ActivationProfile]:
        """Profile all plugins in the marketplace."""
        profiles: list[ActivationProfile] = []

        if not self.plugins_root.exists():
            return profiles

        for plugin_dir in sorted(self.plugins_root.iterdir()):
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                profile = self.profile_plugin(plugin_dir.name)
                profiles.append(profile)

        return profiles


class ActivationProfileReporter:
    """Generates formatted reports for activation profiling results."""

    def generate_report(self, profile: ActivationProfile) -> str:
        """Generate markdown report for a single plugin."""
        lines = []

        # Header
        lines.append(f"# Activation Profile: {profile.plugin_name}")
        lines.append("")
        lines.append(f"**Plugin Path:** `{profile.plugin_path}`")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Activation Time:** {profile.total_activation_time_ms:.2f}ms")
        lines.append(f"- **Target Activation Time:** {AgentActivationProfiler.TARGET_ACTIVATION_TIME_MS}ms")
        lines.append(f"- **Status:** {profile.status_emoji} {profile.status.upper()}")
        lines.append(f"- **Agents:** {profile.agent_count}")
        lines.append(f"- **Keywords:** {profile.keyword_count}")
        lines.append(f"- **Metrics Collected:** {len(profile.metrics)}")
        lines.append(f"- **Errors:** {len(profile.errors)}")
        lines.append(f"- **Warnings:** {len(profile.warnings)}")
        lines.append("")

        # Detailed Metrics
        if profile.metrics:
            lines.append("## Activation Time Breakdown")
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
            lines.append(f"**Status:** {profile.status_emoji} EXCELLENT - Activation time well below target")
            lines.append("")
            lines.append(f"Plugin activates in {profile.total_activation_time_ms:.2f}ms, well under the {AgentActivationProfiler.TARGET_ACTIVATION_TIME_MS}ms target.")
        elif profile.status == 'warn':
            lines.append(f"**Status:** {profile.status_emoji} WARNING - Activation time approaching target")
            lines.append("")
            lines.append(f"Plugin activates in {profile.total_activation_time_ms:.2f}ms, approaching the {AgentActivationProfiler.TARGET_ACTIVATION_TIME_MS}ms target.")
            lines.append("Consider optimizing slow components.")
        elif profile.status == 'fail':
            lines.append(f"**Status:** {profile.status_emoji} OPTIMIZATION NEEDED - Activation time exceeds target")
            lines.append("")
            lines.append(f"Plugin activates in {profile.total_activation_time_ms:.2f}ms, exceeding the {AgentActivationProfiler.TARGET_ACTIVATION_TIME_MS}ms target.")
            lines.append("")
            lines.append("**Recommended Actions:**")
            # Find slowest components
            slow_metrics = [m for m in profile.metrics if m.status in ['warn', 'fail']]
            for metric in slow_metrics:
                lines.append(f"- Optimize {metric.name} (currently {metric.duration_ms:.2f}ms)")

            # Specific recommendations based on metrics
            if profile.keyword_count > 50:
                lines.append(f"- Consider reducing keyword count (currently {profile.keyword_count})")
        else:
            lines.append(f"**Status:** {profile.status_emoji} ERROR - Unable to complete profiling")

        lines.append("")

        return '\n'.join(lines)

    def generate_summary_report(self, profiles: list[ActivationProfile]) -> str:
        """Generate summary report for multiple plugins."""
        lines = []

        # Header
        lines.append("# Activation Profile Summary")
        lines.append("")
        lines.append(f"**Plugins Profiled:** {len(profiles)}")
        lines.append("")

        # Overall Statistics
        total_plugins = len(profiles)
        pass_count = sum(1 for p in profiles if p.status == 'pass')
        warn_count = sum(1 for p in profiles if p.status == 'warn')
        fail_count = sum(1 for p in profiles if p.status == 'fail')
        error_count = sum(1 for p in profiles if p.status == 'error')

        avg_activation_time = sum(p.total_activation_time_ms for p in profiles) / total_plugins if total_plugins > 0 else 0

        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"- **Average Activation Time:** {avg_activation_time:.2f}ms")
        lines.append(f"- **Target Activation Time:** {AgentActivationProfiler.TARGET_ACTIVATION_TIME_MS}ms")
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
        lines.append("| Plugin | Activation Time (ms) | Agents | Keywords | Status |")
        lines.append("|--------|---------------------|--------|----------|--------|")

        for profile in sorted(profiles, key=lambda p: p.total_activation_time_ms, reverse=True):
            lines.append(
                f"| {profile.plugin_name} | {profile.total_activation_time_ms:.2f} | "
                f"{profile.agent_count} | {profile.keyword_count} | {profile.status_emoji} |"
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
    """Main entry point for activation profiler."""

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 activation_profiler.py <plugin-name> [plugins-root]")
        print("       python3 activation_profiler.py --all [plugins-root]")
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
    profiler = AgentActivationProfiler(plugins_root)
    reporter = ActivationProfileReporter()

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
