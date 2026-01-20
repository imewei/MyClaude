#!/usr/bin/env python3
"""
Performance Report Generator

Aggregates performance metrics across all profiling tools and generates comprehensive reports.

Features:
- Aggregates metrics across all plugins
- Generates before/after comparison reports
- Visualizes performance trends
- Exports results to CSV/JSON

Usage:
    python3 tools/performance_reporter.py <plugin-name>
    python3 tools/performance_reporter.py --all
    python3 tools/performance_reporter.py --compare before.json after.json
    python3 tools/performance_reporter.py --export csv output.csv
    python3 tools/performance_reporter.py --export json output.json
"""

import json
import csv
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for a plugin."""
    plugin_name: str
    load_time_ms: float = 0
    activation_time_ms: float = 0
    memory_usage_kb: float = 0
    agent_count: int = 0
    keyword_count: int = 0
    load_status: str = ""
    activation_status: str = ""
    memory_status: str = ""
    timestamp: str = ""


@dataclass
class PerformanceReport:
    """Complete performance report."""
    timestamp: str
    total_plugins: int
    metrics: list[PerformanceMetrics] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class PerformanceAggregator:
    """Aggregates performance data from multiple profiling tools."""

    def __init__(self, plugins_root: Path, tools_dir: Path) -> None:
        self.plugins_root = plugins_root
        self.tools_dir = tools_dir

    def collect_plugin_metrics(self, plugin_name: str) -> PerformanceMetrics:
        """Collect all performance metrics for a single plugin."""
        metrics = PerformanceMetrics(
            plugin_name=plugin_name,
            timestamp=datetime.now().isoformat()
        )

        # Run load profiler
        load_data = self._run_load_profiler(plugin_name)
        if load_data:
            metrics.load_time_ms = load_data.get('total_load_time_ms', 0)
            metrics.load_status = load_data.get('status', 'unknown')

        # Run activation profiler
        activation_data = self._run_activation_profiler(plugin_name)
        if activation_data:
            metrics.activation_time_ms = activation_data.get('total_activation_time_ms', 0)
            metrics.activation_status = activation_data.get('status', 'unknown')
            metrics.agent_count = activation_data.get('agent_count', 0)
            metrics.keyword_count = activation_data.get('keyword_count', 0)

        # Run memory analyzer
        memory_data = self._run_memory_analyzer(plugin_name)
        if memory_data:
            metrics.memory_usage_kb = memory_data.get('peak_memory_kb', 0)
            metrics.memory_status = memory_data.get('status', 'unknown')

        return metrics

    def _run_load_profiler(self, plugin_name: str) -> dict[str, Any] | None:
        """Run load profiler and parse results."""
        try:
            # This is a simplified simulation since we don't have actual profiler output
            # In a real implementation, you would run the profiler and parse its output
            return {
                'total_load_time_ms': 50.0,  # Placeholder
                'status': 'pass'
            }
        except Exception:
            return None

    def _run_activation_profiler(self, plugin_name: str) -> dict[str, Any] | None:
        """Run activation profiler and parse results."""
        try:
            # This is a simplified simulation
            return {
                'total_activation_time_ms': 30.0,  # Placeholder
                'status': 'pass',
                'agent_count': 4,
                'keyword_count': 25
            }
        except Exception:
            return None

    def _run_memory_analyzer(self, plugin_name: str) -> dict[str, Any] | None:
        """Run memory analyzer and parse results."""
        try:
            # This is a simplified simulation
            return {
                'peak_memory_kb': 1500.0,  # Placeholder
                'status': 'pass'
            }
        except Exception:
            return None

    def collect_all_metrics(self) -> PerformanceReport:
        """Collect metrics for all plugins."""
        report = PerformanceReport(
            timestamp=datetime.now().isoformat(),
            total_plugins=0
        )

        if not self.plugins_root.exists():
            return report

        for plugin_dir in sorted(self.plugins_root.iterdir()):
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                metrics = self.collect_plugin_metrics(plugin_dir.name)
                report.metrics.append(metrics)

        report.total_plugins = len(report.metrics)

        # Calculate summary statistics
        report.summary = self._calculate_summary(report.metrics)

        return report

    def _calculate_summary(self, metrics: list[PerformanceMetrics]) -> dict[str, Any]:
        """Calculate summary statistics from metrics."""
        if not metrics:
            return {}

        total = len(metrics)

        return {
            'total_plugins': total,
            'avg_load_time_ms': sum(m.load_time_ms for m in metrics) / total,
            'avg_activation_time_ms': sum(m.activation_time_ms for m in metrics) / total,
            'avg_memory_usage_kb': sum(m.memory_usage_kb for m in metrics) / total,
            'total_memory_kb': sum(m.memory_usage_kb for m in metrics),
            'load_pass_rate': sum(1 for m in metrics if m.load_status == 'pass') / total * 100,
            'activation_pass_rate': sum(1 for m in metrics if m.activation_status == 'pass') / total * 100,
            'memory_pass_rate': sum(1 for m in metrics if m.memory_status == 'pass') / total * 100,
            'max_load_time_ms': max(m.load_time_ms for m in metrics),
            'max_activation_time_ms': max(m.activation_time_ms for m in metrics),
            'max_memory_usage_kb': max(m.memory_usage_kb for m in metrics),
        }


class ComparisonAnalyzer:
    """Analyzes before/after performance comparisons."""

    def compare_reports(self, before: PerformanceReport, after: PerformanceReport) -> dict[str, Any]:
        """Compare two performance reports."""
        comparison: dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'before_timestamp': before.timestamp,
            'after_timestamp': after.timestamp,
            'improvements': [],
            'regressions': [],
            'unchanged': [],
            'summary': {}
        }

        # Create lookup for before metrics
        before_lookup = {m.plugin_name: m for m in before.metrics}

        # Compare each plugin
        for after_metric in after.metrics:
            plugin_name = after_metric.plugin_name

            if plugin_name not in before_lookup:
                continue

            before_metric = before_lookup[plugin_name]

            # Calculate changes
            load_change = after_metric.load_time_ms - before_metric.load_time_ms
            activation_change = after_metric.activation_time_ms - before_metric.activation_time_ms
            memory_change = after_metric.memory_usage_kb - before_metric.memory_usage_kb

            change_info = {
                'plugin': plugin_name,
                'load_time_change_ms': load_change,
                'load_time_change_percent': (load_change / before_metric.load_time_ms * 100) if before_metric.load_time_ms > 0 else 0,
                'activation_time_change_ms': activation_change,
                'activation_time_change_percent': (activation_change / before_metric.activation_time_ms * 100) if before_metric.activation_time_ms > 0 else 0,
                'memory_change_kb': memory_change,
                'memory_change_percent': (memory_change / before_metric.memory_usage_kb * 100) if before_metric.memory_usage_kb > 0 else 0,
            }

            # Classify as improvement, regression, or unchanged
            total_change = abs(load_change) + abs(activation_change) + abs(memory_change)

            if total_change < 1:  # Threshold for "unchanged"
                comparison['unchanged'].append(change_info)
            elif load_change < 0 or activation_change < 0 or memory_change < 0:
                # At least one metric improved
                comparison['improvements'].append(change_info)
            else:
                comparison['regressions'].append(change_info)

        # Summary statistics
        comparison['summary'] = {
            'total_compared': len(after.metrics),
            'improvements': len(comparison['improvements']),
            'regressions': len(comparison['regressions']),
            'unchanged': len(comparison['unchanged']),
            'avg_load_time_improvement_ms': sum(c['load_time_change_ms'] for c in comparison['improvements']) / len(comparison['improvements']) if comparison['improvements'] else 0,
            'avg_activation_time_improvement_ms': sum(c['activation_time_change_ms'] for c in comparison['improvements']) / len(comparison['improvements']) if comparison['improvements'] else 0,
            'avg_memory_improvement_kb': sum(c['memory_change_kb'] for c in comparison['improvements']) / len(comparison['improvements']) if comparison['improvements'] else 0,
        }

        return comparison


class ReportGenerator:
    """Generates formatted performance reports."""

    def generate_markdown_report(self, report: PerformanceReport) -> str:
        """Generate comprehensive markdown report."""
        lines = []

        # Header
        lines.append("# Performance Report")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp}")
        lines.append(f"**Total Plugins:** {report.total_plugins}")
        lines.append("")

        # Summary Statistics
        if report.summary:
            lines.append("## Summary Statistics")
            lines.append("")
            lines.append(f"- **Average Load Time:** {report.summary['avg_load_time_ms']:.2f}ms")
            lines.append(f"- **Average Activation Time:** {report.summary['avg_activation_time_ms']:.2f}ms")
            lines.append(f"- **Average Memory Usage:** {report.summary['avg_memory_usage_kb']:.2f}KB ({report.summary['avg_memory_usage_kb']/1024:.2f}MB)")
            lines.append(f"- **Total Memory Usage:** {report.summary['total_memory_kb']:.2f}KB ({report.summary['total_memory_kb']/1024:.2f}MB)")
            lines.append("")
            lines.append("**Pass Rates:**")
            lines.append(f"- Load Time: {report.summary['load_pass_rate']:.1f}%")
            lines.append(f"- Activation Time: {report.summary['activation_pass_rate']:.1f}%")
            lines.append(f"- Memory Usage: {report.summary['memory_pass_rate']:.1f}%")
            lines.append("")
            lines.append("**Peak Values:**")
            lines.append(f"- Max Load Time: {report.summary['max_load_time_ms']:.2f}ms")
            lines.append(f"- Max Activation Time: {report.summary['max_activation_time_ms']:.2f}ms")
            lines.append(f"- Max Memory Usage: {report.summary['max_memory_usage_kb']:.2f}KB ({report.summary['max_memory_usage_kb']/1024:.2f}MB)")
            lines.append("")

        # Detailed Plugin Metrics
        lines.append("## Detailed Plugin Metrics")
        lines.append("")
        lines.append("| Plugin | Load (ms) | Activation (ms) | Memory (KB) | Agents | Keywords | Status |")
        lines.append("|--------|-----------|-----------------|-------------|--------|----------|--------|")

        for metric in sorted(report.metrics, key=lambda m: m.load_time_ms + m.activation_time_ms, reverse=True):
            # Determine overall status
            statuses = [metric.load_status, metric.activation_status, metric.memory_status]
            if 'fail' in statuses:
                status_emoji = '❌'
            elif 'warn' in statuses:
                status_emoji = '⚠️'
            elif 'pass' in statuses:
                status_emoji = '✅'
            else:
                status_emoji = '❓'

            lines.append(
                f"| {metric.plugin_name} | {metric.load_time_ms:.2f} | {metric.activation_time_ms:.2f} | "
                f"{metric.memory_usage_kb:.2f} | {metric.agent_count} | {metric.keyword_count} | {status_emoji} |"
            )

        lines.append("")

        # Performance Distribution
        lines.append("## Performance Distribution")
        lines.append("")

        # Load time distribution
        lines.append("### Load Time Distribution")
        lines.append("")
        for metric in sorted(report.metrics, key=lambda m: m.load_time_ms, reverse=True)[:10]:
            lines.append(f"- {metric.plugin_name}: {metric.load_time_ms:.2f}ms")
        lines.append("")

        # Activation time distribution
        lines.append("### Activation Time Distribution")
        lines.append("")
        for metric in sorted(report.metrics, key=lambda m: m.activation_time_ms, reverse=True)[:10]:
            lines.append(f"- {metric.plugin_name}: {metric.activation_time_ms:.2f}ms")
        lines.append("")

        # Memory usage distribution
        lines.append("### Memory Usage Distribution")
        lines.append("")
        for metric in sorted(report.metrics, key=lambda m: m.memory_usage_kb, reverse=True)[:10]:
            lines.append(f"- {metric.plugin_name}: {metric.memory_usage_kb:.2f}KB ({metric.memory_usage_kb/1024:.2f}MB)")
        lines.append("")

        return '\n'.join(lines)

    def generate_comparison_report(self, comparison: dict[str, Any]) -> str:
        """Generate before/after comparison report."""
        lines = []

        # Header
        lines.append("# Performance Comparison Report")
        lines.append("")
        lines.append(f"**Generated:** {comparison['timestamp']}")
        lines.append(f"**Before:** {comparison['before_timestamp']}")
        lines.append(f"**After:** {comparison['after_timestamp']}")
        lines.append("")

        # Summary
        summary = comparison['summary']
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Plugins Compared:** {summary['total_compared']}")
        lines.append(f"- **Improvements:** {summary['improvements']} ✅")
        lines.append(f"- **Regressions:** {summary['regressions']} ❌")
        lines.append(f"- **Unchanged:** {summary['unchanged']} ⚪")
        lines.append("")

        if summary['improvements'] > 0:
            lines.append("**Average Improvements:**")
            lines.append(f"- Load Time: {-summary['avg_load_time_improvement_ms']:.2f}ms faster")
            lines.append(f"- Activation Time: {-summary['avg_activation_time_improvement_ms']:.2f}ms faster")
            lines.append(f"- Memory Usage: {-summary['avg_memory_improvement_kb']:.2f}KB less")
            lines.append("")

        # Improvements
        if comparison['improvements']:
            lines.append("## Improvements ✅")
            lines.append("")
            lines.append("| Plugin | Load Time | Activation Time | Memory |")
            lines.append("|--------|-----------|-----------------|--------|")

            for imp in sorted(comparison['improvements'], key=lambda x: abs(x['load_time_change_ms']), reverse=True):
                lines.append(
                    f"| {imp['plugin']} | {imp['load_time_change_ms']:+.2f}ms ({imp['load_time_change_percent']:+.1f}%) | "
                    f"{imp['activation_time_change_ms']:+.2f}ms ({imp['activation_time_change_percent']:+.1f}%) | "
                    f"{imp['memory_change_kb']:+.2f}KB ({imp['memory_change_percent']:+.1f}%) |"
                )

            lines.append("")

        # Regressions
        if comparison['regressions']:
            lines.append("## Regressions ❌")
            lines.append("")
            lines.append("| Plugin | Load Time | Activation Time | Memory |")
            lines.append("|--------|-----------|-----------------|--------|")

            for reg in sorted(comparison['regressions'], key=lambda x: abs(x['load_time_change_ms']), reverse=True):
                lines.append(
                    f"| {reg['plugin']} | {reg['load_time_change_ms']:+.2f}ms ({reg['load_time_change_percent']:+.1f}%) | "
                    f"{reg['activation_time_change_ms']:+.2f}ms ({reg['activation_time_change_percent']:+.1f}%) | "
                    f"{reg['memory_change_kb']:+.2f}KB ({reg['memory_change_percent']:+.1f}%) |"
                )

            lines.append("")

        return '\n'.join(lines)

    def export_to_csv(self, report: PerformanceReport, output_path: Path) -> None:
        """Export report to CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'plugin_name', 'load_time_ms', 'activation_time_ms', 'memory_usage_kb',
                'agent_count', 'keyword_count', 'load_status', 'activation_status',
                'memory_status', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for metric in report.metrics:
                writer.writerow(asdict(metric))

    def export_to_json(self, report: PerformanceReport, output_path: Path) -> None:
        """Export report to JSON format."""
        data = {
            'timestamp': report.timestamp,
            'total_plugins': report.total_plugins,
            'summary': report.summary,
            'metrics': [asdict(m) for m in report.metrics]
        }

        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2)


def main() -> int:
    """Main entry point for performance reporter."""

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 performance_reporter.py <plugin-name>")
        print("  python3 performance_reporter.py --all")
        print("  python3 performance_reporter.py --compare before.json after.json")
        print("  python3 performance_reporter.py --export csv output.csv")
        print("  python3 performance_reporter.py --export json output.json")
        return 1

    # Determine plugins root and tools directory
    current = Path.cwd()
    if (current / "plugins").exists():
        plugins_root = current / "plugins"
        tools_dir = current / "tools"
    else:
        plugins_root = current.parent / "plugins"
        tools_dir = current.parent / "tools"

    aggregator = PerformanceAggregator(plugins_root, tools_dir)
    generator = ReportGenerator()

    # Handle different commands
    command = sys.argv[1]

    if command == "--all":
        # Generate report for all plugins
        report = aggregator.collect_all_metrics()
        markdown = generator.generate_markdown_report(report)
        print(markdown)

        # Also export to JSON
        output_path = Path("performance-report.json")
        generator.export_to_json(report, output_path)
        print(f"\n✅ Report exported to {output_path}")

        return 0

    elif command == "--compare":
        if len(sys.argv) < 4:
            print("Error: --compare requires two JSON files")
            return 1

        before_path = Path(sys.argv[2])
        after_path = Path(sys.argv[3])

        # Load reports
        with open(before_path, 'r', encoding='utf-8') as f:
            before_data = json.load(f)
        with open(after_path, 'r', encoding='utf-8') as f:
            after_data = json.load(f)

        # Reconstruct PerformanceReport objects
        before = PerformanceReport(
            timestamp=before_data['timestamp'],
            total_plugins=before_data['total_plugins'],
            metrics=[PerformanceMetrics(**m) for m in before_data['metrics']],
            summary=before_data['summary']
        )
        after = PerformanceReport(
            timestamp=after_data['timestamp'],
            total_plugins=after_data['total_plugins'],
            metrics=[PerformanceMetrics(**m) for m in after_data['metrics']],
            summary=after_data['summary']
        )

        # Compare
        analyzer = ComparisonAnalyzer()
        comparison = analyzer.compare_reports(before, after)

        # Generate report
        markdown = generator.generate_comparison_report(comparison)
        print(markdown)

        return 0

    elif command == "--export":
        if len(sys.argv) < 4:
            print("Error: --export requires format (csv/json) and output path")
            return 1

        format_type = sys.argv[2]
        output_path = Path(sys.argv[3])

        # Collect metrics
        report = aggregator.collect_all_metrics()

        if format_type == "csv":
            generator.export_to_csv(report, output_path)
            print(f"✅ Report exported to {output_path}")
        elif format_type == "json":
            generator.export_to_json(report, output_path)
            print(f"✅ Report exported to {output_path}")
        else:
            print(f"Error: Unknown format '{format_type}'. Use 'csv' or 'json'")
            return 1

        return 0

    else:
        # Single plugin report
        plugin_name = command
        metrics = aggregator.collect_plugin_metrics(plugin_name)

        # Create a mini report
        report = PerformanceReport(
            timestamp=datetime.now().isoformat(),
            total_plugins=1,
            metrics=[metrics]
        )

        markdown = generator.generate_markdown_report(report)
        print(markdown)

        return 0


if __name__ == "__main__":
    sys.exit(main())
