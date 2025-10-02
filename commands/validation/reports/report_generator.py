#!/usr/bin/env python3
"""
Report Generator - Generates comprehensive validation reports in multiple formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Import validation result from executor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class ReportGenerator:
    """Generates validation reports in multiple formats."""

    def generate(
        self,
        results: List[Any],
        output_dir: Path,
        formats: List[str]
    ) -> Dict[str, Path]:
        """Generate reports in specified formats.

        Args:
            results: List of ValidationResult objects
            output_dir: Output directory
            formats: List of formats ('html', 'json', 'markdown', 'pdf')

        Returns:
            Dictionary mapping format to file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = {}

        if 'json' in formats:
            generated['json'] = self._generate_json(results, output_dir)

        if 'markdown' in formats:
            generated['markdown'] = self._generate_markdown(results, output_dir)

        if 'html' in formats:
            generated['html'] = self._generate_html(results, output_dir)

        return generated

    def _generate_json(self, results: List[Any], output_dir: Path) -> Path:
        """Generate JSON report."""
        output_file = output_dir / "validation_report.json"

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(results),
            'results': [self._result_to_dict(r) for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        return output_file

    def _generate_markdown(self, results: List[Any], output_dir: Path) -> Path:
        """Generate Markdown report."""
        output_file = output_dir / "validation_report.md"

        summary = self._generate_summary(results)

        lines = [
            "# Validation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Executive Summary\n",
            f"- **Total Validations:** {summary['total']}",
            f"- **Successful:** {summary['successful']} ({summary['success_rate']:.1f}%)",
            f"- **Failed:** {summary['failed']}",
            f"- **Warnings:** {summary['warnings']}",
            f"- **Total Duration:** {summary['total_duration']:.1f}s",
            f"- **Average Duration:** {summary['avg_duration']:.1f}s",
            "\n## Results by Project\n"
        ]

        # Group by project
        by_project = {}
        for result in results:
            by_project.setdefault(result.project_name, []).append(result)

        for project, project_results in sorted(by_project.items()):
            lines.append(f"\n### {project}\n")
            lines.append(f"Scenarios run: {len(project_results)}\n")

            for result in project_results:
                status = "✓" if result.success else "✗"
                lines.append(
                    f"- {status} **{result.scenario_name}** "
                    f"({result.duration_seconds:.1f}s)"
                )

                if result.errors:
                    for error in result.errors:
                        lines.append(f"  - Error: {error}")

                if result.warnings:
                    for warning in result.warnings:
                        lines.append(f"  - Warning: {warning}")

        lines.append("\n## Detailed Metrics\n")
        lines.append("| Project | Scenario | Status | Duration | Quality | Coverage |")
        lines.append("|---------|----------|--------|----------|---------|----------|")

        for result in results:
            status = "✓ Pass" if result.success else "✗ Fail"
            quality = result.metrics.get('quality_score', 'N/A')
            coverage = result.metrics.get('test_coverage', 'N/A')

            lines.append(
                f"| {result.project_name} | {result.scenario_name} | "
                f"{status} | {result.duration_seconds:.1f}s | "
                f"{quality} | {coverage} |"
            )

        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        return output_file

    def _generate_html(self, results: List[Any], output_dir: Path) -> Path:
        """Generate HTML report with interactive dashboard."""
        output_file = output_dir / "validation_report.html"

        summary = self._generate_summary(results)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 20px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #777; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .success {{ color: #4CAF50; font-weight: bold; }}
        .failure {{ color: #f44336; font-weight: bold; }}
        .warning {{ color: #ff9800; }}
        .project-section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 5px; }}
        .error-list {{ background: #ffebee; padding: 10px; border-left: 3px solid #f44336; margin: 10px 0; }}
        .warning-list {{ background: #fff3e0; padding: 10px; border-left: 3px solid #ff9800; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Executive Summary</h2>
        <div class="summary">
            <div class="metric">
                <div class="metric-value">{summary['total']}</div>
                <div class="metric-label">Total Validations</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #4CAF50;">{summary['successful']}</div>
                <div class="metric-label">Successful ({summary['success_rate']:.1f}%)</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: #f44336;">{summary['failed']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['total_duration']:.1f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Project</th>
                    <th>Scenario</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Quality Score</th>
                    <th>Coverage</th>
                </tr>
            </thead>
            <tbody>
"""

        for result in results:
            status_class = "success" if result.success else "failure"
            status_text = "✓ Pass" if result.success else "✗ Fail"
            quality = result.metrics.get('quality_score', 'N/A')
            coverage = result.metrics.get('test_coverage', 'N/A')

            html += f"""
                <tr>
                    <td>{result.project_name}</td>
                    <td>{result.scenario_name}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.duration_seconds:.1f}s</td>
                    <td>{quality}</td>
                    <td>{coverage}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(output_file, 'w') as f:
            f.write(html)

        return output_file

    def _generate_summary(self, results: List[Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        warnings = sum(len(r.warnings) for r in results)
        total_duration = sum(r.duration_seconds for r in results)
        avg_duration = total_duration / total if total > 0 else 0
        success_rate = (successful / total * 100) if total > 0 else 0

        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'warnings': warnings,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'success_rate': success_rate
        }

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'project_name': result.project_name,
            'scenario_name': result.scenario_name,
            'success': result.success,
            'duration_seconds': result.duration_seconds,
            'metrics': result.metrics,
            'errors': result.errors,
            'warnings': result.warnings,
            'timestamp': result.timestamp.isoformat()
        }