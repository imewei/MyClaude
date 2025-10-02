#!/usr/bin/env python3
"""
Reflection Command Executor
Reflection engine with advanced AI reasoning and session analysis
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_executor import CommandExecutor, AgentOrchestrator


class ReflectionExecutor(CommandExecutor):
    """Executor for /reflection command"""

    def __init__(self):
        super().__init__("reflection")
        self.orchestrator = AgentOrchestrator()

    def get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='Reflection engine')
        parser.add_argument('--type', type=str, default='comprehensive',
                          choices=['comprehensive', 'focused', 'scientific',
                                 'instruction', 'session'])
        parser.add_argument('--analysis', type=str, default='deep',
                          choices=['deep', 'surface', 'meta'])
        parser.add_argument('--optimize', type=str,
                          choices=['performance', 'accuracy', 'collaboration', 'innovation'])
        parser.add_argument('--export-insights', action='store_true',
                          help='Export insights to file')
        parser.add_argument('--breakthrough-mode', action='store_true',
                          help='Enable breakthrough analysis')
        parser.add_argument('--implement', action='store_true',
                          help='Implement recommendations')
        parser.add_argument('--agents', type=str, default='orchestrator',
                          choices=['orchestrator', 'scientific', 'quality',
                                 'research', 'all'])
        return parser

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("🤔 REFLECTION ENGINE")
        print("="*60 + "\n")

        try:
            reflection_type = args.get('type', 'comprehensive')
            print(f"🎯 Reflection Type: {reflection_type}")

            # Perform reflection analysis
            print("\n🔍 Analyzing project state...")
            reflection = self._perform_reflection(args)

            # Generate insights
            print("\n💡 Generating insights...")
            insights = self._generate_insights(reflection, args)

            # Export if requested
            if args.get('export_insights'):
                export_path = self._export_insights(insights, args)
                print(f"\n💾 Insights exported to: {export_path}")

            return {
                'success': True,
                'summary': f'Reflection completed: {len(insights)} insights generated',
                'details': self._generate_reflection_report(reflection, insights, args),
                'insights_count': len(insights)
            }

        except Exception as e:
            return {
                'success': False,
                'summary': 'Reflection failed',
                'details': str(e)
            }

    def _perform_reflection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reflection analysis"""
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'type': args.get('type'),
            'project': {
                'path': str(self.work_dir),
                'name': self.work_dir.name,
            },
            'metrics': self._collect_metrics(),
            'observations': self._make_observations(),
        }

        return reflection

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect project metrics"""
        metrics = {
            'files': {
                'python': len(list(self.work_dir.rglob('*.py'))),
                'total': len(list(self.work_dir.rglob('*'))),
            },
            'size': sum(f.stat().st_size for f in self.work_dir.rglob('*')
                       if f.is_file()),
        }

        return metrics

    def _make_observations(self) -> List[str]:
        """Make observations about project"""
        observations = []

        # Check for tests
        if list(self.work_dir.rglob('test_*.py')):
            observations.append("Tests are present")
        else:
            observations.append("No tests found")

        # Check for documentation
        if (self.work_dir / 'README.md').exists():
            observations.append("README exists")
        else:
            observations.append("No README found")

        # Check for CI
        if (self.work_dir / '.github' / 'workflows').exists():
            observations.append("CI/CD configured")

        return observations

    def _generate_insights(self, reflection: Dict[str, Any],
                          args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from reflection"""
        insights = []

        # Analyze observations
        for obs in reflection['observations']:
            if 'No tests' in obs:
                insights.append({
                    'type': 'testing',
                    'priority': 'high',
                    'observation': obs,
                    'recommendation': 'Add test coverage to improve code quality',
                    'action': 'Run /generate-tests command'
                })

            if 'No README' in obs:
                insights.append({
                    'type': 'documentation',
                    'priority': 'medium',
                    'observation': obs,
                    'recommendation': 'Create README for project documentation',
                    'action': 'Run /update-docs --type=readme'
                })

        # Check file count
        py_count = reflection['metrics']['files']['python']
        if py_count > 50:
            insights.append({
                'type': 'organization',
                'priority': 'low',
                'observation': f'Large project with {py_count} Python files',
                'recommendation': 'Consider modularization and code organization',
                'action': 'Review project structure'
            })

        return insights

    def _export_insights(self, insights: List[Dict[str, Any]],
                        args: Dict[str, Any]) -> str:
        """Export insights to file"""
        insights_dir = self.work_dir / '.claude' / 'insights'
        insights_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        insights_file = insights_dir / f'reflection_{timestamp}.json'

        with open(insights_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'insights': insights
            }, f, indent=2)

        return str(insights_file)

    def _generate_reflection_report(self, reflection: Dict[str, Any],
                                   insights: List[Dict[str, Any]],
                                   args: Dict[str, Any]) -> str:
        """Generate reflection report"""
        report = "\nREFLECTION REPORT\n" + "="*60 + "\n\n"

        report += f"Project: {reflection['project']['name']}\n"
        report += f"Analysis Type: {reflection['type']}\n\n"

        # Metrics
        report += "PROJECT METRICS:\n"
        report += f"  • Python Files: {reflection['metrics']['files']['python']}\n"
        report += f"  • Total Files: {reflection['metrics']['files']['total']}\n"
        report += f"  • Size: {reflection['metrics']['size'] / 1024:.1f} KB\n\n"

        # Observations
        report += "OBSERVATIONS:\n"
        for obs in reflection['observations']:
            report += f"  • {obs}\n"
        report += "\n"

        # Insights
        if insights:
            report += f"INSIGHTS ({len(insights)}):\n\n"
            for i, insight in enumerate(insights, 1):
                report += f"{i}. {insight['type'].upper()} (Priority: {insight['priority']})\n"
                report += f"   Observation: {insight['observation']}\n"
                report += f"   Recommendation: {insight['recommendation']}\n"
                report += f"   Action: {insight['action']}\n\n"

        return report


def main():
    executor = ReflectionExecutor()
    return executor.run()


if __name__ == "__main__":
    sys.exit(main())