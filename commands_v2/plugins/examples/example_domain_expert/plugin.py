#!/usr/bin/env python3
"""
Domain Expert Agent Plugin
===========================

Custom domain expert agent that provides specialized analysis for specific domains.

Can be configured for different domains:
- web: Web development expertise
- data: Data science and analytics
- security: Security analysis
- performance: Performance optimization
- general: General software engineering
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import AgentPlugin, PluginContext, PluginResult
from api.agent_api import AgentAPI


class DomainExpertPlugin(AgentPlugin):
    """Domain expert agent plugin"""

    def load(self) -> bool:
        """Load plugin"""
        self.logger.info(f"Loading {self.metadata.name} plugin")

        # Get domain configuration
        domain = self.get_config('domain', 'general')
        self.logger.info(f"Configured for domain: {domain}")

        return True

    def get_agent_profile(self) -> Dict[str, Any]:
        """Get agent profile"""
        domain = self.get_config('domain', 'general')

        # Domain-specific profiles
        profiles = {
            'web': AgentAPI.create_agent_profile(
                capabilities=['web_development', 'frontend', 'backend'],
                specializations=['React', 'Node.js', 'REST APIs'],
                languages=['javascript', 'typescript', 'python'],
                frameworks=['react', 'express', 'django', 'flask'],
                priority=8
            ),
            'data': AgentAPI.create_agent_profile(
                capabilities=['data_science', 'analytics', 'visualization'],
                specializations=['data pipelines', 'ETL', 'visualization'],
                languages=['python', 'sql', 'r'],
                frameworks=['pandas', 'numpy', 'plotly', 'airflow'],
                priority=8
            ),
            'security': AgentAPI.create_agent_profile(
                capabilities=['security_analysis', 'vulnerability_detection'],
                specializations=['OWASP', 'penetration testing', 'secure coding'],
                languages=['python', 'bash'],
                frameworks=['security_tools'],
                priority=9
            ),
            'performance': AgentAPI.create_agent_profile(
                capabilities=['performance_optimization', 'profiling'],
                specializations=['performance tuning', 'caching', 'optimization'],
                languages=['python', 'c', 'c++'],
                frameworks=['profiling_tools'],
                priority=8
            ),
            'general': AgentAPI.create_agent_profile(
                capabilities=['code_analysis', 'architecture_review'],
                specializations=['best practices', 'design patterns'],
                languages=['python', 'javascript', 'java'],
                frameworks=['general'],
                priority=7
            )
        }

        return profiles.get(domain, profiles['general'])

    def analyze(self, context: PluginContext) -> Dict[str, Any]:
        """
        Perform domain-specific analysis.

        Args:
            context: Analysis context

        Returns:
            Analysis results
        """
        domain = self.get_config('domain', 'general')
        analysis_depth = self.get_config('analysis_depth', 'thorough')

        work_dir = context.work_dir

        # Perform domain-specific analysis
        if domain == 'web':
            return self._analyze_web(work_dir)
        elif domain == 'data':
            return self._analyze_data(work_dir)
        elif domain == 'security':
            return self._analyze_security(work_dir)
        elif domain == 'performance':
            return self._analyze_performance(work_dir)
        else:
            return self._analyze_general(work_dir)

    def execute(self, context: PluginContext) -> PluginResult:
        """Execute agent analysis"""
        try:
            analysis = self.analyze(context)

            findings = analysis.get('findings', [])
            recommendations = analysis.get('recommendations', [])

            return AgentAPI.success_result(
                plugin_name=self.metadata.name,
                findings=findings,
                recommendations=recommendations,
                data=analysis
            )

        except Exception as e:
            return AgentAPI.error_result(
                self.metadata.name,
                f"Analysis failed: {str(e)}"
            )

    def _analyze_web(self, work_dir: Path) -> Dict[str, Any]:
        """Analyze web development project"""
        findings = []
        recommendations = []

        # Check for package.json
        if (work_dir / "package.json").exists():
            findings.append("Node.js project detected")
            recommendations.append("Ensure dependencies are up to date")

        # Check for frontend frameworks
        if (work_dir / "src" / "App.jsx").exists() or (work_dir / "src" / "App.tsx").exists():
            findings.append("React application detected")
            recommendations.append("Consider code splitting for better performance")

        return {
            "domain": "web",
            "findings": findings,
            "recommendations": recommendations,
            "files_analyzed": len(list(work_dir.rglob("*.js"))) + len(list(work_dir.rglob("*.jsx")))
        }

    def _analyze_data(self, work_dir: Path) -> Dict[str, Any]:
        """Analyze data science project"""
        findings = []
        recommendations = []

        # Check for Jupyter notebooks
        notebooks = list(work_dir.rglob("*.ipynb"))
        if notebooks:
            findings.append(f"Found {len(notebooks)} Jupyter notebooks")
            recommendations.append("Consider converting notebooks to scripts for production")

        # Check for data files
        csv_files = list(work_dir.rglob("*.csv"))
        if csv_files:
            findings.append(f"Found {len(csv_files)} CSV files")

        return {
            "domain": "data",
            "findings": findings,
            "recommendations": recommendations,
            "notebooks": len(notebooks),
            "data_files": len(csv_files)
        }

    def _analyze_security(self, work_dir: Path) -> Dict[str, Any]:
        """Analyze security aspects"""
        findings = []
        recommendations = []

        # Check for common security issues
        py_files = list(work_dir.rglob("*.py"))

        for py_file in py_files[:10]:  # Sample first 10
            content = py_file.read_text()

            # Basic security checks
            if "eval(" in content:
                findings.append(f"Potentially unsafe eval() usage in {py_file.name}")
                recommendations.append("Avoid using eval() - use safer alternatives")

            if "pickle.load" in content:
                findings.append(f"Pickle usage detected in {py_file.name}")
                recommendations.append("Be cautious with pickle - only load trusted data")

        return {
            "domain": "security",
            "findings": findings,
            "recommendations": recommendations,
            "files_scanned": len(py_files)
        }

    def _analyze_performance(self, work_dir: Path) -> Dict[str, Any]:
        """Analyze performance aspects"""
        findings = []
        recommendations = []

        # Check for common performance issues
        py_files = list(work_dir.rglob("*.py"))

        has_loops = False
        has_comprehensions = False

        for py_file in py_files[:10]:  # Sample
            content = py_file.read_text()

            if "for " in content and "append(" in content:
                has_loops = True
                findings.append("Found loops with append operations")

            if "[" in content and "for " in content and "]" in content:
                has_comprehensions = True

        if has_loops and not has_comprehensions:
            recommendations.append("Consider using list comprehensions for better performance")

        return {
            "domain": "performance",
            "findings": findings,
            "recommendations": recommendations,
            "files_analyzed": len(py_files)
        }

    def _analyze_general(self, work_dir: Path) -> Dict[str, Any]:
        """General code analysis"""
        findings = []
        recommendations = []

        # Basic project structure analysis
        has_tests = bool(list(work_dir.rglob("test_*.py"))) or (work_dir / "tests").exists()
        has_readme = (work_dir / "README.md").exists()
        has_requirements = (work_dir / "requirements.txt").exists()

        if has_tests:
            findings.append("Test suite detected")
        else:
            recommendations.append("Consider adding tests for better code quality")

        if has_readme:
            findings.append("Documentation present")
        else:
            recommendations.append("Add README.md for project documentation")

        if has_requirements:
            findings.append("Dependencies documented")

        return {
            "domain": "general",
            "findings": findings,
            "recommendations": recommendations,
            "has_tests": has_tests,
            "has_readme": has_readme,
            "has_requirements": has_requirements
        }


def main():
    """Test plugin"""
    from core.plugin_base import PluginMetadata, PluginType

    metadata = PluginMetadata(
        name="domain-expert",
        version="1.0.0",
        plugin_type=PluginType.AGENT,
        description="Domain expert agent",
        author="Test"
    )

    config = {"domain": "web"}

    plugin = DomainExpertPlugin(metadata, config)
    plugin.load()

    print("Agent Profile:")
    profile = plugin.get_agent_profile()
    for key, value in profile.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())