#!/usr/bin/env python3
"""
Security Scanner Plugin
=======================

Scans Python code for common security vulnerabilities.

Checks for:
- SQL injection vulnerabilities
- XSS vulnerabilities
- Unsafe eval/exec usage
- Hardcoded secrets
- Insecure random usage
- Pickle vulnerabilities
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import CommandPlugin, PluginContext, PluginResult
from api.command_api import CommandAPI


class SecurityScannerPlugin(CommandPlugin):
    """Security vulnerability scanner"""

    # Security patterns
    PATTERNS = {
        'sql_injection': [
            (r'execute\s*\([^)]*%s', 'Potential SQL injection via string formatting'),
            (r'execute\s*\([^)]*\+\s*', 'Potential SQL injection via string concatenation'),
        ],
        'xss': [
            (r'render_template_string\([^)]*request\.', 'Potential XSS via unsafe template rendering'),
        ],
        'unsafe_eval': [
            (r'\beval\s*\(', 'Unsafe use of eval()'),
            (r'\bexec\s*\(', 'Unsafe use of exec()'),
        ],
        'secrets': [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
        ],
        'random': [
            (r'import\s+random\b', 'Use of insecure random (use secrets module for security)'),
        ],
        'pickle': [
            (r'pickle\.loads?\s*\(', 'Unsafe pickle usage (can execute arbitrary code)'),
        ]
    }

    def load(self) -> bool:
        """Load plugin"""
        self.logger.info(f"Loading {self.metadata.name} plugin")
        return True

    def execute(self, context: PluginContext) -> PluginResult:
        """Execute security scan"""
        work_dir = context.work_dir

        # Configuration
        check_sql = self.get_config('check_sql_injection', True)
        check_xss = self.get_config('check_xss', True)
        check_unsafe_eval = self.get_config('check_unsafe_eval', True)

        # Find Python files
        py_files = list(work_dir.rglob("*.py"))

        if not py_files:
            return CommandAPI.error_result(
                self.metadata.name,
                "No Python files found"
            )

        # Scan files
        vulnerabilities = []

        for py_file in py_files:
            file_vulns = self._scan_file(py_file, work_dir)
            vulnerabilities.extend(file_vulns)

        # Categorize by severity
        critical = [v for v in vulnerabilities if v['severity'] == 'critical']
        high = [v for v in vulnerabilities if v['severity'] == 'high']
        medium = [v for v in vulnerabilities if v['severity'] == 'medium']
        low = [v for v in vulnerabilities if v['severity'] == 'low']

        # Generate recommendations
        recommendations = []
        if critical:
            recommendations.append(
                f"Found {len(critical)} critical vulnerabilities - immediate action required!"
            )
        if high:
            recommendations.append(
                f"Found {len(high)} high severity vulnerabilities - should be fixed soon"
            )
        if medium:
            recommendations.append(
                f"Found {len(medium)} medium severity vulnerabilities - plan to fix"
            )

        if not vulnerabilities:
            recommendations.append("No security vulnerabilities detected")

        # Return results
        return CommandAPI.success_result(
            plugin_name=self.metadata.name,
            data={
                "files_scanned": len(py_files),
                "total_vulnerabilities": len(vulnerabilities),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low),
                "vulnerabilities": vulnerabilities[:20],  # First 20
                "recommendations": recommendations
            },
            message=f"Scanned {len(py_files)} files, found {len(vulnerabilities)} vulnerabilities"
        )

    def _scan_file(self, file_path: Path, base_dir: Path) -> List[Dict[str, Any]]:
        """Scan a single file for vulnerabilities"""
        vulnerabilities = []

        try:
            content = file_path.read_text()
            lines = content.split('\n')

            # Check each pattern category
            for category, patterns in self.PATTERNS.items():
                for pattern, description in patterns:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        line_num = content[:match.start()].count('\n') + 1

                        # Determine severity
                        severity = self._get_severity(category)

                        vulnerabilities.append({
                            'file': str(file_path.relative_to(base_dir)),
                            'line': line_num,
                            'category': category,
                            'severity': severity,
                            'description': description,
                            'code_snippet': lines[line_num - 1].strip()
                        })

        except Exception as e:
            self.logger.warning(f"Error scanning {file_path}: {e}")

        return vulnerabilities

    def _get_severity(self, category: str) -> str:
        """Get severity level for vulnerability category"""
        severity_map = {
            'sql_injection': 'critical',
            'xss': 'high',
            'unsafe_eval': 'high',
            'secrets': 'critical',
            'random': 'medium',
            'pickle': 'high'
        }
        return severity_map.get(category, 'low')

    def get_command_info(self) -> dict:
        """Get command information"""
        return {
            "name": "security-scanner",
            "description": "Scan code for security vulnerabilities",
            "usage": "/security-scanner",
            "examples": [
                {
                    "command": "/security-scanner",
                    "description": "Scan codebase for vulnerabilities"
                }
            ]
        }


def main():
    """Test plugin"""
    from core.plugin_base import PluginMetadata, PluginType

    metadata = PluginMetadata(
        name="security-scanner",
        version="1.0.0",
        plugin_type=PluginType.COMMAND,
        description="Security scanner",
        author="Test"
    )

    plugin = SecurityScannerPlugin(metadata)
    plugin.load()

    context = PluginContext(
        plugin_name="security-scanner",
        command_name="security-scanner",
        work_dir=Path.cwd(),
        config={},
        framework_version="2.0.0"
    )

    result = plugin.execute(context)
    print(CommandAPI.format_output(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())