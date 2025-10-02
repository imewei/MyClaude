#!/usr/bin/env python3
"""
Security audit script for Scientific Computing Agents.

This script performs comprehensive security checks on the codebase,
dependencies, and configuration.
"""

import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityAuditor:
    """Performs security audits on the system."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.findings = []
        self.warnings = []
        self.info = []

    def add_finding(self, category: str, severity: str, message: str, details: str = ""):
        """Add a security finding."""
        finding = {
            'category': category,
            'severity': severity,
            'message': message,
            'details': details
        }

        if severity == 'critical':
            self.findings.append(finding)
            logger.error(f"[CRITICAL] {category}: {message}")
        elif severity == 'warning':
            self.warnings.append(finding)
            logger.warning(f"[WARNING] {category}: {message}")
        else:
            self.info.append(finding)
            logger.info(f"[INFO] {category}: {message}")

    def check_dependency_vulnerabilities(self) -> bool:
        """Check for known vulnerabilities in dependencies."""
        logger.info("Checking dependency vulnerabilities...")

        try:
            # Try to run safety check
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.add_finding(
                    'Dependencies',
                    'info',
                    'No known vulnerabilities in dependencies'
                )
                return True
            else:
                try:
                    vulns = json.loads(result.stdout)
                    for vuln in vulns:
                        self.add_finding(
                            'Dependencies',
                            'critical',
                            f"Vulnerability in {vuln.get('package', 'unknown')}",
                            f"Version: {vuln.get('installed_version')}, "
                            f"CVE: {vuln.get('vulnerability_id')}"
                        )
                except json.JSONDecodeError:
                    self.add_finding(
                        'Dependencies',
                        'warning',
                        'Safety check completed with warnings',
                        result.stdout
                    )
                return False

        except FileNotFoundError:
            self.add_finding(
                'Dependencies',
                'warning',
                'Safety tool not installed - skipping vulnerability check',
                'Install with: pip install safety'
            )
            return True
        except Exception as e:
            self.add_finding(
                'Dependencies',
                'warning',
                f'Dependency check failed: {e}'
            )
            return True

    def check_secrets_in_code(self) -> bool:
        """Check for potential secrets in code."""
        logger.info("Checking for secrets in code...")

        patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
            (r'aws_access_key_id', 'AWS credentials'),
            (r'PRIVATE[_-]?KEY', 'Private key'),
        ]

        issues_found = False

        # Check Python files
        python_files = list(self.project_root.rglob('*.py'))

        for py_file in python_files:
            # Skip test files and examples
            if 'test' in str(py_file) or 'example' in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8').lower()

                for pattern, desc in patterns:
                    import re
                    if re.search(pattern, content, re.IGNORECASE):
                        self.add_finding(
                            'Secrets',
                            'warning',
                            f'Potential {desc} in {py_file.name}',
                            f'File: {py_file}'
                        )
                        issues_found = True
            except Exception as e:
                logger.debug(f"Error checking {py_file}: {e}")

        if not issues_found:
            self.add_finding(
                'Secrets',
                'info',
                'No obvious secrets found in code'
            )

        return not issues_found

    def check_file_permissions(self) -> bool:
        """Check for insecure file permissions."""
        logger.info("Checking file permissions...")

        issues_found = False

        # Check for world-writable files
        try:
            import stat
            for path in self.project_root.rglob('*'):
                if path.is_file():
                    mode = path.stat().st_mode
                    if mode & stat.S_IWOTH:
                        self.add_finding(
                            'Permissions',
                            'warning',
                            f'World-writable file: {path.name}',
                            f'File: {path}'
                        )
                        issues_found = True
        except Exception as e:
            logger.debug(f"Permission check error: {e}")

        if not issues_found:
            self.add_finding(
                'Permissions',
                'info',
                'No insecure file permissions found'
            )

        return not issues_found

    def check_input_validation(self) -> bool:
        """Check for potential input validation issues."""
        logger.info("Checking input validation patterns...")

        dangerous_patterns = [
            ('eval(', 'Use of eval()'),
            ('exec(', 'Use of exec()'),
            ('__import__(', 'Dynamic import'),
            ('pickle.loads(', 'Unsafe deserialization'),
        ]

        issues_found = False
        python_files = list(self.project_root.rglob('*.py'))

        for py_file in python_files:
            # Skip test files
            if 'test' in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding='utf-8')

                for pattern, desc in dangerous_patterns:
                    if pattern in content:
                        self.add_finding(
                            'Input Validation',
                            'warning',
                            f'{desc} in {py_file.name}',
                            f'File: {py_file}'
                        )
                        issues_found = True
            except Exception as e:
                logger.debug(f"Error checking {py_file}: {e}")

        if not issues_found:
            self.add_finding(
                'Input Validation',
                'info',
                'No obvious dangerous patterns found'
            )

        return not issues_found

    def check_configuration_security(self) -> bool:
        """Check security of configuration files."""
        logger.info("Checking configuration security...")

        issues_found = False

        # Check for .env files in git
        gitignore_path = self.project_root / '.gitignore'
        if gitignore_path.exists():
            gitignore = gitignore_path.read_text()
            if '.env' not in gitignore:
                self.add_finding(
                    'Configuration',
                    'warning',
                    '.env not in .gitignore',
                    'Environment files may contain secrets'
                )
                issues_found = True
        else:
            self.add_finding(
                'Configuration',
                'warning',
                'No .gitignore file found'
            )
            issues_found = True

        # Check for exposed configuration files
        sensitive_files = ['.env', 'credentials.json', 'secrets.yaml']
        for filename in sensitive_files:
            if (self.project_root / filename).exists():
                self.add_finding(
                    'Configuration',
                    'warning',
                    f'Sensitive file exists: {filename}',
                    'Ensure it is not committed to version control'
                )
                issues_found = True

        if not issues_found:
            self.add_finding(
                'Configuration',
                'info',
                'Configuration security looks good'
            )

        return not issues_found

    def check_docker_security(self) -> bool:
        """Check Docker security best practices."""
        logger.info("Checking Docker security...")

        dockerfile_path = self.project_root / 'Dockerfile'
        if not dockerfile_path.exists():
            self.add_finding(
                'Docker',
                'info',
                'No Dockerfile found - skipping Docker security check'
            )
            return True

        issues_found = False

        try:
            dockerfile = dockerfile_path.read_text()

            # Check for root user
            if 'USER root' in dockerfile and 'USER' not in dockerfile.split('USER root')[1]:
                self.add_finding(
                    'Docker',
                    'warning',
                    'Container may run as root',
                    'Consider using non-root user'
                )
                issues_found = True
            elif 'USER' in dockerfile:
                self.add_finding(
                    'Docker',
                    'info',
                    'Non-root user configured in Dockerfile'
                )

            # Check for latest tag
            if ':latest' in dockerfile:
                self.add_finding(
                    'Docker',
                    'warning',
                    'Using :latest tag in Dockerfile',
                    'Pin specific versions for reproducibility'
                )
                issues_found = True

        except Exception as e:
            logger.debug(f"Docker check error: {e}")

        return not issues_found

    def run_all_checks(self) -> bool:
        """Run all security checks."""
        logger.info("=" * 60)
        logger.info("Scientific Computing Agents - Security Audit")
        logger.info("=" * 60)

        checks = [
            ('Dependency Vulnerabilities', self.check_dependency_vulnerabilities),
            ('Secrets in Code', self.check_secrets_in_code),
            ('File Permissions', self.check_file_permissions),
            ('Input Validation', self.check_input_validation),
            ('Configuration Security', self.check_configuration_security),
            ('Docker Security', self.check_docker_security),
        ]

        all_passed = True

        for name, check_func in checks:
            try:
                passed = check_func()
                if not passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"Check '{name}' failed with error: {e}")
                all_passed = False

        return all_passed

    def generate_report(self, output_file: str = "security_audit_report.json"):
        """Generate security audit report."""
        report = {
            'timestamp': subprocess.run(
                ['date', '+%Y-%m-%d %H:%M:%S'],
                capture_output=True,
                text=True
            ).stdout.strip(),
            'summary': {
                'critical': len(self.findings),
                'warnings': len(self.warnings),
                'info': len(self.info),
                'total_checks': len(self.findings) + len(self.warnings) + len(self.info)
            },
            'findings': {
                'critical': self.findings,
                'warnings': self.warnings,
                'info': self.info
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("=" * 60)
        logger.info("Security Audit Summary")
        logger.info("=" * 60)
        logger.info(f"Critical issues: {len(self.findings)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info(f"Info: {len(self.info)}")
        logger.info(f"\nReport saved to: {output_file}")

        return report


def main():
    """Run security audit."""
    auditor = SecurityAuditor()
    all_passed = auditor.run_all_checks()
    auditor.generate_report()

    # Exit code based on critical findings
    if auditor.findings:
        logger.error(f"\n❌ Security audit failed with {len(auditor.findings)} critical issue(s)")
        return 1
    elif auditor.warnings:
        logger.warning(f"\n⚠️  Security audit passed with {len(auditor.warnings)} warning(s)")
        return 0
    else:
        logger.info("\n✅ Security audit passed - no issues found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
