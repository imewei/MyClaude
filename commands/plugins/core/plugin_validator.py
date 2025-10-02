#!/usr/bin/env python3
"""
Plugin Validator
================

Security and validation system for plugins.

Features:
- Manifest validation
- Dependency security scanning
- Permission validation
- Code analysis (basic)
- Resource limits enforcement
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .plugin_base import PluginMetadata, BasePlugin


@dataclass
class ValidationIssue:
    """Validation issue"""
    severity: str  # error, warning, info
    category: str  # manifest, dependencies, code, permissions
    message: str
    details: Optional[str] = None


class PluginValidator:
    """
    Plugin validation and security system.

    Validates:
    - Plugin manifest
    - Dependencies
    - Permissions
    - Code (basic static analysis)
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Security patterns
        self.dangerous_patterns = [
            (r'eval\s*\(', 'Use of eval() is dangerous'),
            (r'exec\s*\(', 'Use of exec() is dangerous'),
            (r'__import__\s*\(', 'Dynamic imports should be reviewed'),
            (r'open\s*\([^)]*[\'"]w', 'File write operations detected'),
            (r'subprocess\.', 'Subprocess execution detected'),
            (r'os\.system', 'OS system calls detected'),
        ]

    def validate_plugin(
        self,
        plugin: BasePlugin,
        strict: bool = False
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate a plugin instance.

        Args:
            plugin: Plugin instance to validate
            strict: Strict validation mode

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        # Validate metadata
        issues.extend(self.validate_metadata(plugin.metadata))

        # Validate configuration
        issues.extend(self.validate_config(plugin.config, plugin.metadata.config_schema))

        # Validate permissions
        issues.extend(self.validate_permissions(plugin.metadata.permissions))

        # Check for errors
        errors = [i for i in issues if i.severity == 'error']
        is_valid = len(errors) == 0

        if strict:
            warnings = [i for i in issues if i.severity == 'warning']
            is_valid = is_valid and len(warnings) == 0

        return is_valid, issues

    def validate_metadata(self, metadata: PluginMetadata) -> List[ValidationIssue]:
        """Validate plugin metadata"""
        issues = []

        # Required fields
        if not metadata.name:
            issues.append(ValidationIssue(
                severity='error',
                category='manifest',
                message='Plugin name is required'
            ))

        if not metadata.version:
            issues.append(ValidationIssue(
                severity='error',
                category='manifest',
                message='Plugin version is required'
            ))

        # Name format
        if metadata.name and not re.match(r'^[a-z0-9-]+$', metadata.name):
            issues.append(ValidationIssue(
                severity='error',
                category='manifest',
                message='Plugin name must be kebab-case (lowercase letters, numbers, hyphens)'
            ))

        # Version format
        if metadata.version and not re.match(r'^\d+\.\d+\.\d+', metadata.version):
            issues.append(ValidationIssue(
                severity='warning',
                category='manifest',
                message='Version should follow semantic versioning (e.g., 1.0.0)'
            ))

        # Description
        if not metadata.description or len(metadata.description) < 10:
            issues.append(ValidationIssue(
                severity='warning',
                category='manifest',
                message='Description should be at least 10 characters'
            ))

        # Author
        if not metadata.author:
            issues.append(ValidationIssue(
                severity='warning',
                category='manifest',
                message='Author field is recommended'
            ))

        # License
        if not metadata.license:
            issues.append(ValidationIssue(
                severity='info',
                category='manifest',
                message='License field is recommended'
            ))

        return issues

    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate plugin configuration against schema"""
        issues = []

        if not schema:
            return issues

        # Check required fields
        for key, spec in schema.items():
            if isinstance(spec, dict) and spec.get('required', False):
                if key not in config:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='config',
                        message=f'Required configuration key missing: {key}'
                    ))

        return issues

    def validate_permissions(self, permissions: List[str]) -> List[ValidationIssue]:
        """Validate requested permissions"""
        issues = []

        valid_permissions = ['read', 'write', 'network', 'execute', 'admin']
        dangerous_permissions = ['execute', 'admin']

        for perm in permissions:
            if perm not in valid_permissions:
                issues.append(ValidationIssue(
                    severity='error',
                    category='permissions',
                    message=f'Invalid permission: {perm}'
                ))

            if perm in dangerous_permissions:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='permissions',
                    message=f'Dangerous permission requested: {perm}',
                    details='This permission requires user approval'
                ))

        return issues

    def validate_dependencies(self, dependencies: List[str]) -> List[ValidationIssue]:
        """Validate and check dependencies for known vulnerabilities"""
        issues = []

        # Known vulnerable packages (example list)
        vulnerable_packages = {
            'pillow': ['<8.1.1', 'CVE-2021-25287'],
            'requests': ['<2.20.0', 'CVE-2018-18074'],
        }

        for dep in dependencies:
            # Parse dependency
            if '==' in dep:
                pkg_name, version = dep.split('==', 1)
            elif '>=' in dep:
                pkg_name = dep.split('>=')[0]
                version = None
            else:
                pkg_name = dep
                version = None

            # Check for known vulnerabilities
            if pkg_name in vulnerable_packages:
                vuln_version, cve = vulnerable_packages[pkg_name]
                if version and self._version_matches(version, vuln_version):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='dependencies',
                        message=f'Vulnerable dependency: {pkg_name} {version}',
                        details=f'Known vulnerability: {cve}'
                    ))

        return issues

    def validate_code(self, code_path: Path) -> List[ValidationIssue]:
        """Perform basic static code analysis"""
        issues = []

        if not code_path.exists():
            return issues

        try:
            code = code_path.read_text()

            # Check for dangerous patterns
            for pattern, message in self.dangerous_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='code',
                        message=message,
                        details=f'Found at line {line_num}'
                    ))

        except Exception as e:
            self.logger.error(f"Error analyzing code: {e}")

        return issues

    def _version_matches(self, version: str, constraint: str) -> bool:
        """Check if version matches constraint"""
        # Simplified version comparison
        if constraint.startswith('<'):
            target = constraint[1:]
            return version < target
        elif constraint.startswith('>'):
            target = constraint[1:]
            return version > target
        return False

    def format_issues(self, issues: List[ValidationIssue]) -> str:
        """Format validation issues for display"""
        if not issues:
            return "✅ No validation issues found"

        lines = []

        # Group by severity
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        info = [i for i in issues if i.severity == 'info']

        if errors:
            lines.append("❌ Errors:")
            for issue in errors:
                lines.append(f"  - [{issue.category}] {issue.message}")
                if issue.details:
                    lines.append(f"    {issue.details}")

        if warnings:
            lines.append("\n⚠️  Warnings:")
            for issue in warnings:
                lines.append(f"  - [{issue.category}] {issue.message}")
                if issue.details:
                    lines.append(f"    {issue.details}")

        if info:
            lines.append("\nℹ️  Info:")
            for issue in info:
                lines.append(f"  - [{issue.category}] {issue.message}")

        return "\n".join(lines)


def main():
    """Test validator"""
    logging.basicConfig(level=logging.INFO)

    print("Plugin Validator")
    print("================\n")

    from plugin_base import PluginMetadata, PluginType

    # Create test metadata
    metadata = PluginMetadata(
        name="test-plugin",
        version="1.0.0",
        plugin_type=PluginType.COMMAND,
        description="Test plugin for validation",
        author="Test Author",
        permissions=['read', 'write']
    )

    validator = PluginValidator()

    # Validate metadata
    issues = validator.validate_metadata(metadata)

    print(validator.format_issues(issues))

    return 0


if __name__ == "__main__":
    sys.exit(main())