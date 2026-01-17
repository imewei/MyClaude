#!/usr/bin/env python3
"""
Comprehensive security scanning across multiple dimensions.

This script runs various security tools:
1. Dependency vulnerability scanning (npm audit, pip-audit, cargo audit)
2. Static analysis security testing - SAST (semgrep, bandit)
3. Secret detection (gitleaks, trufflehog)
4. Code quality security checks

Usage:
    python security_scan.py [--fast] [--json]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import shutil


@dataclass
class SecurityFinding:
    """A security finding from a tool."""
    tool: str
    severity: str  # critical, high, medium, low, info
    category: str  # dependency, sast, secret, etc.
    message: str
    file: Optional[str] = None
    line: Optional[int] = None


class SecurityScanner:
    """Orchestrates security scanning across multiple tools."""

    def __init__(self, fast_mode: bool = False, json_output: bool = False):
        self.fast_mode = fast_mode
        self.json_output = json_output
        self.findings: List[SecurityFinding] = []
        self.root_dir = Path.cwd()

    def command_exists(self, cmd: str) -> bool:
        """Check if a command exists in PATH."""
        return shutil.which(cmd) is not None

    def run_command(self, cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                cwd=self.root_dir,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            print(f"⚠️  Command timed out: {' '.join(cmd)}", file=sys.stderr)
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="Command timed out"
            )
        except Exception as e:
            print(f"⚠️  Command failed: {' '.join(cmd)}: {e}", file=sys.stderr)
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=str(e)
            )

    def scan_npm_dependencies(self):
        """Scan npm dependencies for vulnerabilities."""
        if not (self.root_dir / "package.json").exists():
            return

        if not self.command_exists("npm"):
            print("⚠️  npm not found, skipping npm audit")
            return

        print("  📦 Running npm audit...")
        result = self.run_command(["npm", "audit", "--json"])

        if result.returncode != 0:
            try:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get("vulnerabilities", {})

                for pkg_name, vuln_data in vulnerabilities.items():
                    severity = vuln_data.get("severity", "unknown")
                    via = vuln_data.get("via", [])

                    for v in via:
                        if isinstance(v, dict):
                            self.findings.append(SecurityFinding(
                                tool="npm-audit",
                                severity=severity,
                                category="dependency",
                                message=f"{pkg_name}: {v.get('title', 'Vulnerability found')}",
                                file="package.json"
                            ))

            except json.JSONDecodeError:
                self.findings.append(SecurityFinding(
                    tool="npm-audit",
                    severity="high",
                    category="dependency",
                    message="npm audit found vulnerabilities (parse error)",
                    file="package.json"
                ))

    def scan_python_dependencies(self):
        """Scan Python dependencies for vulnerabilities."""
        has_requirements = (self.root_dir / "requirements.txt").exists()
        has_pyproject = (self.root_dir / "pyproject.toml").exists()

        if not (has_requirements or has_pyproject):
            return

        # Try pip-audit first
        if self.command_exists("pip-audit"):
            print("  📦 Running pip-audit...")
            result = self.run_command(["pip-audit", "--format", "json"])

            if result.returncode != 0:
                try:
                    audit_data = json.loads(result.stdout)
                    for vuln in audit_data.get("vulnerabilities", []):
                        self.findings.append(SecurityFinding(
                            tool="pip-audit",
                            severity="high",
                            category="dependency",
                            message=f"{vuln.get('package')}: {vuln.get('id')} - {vuln.get('description', '')[:100]}",
                            file="requirements.txt" if has_requirements else "pyproject.toml"
                        ))
                except (json.JSONDecodeError, KeyError):
                    pass

        # Try safety as fallback
        elif self.command_exists("safety"):
            print("  📦 Running safety check...")
            result = self.run_command(["safety", "check", "--json"])

            if result.returncode != 0:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        self.findings.append(SecurityFinding(
                            tool="safety",
                            severity="high",
                            category="dependency",
                            message=f"{vuln[0]}: {vuln[3][:100]}",
                            file="requirements.txt"
                        ))
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    def scan_rust_dependencies(self):
        """Scan Rust dependencies for vulnerabilities."""
        if not (self.root_dir / "Cargo.toml").exists():
            return

        if not self.command_exists("cargo"):
            return

        print("  📦 Running cargo audit...")
        result = self.run_command(["cargo", "audit", "--json"])

        if result.returncode != 0:
            try:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get("vulnerabilities", {}).get("list", [])

                for vuln in vulnerabilities:
                    self.findings.append(SecurityFinding(
                        tool="cargo-audit",
                        severity="high",
                        category="dependency",
                        message=f"{vuln.get('package', {}).get('name')}: {vuln.get('advisory', {}).get('title')}",
                        file="Cargo.toml"
                    ))
            except (json.JSONDecodeError, KeyError):
                pass

    def scan_with_semgrep(self):
        """Run semgrep SAST scanning."""
        if not self.command_exists("semgrep"):
            print("⚠️  semgrep not found, skipping SAST scan")
            return

        print("  🔍 Running semgrep...")
        result = self.run_command([
            "semgrep",
            "--config=auto",
            "--json",
            "--quiet",
            "."
        ])

        try:
            semgrep_data = json.loads(result.stdout)
            for finding in semgrep_data.get("results", []):
                severity = finding.get("extra", {}).get("severity", "medium")

                self.findings.append(SecurityFinding(
                    tool="semgrep",
                    severity=severity,
                    category="sast",
                    message=finding.get("extra", {}).get("message", "Security issue detected"),
                    file=finding.get("path"),
                    line=finding.get("start", {}).get("line")
                ))
        except (json.JSONDecodeError, KeyError):
            pass

    def scan_with_bandit(self):
        """Run bandit for Python security issues."""
        # Check if this is a Python project
        has_python = any(self.root_dir.rglob("*.py"))
        if not has_python:
            return

        if not self.command_exists("bandit"):
            print("⚠️  bandit not found, skipping Python SAST")
            return

        print("  🔍 Running bandit...")
        result = self.run_command([
            "bandit",
            "-r", ".",
            "-f", "json",
            "--skip", "B101"  # Skip assert warnings in tests
        ])

        try:
            bandit_data = json.loads(result.stdout)
            for finding in bandit_data.get("results", []):
                severity_map = {
                    "HIGH": "high",
                    "MEDIUM": "medium",
                    "LOW": "low"
                }
                severity = severity_map.get(finding.get("issue_severity", "MEDIUM"), "medium")

                self.findings.append(SecurityFinding(
                    tool="bandit",
                    severity=severity,
                    category="sast",
                    message=finding.get("issue_text", "Security issue"),
                    file=finding.get("filename"),
                    line=finding.get("line_number")
                ))
        except (json.JSONDecodeError, KeyError):
            pass

    def scan_for_secrets(self):
        """Scan for secrets in code."""
        # Try gitleaks
        if self.command_exists("gitleaks"):
            print("  🔑 Running gitleaks...")
            result = self.run_command([
                "gitleaks", "detect",
                "--no-git",
                "--report-format", "json",
                "--report-path", "/dev/stdout"
            ])

            try:
                gitleaks_data = json.loads(result.stdout)
                for finding in gitleaks_data:
                    self.findings.append(SecurityFinding(
                        tool="gitleaks",
                        severity="critical",
                        category="secret",
                        message=f"Secret detected: {finding.get('Description', 'Unknown')}",
                        file=finding.get("File"),
                        line=finding.get("StartLine")
                    ))
            except (json.JSONDecodeError, KeyError):
                pass

        # Try trufflehog as fallback
        elif self.command_exists("trufflehog"):
            print("  🔑 Running trufflehog...")
            result = self.run_command([
                "trufflehog", "filesystem",
                "--json",
                str(self.root_dir)
            ])

            # trufflehog outputs JSONL (one JSON per line)
            for line in result.stdout.split("\n"):
                if line.strip():
                    try:
                        finding = json.loads(line)
                        self.findings.append(SecurityFinding(
                            tool="trufflehog",
                            severity="critical",
                            category="secret",
                            message="Secret detected in repository",
                            file=finding.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("file")
                        ))
                    except (json.JSONDecodeError, KeyError):
                        pass

    def generate_report(self) -> str:
        """Generate a human-readable security report."""
        # Group findings by severity
        critical = [f for f in self.findings if f.severity == "critical"]
        high = [f for f in self.findings if f.severity == "high"]
        medium = [f for f in self.findings if f.severity == "medium"]
        low = [f for f in self.findings if f.severity == "low"]

        report = []
        report.append("\n" + "="*80)
        report.append("SECURITY SCAN RESULTS")
        report.append("="*80)
        report.append(f"\n📊 Summary:")
        report.append(f"  🔴 Critical: {len(critical)}")
        report.append(f"  🟠 High:     {len(high)}")
        report.append(f"  🟡 Medium:   {len(medium)}")
        report.append(f"  🟢 Low:      {len(low)}")
        report.append(f"  📝 Total:    {len(self.findings)}")

        def print_findings(findings: List[SecurityFinding], title: str):
            if not findings:
                return

            report.append(f"\n{'-'*80}")
            report.append(f"{title}")
            report.append(f"{'-'*80}")

            for f in findings:
                location = ""
                if f.file:
                    location = f" [{f.file}"
                    if f.line:
                        location += f":{f.line}"
                    location += "]"

                report.append(f"\n  [{f.tool}] {f.message}{location}")

        print_findings(critical, "🔴 CRITICAL Issues")
        print_findings(high, "🟠 HIGH Severity Issues")

        if not self.fast_mode:
            print_findings(medium, "🟡 MEDIUM Severity Issues")
            print_findings(low, "🟢 LOW Severity Issues")

        report.append("\n" + "="*80)

        if len(critical) + len(high) == 0:
            report.append("✅ No critical or high severity issues found!")
        else:
            report.append(f"⚠️  Found {len(critical) + len(high)} critical/high severity issues!")
            report.append("    Please address these before deployment.")

        report.append("="*80 + "\n")

        return "\n".join(report)

    def run_all_scans(self):
        """Run all security scans."""
        print("\n🔒 Starting Security Scan\n")

        # Phase 1: Dependency vulnerabilities
        print("Phase 1: Dependency Vulnerabilities")
        self.scan_npm_dependencies()
        self.scan_python_dependencies()
        self.scan_rust_dependencies()

        # Phase 2: SAST
        print("\nPhase 2: Static Analysis Security Testing (SAST)")
        self.scan_with_semgrep()
        self.scan_with_bandit()

        # Phase 3: Secret detection
        print("\nPhase 3: Secret Detection")
        self.scan_for_secrets()

        # Generate report
        if self.json_output:
            print(json.dumps([asdict(f) for f in self.findings], indent=2))
        else:
            print(self.generate_report())

        # Exit with error code if critical/high issues found
        critical_high_count = sum(
            1 for f in self.findings if f.severity in ["critical", "high"]
        )
        sys.exit(1 if critical_high_count > 0 else 0)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive security scanning"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip low/medium severity reporting"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    scanner = SecurityScanner(fast_mode=args.fast, json_output=args.json)
    scanner.run_all_scans()


if __name__ == "__main__":
    main()
