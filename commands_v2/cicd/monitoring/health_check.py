#!/usr/bin/env python3
"""
System health monitoring.

This script checks the health of deployed systems and services.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class HealthChecker:
    """Monitor system health."""

    def __init__(self):
        self.checks = []
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "overall_status": "UNKNOWN",
        }

    def check_pypi_package(self, package_name: str) -> Dict:
        """Check if package is available on PyPI."""
        import urllib.request
        import urllib.error

        try:
            url = f"https://pypi.org/pypi/{package_name}/json"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
                return {
                    "name": "PyPI Package",
                    "status": "HEALTHY",
                    "version": data["info"]["version"],
                    "url": data["info"]["package_url"],
                }
        except urllib.error.HTTPError as e:
            return {
                "name": "PyPI Package",
                "status": "UNHEALTHY",
                "error": f"HTTP {e.code}",
            }
        except Exception as e:
            return {
                "name": "PyPI Package",
                "status": "UNHEALTHY",
                "error": str(e),
            }

    def check_documentation_site(self, url: str) -> Dict:
        """Check if documentation site is accessible."""
        import urllib.request
        import urllib.error

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                status_code = response.getcode()
                return {
                    "name": "Documentation Site",
                    "status": "HEALTHY" if status_code == 200 else "DEGRADED",
                    "status_code": status_code,
                    "url": url,
                }
        except Exception as e:
            return {
                "name": "Documentation Site",
                "status": "UNHEALTHY",
                "error": str(e),
            }

    def check_github_repo(self, repo: str) -> Dict:
        """Check if GitHub repository is accessible."""
        import urllib.request
        import urllib.error

        try:
            url = f"https://api.github.com/repos/{repo}"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'HealthChecker/1.0')

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
                return {
                    "name": "GitHub Repository",
                    "status": "HEALTHY",
                    "stars": data.get("stargazers_count", 0),
                    "issues": data.get("open_issues_count", 0),
                    "url": data.get("html_url", ""),
                }
        except Exception as e:
            return {
                "name": "GitHub Repository",
                "status": "UNHEALTHY",
                "error": str(e),
            }

    def check_docker_hub(self, image: str) -> Dict:
        """Check if Docker image is available."""
        import urllib.request
        import urllib.error

        try:
            url = f"https://hub.docker.com/v2/repositories/{image}"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
                return {
                    "name": "Docker Hub",
                    "status": "HEALTHY",
                    "pulls": data.get("pull_count", 0),
                    "stars": data.get("star_count", 0),
                }
        except Exception as e:
            return {
                "name": "Docker Hub",
                "status": "UNHEALTHY",
                "error": str(e),
            }

    def run_all_checks(
        self,
        package_name: str = "claude-commands",
        docs_url: str = "https://docs.claude-commands.dev",
        repo: str = "anthropic/claude-commands",
        docker_image: str = "claudecommands/executor",
    ) -> None:
        """Run all health checks."""
        print("Running health checks...\n")

        # Check PyPI
        print("Checking PyPI package...")
        result = self.check_pypi_package(package_name)
        self.results["checks"].append(result)
        self._print_check_result(result)

        # Check documentation
        print("\nChecking documentation site...")
        result = self.check_documentation_site(docs_url)
        self.results["checks"].append(result)
        self._print_check_result(result)

        # Check GitHub
        print("\nChecking GitHub repository...")
        result = self.check_github_repo(repo)
        self.results["checks"].append(result)
        self._print_check_result(result)

        # Check Docker Hub
        print("\nChecking Docker Hub...")
        result = self.check_docker_hub(docker_image)
        self.results["checks"].append(result)
        self._print_check_result(result)

        # Determine overall status
        statuses = [check["status"] for check in self.results["checks"]]
        if all(s == "HEALTHY" for s in statuses):
            self.results["overall_status"] = "HEALTHY"
        elif any(s == "UNHEALTHY" for s in statuses):
            self.results["overall_status"] = "UNHEALTHY"
        else:
            self.results["overall_status"] = "DEGRADED"

    def _print_check_result(self, result: Dict) -> None:
        """Print a check result."""
        status = result["status"]
        if status == "HEALTHY":
            print(f"  ✓ {result['name']}: {status}")
        elif status == "DEGRADED":
            print(f"  ⚠ {result['name']}: {status}")
        else:
            print(f"  ❌ {result['name']}: {status}")
            if "error" in result:
                print(f"     Error: {result['error']}")

    def print_summary(self) -> None:
        """Print health check summary."""
        print("\n" + "=" * 60)
        print("HEALTH CHECK SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"\nChecks Run: {len(self.results['checks'])}")

        healthy = sum(1 for c in self.results["checks"] if c["status"] == "HEALTHY")
        degraded = sum(1 for c in self.results["checks"] if c["status"] == "DEGRADED")
        unhealthy = sum(1 for c in self.results["checks"] if c["status"] == "UNHEALTHY")

        print(f"  Healthy: {healthy}")
        print(f"  Degraded: {degraded}")
        print(f"  Unhealthy: {unhealthy}")
        print("=" * 60)

    def save_results(self, output_file: Path) -> None:
        """Save results to JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run health checks")
    parser.add_argument(
        "--package",
        default="claude-commands",
        help="PyPI package name",
    )
    parser.add_argument(
        "--docs-url",
        default="https://docs.claude-commands.dev",
        help="Documentation URL",
    )
    parser.add_argument(
        "--repo",
        default="anthropic/claude-commands",
        help="GitHub repository (owner/repo)",
    )
    parser.add_argument(
        "--docker-image",
        default="claudecommands/executor",
        help="Docker Hub image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("health-check-results.json"),
        help="Output file for results",
    )

    args = parser.parse_args()

    checker = HealthChecker()

    try:
        checker.run_all_checks(
            package_name=args.package,
            docs_url=args.docs_url,
            repo=args.repo,
            docker_image=args.docker_image,
        )

        checker.print_summary()
        checker.save_results(args.output)

        if checker.results["overall_status"] == "UNHEALTHY":
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()