#!/usr/bin/env python3
"""
Phase 4 Week 35-36: Comprehensive Test Coverage Analysis

This script analyzes test coverage across all Phase 4 modules to identify gaps
and guide the final test coverage push to 95%+.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Phase 4 modules to analyze
PHASE4_MODULES = {
    "ML Optimal Control": [
        "ml_optimal_control/networks.py",
        "ml_optimal_control/advanced_rl.py",
        "ml_optimal_control/model_based_rl.py",
        "ml_optimal_control/pinn_optimal_control.py",
        "ml_optimal_control/transfer_learning.py",
        "ml_optimal_control/curriculum_learning.py",
        "ml_optimal_control/multitask_metalearning.py",
        "ml_optimal_control/meta_learning.py",
        "ml_optimal_control/robust_control.py",
        "ml_optimal_control/advanced_optimization.py",
        "ml_optimal_control/performance.py",
    ],
    "HPC Integration": [
        "hpc/schedulers.py",
        "hpc/distributed.py",
        "hpc/parallel.py",
    ],
    "GPU Kernels": [
        "gpu_kernels/quantum_evolution.py",
    ],
    "Solvers": [
        "solvers/pontryagin.py",
        "solvers/pontryagin_jax.py",
        "solvers/magnus_expansion.py",
        "solvers/collocation.py",
    ],
}

TEST_DIRECTORIES = [
    "tests/ml",
    "tests/hpc",
    "tests/gpu",
    "tests/solvers",
]


def run_coverage_analysis() -> Dict:
    """Run pytest with coverage and return results."""
    print("=" * 80)
    print("RUNNING TEST COVERAGE ANALYSIS")
    print("=" * 80)

    # Run coverage
    test_paths = " ".join(TEST_DIRECTORIES)
    cmd = f"python3 -m coverage run --source=. -m pytest {test_paths} -v --tb=short"

    print(f"\nCommand: {cmd}\n")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300
    )

    # Get coverage report
    coverage_result = subprocess.run(
        "python3 -m coverage report --skip-empty",
        shell=True,
        capture_output=True,
        text=True
    )

    return {
        "test_output": result.stdout + result.stderr,
        "coverage_report": coverage_result.stdout,
        "return_code": result.returncode
    }


def analyze_test_results(output: str) -> Dict:
    """Parse pytest output to extract test statistics."""
    stats = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "total": 0
    }

    lines = output.split("\n")
    for line in lines:
        if " passed" in line or " failed" in line or " skipped" in line:
            # Parse summary line
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed":
                    stats["passed"] = int(parts[i-1])
                elif part == "failed":
                    stats["failed"] = int(parts[i-1])
                elif part == "skipped":
                    stats["skipped"] = int(parts[i-1])
                elif part == "error" or part == "errors":
                    stats["errors"] = int(parts[i-1])

    stats["total"] = stats["passed"] + stats["failed"] + stats["skipped"] + stats["errors"]

    return stats


def analyze_coverage_report(report: str) -> Dict:
    """Parse coverage report to extract statistics."""
    lines = report.strip().split("\n")

    module_coverage = {}
    total_coverage = 0.0

    for line in lines:
        if "%" in line and not line.startswith("TOTAL"):
            parts = line.split()
            if len(parts) >= 4:
                module = parts[0]
                try:
                    coverage_pct = float(parts[-1].rstrip("%"))
                    module_coverage[module] = coverage_pct
                except ValueError:
                    pass
        elif line.startswith("TOTAL"):
            parts = line.split()
            try:
                total_coverage = float(parts[-1].rstrip("%"))
            except (ValueError, IndexError):
                pass

    return {
        "module_coverage": module_coverage,
        "total_coverage": total_coverage
    }


def identify_coverage_gaps(module_coverage: Dict[str, float]) -> List[Tuple[str, float]]:
    """Identify modules with coverage below target (95%)."""
    TARGET_COVERAGE = 95.0

    gaps = []
    for module, coverage in module_coverage.items():
        if coverage < TARGET_COVERAGE:
            gap = TARGET_COVERAGE - coverage
            gaps.append((module, coverage, gap))

    # Sort by gap size (descending)
    gaps.sort(key=lambda x: x[2], reverse=True)

    return gaps


def count_source_lines() -> Dict[str, int]:
    """Count source lines in Phase 4 modules."""
    line_counts = {}

    for category, modules in PHASE4_MODULES.items():
        category_lines = 0
        for module in modules:
            path = Path(module)
            if path.exists():
                with open(path, 'r') as f:
                    lines = len(f.readlines())
                    line_counts[module] = lines
                    category_lines += lines

        line_counts[f"{category} (total)"] = category_lines

    return line_counts


def generate_coverage_report():
    """Generate comprehensive coverage analysis report."""
    print("\n" + "=" * 80)
    print("PHASE 4 TEST COVERAGE ANALYSIS - WEEK 35-36")
    print("=" * 80 + "\n")

    # Count source lines
    print("📊 Source Code Statistics:")
    print("-" * 80)
    line_counts = count_source_lines()
    total_lines = 0
    for module, count in sorted(line_counts.items()):
        print(f"  {module:60s} {count:6d} lines")
        if "(total)" not in module:
            total_lines += count
    print("-" * 80)
    print(f"  {'TOTAL PHASE 4 CODE':60s} {total_lines:6d} lines")
    print()

    # Run coverage analysis
    print("🧪 Running Test Suite with Coverage...")
    print("-" * 80)

    try:
        results = run_coverage_analysis()
    except subprocess.TimeoutExpired:
        print("ERROR: Test execution timed out (> 5 minutes)")
        return
    except Exception as e:
        print(f"ERROR: Test execution failed: {e}")
        return

    # Analyze test results
    test_stats = analyze_test_results(results["test_output"])

    print(f"\n✓ Tests Passed:   {test_stats['passed']:4d}")
    print(f"✗ Tests Failed:   {test_stats['failed']:4d}")
    print(f"⊘ Tests Skipped:  {test_stats['skipped']:4d}")
    print(f"⚠ Errors:        {test_stats['errors']:4d}")
    print(f"━ Total Tests:    {test_stats['total']:4d}")

    if test_stats['total'] > 0:
        pass_rate = (test_stats['passed'] / test_stats['total']) * 100
        print(f"\n📈 Pass Rate: {pass_rate:.1f}%")

    # Analyze coverage
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80 + "\n")

    coverage_stats = analyze_coverage_report(results["coverage_report"])

    print(f"📊 Overall Coverage: {coverage_stats['total_coverage']:.1f}%\n")

    # Identify gaps
    gaps = identify_coverage_gaps(coverage_stats["module_coverage"])

    if gaps:
        print("🎯 Coverage Gaps (< 95% target):")
        print("-" * 80)
        print(f"  {'Module':55s} {'Coverage':>10s} {'Gap':>10s}")
        print("-" * 80)

        for module, coverage, gap in gaps[:20]:  # Top 20 gaps
            print(f"  {module:55s} {coverage:9.1f}% {gap:9.1f}%")

        print("-" * 80)
        print(f"  Total modules below target: {len(gaps)}")
    else:
        print("✓ All modules meet 95% coverage target!")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR WEEK 35-36")
    print("=" * 80 + "\n")

    if coverage_stats['total_coverage'] < 95.0:
        gap = 95.0 - coverage_stats['total_coverage']
        print(f"🎯 Target: Increase coverage by {gap:.1f}% to reach 95%\n")

        print("Priority Actions:")
        print("  1. Focus on modules with largest gaps (listed above)")
        print("  2. Add edge case tests (boundary conditions, error handling)")
        print("  3. Increase branch coverage (if/else, try/except)")
        print("  4. Add integration tests for complex workflows")
        print("  5. Test error paths and exception handling")
    else:
        print("✓ Coverage target already met!")
        print("\nContinue with:")
        print("  1. Performance regression tests")
        print("  2. Integration test expansion")
        print("  3. Load testing and stress tests")

    # Save detailed report
    print("\n" + "=" * 80)
    print("Saving detailed coverage HTML report...")
    subprocess.run("python3 -m coverage html", shell=True)
    print("✓ Report saved to htmlcov/index.html")

    # Export JSON report
    report_data = {
        "source_lines": line_counts,
        "test_statistics": test_stats,
        "coverage_statistics": coverage_stats,
        "coverage_gaps": [
            {"module": m, "coverage": c, "gap": g}
            for m, c, g in gaps
        ]
    }

    with open("coverage_analysis.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print("✓ JSON report saved to coverage_analysis.json")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        generate_coverage_report()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
