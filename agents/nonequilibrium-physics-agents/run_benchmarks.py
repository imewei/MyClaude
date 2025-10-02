#!/usr/bin/env python3
"""
Phase 4 Week 37-38: Master Benchmark Runner

Comprehensive performance benchmarking suite that runs all benchmarks
and generates detailed performance reports.

Usage:
    python run_benchmarks.py [--all] [--standard] [--scaling] [--gpu] [--report]

Author: Nonequilibrium Physics Agents
Date: 2025-10-01
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import warnings

# Import benchmark modules
from benchmarks.standard_problems import (
    run_standard_benchmark_suite,
    compare_performance
)
from benchmarks.scalability import (
    run_scalability_suite,
    print_scalability_summary
)
from benchmarks.gpu_performance import (
    run_gpu_benchmark_suite,
    print_gpu_summary
)


class BenchmarkRunner:
    """Master benchmark runner.

    Coordinates execution of all benchmark suites and generates reports.

    Parameters
    ----------
    output_dir : Path
        Directory for benchmark results
    """

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            self.output_dir = Path("benchmark_results")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            'standard_problems': {},
            'scalability': {},
            'gpu_performance': {}
        }

    def run_standard_benchmarks(self, problem_sizes: List[int] = None) -> None:
        """Run standard problem benchmarks.

        Parameters
        ----------
        problem_sizes : List[int], optional
            Problem sizes to test
        """
        print("\n" + "=" * 80)
        print("RUNNING STANDARD PROBLEM BENCHMARKS")
        print("=" * 80)

        results = run_standard_benchmark_suite(problem_sizes)

        # Convert to serializable format
        for problem_name, problem_results in results.items():
            self.results['standard_problems'][problem_name] = [
                r.to_dict() for r in problem_results
            ]

        # Print comparison
        compare_performance(results)

    def run_scalability_benchmarks(self) -> None:
        """Run scalability benchmarks."""
        print("\n" + "=" * 80)
        print("RUNNING SCALABILITY BENCHMARKS")
        print("=" * 80)

        results = run_scalability_suite()

        # Convert to serializable format
        if 'strong_scaling' in results:
            self.results['scalability']['strong_scaling'] = [
                r.to_dict() for r in results['strong_scaling']
            ]

        if 'weak_scaling' in results:
            self.results['scalability']['weak_scaling'] = [
                r.to_dict() for r in results['weak_scaling']
            ]

        if 'network_overhead' in results:
            self.results['scalability']['network_overhead'] = results['network_overhead']

        # Print summary
        print_scalability_summary(results)

    def run_gpu_benchmarks(self) -> None:
        """Run GPU performance benchmarks."""
        print("\n" + "=" * 80)
        print("RUNNING GPU PERFORMANCE BENCHMARKS")
        print("=" * 80)

        results = run_gpu_benchmark_suite()

        if not results:
            print("GPU benchmarks not available (JAX not installed or no GPU)")
            return

        # Convert to serializable format
        for operation_name, operation_results in results.items():
            self.results['gpu_performance'][operation_name] = [
                r.to_dict() for r in operation_results
            ]

        # Print summary
        print_gpu_summary(results)

    def save_results(self) -> None:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report.

        Returns
        -------
        str
            Markdown-formatted report
        """
        lines = []

        lines.append("# Performance Benchmark Report")
        lines.append("")
        lines.append(f"**Date**: {self.results['metadata']['timestamp']}")
        lines.append(f"**Version**: {self.results['metadata']['version']}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Standard problems section
        if self.results['standard_problems']:
            lines.append("## Standard Problem Benchmarks")
            lines.append("")

            for problem_name, problem_results in self.results['standard_problems'].items():
                lines.append(f"### {problem_name}")
                lines.append("")
                lines.append("| Problem Size | Time (s) | Cost | Converged |")
                lines.append("|--------------|----------|------|-----------|")

                for result in problem_results:
                    lines.append(f"| {result['problem_size']} | "
                               f"{result['execution_time']:.6f} | "
                               f"{result['final_cost']:.4f} | "
                               f"{result['convergence']} |")

                lines.append("")

                # Scaling analysis
                if len(problem_results) >= 2:
                    first = problem_results[0]
                    last = problem_results[-1]
                    size_ratio = last['problem_size'] / first['problem_size']
                    time_ratio = last['execution_time'] / first['execution_time']

                    lines.append(f"**Scaling**: {size_ratio:.0f}x problem size → "
                               f"{time_ratio:.2f}x execution time")
                    lines.append("")

        # Scalability section
        if self.results['scalability']:
            lines.append("## Scalability Benchmarks")
            lines.append("")

            if 'strong_scaling' in self.results['scalability']:
                lines.append("### Strong Scaling (Fixed Problem Size)")
                lines.append("")
                lines.append("| Workers | Time (s) | Speedup | Efficiency |")
                lines.append("|---------|----------|---------|------------|")

                for result in self.results['scalability']['strong_scaling']:
                    lines.append(f"| {result['num_workers']} | "
                               f"{result['execution_time']:.4f} | "
                               f"{result['speedup']:.2f}x | "
                               f"{result['efficiency']*100:.1f}% |")

                lines.append("")

            if 'weak_scaling' in self.results['scalability']:
                lines.append("### Weak Scaling (Constant Work Per Worker)")
                lines.append("")
                lines.append("| Workers | Problem Size | Time (s) | Efficiency |")
                lines.append("|---------|--------------|----------|------------|")

                for result in self.results['scalability']['weak_scaling']:
                    lines.append(f"| {result['num_workers']} | "
                               f"{result['problem_size']} | "
                               f"{result['execution_time']:.4f} | "
                               f"{result['efficiency']*100:.1f}% |")

                lines.append("")

        # GPU performance section
        if self.results['gpu_performance']:
            lines.append("## GPU Performance Benchmarks")
            lines.append("")

            for operation_name, operation_results in self.results['gpu_performance'].items():
                lines.append(f"### {operation_name.replace('_', ' ').title()}")
                lines.append("")
                lines.append("| Problem Size | CPU (ms) | GPU (ms) | Speedup |")
                lines.append("|--------------|----------|----------|---------|")

                for result in operation_results:
                    lines.append(f"| {result['problem_size']} | "
                               f"{result['cpu_time']*1000:.4f} | "
                               f"{result['gpu_time']*1000:.4f} | "
                               f"{result['speedup']:.2f}x |")

                lines.append("")

                # Average speedup
                if operation_results:
                    avg_speedup = sum(r['speedup'] for r in operation_results) / len(operation_results)
                    lines.append(f"**Average Speedup**: {avg_speedup:.2f}x")
                    lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")

        # Count benchmarks
        n_standard = sum(len(v) for v in self.results['standard_problems'].values())
        n_scaling = sum(len(v) for v in self.results['scalability'].values() if isinstance(v, list))
        n_gpu = sum(len(v) for v in self.results['gpu_performance'].values())

        lines.append(f"- **Standard Problem Benchmarks**: {n_standard}")
        lines.append(f"- **Scalability Benchmarks**: {n_scaling}")
        lines.append(f"- **GPU Benchmarks**: {n_gpu}")
        lines.append(f"- **Total Benchmarks**: {n_standard + n_scaling + n_gpu}")
        lines.append("")

        # Key findings
        lines.append("### Key Findings")
        lines.append("")

        # GPU speedup
        if self.results['gpu_performance']:
            all_speedups = []
            for operation_results in self.results['gpu_performance'].values():
                all_speedups.extend([r['speedup'] for r in operation_results])

            if all_speedups:
                avg_gpu_speedup = sum(all_speedups) / len(all_speedups)
                max_gpu_speedup = max(all_speedups)
                lines.append(f"- **Average GPU Speedup**: {avg_gpu_speedup:.2f}x")
                lines.append(f"- **Maximum GPU Speedup**: {max_gpu_speedup:.2f}x")

        # Parallel efficiency
        if 'strong_scaling' in self.results['scalability']:
            strong_results = self.results['scalability']['strong_scaling']
            if strong_results:
                max_workers_result = max(strong_results, key=lambda r: r['num_workers'])
                lines.append(f"- **Parallel Efficiency ({max_workers_result['num_workers']} workers)**: "
                           f"{max_workers_result['efficiency']*100:.1f}%")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Phase 4 Week 37-38 Benchmarking Suite*")

        return "\n".join(lines)

    def save_report(self, report: str) -> None:
        """Save report to markdown file.

        Parameters
        ----------
        report : str
            Report content
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_report_{timestamp}.md"

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to: {report_file}")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run Phase 4 performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    parser.add_argument('--standard', action='store_true',
                       help='Run standard problem benchmarks')
    parser.add_argument('--scaling', action='store_true',
                       help='Run scalability benchmarks')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU performance benchmarks')
    parser.add_argument('--report', action='store_true',
                       help='Generate and save detailed report')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--problem-sizes', type=int, nargs='+',
                       help='Problem sizes for standard benchmarks (default: 10 50 100)')

    args = parser.parse_args()

    # Default to all if no specific benchmarks selected
    if not (args.standard or args.scaling or args.gpu or args.all):
        args.all = True

    # Create runner
    runner = BenchmarkRunner(output_dir=Path(args.output_dir))

    print("=" * 80)
    print("PHASE 4 WEEK 37-38: PERFORMANCE BENCHMARKING SUITE")
    print("=" * 80)
    print()

    # Run selected benchmarks
    try:
        if args.all or args.standard:
            problem_sizes = args.problem_sizes if args.problem_sizes else [10, 50, 100]
            runner.run_standard_benchmarks(problem_sizes)

        if args.all or args.scaling:
            runner.run_scalability_benchmarks()

        if args.all or args.gpu:
            runner.run_gpu_benchmarks()

        # Save results
        runner.save_results()

        # Generate and save report if requested
        if args.report or args.all:
            print("\n" + "=" * 80)
            print("GENERATING BENCHMARK REPORT")
            print("=" * 80)

            report = runner.generate_report()
            runner.save_report(report)

            # Also print to console
            print("\n" + report)

        print("\n" + "=" * 80)
        print("BENCHMARKING COMPLETE")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
