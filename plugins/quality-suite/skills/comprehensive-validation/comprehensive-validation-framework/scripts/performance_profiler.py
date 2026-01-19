#!/usr/bin/env python3
"""
Performance profiling for Python, JavaScript, and other languages.

Profiles code execution and identifies bottlenecks.

Usage:
    python performance_profiler.py <script-to-profile> [args...]
"""

import argparse
import cProfile
import pstats
import subprocess
import sys
from pathlib import Path
from io import StringIO


class PerformanceProfiler:
    """Profile code execution and identify bottlenecks."""

    def __init__(self, target: str, args: list, top_n: int = 20):
        self.target = Path(target)
        self.args = args
        self.top_n = top_n

    def profile_python(self):
        """Profile Python code."""
        print(f"🔍 Profiling Python: {self.target}")

        # Run cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            # Execute the target script
            with open(self.target) as f:
                code = compile(f.read(), self.target, 'exec')
                exec(code, {'__name__': '__main__'})
        finally:
            profiler.disable()

        # Generate report
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        print("\n" + "="*80)
        print("TOP FUNCTIONS BY CUMULATIVE TIME")
        print("="*80)
        stats.print_stats(self.top_n)

        print("\n" + "="*80)
        print("TOP FUNCTIONS BY TOTAL TIME")
        print("="*80)
        stats.sort_stats('tottime')
        stats.print_stats(self.top_n)

        print(stream.getvalue())

    def profile_javascript(self):
        """Profile JavaScript code."""
        print(f"🔍 Profiling JavaScript: {self.target}")
        print("💡 For detailed profiling, use Chrome DevTools or node --prof")

        # Run with basic time measurement
        cmd = ["node", "--prof", str(self.target)] + self.args
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print("\n💡 Profile saved. Process with: node --prof-process isolate-*.log")

    def run(self):
        """Run profiling based on file type."""
        if self.target.suffix == ".py":
            self.profile_python()
        elif self.target.suffix == ".js":
            self.profile_javascript()
        else:
            print(f"❌ Unsupported file type: {self.target.suffix}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Profile code performance")
    parser.add_argument("target", help="Script to profile")
    parser.add_argument("args", nargs="*", help="Arguments to pass to the script")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top functions to show")

    args = parser.parse_args()

    profiler = PerformanceProfiler(args.target, args.args, args.top)
    profiler.run()


if __name__ == "__main__":
    main()
