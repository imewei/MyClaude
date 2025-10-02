#!/usr/bin/env python3
"""Run all Phase 4 tests without pytest.

This script directly executes the test modules which have built-in
test runners, avoiding pytest import issues.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_all_phase4_tests():
    """Run all Phase 4 tests and report results."""

    print("\n" + "="*80)
    print(" "*20 + "PHASE 4 TEST SUITE")
    print("="*80)

    total_passed = 0
    total_tests = 0

    # Test 1: Magnus Expansion Solver
    print("\n" + "-"*80)
    print("1. Magnus Expansion Solver Tests")
    print("-"*80)
    try:
        # Import the test module
        sys.path.insert(0, str(Path(__file__).parent / 'tests' / 'solvers'))
        import test_magnus
        passed, total = test_magnus.run_all_tests()
        total_passed += passed
        total_tests += total
        print(f"✓ Magnus tests: {passed}/{total} passed")
    except Exception as e:
        print(f"✗ Magnus tests failed to run: {e}")

    # Test 2: Pontryagin Solver
    print("\n" + "-"*80)
    print("2. Pontryagin Maximum Principle Solver Tests")
    print("-"*80)
    try:
        import test_pontryagin
        passed, total = test_pontryagin.run_all_tests()
        total_passed += passed
        total_tests += total
        print(f"✓ PMP tests: {passed}/{total} passed")
    except Exception as e:
        print(f"✗ PMP tests failed to run: {e}")

    # Summary
    print("\n" + "="*80)
    print(f" "*25 + "TEST SUMMARY")
    print("="*80)
    print(f"Total: {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests if total_tests > 0 else 0:.1f}%)")
    print("="*80 + "\n")

    return total_passed, total_tests


if __name__ == '__main__':
    passed, total = run_all_phase4_tests()
    sys.exit(0 if passed == total else 1)
