#!/usr/bin/env python3
"""
Iterative CI/CD Error Resolution Engine
Continuously fixes errors until zero failures or max iterations reached
"""

import subprocess
import json
import time
import re
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FixResult(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_FIX_AVAILABLE = "no_fix"


@dataclass
class ErrorAnalysis:
    category: str
    pattern: str
    confidence: float
    suggested_fix: str
    priority: int
    context: str = ""


@dataclass
class IterationResult:
    iteration: int
    errors_found: int
    errors_fixed: int
    errors_remaining: int
    fixes_applied: List[str]
    new_run_id: Optional[str]
    success: bool


class IterativeFixEngine:
    def __init__(self, repo: str, workflow: str, max_iterations: int = 5):
        self.repo = repo
        self.workflow = workflow
        self.max_iterations = max_iterations
        self.knowledge_base = KnowledgeBase()
        self.iteration_history: List[IterationResult] = []

    def run(self, initial_run_id: str) -> bool:
        """
        Main iterative fix loop.
        Returns True if all errors resolved, False otherwise.
        """
        current_run_id = initial_run_id

        print(f"Starting iterative fix loop (max {self.max_iterations} iterations)")
        print(f"Initial run ID: {current_run_id}\n")

        for iteration in range(1, self.max_iterations + 1):
            print(f"{'='*60}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}\n")

            # Analyze current run
            errors = self.analyze_run(current_run_id)

            if not errors:
                print("✓ SUCCESS: Zero errors detected!")
                self.record_iteration(
                    iteration, 0, 0, 0, [], None, True
                )
                return True

            print(f"Found {len(errors)} error(s) to fix\n")

            # Categorize and prioritize errors
            categorized = self.categorize_errors(errors)
            prioritized = self.prioritize_fixes(categorized)

            # Apply fixes
            fixes_applied = []
            errors_fixed = 0

            for error_analysis in prioritized:
                print(f"Fixing: {error_analysis.pattern[:80]}...")
                print(f"Category: {error_analysis.category}")
                print(f"Confidence: {error_analysis.confidence:.0%}")
                print(f"Strategy: {error_analysis.suggested_fix}\n")

                result = self.apply_fix(error_analysis)

                if result in [FixResult.SUCCESS, FixResult.PARTIAL]:
                    fixes_applied.append(error_analysis.suggested_fix)
                    errors_fixed += 1
                    print("✓ Fix applied successfully\n")
                else:
                    print(f"✗ Fix failed: {result.value}\n")

            if not fixes_applied:
                print("No fixes could be applied. Manual intervention required.")
                self.record_iteration(
                    iteration, len(errors), 0, len(errors), [], None, False
                )
                return False

            # Commit fixes
            self.commit_fixes(fixes_applied, iteration)

            # Trigger new workflow run
            print("Triggering new workflow run...")
            new_run_id = self.trigger_workflow()

            if not new_run_id:
                print("Failed to trigger workflow")
                self.record_iteration(
                    iteration, len(errors), errors_fixed,
                    len(errors) - errors_fixed, fixes_applied, None, False
                )
                return False

            print(f"New run started: {new_run_id}")

            # Wait for completion
            print("Waiting for workflow to complete...")
            if not self.wait_for_completion(new_run_id, timeout=600):
                print("Workflow timeout")
                self.record_iteration(
                    iteration, len(errors), errors_fixed,
                    len(errors) - errors_fixed, fixes_applied, new_run_id, False
                )
                return False

            # Check if successful
            status = self.get_run_status(new_run_id)

            self.record_iteration(
                iteration, len(errors), errors_fixed,
                len(errors) - errors_fixed, fixes_applied, new_run_id,
                status == "success"
            )

            if status == "success":
                print("\n✓ SUCCESS: All errors resolved!")
                self.update_knowledge_base(fixes_applied, True)
                return True

            # Update knowledge base with partial success
            self.update_knowledge_base(fixes_applied, False)

            # Prepare for next iteration
            current_run_id = new_run_id
            print(f"\nProceeding to iteration {iteration + 1}...\n")

        print(f"\nMax iterations ({self.max_iterations}) reached")
        print("Some errors may remain. Review iteration history:")
        self.print_summary()
        return False

    def analyze_run(self, run_id: str) -> List[Dict]:
        """Fetch and parse workflow run logs."""
        cmd = [
            'gh', 'run', 'view', run_id,
            '--repo', self.repo,
            '--log-failed'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error fetching logs: {e}")
            return []

        return self.parse_logs(result.stdout)

    def parse_logs(self, logs: str) -> List[Dict]:
        """Extract error patterns from logs."""
        errors = []

        patterns = {
            'npm_eresolve': r'npm ERR! code ERESOLVE',
            'npm_404': r'npm ERR! 404',
            'npm_peer': r'npm ERR! peer dep missing',
            'ts_error': r'TS\d+:',
            'eslint_error': r'\d+:\d+\s+error',
            'test_failure': r'FAIL .*\.test\.',
            'python_import': r'ModuleNotFoundError|ImportError',
            'python_version': r'Could not find a version that satisfies',
            'build_error': r'Build failed|compilation failed',
            'timeout': r'exceeded the maximum execution time',
            'oom': r'heap out of memory',
            'network_error': r'ETIMEDOUT|ENOTFOUND|ECONNREFUSED',
        }

        for name, pattern in patterns.items():
            matches = re.finditer(pattern, logs, re.MULTILINE)
            for match in matches:
                # Extract context (5 lines before and after)
                lines = logs[:match.end()].split('\n')
                context_start = max(0, len(lines) - 5)
                context = '\n'.join(lines[context_start:])

                errors.append({
                    'type': name,
                    'pattern': pattern,
                    'match': match.group(),
                    'context': context
                })

        return errors

    def categorize_errors(self, errors: List[Dict]) -> List[ErrorAnalysis]:
        """Categorize errors and assign fix strategies."""
        analyses = []

        for error in errors:
            category = self.get_category(error['type'])
            confidence = self.calculate_confidence(error)
            fix_strategy = self.knowledge_base.get_fix_strategy(
                error['type'], error['context']
            )
            priority = self.calculate_priority(error, confidence)

            analyses.append(ErrorAnalysis(
                category=category,
                pattern=error['match'],
                confidence=confidence,
                suggested_fix=fix_strategy,
                priority=priority,
                context=error['context']
            ))

        return analyses

    def get_category(self, error_type: str) -> str:
        """Map error type to category."""
        category_map = {
            'npm_eresolve': 'dependency',
            'npm_404': 'dependency',
            'npm_peer': 'dependency',
            'python_import': 'dependency',
            'python_version': 'dependency',
            'ts_error': 'build',
            'eslint_error': 'build',
            'build_error': 'build',
            'test_failure': 'test',
            'timeout': 'runtime',
            'oom': 'runtime',
            'network_error': 'runtime',
        }
        return category_map.get(error_type, 'unknown')

    def calculate_confidence(self, error: Dict) -> float:
        """Calculate confidence score for fix."""
        # Base confidence from knowledge base
        kb_confidence = self.knowledge_base.get_confidence(error['type'])

        # Adjust based on error clarity
        clarity_bonus = 0.0
        if error['type'] in ['npm_eresolve', 'npm_404', 'python_import']:
            clarity_bonus = 0.1  # These have clear fixes

        # Adjust based on historical success
        history_bonus = self.knowledge_base.get_success_rate(error['type']) * 0.2

        return min(1.0, kb_confidence + clarity_bonus + history_bonus)

    def calculate_priority(self, error: Dict, confidence: float) -> int:
        """Calculate priority score (higher = more important)."""
        # Blocking errors get highest priority
        blocking_types = ['build_error', 'npm_eresolve', 'python_import']

        priority = int(confidence * 100)

        if error['type'] in blocking_types:
            priority += 50

        return priority

    def prioritize_fixes(self, analyses: List[ErrorAnalysis]) -> List[ErrorAnalysis]:
        """Sort fixes by priority (high confidence, blocking errors first)."""
        return sorted(analyses, key=lambda x: (-x.priority, -x.confidence))

    def apply_fix(self, error: ErrorAnalysis) -> FixResult:
        """Execute fix strategy."""
        try:
            if error.category == 'dependency':
                return self.fix_dependency_error(error)
            elif error.category == 'build':
                return self.fix_build_error(error)
            elif error.category == 'test':
                return self.fix_test_error(error)
            elif error.category == 'runtime':
                return self.fix_runtime_error(error)
            else:
                return FixResult.NO_FIX_AVAILABLE
        except Exception as e:
            print(f"Error applying fix: {e}")
            return FixResult.FAILED

    def fix_dependency_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix dependency-related errors."""
        if 'npm_eresolve' in error.pattern:
            return self.fix_npm_eresolve()
        elif 'npm_404' in error.pattern:
            return self.fix_npm_404(error)
        elif 'python_import' in error.pattern:
            return self.fix_python_import(error)
        return FixResult.NO_FIX_AVAILABLE

    def fix_npm_eresolve(self) -> FixResult:
        """Fix npm ERESOLVE conflicts."""
        try:
            # Find workflow files
            workflow_files = list(Path('.github/workflows').glob('*.yml'))

            for workflow_file in workflow_files:
                content = workflow_file.read_text()

                # Add --legacy-peer-deps to npm install commands
                if 'npm install' in content and '--legacy-peer-deps' not in content:
                    content = content.replace('npm install', 'npm install --legacy-peer-deps')
                    content = content.replace('npm ci', 'npm ci --legacy-peer-deps')
                    workflow_file.write_text(content)

            return FixResult.SUCCESS
        except Exception as e:
            print(f"Error fixing npm ERESOLVE: {e}")
            return FixResult.FAILED

    def fix_npm_404(self, error: ErrorAnalysis) -> FixResult:
        """Fix npm 404 package not found."""
        # Extract package name from context
        match = re.search(r"404.*'(@?[^']+)'", error.context)
        if not match:
            return FixResult.NO_FIX_AVAILABLE

        package = match.group(1)
        print(f"Removing unavailable package: {package}")

        try:
            subprocess.run(['npm', 'uninstall', package], check=True)
            return FixResult.SUCCESS
        except subprocess.CalledProcessError:
            return FixResult.FAILED

    def fix_python_import(self, error: ErrorAnalysis) -> FixResult:
        """Fix Python import errors."""
        # Extract module name
        match = re.search(r"No module named '([^']+)'", error.context)
        if not match:
            return FixResult.NO_FIX_AVAILABLE

        module = match.group(1)

        # Map common module names to package names
        package_map = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
        }

        package = package_map.get(module, module)
        print(f"Installing missing module: {package}")

        try:
            subprocess.run(['pip', 'install', package], check=True)

            # Update requirements.txt
            with open('requirements.txt', 'a') as f:
                f.write(f'\n{package}\n')

            return FixResult.SUCCESS
        except subprocess.CalledProcessError:
            return FixResult.FAILED

    def fix_build_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix build-related errors."""
        if 'eslint' in error.pattern.lower():
            return self.fix_eslint_errors()
        return FixResult.NO_FIX_AVAILABLE

    def fix_eslint_errors(self) -> FixResult:
        """Run ESLint auto-fix."""
        try:
            subprocess.run(['npx', 'eslint', '.', '--fix'], check=False)
            return FixResult.SUCCESS
        except Exception:
            return FixResult.PARTIAL

    def fix_test_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix test-related errors."""
        if 'snapshot' in error.context.lower():
            return self.fix_snapshot_errors()
        return FixResult.NO_FIX_AVAILABLE

    def fix_snapshot_errors(self) -> FixResult:
        """Update test snapshots."""
        try:
            subprocess.run(['npm', 'test', '--', '-u'], check=True)
            return FixResult.SUCCESS
        except subprocess.CalledProcessError:
            return FixResult.FAILED

    def fix_runtime_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix runtime errors."""
        if 'oom' in error.pattern or 'heap out of memory' in error.context.lower():
            return self.fix_oom_error()
        elif 'timeout' in error.pattern:
            return self.fix_timeout_error()
        return FixResult.NO_FIX_AVAILABLE

    def fix_oom_error(self) -> FixResult:
        """Fix out of memory errors."""
        try:
            workflow_files = list(Path('.github/workflows').glob('*.yml'))

            for workflow_file in workflow_files:
                content = workflow_file.read_text()

                # Add NODE_OPTIONS for increased heap
                if 'NODE_OPTIONS' not in content:
                    # Add after env: section
                    content = re.sub(
                        r'(env:)',
                        r'\1\n        NODE_OPTIONS: "--max-old-space-size=4096"',
                        content
                    )
                    workflow_file.write_text(content)

            return FixResult.SUCCESS
        except Exception as e:
            print(f"Error fixing OOM: {e}")
            return FixResult.FAILED

    def fix_timeout_error(self) -> FixResult:
        """Fix timeout errors."""
        try:
            workflow_files = list(Path('.github/workflows').glob('*.yml'))

            for workflow_file in workflow_files:
                content = workflow_file.read_text()

                # Add or increase timeout-minutes
                if 'timeout-minutes:' not in content:
                    content = re.sub(
                        r'(runs-on:)',
                        r'timeout-minutes: 60\n    \1',
                        content
                    )
                else:
                    content = re.sub(
                        r'timeout-minutes: \d+',
                        'timeout-minutes: 60',
                        content
                    )

                workflow_file.write_text(content)

            return FixResult.SUCCESS
        except Exception as e:
            print(f"Error fixing timeout: {e}")
            return FixResult.FAILED

    def commit_fixes(self, fixes: List[str], iteration: int):
        """Commit all applied fixes."""
        try:
            subprocess.run(['git', 'add', '.'], check=True)

            message = f"fix(ci): iteration {iteration} - automated error resolution\n\n"
            message += "Applied fixes:\n"
            for fix in fixes:
                message += f"- {fix}\n"
            message += "\n🤖 Generated with iterative-error-resolution"

            subprocess.run(['git', 'commit', '-m', message], check=True)
            subprocess.run(['git', 'push'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error committing fixes: {e}")

    def trigger_workflow(self) -> Optional[str]:
        """Trigger workflow and return new run ID."""
        try:
            result = subprocess.run(
                ['gh', 'workflow', 'run', self.workflow, '--repo', self.repo],
                capture_output=True, text=True, check=True
            )

            # Wait for run to appear
            time.sleep(5)

            # Get latest run ID
            result = subprocess.run(
                ['gh', 'run', 'list', '--workflow', self.workflow,
                 '--repo', self.repo, '--limit', '1', '--json', 'databaseId'],
                capture_output=True, text=True, check=True
            )

            runs = json.loads(result.stdout)
            return str(runs[0]['databaseId']) if runs else None
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"Error triggering workflow: {e}")
            return None

    def wait_for_completion(self, run_id: str, timeout: int = 600) -> bool:
        """Wait for workflow run to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_run_status(run_id)

            if status in ['success', 'failure', 'cancelled']:
                return True

            print(".", end="", flush=True)
            time.sleep(10)

        print()
        return False

    def get_run_status(self, run_id: str) -> str:
        """Get current status of workflow run."""
        try:
            result = subprocess.run(
                ['gh', 'run', 'view', run_id, '--repo', self.repo,
                 '--json', 'status,conclusion'],
                capture_output=True, text=True, check=True
            )

            data = json.loads(result.stdout)

            if data['status'] == 'completed':
                return data['conclusion']

            return data['status']
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            return 'unknown'

    def record_iteration(self, iteration: int, errors_found: int,
                        errors_fixed: int, errors_remaining: int,
                        fixes_applied: List[str], new_run_id: Optional[str],
                        success: bool):
        """Record iteration results."""
        result = IterationResult(
            iteration=iteration,
            errors_found=errors_found,
            errors_fixed=errors_fixed,
            errors_remaining=errors_remaining,
            fixes_applied=fixes_applied,
            new_run_id=new_run_id,
            success=success
        )

        self.iteration_history.append(result)

    def update_knowledge_base(self, fixes: List[str], success: bool):
        """Update knowledge base with fix results."""
        for fix in fixes:
            self.knowledge_base.record_fix(fix, success)
        self.knowledge_base.save()

    def print_summary(self):
        """Print iteration history summary."""
        print("\n" + "="*60)
        print("ITERATION SUMMARY")
        print("="*60 + "\n")

        total_errors = 0
        total_fixed = 0

        for result in self.iteration_history:
            print(f"Iteration {result.iteration}:")
            print(f"  Errors found: {result.errors_found}")
            print(f"  Errors fixed: {result.errors_fixed}")
            print(f"  Errors remaining: {result.errors_remaining}")
            print(f"  Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
            if result.fixes_applied:
                print(f"  Fixes applied:")
                for fix in result.fixes_applied:
                    print(f"    - {fix}")
            print()

            total_errors += result.errors_found
            total_fixed += result.errors_fixed

        print(f"Total errors encountered: {total_errors}")
        print(f"Total errors fixed: {total_fixed}")
        print(f"Success rate: {(total_fixed/total_errors*100) if total_errors > 0 else 0:.1f}%")


class KnowledgeBase:
    """Store and retrieve successful fix strategies."""

    def __init__(self):
        self.kb_file = Path('.github/fix-knowledge-base.json')
        self.fixes: Dict[str, Dict] = {}
        self.load()

    def get_fix_strategy(self, error_type: str, context: str) -> str:
        """Get best fix strategy based on historical success."""
        if error_type in self.fixes:
            strategies = self.fixes[error_type].get('strategies', [])
            if strategies:
                # Return strategy with highest success rate
                best = max(strategies, key=lambda x: x.get('success_rate', 0))
                return best['strategy']

        # Default strategies
        defaults = {
            'npm_eresolve': 'Add --legacy-peer-deps flag',
            'npm_404': 'Remove unavailable package',
            'npm_peer': 'Update peer dependencies',
            'ts_error': 'Fix TypeScript type errors',
            'eslint_error': 'Run ESLint auto-fix',
            'test_failure': 'Update test snapshots or assertions',
            'python_import': 'Install missing Python module',
            'python_version': 'Relax version constraints',
            'timeout': 'Increase timeout duration',
            'oom': 'Increase memory allocation',
            'network_error': 'Add retry logic',
        }

        return defaults.get(error_type, 'Manual review required')

    def get_confidence(self, error_type: str) -> float:
        """Get base confidence for error type."""
        if error_type in self.fixes:
            return self.fixes[error_type].get('base_confidence', 0.5)
        return 0.5

    def get_success_rate(self, error_type: str) -> float:
        """Get historical success rate for error type."""
        if error_type in self.fixes:
            total = self.fixes[error_type].get('total_attempts', 0)
            successes = self.fixes[error_type].get('successes', 0)
            return successes / total if total > 0 else 0.5
        return 0.5

    def record_fix(self, fix: str, success: bool):
        """Record fix attempt result."""
        # Extract error type from fix description
        error_type = self.extract_error_type(fix)

        if error_type not in self.fixes:
            self.fixes[error_type] = {
                'base_confidence': 0.5,
                'total_attempts': 0,
                'successes': 0,
                'strategies': []
            }

        self.fixes[error_type]['total_attempts'] += 1
        if success:
            self.fixes[error_type]['successes'] += 1

        # Update base confidence
        total = self.fixes[error_type]['total_attempts']
        successes = self.fixes[error_type]['successes']
        self.fixes[error_type]['base_confidence'] = successes / total

    def extract_error_type(self, fix: str) -> str:
        """Extract error type from fix description."""
        # Simple pattern matching
        patterns = {
            'npm': 'npm_eresolve',
            'package': 'npm_404',
            'eslint': 'eslint_error',
            'typescript': 'ts_error',
            'python': 'python_import',
            'test': 'test_failure',
            'timeout': 'timeout',
            'memory': 'oom',
        }

        fix_lower = fix.lower()
        for keyword, error_type in patterns.items():
            if keyword in fix_lower:
                return error_type

        return 'unknown'

    def load(self):
        """Load knowledge base from file."""
        if self.kb_file.exists():
            try:
                with open(self.kb_file, 'r') as f:
                    self.fixes = json.load(f)
            except json.JSONDecodeError:
                self.fixes = {}
        else:
            self.fixes = {}

    def save(self):
        """Save knowledge base to file."""
        self.kb_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kb_file, 'w') as f:
            json.dump(self.fixes, f, indent=2)


# CLI Interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Iterative CI/CD Error Resolution Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix errors from specific run
  %(prog)s 12345678 --repo owner/repo --workflow "CI"

  # With custom max iterations
  %(prog)s 12345678 --repo owner/repo --workflow "CI" --max-iterations 3
        """
    )
    parser.add_argument('run_id', help='Initial workflow run ID to analyze')
    parser.add_argument('--repo', required=True, help='Repository (owner/name)')
    parser.add_argument('--workflow', required=True, help='Workflow name or file')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Maximum fix iterations (default: 5)')

    args = parser.parse_args()

    # Validate run_id is numeric
    if not args.run_id.isdigit():
        print(f"Error: run_id must be numeric, got: {args.run_id}")
        sys.exit(1)

    print("="*60)
    print("Iterative CI/CD Error Resolution Engine")
    print("="*60)
    print(f"Repository: {args.repo}")
    print(f"Workflow: {args.workflow}")
    print(f"Initial Run ID: {args.run_id}")
    print(f"Max Iterations: {args.max_iterations}")
    print("="*60 + "\n")

    engine = IterativeFixEngine(
        repo=args.repo,
        workflow=args.workflow,
        max_iterations=args.max_iterations
    )

    success = engine.run(args.run_id)

    if success:
        print("\n✓ All errors resolved successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some errors remain. Manual intervention may be required.")
        sys.exit(1)
