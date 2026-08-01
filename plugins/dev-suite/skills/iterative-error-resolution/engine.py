#!/usr/bin/env python3
"""
Iterative CI/CD Error Resolution Engine
Continuously fixes errors until zero failures or max iterations reached
"""

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class FixResult(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_FIX_AVAILABLE = "no_fix"
    ERRORED = "errored"


# Minimum confidence for a strategy to be auto-applied. Below this the error is
# reported for manual review instead of dispatched.
AUTO_APPLY_CONFIDENCE = 0.7

# Strategies that stop a check from failing without addressing why it failed.
# Regenerating snapshots rewrites the expectation to match current (broken)
# output, so a green CI run afterwards proves nothing. Never auto-applied.
SUPPRESSION_TYPES = frozenset({"test_failure"})

# Strategies whose blast radius (deleting a real dependency) is too large to
# take on the word of a log line that a registry outage can also produce.
MANUAL_REVIEW_TYPES = frozenset({"npm_404"})

# Only these modules may be auto-installed. Not a lookup table with a fallback:
# an unmapped module is reported, never installed.
PYTHON_MODULE_PACKAGES = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
}


@dataclass
class ErrorAnalysis:
    category: str
    error_type: str
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
    fixes_applied: list[str]
    new_run_id: str | None
    success: bool


TARGET_TIMEOUT_MINUTES = 60

def _workflow_files() -> list[Path]:
    workflows = Path(".github/workflows")
    return sorted(workflows.glob("*.yml")) + sorted(workflows.glob("*.yaml"))


def _validate_gh_arg(value: str, label: str) -> str:
    """Validate CLI arguments passed to gh subprocess calls."""
    if not re.match(r"^[a-zA-Z0-9_\-./]+$", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


class IterativeFixEngine:
    def __init__(
        self,
        repo: str,
        workflow: str,
        max_iterations: int = 5,
        auto_commit: bool = False,
        allow_suppression: bool = False,
    ):
        self.repo = _validate_gh_arg(repo, "repo")
        self.workflow = _validate_gh_arg(workflow, "workflow")
        self.max_iterations = max_iterations
        self.auto_commit = auto_commit
        self.allow_suppression = allow_suppression
        self.knowledge_base = KnowledgeBase()
        self.iteration_history: list[IterationResult] = []
        self.errored_count = 0
        self.suppression_applied = False
        self._modified_files: list[str] = []

    def run(self, initial_run_id: str) -> bool:
        """
        Main iterative fix loop.
        Returns True if all errors resolved, False otherwise.
        """
        current_run_id = initial_run_id

        print(f"Starting iterative fix loop (max {self.max_iterations} iterations)")
        print(f"Initial run ID: {current_run_id}\n")

        for iteration in range(1, self.max_iterations + 1):
            print(f"{'=' * 60}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'=' * 60}\n")

            # Analyze current run
            errors = self.analyze_run(current_run_id)

            if not errors:
                print("✓ SUCCESS: Zero errors detected!")
                self.record_iteration(iteration, 0, 0, 0, [], None, True)
                return True

            print(f"Found {len(errors)} error(s) to fix\n")

            # Categorize and prioritize errors
            categorized = self.categorize_errors(errors)
            prioritized = self.prioritize_fixes(categorized)

            # Decide what may be touched BEFORE anything is touched
            actionable, skipped = self.plan_fixes(prioritized)
            self.print_plan(actionable, skipped)

            if not self.auto_commit:
                print(
                    "\n[PLAN ONLY] --auto-commit not set. Nothing was modified: no "
                    "files written, no packages installed or removed, no commands run."
                )
                print("Re-run with --auto-commit to apply the plan above and push.")
                sys.exit(0)

            if not actionable:
                print("No fixes could be applied. Manual intervention required.")
                self.record_iteration(
                    iteration, len(errors), 0, len(errors), [], None, False
                )
                return False

            # Apply fixes
            fixes_applied = []
            errors_fixed = 0
            errored = 0

            for error_analysis in actionable:
                print(f"Fixing: {error_analysis.pattern[:80]}...")
                print(f"Strategy: {error_analysis.suggested_fix}\n")

                result = self.apply_fix(error_analysis)

                if result in [FixResult.SUCCESS, FixResult.PARTIAL]:
                    fixes_applied.append(error_analysis.suggested_fix)
                    errors_fixed += 1
                    print("✓ Fix applied successfully\n")
                else:
                    if result is FixResult.ERRORED:
                        errored += 1
                    print(f"✗ Fix not applied: {result.value}\n")

            self.errored_count += errored

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
                    iteration,
                    len(errors),
                    errors_fixed,
                    len(errors) - errors_fixed,
                    fixes_applied,
                    None,
                    False,
                )
                return False

            print(f"New run started: {new_run_id}")

            # Wait for completion
            print("Waiting for workflow to complete...")
            if not self.wait_for_completion(new_run_id, timeout=600):
                print("Workflow timeout")
                self.record_iteration(
                    iteration,
                    len(errors),
                    errors_fixed,
                    len(errors) - errors_fixed,
                    fixes_applied,
                    new_run_id,
                    False,
                )
                return False

            # Check if successful
            status = self.get_run_status(new_run_id)

            self.record_iteration(
                iteration,
                len(errors),
                errors_fixed,
                len(errors) - errors_fixed,
                fixes_applied,
                new_run_id,
                status == "success",
            )

            if status == "success":
                if self.suppression_applied:
                    print(
                        "\n⚠ CI is green, but a suppression strategy was applied "
                        "this run — green is the signal that strategy manipulates, "
                        "so it is not evidence the errors were resolved. Review "
                        "the diff manually."
                    )
                else:
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

    def analyze_run(self, run_id: str) -> list[dict]:
        """Fetch and parse workflow run logs."""
        cmd = ["gh", "run", "view", run_id, "--repo", self.repo, "--log-failed"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error fetching logs: {e}")
            return []

        return self.parse_logs(result.stdout)

    def parse_logs(self, logs: str) -> list[dict]:
        """Extract error patterns from logs."""
        errors = []

        patterns = {
            "npm_eresolve": r"npm ERR! code ERESOLVE",
            "npm_404": r"npm ERR! 404",
            "npm_peer": r"npm ERR! peer dep missing",
            "ts_error": r"TS\d+:",
            "eslint_error": r"\d+:\d+\s+error",
            "test_failure": r"FAIL .*\.test\.",
            "python_import": r"ModuleNotFoundError|ImportError",
            "python_version": r"Could not find a version that satisfies",
            "build_error": r"Build failed|compilation failed",
            "timeout": r"exceeded the maximum execution time",
            "oom": r"heap out of memory",
            "network_error": r"ETIMEDOUT|ENOTFOUND|ECONNREFUSED",
        }

        for name, pattern in patterns.items():
            matches = re.finditer(pattern, logs, re.MULTILINE)
            for match in matches:
                # Extract context (5 lines before and after)
                lines = logs[: match.end()].split("\n")
                context_start = max(0, len(lines) - 5)
                context = "\n".join(lines[context_start:])

                errors.append(
                    {
                        "type": name,
                        "pattern": pattern,
                        "match": match.group(),
                        "context": context,
                    }
                )

        return errors

    def categorize_errors(self, errors: list[dict]) -> list[ErrorAnalysis]:
        """Categorize errors and assign fix strategies."""
        analyses = []

        for error in errors:
            category = self.get_category(error["type"])
            confidence = self.calculate_confidence(error)
            fix_strategy = self.knowledge_base.get_fix_strategy(
                error["type"], error["context"]
            )
            priority = self.calculate_priority(error, confidence)

            analyses.append(
                ErrorAnalysis(
                    category=category,
                    error_type=error["type"],
                    pattern=error["match"],
                    confidence=confidence,
                    suggested_fix=fix_strategy,
                    priority=priority,
                    context=error["context"],
                )
            )

        return analyses

    def get_category(self, error_type: str) -> str:
        """Map error type to category."""
        category_map = {
            "npm_eresolve": "dependency",
            "npm_404": "dependency",
            "npm_peer": "dependency",
            "python_import": "dependency",
            "python_version": "dependency",
            "ts_error": "build",
            "eslint_error": "build",
            "build_error": "build",
            "test_failure": "test",
            "timeout": "runtime",
            "oom": "runtime",
            "network_error": "runtime",
        }
        return category_map.get(error_type, "unknown")

    def calculate_confidence(self, error: dict) -> float:
        """Calculate confidence score for fix."""
        # Base confidence from knowledge base
        kb_confidence = self.knowledge_base.get_confidence(error["type"])

        # Adjust based on error clarity
        clarity_bonus = 0.0
        if error["type"] in ["npm_eresolve", "npm_404", "python_import"]:
            clarity_bonus = 0.1  # These have clear fixes

        # Adjust based on historical success
        history_bonus = self.knowledge_base.get_success_rate(error["type"]) * 0.2

        return min(1.0, kb_confidence + clarity_bonus + history_bonus)

    def calculate_priority(self, error: dict, confidence: float) -> int:
        """Calculate priority score (higher = more important)."""
        # Blocking errors get highest priority
        blocking_types = ["build_error", "npm_eresolve", "python_import"]

        priority = int(confidence * 100)

        if error["type"] in blocking_types:
            priority += 50

        return priority

    def prioritize_fixes(self, analyses: list[ErrorAnalysis]) -> list[ErrorAnalysis]:
        """Sort fixes by priority (high confidence, blocking errors first)."""
        return sorted(analyses, key=lambda x: (-x.priority, -x.confidence))

    def plan_fixes(
        self, analyses: list[ErrorAnalysis]
    ) -> tuple[list[ErrorAnalysis], list[tuple[ErrorAnalysis, str]]]:
        """Split errors into auto-applicable fixes and ones needing a human.

        Runs before any strategy executes, so the printed plan is the whole set
        of side effects the engine is about to cause.
        """
        actionable: list[ErrorAnalysis] = []
        skipped: list[tuple[ErrorAnalysis, str]] = []

        for error in analyses:
            if error.error_type in SUPPRESSION_TYPES and not self.allow_suppression:
                skipped.append(
                    (
                        error,
                        (
                            "suppresses the failure rather than fixing it — "
                            "requires explicit --allow-suppression"
                        ),
                    )
                )
            elif error.error_type in MANUAL_REVIEW_TYPES:
                skipped.append(
                    (error, "removes a real dependency — manual confirmation required")
                )
            elif error.confidence < AUTO_APPLY_CONFIDENCE:
                skipped.append(
                    (
                        error,
                        (
                            f"confidence {error.confidence:.0%} is below the "
                            f"{AUTO_APPLY_CONFIDENCE:.0%} auto-apply threshold"
                        ),
                    )
                )
            else:
                actionable.append(error)

        return actionable, skipped

    def print_plan(
        self,
        actionable: list[ErrorAnalysis],
        skipped: list[tuple[ErrorAnalysis, str]],
    ) -> None:
        """Show exactly what will and will not be attempted."""
        print("PLAN\n" + "-" * 60)

        if actionable:
            print(f"Will attempt ({len(actionable)}):")
            for error in actionable:
                print(
                    f"  [{error.error_type}] {error.suggested_fix} "
                    f"(confidence {error.confidence:.0%})"
                )
        else:
            print("Will attempt (0): nothing meets the auto-apply bar.")

        if skipped:
            print(f"\nManual review needed ({len(skipped)}):")
            for error, reason in skipped:
                print(f"  [{error.error_type}] {error.suggested_fix} — {reason}")
        print("-" * 60)

    def apply_fix(self, error: ErrorAnalysis) -> FixResult:
        """Execute fix strategy."""
        try:
            if error.category == "dependency":
                return self.fix_dependency_error(error)
            elif error.category == "build":
                return self.fix_build_error(error)
            elif error.category == "test":
                return self.fix_test_error(error)
            elif error.category == "runtime":
                return self.fix_runtime_error(error)
            else:
                return FixResult.NO_FIX_AVAILABLE
        except Exception as e:  # noqa: BLE001 — dispatches arbitrary fix strategies (subprocess/file IO/regex); must degrade to ERRORED, not crash the iteration loop
            print(f"Error applying fix: {e}")
            return FixResult.ERRORED

    def fix_dependency_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix dependency-related errors."""
        if error.error_type == "npm_eresolve":
            return self.fix_npm_eresolve()
        elif error.error_type == "npm_404":
            return self.fix_npm_404(error)
        elif error.error_type == "python_import":
            return self.fix_python_import(error)
        return FixResult.NO_FIX_AVAILABLE

    def fix_npm_eresolve(self) -> FixResult:
        """Add --legacy-peer-deps so npm stops enforcing the peer-dep graph.

        This makes the install succeed without reconciling the conflict, so the
        incompatibility is still present at runtime — treat it as a stopgap.
        """
        changed = False
        for workflow_file in _workflow_files():
            content = workflow_file.read_text()

            if "npm install" in content and "--legacy-peer-deps" not in content:
                updated = content.replace(
                    "npm install", "npm install --legacy-peer-deps"
                ).replace("npm ci", "npm ci --legacy-peer-deps")
                if updated != content:
                    workflow_file.write_text(updated)
                    changed = True

        return FixResult.SUCCESS if changed else FixResult.NO_FIX_AVAILABLE

    def fix_npm_404(self, error: ErrorAnalysis) -> FixResult:
        """Report an npm 404 for manual review — never uninstall automatically.

        A 404 in a CI log is also what a registry outage, a private-scope auth
        failure, or a transient mirror problem looks like. Deleting a real
        dependency on that evidence is not recoverable by re-running CI.
        """
        match = re.search(r"404.*'(@?[^']+)'", error.context)
        package = match.group(1) if match else "<unknown>"
        print(
            f"Manual review needed: npm reported 404 for '{package}'. "
            "Confirm the package is genuinely gone (not a registry/auth outage) "
            "before removing it."
        )
        return FixResult.NO_FIX_AVAILABLE

    def fix_python_import(self, error: ErrorAnalysis) -> FixResult:
        """Fix Python import errors."""
        # Extract module name
        match = re.search(r"No module named '([^']+)'", error.context)
        if not match:
            return FixResult.NO_FIX_AVAILABLE

        module = match.group(1)

        # Closed allowlist. The module name comes from CI log text, which an
        # attacker or a typo can influence; a well-formed name is not evidence
        # that the package is the real one. Anything unmapped goes to a human.
        package = PYTHON_MODULE_PACKAGES.get(module)
        if package is None:
            print(
                f"Manual review needed: module '{module}' is not in the known "
                "module-to-package allowlist. Add it explicitly if the mapping "
                "is correct."
            )
            return FixResult.NO_FIX_AVAILABLE

        print(f"Installing missing module: {package}")

        try:
            subprocess.run(["pip", "install", package], check=True)

            # Update requirements.txt with dedup check
            req_path = Path.cwd() / "requirements.txt"
            existing = set()
            if req_path.exists():
                existing = {
                    line.strip()
                    for line in req_path.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                }
            if package not in existing:
                with open(req_path, "a") as f:
                    f.write(f"{package}\n")
                self._modified_files.append(str(req_path))

            return FixResult.SUCCESS
        except subprocess.CalledProcessError:
            return FixResult.FAILED

    def fix_build_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix build-related errors."""
        if error.error_type == "eslint_error":
            return self.fix_eslint_errors()
        return FixResult.NO_FIX_AVAILABLE

    def fix_eslint_errors(self) -> FixResult:
        """Run ESLint auto-fix."""
        try:
            subprocess.run(["npx", "eslint", ".", "--fix"], check=False)
            return FixResult.SUCCESS
        except OSError:
            return FixResult.PARTIAL

    def fix_test_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix test-related errors."""
        if "snapshot" in error.context.lower():
            return self.fix_snapshot_errors()
        return FixResult.NO_FIX_AVAILABLE

    def fix_snapshot_errors(self) -> FixResult:
        """Regenerate test snapshots — SUPPRESSION, not a fix.

        `npm test -- -u` rewrites the expectation to match whatever the code
        currently produces, so the test passes again whether or not the change
        that broke it was intentional. Only reachable via --allow-suppression,
        and never reported as SUCCESS: the caller must not treat the next green
        CI run as evidence the underlying problem was solved.
        """
        if not self.allow_suppression:
            print("Refusing to regenerate snapshots without --allow-suppression.")
            return FixResult.NO_FIX_AVAILABLE

        try:
            subprocess.run(["npm", "test", "--", "-u"], check=True)
        except subprocess.CalledProcessError:
            return FixResult.FAILED

        self.suppression_applied = True
        print(
            "⚠ Snapshots regenerated. This SUPPRESSED the failure — the "
            "expectation now matches current output. Review the snapshot diff "
            "before trusting a green run."
        )
        return FixResult.PARTIAL

    def fix_runtime_error(self, error: ErrorAnalysis) -> FixResult:
        """Fix runtime errors."""
        if error.error_type == "oom":
            return self.fix_oom_error()
        elif error.error_type == "timeout":
            return self.fix_timeout_error()
        return FixResult.NO_FIX_AVAILABLE

    def fix_oom_error(self) -> FixResult:
        """Raise the Node heap limit via a workflow-level env block.

        ponytail: sets top-level `env:` (applies to every job) rather than
        editing each job — column-0 insertion has unambiguous indentation. A
        job- or step-level NODE_OPTIONS still overrides it; if the OOM persists,
        that override is where to look.
        """
        changed = False
        for workflow_file in _workflow_files():
            content = workflow_file.read_text()
            if "NODE_OPTIONS" in content:
                continue

            try:
                original = yaml.safe_load(content)
            except yaml.YAMLError as e:
                print(f"Skipping {workflow_file}: already invalid YAML ({e})")
                continue
            if not isinstance(original, dict):
                continue

            if "env" in original:
                updated = re.sub(
                    r"^env:\s*$",
                    'env:\n  NODE_OPTIONS: "--max-old-space-size=4096"',
                    content,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                updated = content.rstrip("\n") + (
                    '\n\nenv:\n  NODE_OPTIONS: "--max-old-space-size=4096"\n'
                )

            if updated == content:
                continue

            try:
                reparsed = yaml.safe_load(updated)
            except yaml.YAMLError as e:
                print(f"Aborting {workflow_file}: edit produced invalid YAML ({e})")
                continue
            if (
                not isinstance(reparsed, dict)
                or reparsed.get("env", {}).get("NODE_OPTIONS") is None
            ):
                print(f"Aborting {workflow_file}: NODE_OPTIONS did not land as env")
                continue

            workflow_file.write_text(updated)
            changed = True

        return FixResult.SUCCESS if changed else FixResult.NO_FIX_AVAILABLE

    def fix_timeout_error(self) -> FixResult:
        """Raise CI timeouts. Only ever increases — never lowers a deliberate value."""
        changed = False
        for workflow_file in _workflow_files():
            content = workflow_file.read_text()

            if "timeout-minutes:" not in content:
                updated = re.sub(
                    r"(runs-on:)",
                    f"timeout-minutes: {TARGET_TIMEOUT_MINUTES}\n    \\1",
                    content,
                )
            else:
                updated = re.sub(
                    r"timeout-minutes: (\d+)",
                    lambda m: (
                        "timeout-minutes: "
                        f"{max(int(m.group(1)), TARGET_TIMEOUT_MINUTES)}"
                    ),
                    content,
                )

            if updated == content:
                continue
            try:
                yaml.safe_load(updated)
            except yaml.YAMLError as e:
                print(f"Aborting {workflow_file}: edit produced invalid YAML ({e})")
                continue

            workflow_file.write_text(updated)
            changed = True

        return FixResult.SUCCESS if changed else FixResult.NO_FIX_AVAILABLE

    def commit_fixes(self, fixes: list[str], iteration: int):
        """Commit and push the applied fixes.

        Only reached with --auto-commit set; the plan gate in run() exits before
        any strategy executes otherwise.
        """
        message = f"fix(ci): iteration {iteration} - automated error resolution\n\n"
        message += "Applied fixes:\n"
        for fix in fixes:
            message += f"- {fix}\n"
        message += "\n🤖 Generated with iterative-error-resolution"

        try:
            # Stage only tracked modified files, not untracked files
            subprocess.run(["git", "add", "--update"], check=True)
            # Also stage any explicitly tracked new files from fixes
            for path in self._modified_files:
                subprocess.run(["git", "add", path], check=True)

            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error committing fixes: {e}")

    def trigger_workflow(self) -> str | None:
        """Trigger workflow and return new run ID."""
        try:
            result = subprocess.run(
                ["gh", "workflow", "run", self.workflow, "--repo", self.repo],
                capture_output=True,
                text=True,
                check=True,
            )

            # Wait for run to appear
            time.sleep(5)

            # Get latest run ID
            result = subprocess.run(
                [
                    "gh",
                    "run",
                    "list",
                    "--workflow",
                    self.workflow,
                    "--repo",
                    self.repo,
                    "--limit",
                    "1",
                    "--json",
                    "databaseId",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            runs = json.loads(result.stdout)
            return str(runs[0]["databaseId"]) if runs else None
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"Error triggering workflow: {e}")
            return None

    def wait_for_completion(self, run_id: str, timeout: int = 600) -> bool:
        """Wait for workflow run to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_run_status(run_id)

            if status in ["success", "failure", "cancelled"]:
                return True

            print(".", end="", flush=True)
            time.sleep(10)

        print()
        return False

    def get_run_status(self, run_id: str) -> str:
        """Get current status of workflow run."""
        try:
            result = subprocess.run(
                [
                    "gh",
                    "run",
                    "view",
                    run_id,
                    "--repo",
                    self.repo,
                    "--json",
                    "status,conclusion",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            data = json.loads(result.stdout)

            if data["status"] == "completed":
                return data["conclusion"]

            return data["status"]
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            return "unknown"

    def record_iteration(
        self,
        iteration: int,
        errors_found: int,
        errors_fixed: int,
        errors_remaining: int,
        fixes_applied: list[str],
        new_run_id: str | None,
        success: bool,
    ):
        """Record iteration results."""
        result = IterationResult(
            iteration=iteration,
            errors_found=errors_found,
            errors_fixed=errors_fixed,
            errors_remaining=errors_remaining,
            fixes_applied=fixes_applied,
            new_run_id=new_run_id,
            success=success,
        )

        self.iteration_history.append(result)

    def update_knowledge_base(self, fixes: list[str], success: bool):
        """Update knowledge base with fix results."""
        for fix in fixes:
            self.knowledge_base.record_fix(fix, success)
        self.knowledge_base.save()

    def print_summary(self):
        """Print iteration history summary."""
        print("\n" + "=" * 60)
        print("ITERATION SUMMARY")
        print("=" * 60 + "\n")

        total_errors = 0
        total_fixed = 0

        for result in self.iteration_history:
            print(f"Iteration {result.iteration}:")
            print(f"  Errors found: {result.errors_found}")
            print(f"  Errors fixed: {result.errors_fixed}")
            print(f"  Errors remaining: {result.errors_remaining}")
            print(f"  Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
            if result.fixes_applied:
                print("  Fixes applied:")
                for fix in result.fixes_applied:
                    print(f"    - {fix}")
            print()

            total_errors += result.errors_found
            total_fixed += result.errors_fixed

        print(f"Total errors encountered: {total_errors}")
        print(f"Total errors fixed: {total_fixed}")
        if self.errored_count:
            print(
                f"Strategies that crashed mid-run: {self.errored_count} "
                "(distinct from 'no fix available' — these left partial state)"
            )
        print(
            f"Success rate: {(total_fixed / total_errors * 100) if total_errors > 0 else 0:.1f}%"
        )


# Attempts required before a strategy's recorded success rate is trusted.
MIN_SAMPLES = 3
NEUTRAL_CONFIDENCE = 0.5


class KnowledgeBase:
    """Store and retrieve successful fix strategies."""

    def __init__(self) -> None:
        self.kb_file = Path(".github/fix-knowledge-base.json")
        self.fixes: dict[str, dict] = {}
        self.load()

    def get_fix_strategy(self, error_type: str, context: str) -> str:
        """Get best fix strategy based on historical success."""
        if error_type in self.fixes:
            strategies = self.fixes[error_type].get("strategies", [])
            if strategies:
                # Return strategy with highest success rate
                best = max(strategies, key=lambda x: x.get("success_rate", 0))
                return best["strategy"]

        # Default strategies
        defaults = {
            "npm_eresolve": "Add --legacy-peer-deps flag",
            "npm_404": "Remove unavailable package",
            "npm_peer": "Update peer dependencies",
            "ts_error": "Fix TypeScript type errors",
            "eslint_error": "Run ESLint auto-fix",
            "test_failure": "Update test snapshots or assertions",
            "python_import": "Install missing Python module",
            "python_version": "Relax version constraints",
            "timeout": "Increase timeout duration",
            "oom": "Increase memory allocation",
            "network_error": "Add retry logic",
        }

        return defaults.get(error_type, "Manual review required")

    def get_confidence(self, error_type: str) -> float:
        """Get base confidence for error type, ignoring under-sampled history."""
        if self._has_enough_samples(error_type):
            return self.fixes[error_type].get("base_confidence", NEUTRAL_CONFIDENCE)
        return NEUTRAL_CONFIDENCE

    def get_success_rate(self, error_type: str) -> float:
        """Get historical success rate, ignoring under-sampled history."""
        if self._has_enough_samples(error_type):
            entry = self.fixes[error_type]
            return entry["successes"] / entry["total_attempts"]
        return NEUTRAL_CONFIDENCE

    def _has_enough_samples(self, error_type: str) -> bool:
        """A strategy is unproven until it has been tried MIN_SAMPLES times.

        Without this a single 1/1 result pins a strategy's success rate at 100%
        permanently, which is how a lucky first attempt becomes policy.
        """
        entry = self.fixes.get(error_type)
        if not entry:
            return False
        return entry.get("total_attempts", 0) >= MIN_SAMPLES

    def record_fix(self, fix: str, success: bool):
        """Record fix attempt result."""
        # Extract error type from fix description
        error_type = self.extract_error_type(fix)

        if error_type not in self.fixes:
            self.fixes[error_type] = {
                "base_confidence": 0.5,
                "total_attempts": 0,
                "successes": 0,
                "strategies": [],
            }

        self.fixes[error_type]["total_attempts"] += 1
        if success:
            self.fixes[error_type]["successes"] += 1

        # Update base confidence
        total = self.fixes[error_type]["total_attempts"]
        successes = self.fixes[error_type]["successes"]
        self.fixes[error_type]["base_confidence"] = successes / total

    def extract_error_type(self, fix: str) -> str:
        """Extract error type from fix description."""
        # Simple pattern matching
        patterns = {
            "npm": "npm_eresolve",
            "package": "npm_404",
            "eslint": "eslint_error",
            "typescript": "ts_error",
            "python": "python_import",
            "test": "test_failure",
            "timeout": "timeout",
            "memory": "oom",
        }

        fix_lower = fix.lower()
        for keyword, error_type in patterns.items():
            if keyword in fix_lower:
                return error_type

        return "unknown"

    def load(self) -> None:
        """Load knowledge base from file."""
        if self.kb_file.exists():
            try:
                with open(self.kb_file, "r") as f:
                    self.fixes = json.load(f)
            except json.JSONDecodeError:
                self.fixes = {}
        else:
            self.fixes = {}

    def save(self):
        """Save knowledge base to file."""
        self.kb_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kb_file, "w") as f:
            json.dump(self.fixes, f, indent=2)


# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Iterative CI/CD Error Resolution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix errors from specific run
  %(prog)s 12345678 --repo owner/repo --workflow "CI"

  # With custom max iterations
  %(prog)s 12345678 --repo owner/repo --workflow "CI" --max-iterations 3
        """,
    )
    parser.add_argument("run_id", help="Initial workflow run ID to analyze")
    parser.add_argument("--repo", required=True, help="Repository (owner/name)")
    parser.add_argument("--workflow", required=True, help="Workflow name or file")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum fix iterations (default: 5)",
    )
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        default=False,
        help=(
            "Actually apply fixes, commit and push. Without it the engine prints "
            "the plan and exits without modifying anything."
        ),
    )
    parser.add_argument(
        "--allow-suppression",
        action="store_true",
        default=False,
        help=(
            "Permit strategies that silence a failure without fixing its cause "
            "(e.g. regenerating test snapshots). Off by default."
        ),
    )

    args = parser.parse_args()

    # Validate run_id is numeric
    if not args.run_id.isdigit():
        print(f"Error: run_id must be numeric, got: {args.run_id}")
        sys.exit(1)

    print("=" * 60)
    print("Iterative CI/CD Error Resolution Engine")
    print("=" * 60)
    print(f"Repository: {args.repo}")
    print(f"Workflow: {args.workflow}")
    print(f"Initial Run ID: {args.run_id}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 60 + "\n")

    engine = IterativeFixEngine(
        repo=args.repo,
        workflow=args.workflow,
        max_iterations=args.max_iterations,
        auto_commit=args.auto_commit,
        allow_suppression=args.allow_suppression,
    )

    success = engine.run(args.run_id)

    if success:
        print("\n✓ All errors resolved successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some errors remain. Manual intervention may be required.")
        sys.exit(1)
