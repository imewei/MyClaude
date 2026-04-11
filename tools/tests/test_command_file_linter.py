"""Tests for the Claude Code command file structural linter.

Two test groups:

1. **Unit tests** — synthetic fixture strings covering each rule's
   positive case (issue detected) and negative case (clean content
   produces no issue).
2. **Integration test** — runs the linter over every real command file
   under ``plugins/`` and asserts zero hard errors. This is the most
   valuable test: it converts the linter into a CI-enforced regression
   guard for all command files, not just team-assemble.md.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from tools.validation.command_file_linter import (
    RULE_BROKEN_STEP_REF,
    RULE_DUPLICATE_HEADING,
    RULE_HEADING_SKIP,
    RULE_TRAILING_WS,
    RULE_UNBALANCED_FENCE,
    LintIssue,
    lint_command_file,
    lint_paths,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PLUGINS_ROOT = REPO_ROOT / "plugins"


def _write(tmp_path: pathlib.Path, content: str) -> pathlib.Path:
    """Write content to a temp file and return its path.

    Uses ``textwrap.dedent`` to let the fixture strings be indented
    naturally in the test source.
    """
    path = tmp_path / "fixture.md"
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


def _rules(issues: list[LintIssue]) -> list[str]:
    return [i.rule for i in issues]


# ---------------------------------------------------------------------------
# Rule: unbalanced fence
# ---------------------------------------------------------------------------


class TestUnbalancedFence:
    def test_odd_fence_count_is_flagged(self, tmp_path: pathlib.Path) -> None:
        path = _write(
            tmp_path,
            """
            # Title

            Paragraph.

            ```python
            x = 1
            ```

            ```
            unterminated
            """,
        )
        issues = lint_command_file(path)
        assert RULE_UNBALANCED_FENCE in _rules(issues)

    def test_balanced_fences_pass(self, tmp_path: pathlib.Path) -> None:
        path = _write(
            tmp_path,
            """
            # Title

            ```python
            x = 1
            ```

            ```bash
            echo hi
            ```
            """,
        )
        issues = lint_command_file(path)
        assert RULE_UNBALANCED_FENCE not in _rules(issues)


# ---------------------------------------------------------------------------
# Rule: heading skip
# ---------------------------------------------------------------------------


class TestHeadingSkip:
    def test_h1_to_h3_is_flagged(self, tmp_path: pathlib.Path) -> None:
        path = _write(
            tmp_path,
            """
            # Title

            ### Section

            Body.
            """,
        )
        issues = lint_command_file(path)
        assert RULE_HEADING_SKIP in _rules(issues)

    def test_clean_hierarchy_passes(self, tmp_path: pathlib.Path) -> None:
        path = _write(
            tmp_path,
            """
            # Title

            ## Section

            ### Subsection

            Body.
            """,
        )
        issues = lint_command_file(path)
        assert RULE_HEADING_SKIP not in _rules(issues)

    def test_headings_inside_fences_are_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write(
            tmp_path,
            """
            # Title

            ```
            # Not a real heading
            ### Also not
            ```

            ## Real section
            """,
        )
        issues = lint_command_file(path)
        assert RULE_HEADING_SKIP not in _rules(issues)


# ---------------------------------------------------------------------------
# Rule: broken step reference
# ---------------------------------------------------------------------------


class TestBrokenStepReference:
    def test_unknown_step_is_flagged(self, tmp_path: pathlib.Path) -> None:
        path = _write(
            tmp_path,
            """
            ## Step 1: Foo

            Run **Step 99** next.

            ## Step 2: Bar
            """,
        )
        issues = lint_command_file(path)
        assert RULE_BROKEN_STEP_REF in _rules(issues)

    def test_existing_step_passes(self, tmp_path: pathlib.Path) -> None:
        path = _write(
            tmp_path,
            """
            ## Step 1: Foo

            Then run Step 2.

            ## Step 2: Bar
            """,
        )
        issues = lint_command_file(path)
        assert RULE_BROKEN_STEP_REF not in _rules(issues)

    def test_substep_reference_resolves(self, tmp_path: pathlib.Path) -> None:
        """Step 2.6 should resolve when the file defines Step 2.6a."""
        path = _write(
            tmp_path,
            """
            ## Step 1: Foo

            Run Step 2.6 when ready.

            ## Step 2: Bar

            ### Step 2.6a: Sub

            Details.
            """,
        )
        issues = lint_command_file(path)
        assert RULE_BROKEN_STEP_REF not in _rules(issues)

    def test_file_without_any_steps_skips_check(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The step-ref check only activates when the file has at least
        one Step heading. Prose about 'Step 3' in a file without Steps
        must not false-positive."""
        path = _write(
            tmp_path,
            """
            # Some Command

            This command does Step 3 of a recipe written somewhere else.
            """,
        )
        issues = lint_command_file(path)
        assert RULE_BROKEN_STEP_REF not in _rules(issues)


# ---------------------------------------------------------------------------
# Rule: trailing whitespace
# ---------------------------------------------------------------------------


class TestTrailingWhitespace:
    def test_trailing_space_on_prose_is_flagged(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Write bytes directly to guarantee the trailing space survives
        # (textwrap.dedent preserves it, but some editors don't).
        path = tmp_path / "fixture.md"
        path.write_text("# Title\n\nA line with a trailing space.   \n", encoding="utf-8")
        issues = lint_command_file(path)
        assert RULE_TRAILING_WS in _rules(issues)

    def test_trailing_whitespace_inside_fences_is_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "fixture.md"
        path.write_text(
            "# Title\n\n```python\nx = 1   \n```\n",
            encoding="utf-8",
        )
        issues = lint_command_file(path)
        assert RULE_TRAILING_WS not in _rules(issues)


# ---------------------------------------------------------------------------
# Rule: duplicate heading
# ---------------------------------------------------------------------------


class TestDuplicateHeading:
    def test_duplicate_h3_under_same_h2_is_flagged(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write(
            tmp_path,
            """
            ## Parent

            ### Child

            Body 1.

            ### Child

            Body 2.
            """,
        )
        issues = lint_command_file(path)
        assert RULE_DUPLICATE_HEADING in _rules(issues)

    def test_same_h3_under_different_h2_passes(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = _write(
            tmp_path,
            """
            ## Parent A

            ### Examples

            Body.

            ## Parent B

            ### Examples

            Body.
            """,
        )
        issues = lint_command_file(path)
        assert RULE_DUPLICATE_HEADING not in _rules(issues)


# ---------------------------------------------------------------------------
# Integration: run the linter over every real command file
# ---------------------------------------------------------------------------


class TestRealCommandFiles:
    """Run the linter over every ``plugins/*/commands/*.md`` file.

    If this test starts failing, the fix is almost always to correct
    the offending command file, not to loosen the linter.
    """

    def test_no_hard_errors_in_any_command_file(self) -> None:
        issues = lint_paths([PLUGINS_ROOT])
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            formatted = "\n".join(i.format() for i in errors)
            pytest.fail(
                f"Command file linter found {len(errors)} hard error(s):\n"
                f"{formatted}"
            )

    def test_team_assemble_has_zero_errors_and_warnings(self) -> None:
        """team-assemble.md is the canonical command file that we just
        hardened. It must pass both errors AND warnings (we can tolerate
        info-level trailing whitespace)."""
        target = PLUGINS_ROOT / "agent-core/commands/team-assemble.md"
        issues = lint_command_file(target)
        hard = [i for i in issues if i.severity in ("error", "warning")]
        if hard:
            formatted = "\n".join(i.format() for i in hard)
            pytest.fail(
                f"team-assemble.md has {len(hard)} error/warning issue(s):\n"
                f"{formatted}"
            )
