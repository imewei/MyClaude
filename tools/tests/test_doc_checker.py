#!/usr/bin/env python3
"""Unit tests for tools.validation.doc_checker._check_links.

Covers the regressions fixed in commit 8e8b8641:
- code-block stripping (triple-fenced, double-backtick, single-backtick)
- ${CLAUDE_PLUGIN_ROOT} resolution
- YAML frontmatter scanning for path-shaped string values
"""

import sys
import tempfile
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tools.validation.doc_checker import (  # noqa: E402
    DocCheckResult,
    DocumentationChecker,
)


def _check(content: str, *, name: str = "cmd.md") -> list[str]:
    """Run _check_links on `content` and return a list of error messages."""
    with tempfile.TemporaryDirectory() as td:
        plugin = Path(td) / "test-plugin"
        (plugin / "commands").mkdir(parents=True)
        fp = plugin / "commands" / name
        fp.write_text(content)
        checker = DocumentationChecker()
        result = DocCheckResult(plugin_name="test", plugin_path=plugin)
        checker._check_links(fp, content, content.splitlines(), result)
        return [i.message for i in result.issues if i.severity == "error"]


class TestCheckLinksBody(unittest.TestCase):
    def test_external_urls_skipped(self):
        errors = _check("[anthropic](https://anthropic.com) [docs](http://x.io) [mail](mailto:a@b)")
        self.assertEqual(errors, [])

    def test_anchor_skipped(self):
        errors = _check("# H\n[jump](#section)")
        self.assertEqual(errors, [])

    def test_broken_relative_link_flagged(self):
        errors = _check("[broken](does-not-exist.md)")
        self.assertEqual(len(errors), 1)
        self.assertIn("does-not-exist.md", errors[0])


class TestCheckLinksCodeBlockStripping(unittest.TestCase):
    """Regression tests — these used to false-positive."""

    def test_triple_fenced_code_block_skipped(self):
        content = "```python\nfoo = bar  # this is [text](url)\n```\n"
        self.assertEqual(_check(content), [])

    def test_inline_single_backtick_skipped(self):
        # The historical false positive in agent-core/jax-pro.md
        content = "Call `self.layers[-1](x)` to invoke the last layer."
        self.assertEqual(_check(content), [])

    def test_inline_double_backtick_with_inner_backtick_skipped(self):
        # The exact pattern from team-assemble.md:181
        content = "Strip markdown: replace `` `code` `` and `[text](url)` with `text`."
        self.assertEqual(_check(content), [])

    def test_real_link_after_code_block_still_flagged(self):
        content = "```\n[skip](url)\n```\n\n[real-broken](missing.md)"
        errors = _check(content)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing.md", errors[0])


class TestCheckLinksClaudePluginRoot(unittest.TestCase):
    def test_claude_plugin_root_resolves(self):
        # Create the target file so the link resolves
        with tempfile.TemporaryDirectory() as td:
            plugin = Path(td) / "p"
            (plugin / "commands").mkdir(parents=True)
            (plugin / "docs").mkdir()
            (plugin / "docs" / "real.md").write_text("hello")
            fp = plugin / "commands" / "cmd.md"
            content = "[ok](${CLAUDE_PLUGIN_ROOT}/docs/real.md)"
            fp.write_text(content)
            checker = DocumentationChecker()
            result = DocCheckResult(plugin_name="test", plugin_path=plugin)
            checker._check_links(fp, content, content.splitlines(), result)
            errors = [i for i in result.issues if i.severity == "error"]
            self.assertEqual(errors, [])

    def test_claude_plugin_root_missing_target_flagged(self):
        errors = _check("[bad](${CLAUDE_PLUGIN_ROOT}/docs/missing.md)")
        self.assertEqual(len(errors), 1)
        self.assertIn("${CLAUDE_PLUGIN_ROOT}/docs/missing.md", errors[0])


def _check_formatting(content: str, *, name: str = "cmd.md") -> list[tuple[str, str]]:
    """Run _check_markdown_formatting on `content` and return (severity, message) pairs."""
    with tempfile.TemporaryDirectory() as td:
        plugin = Path(td) / "test-plugin"
        (plugin / "commands").mkdir(parents=True)
        fp = plugin / "commands" / name
        fp.write_text(content)
        checker = DocumentationChecker()
        result = DocCheckResult(plugin_name="test", plugin_path=plugin)
        checker._check_markdown_formatting(fp, content, content.splitlines(), result)
        return [(i.severity, i.message) for i in result.issues]


class TestHeadingSpaceInsideCodeBlocks(unittest.TestCase):
    """Regression: heading-space check must skip content inside fenced code blocks.

    Pre-fix, prompt-syntax examples like ##CONTEXT## inside ```text ... ```
    blocks and Julia docstring markers like #= inside ```julia ... ``` blocks
    were flagged as malformed markdown headings.
    """

    def test_prompt_syntax_in_code_block_not_flagged(self):
        content = (
            "### GPT-4 Style\n"
            "```\n"
            "##CONTEXT##\n"
            "##OBJECTIVE##\n"
            "##INSTRUCTIONS## (numbered)\n"
            "##OUTPUT FORMAT## (JSON/structured)\n"
            "```\n"
        )
        issues = _check_formatting(content)
        heading_warnings = [m for _, m in issues if "Heading should have space" in m]
        self.assertEqual(heading_warnings, [])

    def test_julia_docstring_marker_in_code_block_not_flagged(self):
        content = (
            "Example:\n"
            "```julia\n"
            "#= block docstring\n"
            "   describing the function\n"
            "=#\n"
            "function foo() end\n"
            "```\n"
        )
        issues = _check_formatting(content)
        heading_warnings = [m for _, m in issues if "Heading should have space" in m]
        self.assertEqual(heading_warnings, [])

    def test_real_malformed_heading_outside_code_block_still_flagged(self):
        content = (
            "##NotAHeadingWithoutSpace\n"
            "\n"
            "```\n"
            "##AlsoInsideBlockButWeIgnore\n"
            "```\n"
        )
        issues = _check_formatting(content)
        heading_warnings = [m for _, m in issues if "Heading should have space" in m]
        # Only the one outside the code block should be flagged.
        self.assertEqual(len(heading_warnings), 1)
        self.assertIn("Line 1", heading_warnings[0])

    def test_proper_heading_with_space_never_flagged(self):
        content = "## Proper Heading\n\nBody text.\n"
        issues = _check_formatting(content)
        heading_warnings = [m for _, m in issues if "Heading should have space" in m]
        self.assertEqual(heading_warnings, [])


def _check_common(content: str, *, name: str = "cmd.md") -> list[tuple[str, str]]:
    """Run _check_common_issues on `content` and return (severity, message) pairs."""
    with tempfile.TemporaryDirectory() as td:
        plugin = Path(td) / "test-plugin"
        (plugin / "commands").mkdir(parents=True)
        fp = plugin / "commands" / name
        fp.write_text(content)
        checker = DocumentationChecker()
        result = DocCheckResult(plugin_name="test", plugin_path=plugin)
        checker._check_common_issues(fp, content, content.splitlines(), result)
        return [(i.severity, i.message) for i in result.issues]


class TestPlaceholderWordFalsePositives(unittest.TestCase):
    """Regression: bare word 'placeholder' should not trigger placeholder-text warning.

    It is a common domain term (HTML img placeholders, prompt-template
    placeholders, the 'placeholder auto-fill' team-assemble feature name).
    Specific patterns like 'lorem ipsum', 'TODO:', 'work in progress' still fire.
    """

    def test_bare_placeholder_not_flagged(self):
        content = "Use a placeholder image to reserve layout space.\n"
        issues = _check_common(content)
        flagged = [m for _, m in issues if "placeholder text" in m.lower()]
        self.assertEqual(flagged, [])

    def test_placeholder_auto_fill_feature_name_not_flagged(self):
        content = "Team assembly does placeholder auto-fill from signals.\n"
        issues = _check_common(content)
        flagged = [m for _, m in issues if "placeholder text" in m.lower()]
        self.assertEqual(flagged, [])

    def test_lorem_ipsum_still_flagged(self):
        content = "Intro: Lorem ipsum dolor sit amet.\n"
        issues = _check_common(content)
        flagged = [m for _, m in issues if "placeholder text" in m.lower()]
        self.assertEqual(len(flagged), 1)

    def test_work_in_progress_still_flagged(self):
        content = "Status: work in progress — more details soon.\n"
        issues = _check_common(content)
        flagged = [m for _, m in issues if "placeholder text" in m.lower()]
        self.assertEqual(len(flagged), 1)


class TestCheckLinksFrontmatter(unittest.TestCase):
    """New behavior added by Task 6 — frontmatter path scanning."""

    def test_broken_documentation_field_flagged(self):
        content = (
            "---\n"
            "name: x\n"
            'documentation:\n'
            '  ref: "${CLAUDE_PLUGIN_ROOT}/docs/missing/page.md"\n'
            "---\n"
            "# Body\n"
        )
        errors = _check(content)
        self.assertTrue(any("missing/page.md" in e for e in errors))

    def test_broken_relative_frontmatter_path_flagged(self):
        content = (
            "---\n"
            "name: x\n"
            'references:\n'
            '  doc: "../missing/path.md"\n'
            "---\n"
            "# Body\n"
        )
        errors = _check(content)
        self.assertTrue(any("../missing/path.md" in e for e in errors))

    def test_frontmatter_non_path_strings_ignored(self):
        # Description text that happens to contain a colon should not be
        # treated as a path reference.
        content = (
            "---\n"
            "name: x\n"
            "description: 'Multi-step workflow: design, build, review'\n"
            "model: opus\n"
            "---\n"
            "# Body\n"
        )
        errors = _check(content)
        self.assertEqual(errors, [])

    def test_body_link_inside_frontmatter_not_double_counted(self):
        # If frontmatter happens to contain a markdown-link-shaped string
        # (unusual), the body scanner should not count it again.
        content = (
            "---\n"
            "name: x\n"
            "tagline: '[skip me](url)'\n"
            "---\n"
            "# Body\n[real-broken](missing.md)\n"
        )
        errors = _check(content)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing.md", errors[0])


if __name__ == "__main__":
    unittest.main()
