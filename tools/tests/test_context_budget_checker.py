#!/usr/bin/env python3
"""Unit tests for tools.validation.context_budget_checker.check_skill_budget.

Validates the policy from MIGRATION.md Task 3:
- 4 KB (200K * 2%) is the absolute gate.
- 20 KB (1M * 2%) is informational headroom.
- Both budgets are reported simultaneously.
"""

import sys
import tempfile
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tools.validation.context_budget_checker import (  # noqa: E402
    CHARS_PER_TOKEN,
    CONTEXT_SIZES,
    SKILL_BUDGET_PERCENT,
    check_skill_budget,
)

# Pre-computed from the constants the validator uses
BUDGET_200K = int(CONTEXT_SIZES["200k"] * SKILL_BUDGET_PERCENT)  # 4,000 tokens
BUDGET_1M = int(CONTEXT_SIZES["1m"] * SKILL_BUDGET_PERCENT)      # 20,000 tokens


def _make_skill(td: Path, name: str, *, body_chars: int) -> Path:
    """Create a SKILL.md file with `body_chars` characters of body content."""
    skill_dir = td / name
    skill_dir.mkdir()
    body = "x" * body_chars
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: test\n---\n# Heading\n\n{body}\n"
    )
    return skill_dir


class TestCheckSkillBudget(unittest.TestCase):
    def test_returns_none_when_skill_md_missing(self):
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "no-skill"
            empty.mkdir()
            self.assertIsNone(check_skill_budget(empty, "test"))

    def test_small_skill_fits_both_budgets(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_skill(Path(td), "small", body_chars=200)
            r = check_skill_budget(sd, "test-plugin")
            assert r is not None
            self.assertTrue(r.fits_200k)
            self.assertTrue(r.fits_1m)
            self.assertEqual(r.budget_200k, BUDGET_200K)
            self.assertEqual(r.budget_1m, BUDGET_1M)
            self.assertEqual(r.plugin_name, "test-plugin")
            self.assertEqual(r.skill_name, "small")

    def test_skill_just_under_200k_budget_fits(self):
        # ~3,800 tokens worth of body — under the 4,000-token 200K cap.
        chars = int(BUDGET_200K * CHARS_PER_TOKEN * 0.95)
        with tempfile.TemporaryDirectory() as td:
            sd = _make_skill(Path(td), "headroom", body_chars=chars)
            r = check_skill_budget(sd, "test-plugin")
            assert r is not None
            self.assertTrue(r.fits_200k)
            self.assertTrue(r.fits_1m)

    def test_skill_over_200k_budget_fails_200k_passes_1m(self):
        # ~5,000 tokens of body — exceeds 200K cap (4,000) but well under 1M (20,000).
        chars = int(BUDGET_200K * CHARS_PER_TOKEN * 1.25)
        with tempfile.TemporaryDirectory() as td:
            sd = _make_skill(Path(td), "oversized-200k", body_chars=chars)
            r = check_skill_budget(sd, "test-plugin")
            assert r is not None
            self.assertFalse(r.fits_200k)
            self.assertTrue(r.fits_1m)

    def test_skill_over_1m_budget_fails_both(self):
        # ~22,000 tokens — exceeds 1M cap.
        chars = int(BUDGET_1M * CHARS_PER_TOKEN * 1.1)
        with tempfile.TemporaryDirectory() as td:
            sd = _make_skill(Path(td), "huge", body_chars=chars)
            r = check_skill_budget(sd, "test-plugin")
            assert r is not None
            self.assertFalse(r.fits_200k)
            self.assertFalse(r.fits_1m)

    def test_frontmatter_detection(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_skill(Path(td), "with-fm", body_chars=100)
            r = check_skill_budget(sd, "test")
            assert r is not None
            self.assertTrue(r.has_frontmatter)

    def test_no_frontmatter_detection(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td) / "raw"
            sd.mkdir()
            (sd / "SKILL.md").write_text("# Just a heading and body, no frontmatter.\n")
            r = check_skill_budget(sd, "test")
            assert r is not None
            self.assertFalse(r.has_frontmatter)

    def test_first_section_tokens_reported(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_skill(Path(td), "fs", body_chars=2000)
            r = check_skill_budget(sd, "test")
            assert r is not None
            # First section spans frontmatter + heading + body up to next ##
            self.assertGreater(r.first_section_tokens, 0)

    def test_budget_constants_match_documented_policy(self):
        """Lock in the policy: 4KB at 200K, 20KB at 1M (both = 2%)."""
        self.assertEqual(BUDGET_200K, 4_000)
        self.assertEqual(BUDGET_1M, 20_000)
        self.assertEqual(SKILL_BUDGET_PERCENT, 0.02)


if __name__ == "__main__":
    unittest.main()
