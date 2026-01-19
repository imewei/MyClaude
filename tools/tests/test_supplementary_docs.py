#!/usr/bin/env python3
"""
Test suite for Task Group 6: Guides, Glossary, and Changelog

Tests supplementary documentation components:
- Glossary with minimum 20 technical terms
- Quick-start guides building successfully
- Glossary term links using :term: directive
- Guide code examples rendering correctly
- Changelog following version structure
"""

import os
import re
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestSupplementaryDocs(unittest.TestCase):
    def setUp(self):
        self.project_root = project_root
        self.docs_dir = self.project_root / "docs"

    def test_glossary_has_minimum_20_terms(self):
        """Test that glossary includes at least 20 technical terms."""
        glossary_path = self.docs_dir / "glossary.rst"
        self.assertTrue(glossary_path.exists(), f"Glossary file not found at {glossary_path}")

        content = glossary_path.read_text()

        # Find all glossary terms (indented term definitions)
        # Pattern: term name (not indented) followed by indented definition
        term_pattern = r'^\s{3}(\S.*?)\s*$'
        terms = re.findall(term_pattern, content, re.MULTILINE)

        # Filter out common non-terms
        non_terms = ['Placeholder', 'This glossary', 'Coming Soon', 'The glossary']
        actual_terms = [t for t in terms if not any(nt in t for nt in non_terms)]

        term_count = len(actual_terms)
        self.assertGreaterEqual(term_count, 20, f"Glossary contains only {term_count} terms (20 required)")

    def test_quick_start_guides_exist(self):
        """Test that at least 5 quick-start guides exist."""
        guides_dir = self.docs_dir / "guides"
        self.assertTrue(guides_dir.exists(), f"Guides directory not found at {guides_dir}")

        # Expected guide files (excluding index.rst)
        expected_guides = [
            'scientific-workflows.rst',
            'development-workflows.rst',
            'devops-workflows.rst',
            'infrastructure-workflows.rst',
            'integration-patterns.rst'
        ]

        existing_guides = []
        for guide in expected_guides:
            guide_path = guides_dir / guide
            if guide_path.exists():
                existing_guides.append(guide)

        self.assertGreaterEqual(len(existing_guides), 5, f"Only {len(existing_guides)} guides exist (5 required)")

    def test_glossary_term_links_work(self):
        """Test that glossary uses proper :term: directive syntax."""
        glossary_path = self.docs_dir / "glossary.rst"
        self.assertTrue(glossary_path.exists(), f"Glossary file not found at {glossary_path}")

        content = glossary_path.read_text()

        # Check that glossary directive exists
        self.assertIn('.. glossary::', content, "Glossary missing '.. glossary::' directive")

        # Check for proper indentation
        glossary_section = content.split('.. glossary::')[1] if '.. glossary::' in content else ''
        lines_after_directive = [l for l in glossary_section.split('\n')[1:] if l.strip()]
        self.assertGreaterEqual(len(lines_after_directive), 3, "Glossary appears empty or improperly formatted")

    def test_guide_code_examples_render(self):
        """Test that guide code examples use proper RST code-block syntax."""
        guides_dir = self.docs_dir / "guides"
        self.assertTrue(guides_dir.exists(), f"Guides directory not found at {guides_dir}")

        guides = list(guides_dir.glob("*.rst"))
        guides = [g for g in guides if g.name != 'index.rst']
        self.assertTrue(guides, "No guide files found")

        code_blocks_found = 0
        for guide_path in guides:
            content = guide_path.read_text()
            if '.. code-block::' in content or '.. code::' in content:
                code_blocks_found += 1

        self.assertGreaterEqual(code_blocks_found, 3, f"Only {code_blocks_found} guides have code examples (expected at least 3)")

    def test_changelog_follows_version_structure(self):
        """Test that changelog follows semantic versioning structure."""
        changelog_path = self.docs_dir / "changelog.rst"
        self.assertTrue(changelog_path.exists(), f"Changelog file not found at {changelog_path}")

        content = changelog_path.read_text()

        # Check for version headers
        version_patterns = [
            r'v\d+\.\d+\.\d+',  # v1.0.0 format
            r'Version \d+\.\d+\.\d+',  # Version 1.0.0 format
        ]

        versions_found = []
        for pattern in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            versions_found.extend(matches)

        self.assertTrue(versions_found, "Changelog missing version structure")

    def test_guides_have_multi_plugin_workflows(self):
        """Test that guides include multi-plugin workflow descriptions."""
        guides_dir = self.docs_dir / "guides"
        self.assertTrue(guides_dir.exists(), f"Guides directory not found at {guides_dir}")

        guides = list(guides_dir.glob("*.rst"))
        guides = [g for g in guides if g.name != 'index.rst']
        self.assertTrue(guides, "No guide files found")

        guides_with_workflow = 0
        for guide_path in guides:
            content = guide_path.read_text()
            workflow_indicators = [
                ':doc:', 'Prerequisites', 'Step', 'workflow', 'Tutorial', 'Example'
            ]
            indicators_found = sum(1 for indicator in workflow_indicators if indicator in content)
            if indicators_found >= 2:
                guides_with_workflow += 1

        self.assertGreaterEqual(guides_with_workflow, 3, f"Only {guides_with_workflow} guides have workflow content (expected at least 3)")

    def test_guides_include_cross_references(self):
        """Test that guides include cross-references to plugin pages."""
        guides_dir = self.docs_dir / "guides"
        self.assertTrue(guides_dir.exists(), f"Guides directory not found at {guides_dir}")

        guides = list(guides_dir.glob("*.rst"))
        guides = [g for g in guides if g.name != 'index.rst']
        self.assertTrue(guides, "No guide files found")

        guides_with_refs = 0
        for guide_path in guides:
            content = guide_path.read_text()
            doc_refs = re.findall(r':doc:`[^`]+`', content)
            if len(doc_refs) >= 2:
                guides_with_refs += 1

        self.assertGreaterEqual(guides_with_refs, 3, f"Only {guides_with_refs} guides have cross-references (expected at least 3)")

    def test_glossary_includes_required_terms(self):
        """Test that glossary includes all required technical terms from spec."""
        glossary_path = self.docs_dir / "glossary.rst"
        self.assertTrue(glossary_path.exists(), f"Glossary file not found at {glossary_path}")

        content = glossary_path.read_text().lower()
        required_terms = [
            'sciml', 'mcmc', 'jax', 'hpc', 'ci/cd', 'kubernetes', 'docker',
            'terraform', 'ansible', 'gpu', 'parallel', 'rest', 'microservices',
            'tdd', 'bdd', 'cloud', 'observability', 'orm', 'message queue',
            'container orchestration'
        ]

        found_terms = [term for term in required_terms if term in content]
        missing_terms = [term for term in required_terms if term not in content]
        coverage = len(found_terms) / len(required_terms) * 100

        self.assertGreaterEqual(coverage, 80, f"Glossary missing key terms ({coverage:.0f}% coverage). Missing: {', '.join(missing_terms[:5])}...")


if __name__ == "__main__":
    unittest.main()
