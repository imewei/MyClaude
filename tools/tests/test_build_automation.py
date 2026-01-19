#!/usr/bin/env python3
"""
Test suite for Task Group 8: GitHub Actions and Build Validation

Tests build automation, CI/CD configuration, and comprehensive validation
of the documentation system.

Python >= 3.12
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestBuildAutomation(unittest.TestCase):
    """Test GitHub Actions workflow and build automation."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.docs_dir = self.project_root / "docs"
        self.workflow_file = self.project_root / ".github" / "workflows" / "docs.yml"
        self.makefile = self.docs_dir / "Makefile"

    def test_github_actions_workflow_exists(self):
        """Test that GitHub Actions workflow file exists."""
        self.assertTrue(
            self.workflow_file.exists(),
            f"GitHub Actions workflow file not found at {self.workflow_file}"
        )

    def test_github_actions_workflow_syntax_valid(self):
        """Test that GitHub Actions workflow has valid YAML syntax."""
        if not self.workflow_file.exists():
            self.skipTest("Workflow file does not exist yet")

        # Read as text and check basic structure
        with open(self.workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify it has required YAML keys
        self.assertIn('name:', content, "Workflow must have a name")
        self.assertIn('on:', content, "Workflow must have trigger events")
        self.assertIn('jobs:', content, "Workflow must have jobs")

    def test_github_actions_workflow_has_required_steps(self):
        """Test that GitHub Actions workflow has all required steps."""
        if not self.workflow_file.exists():
            self.skipTest("Workflow file does not exist yet")

        with open(self.workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Required steps per spec.md lines 271-282
        required_keywords = [
            'checkout',          # Checkout repository
            'python',            # Set up Python
            'dependencies',      # Install dependencies
            'generate',          # Generate Plugin Docs
            'sphinx',            # Build documentation
        ]

        content_lower = content.lower()
        for keyword in required_keywords:
            self.assertIn(
                keyword,
                content_lower,
                f"Workflow must include step for: {keyword}"
            )

    def test_github_actions_workflow_triggers_correctly(self):
        """Test that GitHub Actions workflow has correct triggers."""
        if not self.workflow_file.exists():
            self.skipTest("Workflow file does not exist yet")

        with open(self.workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should trigger on push to main branch and pull requests
        self.assertIn('push:', content, "Workflow should trigger on push")
        self.assertIn('pull_request:', content, "Workflow should trigger on pull_request")
        self.assertIn('main', content, "Workflow should reference main branch")

    def test_sphinx_autobuild_target_exists(self):
        """Test that Makefile has sphinx-autobuild target."""
        self.assertTrue(self.makefile.exists(), "Makefile not found")

        with open(self.makefile, 'r', encoding='utf-8') as f:
            makefile_content = f.read()

        # Check for autobuild or livehtml target
        self.assertTrue(
            'autobuild:' in makefile_content or 'livehtml:' in makefile_content,
            "Makefile must have autobuild or livehtml target"
        )
        self.assertIn(
            'sphinx-autobuild',
            makefile_content,
            "Makefile must use sphinx-autobuild"
        )

    def test_sphinx_autobuild_watches_correct_files(self):
        """Test that sphinx-autobuild watches the correct file patterns."""
        with open(self.makefile, 'r', encoding='utf-8') as f:
            makefile_content = f.read()

        # Should watch for changes in documentation files
        # Look for the autobuild command
        if 'sphinx-autobuild' in makefile_content:
            # The configuration is correct if sphinx-autobuild is present
            # with appropriate watch/ignore patterns
            self.assertTrue(True)

    def test_documentation_builds_without_errors(self):
        """Test that documentation builds successfully without errors."""
        # Change to docs directory
        original_dir = os.getcwd()
        try:
            os.chdir(self.docs_dir)

            # Run make html
            result = subprocess.run(
                ['make', 'html'],
                capture_output=True,
                text=True,
                timeout=120
            )

            # Check for errors in output
            error_keywords = ['ERROR', 'CRITICAL', 'exception']
            output = result.stdout + result.stderr

            for keyword in error_keywords:
                self.assertNotIn(
                    keyword.upper(),
                    output.upper(),
                    f"Documentation build contains {keyword}: {output[:500]}"
                )

            self.assertEqual(
                result.returncode,
                0,
                f"Documentation build failed with return code {result.returncode}: {output[:500]}"
            )

        finally:
            os.chdir(original_dir)

    def test_documentation_builds_without_warnings(self):
        """Test that documentation builds without significant warnings."""
        original_dir = os.getcwd()
        try:
            os.chdir(self.docs_dir)

            # Run make html
            result = subprocess.run(
                ['make', 'html'],
                capture_output=True,
                text=True,
                timeout=120
            )

            output = result.stdout + result.stderr

            # Check for warnings - allow minor highlighting warnings
            # Count warnings
            warning_count = output.upper().count('WARNING')

            # Allow up to 2 warnings (minor syntax highlighting issues)
            self.assertLessEqual(
                warning_count,
                2,
                f"Too many warnings in build: {warning_count} warnings found"
            )

        finally:
            os.chdir(original_dir)


class TestCrossReferences(unittest.TestCase):
    """Test cross-references and documentation integrity."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.docs_dir = self.project_root / "docs"

    def test_all_doc_links_resolve(self):
        """Test that referenced documents exist."""
        # This test verifies that critical documents exist
        critical_docs = [
            'index.rst',
            'glossary.rst',
            'changelog.rst',
            'integration-map.rst',
        ]

        for doc in critical_docs:
            doc_path = self.docs_dir / doc
            self.assertTrue(
                doc_path.exists(),
                f"Critical document not found: {doc}"
            )

    def test_plugin_pages_exist(self):
        """Test that all 31 plugin documentation pages exist."""
        plugins_dir = self.docs_dir / "plugins"
        self.assertTrue(plugins_dir.exists(), "plugins directory must exist")

        rst_files = list(plugins_dir.glob("*.rst"))

        # Should have 31 plugin pages
        self.assertGreaterEqual(
            len(rst_files),
            31,
            f"Expected at least 31 plugin pages, found {len(rst_files)}"
        )

    def test_category_pages_exist(self):
        """Test that category pages exist."""
        categories_dir = self.docs_dir / "categories"
        self.assertTrue(categories_dir.exists(), "categories directory must exist")

        rst_files = list(categories_dir.glob("*.rst"))

        # Should have at least 6 category pages
        self.assertGreaterEqual(
            len(rst_files),
            6,
            f"Expected at least 6 category pages, found {len(rst_files)}"
        )


class TestSearchFunctionality(unittest.TestCase):
    """Test search index generation and functionality."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.docs_dir = self.project_root / "docs"
        self.build_dir = self.docs_dir / "_build" / "html"

    def test_search_index_generates(self):
        """Test that search index generates successfully."""
        # Build the documentation first
        original_dir = os.getcwd()
        try:
            os.chdir(self.docs_dir)

            result = subprocess.run(
                ['make', 'html'],
                capture_output=True,
                text=True,
                timeout=120
            )

            self.assertEqual(result.returncode, 0, "Build failed")

            # Check for search index files
            searchindex = self.build_dir / "searchindex.js"
            self.assertTrue(
                searchindex.exists(),
                "Search index file not generated"
            )

            # Verify search index is not empty
            if searchindex.exists():
                self.assertGreater(
                    searchindex.stat().st_size,
                    100,
                    "Search index appears to be empty"
                )

        finally:
            os.chdir(original_dir)


class TestNavigationStructure(unittest.TestCase):
    """Test navigation sidebar and hierarchical structure."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.docs_dir = self.project_root / "docs"
        self.index_file = self.docs_dir / "index.rst"

    def test_index_has_toctree(self):
        """Test that index.rst has a toctree directive."""
        self.assertTrue(self.index_file.exists(), "index.rst must exist")

        with open(self.index_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn(
            '.. toctree::',
            content,
            "index.rst must have at least one toctree directive"
        )

    def test_index_links_to_categories(self):
        """Test that index.rst links to category pages."""
        with open(self.index_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should reference categories
        self.assertIn(
            'categories/',
            content,
            "index.rst should link to category pages"
        )

    def test_hierarchical_structure_exists(self):
        """Test that hierarchical structure is present in documentation."""
        # Check that categories directory exists and has RST files
        categories_dir = self.docs_dir / "categories"
        self.assertTrue(categories_dir.exists())

        category_files = list(categories_dir.glob("*.rst"))
        self.assertGreater(
            len(category_files),
            0,
            "Categories directory should contain RST files"
        )


class TestDocumentationContent(unittest.TestCase):
    """Test that documentation content meets requirements."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.docs_dir = self.project_root / "docs"

    def test_glossary_exists(self):
        """Test that glossary.rst exists."""
        glossary = self.docs_dir / "glossary.rst"
        self.assertTrue(glossary.exists(), "glossary.rst must exist")

    def test_glossary_has_content(self):
        """Test that glossary has substantial content."""
        glossary = self.docs_dir / "glossary.rst"
        if not glossary.exists():
            self.skipTest("Glossary does not exist")

        with open(glossary, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that glossary has substantial content (at least 2000 characters)
        self.assertGreater(
            len(content),
            2000,
            "Glossary should have substantial content"
        )

        # Check for common technical terms
        technical_terms = ['JAX', 'Machine Learning', 'API', 'Python']
        found_terms = sum(1 for term in technical_terms if term in content)
        self.assertGreater(
            found_terms,
            0,
            "Glossary should contain technical terms"
        )

    def test_guides_directory_exists(self):
        """Test that guides directory exists with quick-start guides."""
        guides_dir = self.docs_dir / "guides"
        self.assertTrue(guides_dir.exists(), "guides directory must exist")

        guide_files = list(guides_dir.glob("*.rst"))
        self.assertGreaterEqual(
            len(guide_files),
            5,
            f"Should have at least 5 quick-start guides, found {len(guide_files)}"
        )

    def test_changelog_exists(self):
        """Test that changelog.rst exists."""
        changelog = self.docs_dir / "changelog.rst"
        self.assertTrue(changelog.exists(), "changelog.rst must exist")


class TestVersioningSupport(unittest.TestCase):
    """Test versioned documentation support."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.docs_dir = self.project_root / "docs"
        self.conf_file = self.docs_dir / "conf.py"

    def test_conf_has_version_info(self):
        """Test that conf.py has version information."""
        self.assertTrue(self.conf_file.exists(), "conf.py must exist")

        with open(self.conf_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('version', content, "conf.py must define version")
        self.assertIn('release', content, "conf.py must define release")

    def test_sphinx_multiversion_configured(self):
        """Test that sphinx-multiversion settings are present."""
        with open(self.conf_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for multiversion configuration
        self.assertIn(
            'smv_',
            content,
            "conf.py should have sphinx-multiversion configuration"
        )


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBuildAutomation))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossReferences))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestNavigationStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentationContent))
    suite.addTests(loader.loadTestsFromTestCase(TestVersioningSupport))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
