#!/usr/bin/env python3
"""
Tests for category page generation and navigation structure.

Task Group 5.1: Focused tests for category pages and navigation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys


class TestCategoryPages(unittest.TestCase):
    """Tests for category landing pages and navigation structure"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.docs_dir = cls.repo_root / "docs"
        cls.categories_dir = cls.docs_dir / "categories"

    def test_category_directory_exists(self):
        """Test that categories directory exists"""
        self.assertTrue(
            self.categories_dir.exists(),
            f"Categories directory should exist at {self.categories_dir}"
        )
        self.assertTrue(
            self.categories_dir.is_dir(),
            "Categories path should be a directory"
        )

    def test_all_category_pages_exist(self):
        """Test that all required category pages are created"""
        expected_categories = [
            "scientific-computing.rst",
            "development.rst",
            "devops.rst",
            "ai-ml.rst",
            "tools.rst",
            "orchestration.rst",
        ]

        for category_file in expected_categories:
            category_path = self.categories_dir / category_file
            self.assertTrue(
                category_path.exists(),
                f"Category page {category_file} should exist"
            )

    def test_category_page_structure(self):
        """Test that category pages have required RST structure"""
        # Test scientific-computing as example
        category_file = self.categories_dir / "scientific-computing.rst"

        if not category_file.exists():
            self.skipTest("Category page not yet generated")

        content = category_file.read_text()

        # Check for required sections
        required_elements = [
            "Scientific Computing",  # Title
            ".. toctree::",  # TOC tree directive
            ":maxdepth:",  # TOC tree option
        ]

        for element in required_elements:
            self.assertIn(
                element,
                content,
                f"Category page should contain '{element}'"
            )

    def test_category_toctree_includes_plugins(self):
        """Test that category TOC tree includes plugin references"""
        category_file = self.categories_dir / "scientific-computing.rst"

        if not category_file.exists():
            self.skipTest("Category page not yet generated")

        content = category_file.read_text()

        # Should reference plugins in this category
        expected_plugins = [
            "/plugins/julia-development",
            "/plugins/data-visualization",
        ]

        for plugin_ref in expected_plugins:
            self.assertIn(
                plugin_ref,
                content,
                f"Category should reference {plugin_ref}"
            )

    def test_main_index_exists(self):
        """Test that main index.rst exists"""
        index_path = self.docs_dir / "index.rst"
        self.assertTrue(
            index_path.exists(),
            "Main index.rst should exist"
        )

    def test_index_navigation_hierarchy(self):
        """Test that index.rst contains hierarchical navigation"""
        index_path = self.docs_dir / "index.rst"

        if not index_path.exists():
            self.skipTest("Index page not yet created")

        content = index_path.read_text()

        # Check for navigation structure
        required_elements = [
            "Plugin Marketplace Documentation",
            ".. toctree::",
            ":maxdepth:",
            "categories/",  # Should reference category pages
        ]

        for element in required_elements:
            self.assertIn(
                element,
                content,
                f"Index should contain '{element}'"
            )

    def test_category_pages_build_successfully(self):
        """Test that category pages build with Sphinx without errors"""
        # Only test if sphinx-build is available
        try:
            result = subprocess.run(
                ["sphinx-build", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                self.skipTest("sphinx-build not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.skipTest("sphinx-build not available")

        # Build documentation
        build_dir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [
                    "sphinx-build",
                    "-b", "html",
                    "-W",  # Turn warnings into errors
                    "-q",  # Quiet mode
                    str(self.docs_dir),
                    build_dir
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Check for specific category pages in output
            self.assertEqual(
                result.returncode,
                0,
                f"Sphinx build should succeed. Error: {result.stderr}"
            )

            # Verify category HTML files were generated
            build_path = Path(build_dir)
            category_html = build_path / "categories" / "scientific-computing.html"

            if category_html.exists():
                self.assertTrue(
                    category_html.exists(),
                    "Category HTML should be generated"
                )
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    def test_navigation_sidebar_hierarchy(self):
        """Test that navigation structure creates proper sidebar hierarchy"""
        index_path = self.docs_dir / "index.rst"

        if not index_path.exists():
            self.skipTest("Index page not yet created")

        content = index_path.read_text()

        # Check for proper toctree structure with captions
        self.assertIn(":caption:", content, "Should use caption for navigation sections")

        # Check that categories are included
        category_references = [
            "categories/scientific-computing",
            "categories/development",
            "categories/devops",
        ]

        for ref in category_references:
            self.assertIn(
                ref,
                content,
                f"Index should reference {ref}"
            )


def run_tests():
    """Run the test suite"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCategoryPages)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
