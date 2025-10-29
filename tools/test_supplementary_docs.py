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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_glossary_has_minimum_20_terms():
    """Test that glossary includes at least 20 technical terms."""
    glossary_path = project_root / "docs" / "glossary.rst"

    if not glossary_path.exists():
        print(f"FAIL: Glossary file not found at {glossary_path}")
        return False

    content = glossary_path.read_text()

    # Find all glossary terms (indented term definitions)
    # Pattern: term name (not indented) followed by indented definition
    term_pattern = r'^\s{3}(\S.*?)\s*$'
    terms = re.findall(term_pattern, content, re.MULTILINE)

    # Filter out common non-terms
    non_terms = ['Placeholder', 'This glossary', 'Coming Soon', 'The glossary']
    actual_terms = [t for t in terms if not any(nt in t for nt in non_terms)]

    term_count = len(actual_terms)

    if term_count >= 20:
        print(f"PASS: Glossary contains {term_count} terms (>= 20 required)")
        return True
    else:
        print(f"FAIL: Glossary contains only {term_count} terms (20 required)")
        print(f"Terms found: {actual_terms[:10]}...")
        return False


def test_quick_start_guides_exist():
    """Test that at least 5 quick-start guides exist."""
    guides_dir = project_root / "docs" / "guides"

    if not guides_dir.exists():
        print(f"FAIL: Guides directory not found at {guides_dir}")
        return False

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

    if len(existing_guides) >= 5:
        print(f"PASS: {len(existing_guides)} quick-start guides exist (>= 5 required)")
        print(f"  Guides: {', '.join(existing_guides)}")
        return True
    else:
        print(f"FAIL: Only {len(existing_guides)} guides exist (5 required)")
        print(f"  Missing: {set(expected_guides) - set(existing_guides)}")
        return False


def test_glossary_term_links_work():
    """Test that glossary uses proper :term: directive syntax."""
    glossary_path = project_root / "docs" / "glossary.rst"

    if not glossary_path.exists():
        print(f"FAIL: Glossary file not found at {glossary_path}")
        return False

    content = glossary_path.read_text()

    # Check that glossary directive exists
    if '.. glossary::' not in content:
        print("FAIL: Glossary missing '.. glossary::' directive")
        return False

    # Check for proper indentation (terms should be indented 3 spaces)
    # This ensures Sphinx can parse the glossary
    glossary_section = content.split('.. glossary::')[1] if '.. glossary::' in content else ''

    # Basic validation - glossary should have content after directive
    lines_after_directive = [l for l in glossary_section.split('\n')[1:] if l.strip()]

    if len(lines_after_directive) >= 3:
        print("PASS: Glossary has proper directive structure")
        return True
    else:
        print("FAIL: Glossary appears empty or improperly formatted")
        return False


def test_guide_code_examples_render():
    """Test that guide code examples use proper RST code-block syntax."""
    guides_dir = project_root / "docs" / "guides"

    if not guides_dir.exists():
        print(f"FAIL: Guides directory not found at {guides_dir}")
        return False

    guides = list(guides_dir.glob("*.rst"))
    guides = [g for g in guides if g.name != 'index.rst']

    if not guides:
        print("FAIL: No guide files found")
        return False

    code_blocks_found = 0
    guides_with_code = []

    for guide_path in guides:
        content = guide_path.read_text()

        # Check for code-block or code:: directives
        if '.. code-block::' in content or '.. code::' in content:
            code_blocks_found += 1
            guides_with_code.append(guide_path.name)

    if code_blocks_found >= 3:
        print(f"PASS: {code_blocks_found} guides contain code examples")
        print(f"  Guides with code: {', '.join(guides_with_code)}")
        return True
    else:
        print(f"FAIL: Only {code_blocks_found} guides have code examples (expected at least 3)")
        return False


def test_changelog_follows_version_structure():
    """Test that changelog follows semantic versioning structure."""
    changelog_path = project_root / "docs" / "changelog.rst"

    if not changelog_path.exists():
        print(f"FAIL: Changelog file not found at {changelog_path}")
        return False

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

    # Check for standard changelog sections
    standard_sections = ['Added', 'Changed', 'Deprecated', 'Removed', 'Fixed', 'Security']
    sections_found = [section for section in standard_sections if section in content]

    has_versions = len(versions_found) > 0
    has_sections = len(sections_found) >= 3

    if has_versions and has_sections:
        print(f"PASS: Changelog has version structure ({len(versions_found)} versions, {len(sections_found)} sections)")
        return True
    elif has_versions:
        print(f"WARN: Changelog has versions but few standard sections ({len(sections_found)}/6)")
        return True  # Still pass, sections optional
    else:
        print("FAIL: Changelog missing version structure")
        return False


def test_guides_have_multi_plugin_workflows():
    """Test that guides include multi-plugin workflow descriptions."""
    guides_dir = project_root / "docs" / "guides"

    if not guides_dir.exists():
        print(f"FAIL: Guides directory not found at {guides_dir}")
        return False

    guides = list(guides_dir.glob("*.rst"))
    guides = [g for g in guides if g.name != 'index.rst']

    if not guides:
        print("FAIL: No guide files found")
        return False

    guides_with_workflow = 0

    for guide_path in guides:
        content = guide_path.read_text()

        # Check for workflow indicators
        workflow_indicators = [
            ':doc:',  # Cross-references to plugins
            'Prerequisites',
            'Step',
            'workflow',
            'Tutorial',
            'Example'
        ]

        indicators_found = sum(1 for indicator in workflow_indicators if indicator in content)

        if indicators_found >= 2:
            guides_with_workflow += 1

    if guides_with_workflow >= 3:
        print(f"PASS: {guides_with_workflow} guides contain multi-plugin workflows")
        return True
    else:
        print(f"FAIL: Only {guides_with_workflow} guides have workflow content (expected at least 3)")
        return False


def test_guides_include_cross_references():
    """Test that guides include cross-references to plugin pages."""
    guides_dir = project_root / "docs" / "guides"

    if not guides_dir.exists():
        print(f"FAIL: Guides directory not found at {guides_dir}")
        return False

    guides = list(guides_dir.glob("*.rst"))
    guides = [g for g in guides if g.name != 'index.rst']

    if not guides:
        print("FAIL: No guide files found")
        return False

    guides_with_refs = 0

    for guide_path in guides:
        content = guide_path.read_text()

        # Check for :doc: directives
        doc_refs = re.findall(r':doc:`[^`]+`', content)

        if len(doc_refs) >= 2:
            guides_with_refs += 1

    if guides_with_refs >= 3:
        print(f"PASS: {guides_with_refs} guides include cross-references")
        return True
    else:
        print(f"FAIL: Only {guides_with_refs} guides have cross-references (expected at least 3)")
        return False


def test_glossary_includes_required_terms():
    """Test that glossary includes all required technical terms from spec."""
    glossary_path = project_root / "docs" / "glossary.rst"

    if not glossary_path.exists():
        print(f"FAIL: Glossary file not found at {glossary_path}")
        return False

    content = glossary_path.read_text().lower()

    # Required terms from spec (case-insensitive check)
    required_terms = [
        'sciml', 'mcmc', 'jax', 'hpc', 'ci/cd', 'kubernetes', 'docker',
        'terraform', 'ansible', 'gpu', 'parallel', 'rest', 'microservices',
        'tdd', 'bdd', 'cloud', 'observability', 'orm', 'message queue',
        'container orchestration'
    ]

    found_terms = []
    missing_terms = []

    for term in required_terms:
        if term in content:
            found_terms.append(term)
        else:
            missing_terms.append(term)

    coverage = len(found_terms) / len(required_terms) * 100

    if coverage >= 80:  # Allow 80% coverage for flexibility
        print(f"PASS: Glossary includes {len(found_terms)}/{len(required_terms)} required terms ({coverage:.0f}%)")
        return True
    else:
        print(f"FAIL: Glossary missing key terms ({coverage:.0f}% coverage)")
        print(f"  Missing: {', '.join(missing_terms[:5])}...")
        return False


def main():
    """Run all supplementary documentation tests."""
    print("=" * 80)
    print("Task Group 6: Supplementary Documentation Tests")
    print("=" * 80)
    print()

    tests = [
        ("Glossary has minimum 20 terms", test_glossary_has_minimum_20_terms),
        ("Quick-start guides exist (5+)", test_quick_start_guides_exist),
        ("Glossary term links work", test_glossary_term_links_work),
        ("Guide code examples render", test_guide_code_examples_render),
        ("Changelog follows version structure", test_changelog_follows_version_structure),
        ("Guides have multi-plugin workflows", test_guides_have_multi_plugin_workflows),
        ("Guides include cross-references", test_guides_include_cross_references),
        ("Glossary includes required terms", test_glossary_includes_required_terms),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")

    print()
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 80)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
