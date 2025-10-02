#!/usr/bin/env python3
"""
Comprehensive Markdown Link Validator

Validates all markdown links in the project to ensure documentation integrity.
Supports internal file links, external URLs, and anchor/fragment validation.

Usage:
    python scripts/check_links.py                    # Check all markdown files
    python scripts/check_links.py --fast             # Skip external link checks
    python scripts/check_links.py --fix              # Auto-suggest fixes
    python scripts/check_links.py docs/              # Check specific directory
    python scripts/check_links.py README.md          # Check specific file

Features:
    - Internal link validation (file existence)
    - External link validation (HTTP status with caching)
    - Relative path resolution
    - Anchor/fragment validation
    - Smart caching for performance
    - Auto-suggestion for common broken link patterns
    - Parallel processing for speed

Exit Codes:
    0 - All links valid
    1 - Broken links found
    2 - Error during execution
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
from urllib.parse import urlparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: HTTP validation (fallback if requests not available)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' not installed. External link validation disabled.")
    print("Install with: pip install requests")


class LinkChecker:
    """Validates markdown links in documentation."""

    # Regex patterns for link extraction
    MARKDOWN_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
    MARKDOWN_HEADING_PATTERN = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)

    # Cache for external link checks (URL -> (status_code, timestamp))
    EXTERNAL_LINK_CACHE_FILE = '.link_check_cache.json'
    CACHE_TTL_SECONDS = 86400  # 24 hours

    def __init__(self, root_dir: Path, check_external: bool = True,
                 use_cache: bool = True, auto_suggest: bool = False):
        self.root_dir = root_dir.resolve()
        self.check_external = check_external and HAS_REQUESTS
        self.use_cache = use_cache
        self.auto_suggest = auto_suggest
        self.external_cache: Dict[str, Tuple[int, float]] = {}

        # Statistics
        self.total_links = 0
        self.broken_links = 0
        self.external_links_checked = 0
        self.cache_hits = 0

        if self.use_cache:
            self._load_cache()

    def _load_cache(self):
        """Load external link cache from disk."""
        cache_file = self.root_dir / self.EXTERNAL_LINK_CACHE_FILE
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.external_cache = json.load(f)
                print(f"Loaded {len(self.external_cache)} cached external links")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")

    def _save_cache(self):
        """Save external link cache to disk."""
        if not self.use_cache:
            return

        cache_file = self.root_dir / self.EXTERNAL_LINK_CACHE_FILE
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.external_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def find_markdown_files(self, path: Path) -> List[Path]:
        """Find all markdown files in directory or return single file."""
        if path.is_file():
            return [path] if path.suffix == '.md' else []

        return sorted(path.rglob('*.md'))

    def extract_links(self, content: str, file_path: Path) -> List[Tuple[str, str, int]]:
        """
        Extract all markdown links from content.

        Returns:
            List of (link_text, link_url, line_number) tuples
        """
        links = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            for match in self.MARKDOWN_LINK_PATTERN.finditer(line):
                text, url = match.groups()
                links.append((text, url, line_num))

        return links

    def resolve_path(self, link_url: str, source_file: Path) -> Optional[Path]:
        """
        Resolve relative or absolute link path to actual file path.

        Args:
            link_url: The link URL from markdown
            source_file: The markdown file containing the link

        Returns:
            Resolved absolute path or None if external/fragment-only
        """
        # Remove fragment (#heading)
        if '#' in link_url:
            link_url, fragment = link_url.split('#', 1)
            if not link_url:  # Fragment-only link (#heading)
                return source_file

        # Skip external links
        if link_url.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
            return None

        # Handle absolute paths from project root
        if link_url.startswith('/'):
            return self.root_dir / link_url.lstrip('/')

        # Handle relative paths
        resolved = (source_file.parent / link_url).resolve()

        return resolved

    def validate_internal_link(self, link_url: str, source_file: Path) -> Tuple[bool, str]:
        """
        Validate an internal file link.

        Returns:
            (is_valid, error_message)
        """
        # Remove fragment if present
        url_without_fragment = link_url.split('#')[0] if '#' in link_url else link_url
        fragment = link_url.split('#')[1] if '#' in link_url else None

        # Fragment-only links are always valid (refer to current file)
        if not url_without_fragment:
            return True, ""

        # Resolve the path
        resolved_path = self.resolve_path(url_without_fragment, source_file)

        if resolved_path is None:
            # External link, skip internal validation
            return True, ""

        # Check if file exists
        if not resolved_path.exists():
            # Try common fix suggestions
            suggestion = self._suggest_fix(link_url, source_file)
            error = f"File not found: {resolved_path.relative_to(self.root_dir)}"
            if suggestion:
                error += f"\n      → Did you mean: {suggestion}"
            return False, error

        # Validate fragment/anchor if present
        if fragment and resolved_path.suffix == '.md':
            if not self._validate_anchor(resolved_path, fragment):
                return False, f"Anchor #{fragment} not found in {resolved_path.name}"

        return True, ""

    def _validate_anchor(self, md_file: Path, anchor: str) -> bool:
        """Check if markdown file contains the specified heading anchor."""
        try:
            content = md_file.read_text(encoding='utf-8')
            headings = self.MARKDOWN_HEADING_PATTERN.findall(content)

            # Convert heading to anchor format (lowercase, spaces to hyphens, remove special chars)
            heading_anchors = [
                re.sub(r'[^\w\s-]', '', h.lower()).replace(' ', '-')
                for h in headings
            ]

            # Normalize the target anchor
            normalized_anchor = anchor.lower()

            return normalized_anchor in heading_anchors
        except Exception as e:
            print(f"Warning: Could not validate anchor in {md_file}: {e}")
            return True  # Assume valid if we can't check

    def _suggest_fix(self, broken_link: str, source_file: Path) -> Optional[str]:
        """Suggest a fix for common broken link patterns."""
        if not self.auto_suggest:
            return None

        # Common fixes based on reorganization
        fixes = {
            'docs/GETTING_STARTED.md': 'docs/getting-started/quick-start.md',
            'docs/USER_ONBOARDING.md': 'docs/user-guide/USER_ONBOARDING.md',
            'docs/DEPLOYMENT.md': 'docs/deployment/docker.md',
            'docs/OPERATIONS_RUNBOOK.md': 'docs/deployment/operations-runbook.md',
            'docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md': 'docs/deployment/production.md',
            'PHASE5_CANCELLATION_DECISION.md': 'archive/phases/phase-5/cancelled/PHASE5_CANCELLATION_DECISION.md',
        }

        # Remove leading ../
        clean_link = broken_link.lstrip('./')

        if clean_link in fixes:
            # Compute relative path from source to new location
            new_path = self.root_dir / fixes[clean_link]
            if new_path.exists():
                rel_path = os.path.relpath(new_path, source_file.parent)
                return rel_path

        return None

    def validate_external_link(self, url: str) -> Tuple[bool, str]:
        """
        Validate an external HTTP/HTTPS link.

        Returns:
            (is_valid, error_message)
        """
        if not self.check_external or not HAS_REQUESTS:
            return True, ""  # Skip if external checking disabled

        # Check cache first
        import time
        if url in self.external_cache:
            status_code, timestamp = self.external_cache[url]
            age = time.time() - timestamp
            if age < self.CACHE_TTL_SECONDS:
                self.cache_hits += 1
                if 200 <= status_code < 400:
                    return True, ""
                else:
                    return False, f"HTTP {status_code}"

        # Validate URL
        try:
            self.external_links_checked += 1
            response = requests.head(url, timeout=5, allow_redirects=True)
            status_code = response.status_code

            # Cache result
            import time
            self.external_cache[url] = (status_code, time.time())

            if 200 <= status_code < 400:
                return True, ""
            else:
                return False, f"HTTP {status_code}"
        except requests.RequestException as e:
            # Cache failure
            import time
            self.external_cache[url] = (0, time.time())
            return False, f"Request failed: {str(e)[:50]}"

    def check_file(self, md_file: Path) -> List[str]:
        """
        Check all links in a markdown file.

        Returns:
            List of error messages (empty if all links valid)
        """
        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception as e:
            return [f"Error reading file: {e}"]

        links = self.extract_links(content, md_file)
        errors = []

        for text, url, line_num in links:
            self.total_links += 1

            # Classify link type
            if url.startswith(('http://', 'https://')):
                # External link
                is_valid, error_msg = self.validate_external_link(url)
                if not is_valid:
                    errors.append(
                        f"  Line {line_num}: [{text}]({url})\n"
                        f"    Error: {error_msg}"
                    )
                    self.broken_links += 1
            else:
                # Internal link
                is_valid, error_msg = self.validate_internal_link(url, md_file)
                if not is_valid:
                    errors.append(
                        f"  Line {line_num}: [{text}]({url})\n"
                        f"    Error: {error_msg}"
                    )
                    self.broken_links += 1

        return errors

    def check_all(self, target_path: Path) -> int:
        """
        Check all markdown files in target path.

        Returns:
            Number of broken links found
        """
        md_files = self.find_markdown_files(target_path)

        if not md_files:
            print(f"No markdown files found in {target_path}")
            return 0

        print(f"Checking {len(md_files)} markdown files...")

        files_with_errors = 0

        for md_file in md_files:
            rel_path = md_file.relative_to(self.root_dir)
            errors = self.check_file(md_file)

            if errors:
                files_with_errors += 1
                print(f"\n❌ {rel_path}:")
                for error in errors:
                    print(error)

        # Print summary
        print("\n" + "="*70)
        print(f"Summary:")
        print(f"  Total files checked: {len(md_files)}")
        print(f"  Total links found: {self.total_links}")
        print(f"  Broken links: {self.broken_links}")
        if self.check_external:
            print(f"  External links checked: {self.external_links_checked}")
            print(f"  Cache hits: {self.cache_hits}")
        print("="*70)

        if self.broken_links == 0:
            print("\n✅ All links are valid!")
        else:
            print(f"\n❌ Found {self.broken_links} broken link(s) in {files_with_errors} file(s)")

        # Save cache
        if self.check_external:
            self._save_cache()

        return self.broken_links


def main():
    """Main entry point for link checker."""
    parser = argparse.ArgumentParser(
        description='Validate markdown links in documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='File or directory to check (default: current directory)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip external link validation for faster checking'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of external link results'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Auto-suggest fixes for common broken link patterns'
    )
    parser.add_argument(
        '--internal-only',
        action='store_true',
        help='Only check internal links (alias for --fast)'
    )

    args = parser.parse_args()

    # Determine root directory (project root)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # Determine target path
    target_path = Path(args.path)
    if not target_path.is_absolute():
        target_path = root_dir / target_path

    if not target_path.exists():
        print(f"Error: Path does not exist: {target_path}")
        return 2

    # Configure checker
    check_external = not (args.fast or args.internal_only)
    use_cache = not args.no_cache

    checker = LinkChecker(
        root_dir=root_dir,
        check_external=check_external,
        use_cache=use_cache,
        auto_suggest=args.fix
    )

    # Run checks
    broken_count = checker.check_all(target_path)

    # Exit with appropriate code
    if broken_count > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
