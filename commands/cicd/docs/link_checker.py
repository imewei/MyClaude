#!/usr/bin/env python3
"""
Check for broken links in documentation.

This script checks all links in markdown files and HTML documentation.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed")
    print("Install with: pip install requests")
    sys.exit(1)


class LinkChecker:
    """Check links in documentation."""

    def __init__(self, base_path: Path, check_external: bool = True):
        self.base_path = base_path
        self.check_external = check_external
        self.broken_links = []
        self.checked_urls = {}

    def extract_links_from_markdown(self, file_path: Path) -> List[Tuple[str, int]]:
        """Extract links from markdown file."""
        links = []
        content = file_path.read_text()

        # Markdown links: [text](url)
        for match in re.finditer(r'\[([^\]]+)\]\(([^\)]+)\)', content):
            url = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            links.append((url, line_num))

        # Reference-style links: [text]: url
        for match in re.finditer(r'^\[([^\]]+)\]:\s*(.+)$', content, re.MULTILINE):
            url = match.group(2).strip()
            line_num = content[:match.start()].count('\n') + 1
            links.append((url, line_num))

        # HTML links in markdown
        for match in re.finditer(r'<a\s+href="([^"]+)"', content):
            url = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            links.append((url, line_num))

        return links

    def is_external_link(self, url: str) -> bool:
        """Check if URL is external."""
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https')

    def check_local_link(self, url: str, source_file: Path) -> bool:
        """Check if local link exists."""
        # Remove anchor
        url = url.split('#')[0]

        if not url:  # Just an anchor
            return True

        # Resolve relative path
        if url.startswith('/'):
            target = self.base_path / url.lstrip('/')
        else:
            target = (source_file.parent / url).resolve()

        return target.exists()

    def check_external_link(self, url: str) -> bool:
        """Check if external link is accessible."""
        # Use cache
        if url in self.checked_urls:
            return self.checked_urls[url]

        try:
            response = requests.head(
                url,
                timeout=10,
                allow_redirects=True,
                headers={'User-Agent': 'LinkChecker/1.0'},
            )

            # Try GET if HEAD fails
            if response.status_code >= 400:
                response = requests.get(
                    url,
                    timeout=10,
                    allow_redirects=True,
                    headers={'User-Agent': 'LinkChecker/1.0'},
                )

            is_ok = response.status_code < 400
            self.checked_urls[url] = is_ok
            return is_ok

        except Exception as e:
            print(f"  Warning: Failed to check {url}: {e}")
            self.checked_urls[url] = False
            return False

    def check_file(self, file_path: Path) -> None:
        """Check all links in a file."""
        print(f"\nChecking {file_path.relative_to(self.base_path)}...")

        links = self.extract_links_from_markdown(file_path)

        for url, line_num in links:
            # Skip special URLs
            if url.startswith(('mailto:', 'javascript:', 'data:')):
                continue

            is_external = self.is_external_link(url)

            if is_external:
                if not self.check_external:
                    continue

                if not self.check_external_link(url):
                    self.broken_links.append({
                        'file': file_path,
                        'line': line_num,
                        'url': url,
                        'type': 'external',
                    })
                    print(f"  ❌ Line {line_num}: {url}")
            else:
                if not self.check_local_link(url, file_path):
                    self.broken_links.append({
                        'file': file_path,
                        'line': line_num,
                        'url': url,
                        'type': 'local',
                    })
                    print(f"  ❌ Line {line_num}: {url}")

    def check_all(self) -> bool:
        """Check all markdown files."""
        markdown_files = list(self.base_path.rglob("*.md"))

        print(f"Found {len(markdown_files)} markdown files")

        for md_file in markdown_files:
            self.check_file(md_file)

        return len(self.broken_links) == 0

    def print_summary(self) -> None:
        """Print summary of broken links."""
        print("\n" + "=" * 60)
        print("LINK CHECK SUMMARY")
        print("=" * 60)

        if self.broken_links:
            print(f"❌ Found {len(self.broken_links)} broken links\n")

            # Group by file
            by_file = {}
            for link in self.broken_links:
                file_path = link['file']
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(link)

            for file_path, links in sorted(by_file.items()):
                print(f"\n{file_path.relative_to(self.base_path)}:")
                for link in links:
                    print(f"  Line {link['line']}: {link['url']} ({link['type']})")
        else:
            print("✅ All links are valid")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check documentation links")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Base path to check (default: current directory)",
    )
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Skip external link checking",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error code if broken links found",
    )

    args = parser.parse_args()

    checker = LinkChecker(
        args.path,
        check_external=not args.no_external,
    )

    try:
        all_valid = checker.check_all()
        checker.print_summary()

        if not all_valid and args.fail_on_error:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()