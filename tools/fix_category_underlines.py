#!/usr/bin/env python3
"""
Fix RST underline length issues in category pages.

Python >= 3.12
"""

from pathlib import Path


def fix_underlines(content: str) -> str:
    """
    Fix RST section underlines to match title length.

    Args:
        content: RST content to fix

    Returns:
        Fixed content
    """
    lines = content.split('\n')
    fixed_lines = []

    i = 0
    while i < len(lines):
        fixed_lines.append(lines[i])

        # Check if next line is an underline
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            current_line = lines[i]

            # Check if it's a section marker (all same character, dashes or equals)
            if next_line and all(c in '-=' for c in next_line) and len(set(next_line)) == 1:
                # Adjust underline length to match title
                char = next_line[0]
                correct_underline = char * len(current_line)
                fixed_lines.append(correct_underline)
                i += 2
                continue

        i += 1

    return '\n'.join(fixed_lines)


def fix_category_file(file_path: Path) -> bool:
    """
    Fix a single category RST file.

    Args:
        file_path: Path to the RST file

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        content = fix_underlines(original_content)

        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path.name}")
            return True

        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Fix all category RST files."""
    project_root = Path(__file__).parent.parent
    categories_dir = project_root / "docs" / "categories"

    if not categories_dir.exists():
        print(f"Categories directory not found: {categories_dir}")
        return

    fixed_count = 0
    for rst_file in categories_dir.glob("*.rst"):
        if fix_category_file(rst_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} category files")


if __name__ == '__main__':
    main()
