#!/usr/bin/env python3
"""
Fix plugin RST files to remove custom directives that aren't recognized by Sphinx.

Converts:
- .. agent:: name -> **Agent: name**
- .. command:: /name -> **Command: /name**
- .. skill:: name -> **Skill: name**

Python >= 3.12
"""

import re
from pathlib import Path


def fix_directive(content: str, directive_type: str) -> str:
    """
    Fix a specific directive type in the content.

    Args:
        content: The RST content to fix
        directive_type: The directive type (agent, command, skill)

    Returns:
        Fixed content
    """
    # Pattern to match the directive
    pattern = rf'\.\. {directive_type}:: (.+)\n\n((?:   .+\n)*)'

    def replace_directive(match):
        name = match.group(1)
        description_lines = match.group(2)

        # Remove the 3-space indentation from description lines
        description = '\n'.join(
            line[3:] if line.startswith('   ') else line
            for line in description_lines.split('\n')
        ).strip()

        # Format as a definition list item
        result = f'**{directive_type.title()}: {name}**\n\n'
        if description:
            result += description + '\n'
        result += '\n'

        return result

    return re.sub(pattern, replace_directive, content, flags=re.MULTILINE)


def fix_plugin_file(file_path: Path) -> bool:
    """
    Fix a single plugin RST file.

    Args:
        file_path: Path to the RST file

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        content = original_content

        # Fix each directive type
        content = fix_directive(content, 'agent')
        content = fix_directive(content, 'command')
        content = fix_directive(content, 'skill')

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
    """Fix all plugin RST files."""
    project_root = Path(__file__).parent.parent
    plugins_dir = project_root / "docs" / "plugins"

    if not plugins_dir.exists():
        print(f"Plugins directory not found: {plugins_dir}")
        return

    fixed_count = 0
    for rst_file in plugins_dir.glob("*.rst"):
        if fix_plugin_file(rst_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} plugin files")


if __name__ == '__main__':
    main()
