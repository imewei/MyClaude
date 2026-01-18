import json
import os
import sys
import re

def validate_integrity():
    # Determine .agent directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir) # .agent directory
    index_path = os.path.join(base_dir, "skills_index.json")

    print(f"Deep validating agent files in: {base_dir}")

    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        return

    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing {index_path}: {e}")
        return

    # 1. Collect indexed files
    indexed_files = set()
    categories = ['skills', 'workflows']
    for category in categories:
        if category in index:
            for name, data in index[category].items():
                if isinstance(data, dict) and 'path' in data:
                    # Normalize path to absolute
                    full_path = os.path.join(base_dir, data['path'])
                    indexed_files.add(os.path.abspath(full_path))

    # 2. Walk directory for orphans
    print("\n--- Orphaned File Check ---")
    orphan_count = 0
    ignored_dirs = {'scripts', 'assets', 'references', 'docs', '.git'} # Common non-indexed dirs

    for root, dirs, files in os.walk(base_dir):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]

        for file in files:
            if file.endswith('.md') or file.endswith('.py'):
                file_path = os.path.abspath(os.path.join(root, file))

                # Skip the index itself and scripts in the scripts dir
                if file_path == os.path.abspath(index_path):
                    continue
                if os.path.dirname(file_path) == os.path.abspath(script_dir):
                    continue

                if file_path not in indexed_files:
                    # Check if it's a supporting file (in a subdirectory of a skill)
                    # This is a heuristic: if it's not the main SKILL.md, it might be an asset
                    rel_path = os.path.relpath(file_path, base_dir)
                    print(f"[ORPHAN] {rel_path}")
                    orphan_count += 1

    if orphan_count == 0:
        print("No orphaned files found.")
    else:
        print(f"Found {orphan_count} orphaned files (not in skills_index.json).")

    # 3. Check Broken Links
    print("\n--- Broken Link Check ---")
    broken_count = 0
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, base_dir)

                try:
                    with open(file_path, 'r') as f:
                        content = f.read()

                    matches = link_pattern.findall(content)
                    for text, target in matches:
                        # Skip external links, anchors, and absolute paths
                        if target.startswith(('http', 'https', 'mailto', '#', '/')):
                            continue

                        # Handle "Coming Soon" placeholders or other non-path text
                        if target.lower() == "coming soon":
                            continue

                        # Resolve relative path
                        # If target starts with '/', it's absolute (already skipped above usually, but be safe)
                        # We treat paths relative to the current file
                        target_path = target.split('#')[0] # Remove anchor
                        if not target_path:
                            continue

                        # Clean up path (sometimes people leave spaces?)
                        target_path = target_path.strip()

                        abs_target_path = os.path.abspath(os.path.join(root, target_path))

                        if not os.path.exists(abs_target_path):
                            # Heuristic to filter code false positives
                            # Code often looks like [index](arg)
                            if " " in target or "," in target or "(" in target:
                                # Likely code, ignore
                                continue

                            print(f"[BROKEN] In {rel_file_path}: Link '{text}' -> '{target}' not found")
                            broken_count += 1

                except Exception as e:
                    print(f"Error reading {rel_file_path}: {e}")

    if broken_count == 0:
        print("No broken links found.")
    else:
        print(f"Found {broken_count} broken links.")

    print("\n" + "="*30)
    print(f"Deep Validation Complete.")
    print("="*30)

if __name__ == "__main__":
    validate_integrity()
