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

    # 0. Build File Set for Fast Lookups (Optimization)
    print("Building file map...")
    all_files_set = set()
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories and .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for d in dirs:
            all_files_set.add(os.path.abspath(os.path.join(root, d)))
        for file in files:
            all_files_set.add(os.path.abspath(os.path.join(root, file)))

    print(f"Indexed {len(all_files_set)} files and directories in filesystem.")

    # 1. Collect indexed files from JSON
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
    ignored_files = {'engine.py', '__init__.py'} # Allowed helper scripts

    for root, dirs, files in os.walk(base_dir):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]

        for file in files:
            if file in ignored_files:
                continue

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

    # 3. Check Broken Links (Optimized)
    print("\n--- Broken Link Check ---")
    broken_count = 0
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    # Pre-compile JAX check regex
    torch_import_pattern = re.compile(r'^\s*(import torch|from torch)', re.MULTILINE)

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            file_path = os.path.join(root, file)
            rel_file_path = os.path.relpath(file_path, base_dir)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # A. Link Checking
                if file.endswith('.md'):
                    matches = link_pattern.findall(content)
                    for text, target in matches:
                        # Skip external links, anchors, and absolute paths
                        if target.startswith(('http', 'https', 'mailto', '#', '/')):
                            continue

                        # Handle "Coming Soon" placeholders or other non-path text
                        if target.lower() == "coming soon":
                            continue

                        # Resolve relative path
                        target_path = target.split('#')[0] # Remove anchor
                        if not target_path:
                            continue

                        # Clean up path
                        target_path = target_path.strip()

                        abs_target_path = os.path.abspath(os.path.join(root, target_path))

                        # OPTIMIZATION: Check against set instead of disk
                        # But ONLY if the file is inside the .agent directory (base_dir)
                        # If it points outside (e.g. to plugins/), we must check disk
                        is_internal = abs_target_path.startswith(base_dir)

                        exists = False
                        if is_internal:
                            exists = abs_target_path in all_files_set
                        else:
                            exists = os.path.exists(abs_target_path)

                        if not exists:
                            # Heuristic to filter code false positives
                            if " " in target or "," in target or "(" in target:
                                continue
                            if len(target) <= 3 and "." not in target and "/" not in target:
                                continue

                            print(f"[BROKEN] In {rel_file_path}: Link '{text}' -> '{target}' not found")
                            broken_count += 1

                # B. JAX Compliance Check
                if file.endswith('.py') or file.endswith('.md'):
                    if torch_import_pattern.search(content):
                        # Allow explicit whitelisting via comment
                        if "# allow-torch" not in content:
                            print(f"[POLICY] In {rel_file_path}: Found PyTorch import. Project is JAX-first. Add '# allow-torch' if strictly necessary.")
                            # We treat policy violations as warnings for now, not broken counts, unless we want to enforce strictly
                            # broken_count += 1

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
