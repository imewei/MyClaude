#!/usr/bin/env python3
import json
import os
import sys
import re
import yaml
import argparse

def validate_agent(base_dir=None, strict=True):
    """
    Unified validation script for the .agent system.
    Combines index validation, orphan detection, link checking, and policy enforcement.
    """
    if base_dir is None:
        # Determine .agent directory relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir) # .agent directory

    index_path = os.path.join(base_dir, "skills_index.json")

    print(f"Validating agent system in: {base_dir}")
    print("=" * 60)

    errors = []
    warnings = []

    # --- 1. Load Index ---
    if not os.path.exists(index_path):
        print(f"CRITICAL: {index_path} not found.")
        sys.exit(1)

    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        print(f"CRITICAL: Error parsing {index_path}: {e}")
        sys.exit(1)

    # --- 2. Build File Map (Optimization) ---
    print("Building filesystem map...")
    all_files_set = set()
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for d in dirs:
            all_files_set.add(os.path.abspath(os.path.join(root, d)))
        for file in files:
            all_files_set.add(os.path.abspath(os.path.join(root, file)))

    print(f"Indexed {len(all_files_set)} filesystem items.")

    # --- 3. Validate Index & Frontmatter ---
    print("\n--- Phase 1: Index & Metadata Validation ---")
    indexed_files = set()
    categories = ['skills', 'workflows']

    for category in categories:
        if category not in index:
            warnings.append(f"Index missing category '{category}'")
            continue

        items = index[category]
        for name, data in items.items():
            if not isinstance(data, dict) or 'path' not in data:
                errors.append(f"Index entry '{name}' invalid format")
                continue

            rel_path = data['path']
            full_path = os.path.join(base_dir, rel_path)
            abs_path = os.path.abspath(full_path)
            indexed_files.add(abs_path)

            # Check existence
            if abs_path not in all_files_set:
                errors.append(f"Missing file for '{name}': {rel_path}")
                continue

            # Check Frontmatter (MD files only)
            if full_path.endswith('.md'):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.startswith('---'):
                        errors.append(f"No frontmatter in {rel_path}")
                        continue

                    parts = content.split('---', 2)
                    if len(parts) < 3:
                        errors.append(f"Malformed frontmatter in {rel_path}")
                        continue

                    meta = yaml.safe_load(parts[1])
                    if not meta:
                        errors.append(f"Empty frontmatter in {rel_path}")
                        continue

                    required = ['version', 'description']
                    if category == 'workflows':
                        required.append('allowed-tools')

                    for field in required:
                        if field not in meta:
                            errors.append(f"{rel_path}: Missing '{field}'")

                except Exception as e:
                    errors.append(f"Error reading {rel_path}: {str(e)}")

    # --- 4. Orphan Detection ---
    print("\n--- Phase 2: Orphan Detection ---")
    ignored_dirs = {'scripts', 'assets', 'references', 'docs', '.git', 'test-corpus', 'reports'}
    ignored_files = {'engine.py', '__init__.py', 'COMPREHENSIVE_REVIEW_REPORT.md', 'TECH_DEBT_REPORT.md'}

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]

        for file in files:
            if file in ignored_files:
                continue

            if file.endswith('.md') or file.endswith('.py'):
                abs_path = os.path.abspath(os.path.join(root, file))

                # Skip index and this script
                if abs_path == os.path.abspath(index_path):
                    continue
                if os.path.dirname(abs_path) == os.path.abspath(script_dir):
                    continue

                if abs_path not in indexed_files:
                    rel_orphan = os.path.relpath(abs_path, base_dir)
                    # Heuristic: only flag orphans in root skills/workflows dirs, not nested assets
                    # Actually, deep_validate flagged everything not indexed. Let's keep it strict.
                    warnings.append(f"Orphaned file: {rel_orphan}")

    # --- 5. Link & Policy Check ---
    print("\n--- Phase 3: Links & Policy ---")
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    torch_import_pattern = re.compile(r'^\s*(import torch|from torch)', re.MULTILINE)

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, base_dir)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Link Checking (Markdown only)
                if file.endswith('.md'):
                    matches = link_pattern.findall(content)
                    for text, target in matches:
                        if target.startswith(('http', 'https', 'mailto', '#', '/')):
                            continue
                        if target.lower() == "coming soon":
                            continue

                        target_clean = target.split('#')[0].strip()
                        if not target_clean:
                            continue

                        abs_target = os.path.abspath(os.path.join(root, target_clean))

                        # Optimization: Check internal files against set, external against disk
                        is_internal = abs_target.startswith(base_dir)
                        exists = False
                        if is_internal:
                            exists = abs_target in all_files_set
                        else:
                            exists = os.path.exists(abs_target)

                        if not exists:
                            # Heuristics for false positives
                            if any(c in target for c in " ,("): continue
                            if len(target) <= 3 and "." not in target and "/" not in target: continue

                            errors.append(f"Broken link in {rel_path}: {target}")

                # Policy Checking (PyTorch)
                if file.endswith('.py') or file.endswith('.md'):
                    if torch_import_pattern.search(content):
                        if "# allow-torch" not in content:
                            msg = f"Policy Violation in {rel_path}: PyTorch usage detected without '# allow-torch' whitelist."
                            if strict:
                                errors.append(msg)
                            else:
                                warnings.append(msg)

            except Exception as e:
                errors.append(f"Error processing {rel_path}: {str(e)}")

    # --- Reporting ---
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n❌  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print(f"\nFAILURE: Found {len(errors)} critical issues.")
        sys.exit(1)
    else:
        print("\n✅  SUCCESS: No critical issues found.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Agent System Integrity")
    parser.add_argument("--relaxed", action="store_true", help="Treat policy violations as warnings instead of errors")
    args = parser.parse_args()

    validate_agent(strict=not args.relaxed)
