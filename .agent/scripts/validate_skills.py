import json
import os
import sys
import yaml

def validate_index():
    # Determine .agent directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir) # .agent directory
    index_path = os.path.join(base_dir, "skills_index.json")

    print(f"Validating agent skills from: {base_dir}")

    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        return

    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing {index_path}: {e}")
        return

    pass_count = 0
    fail_count = 0

    # Check for both skills and workflows keys
    categories = ['skills', 'workflows']

    for category in categories:
        if category not in index:
            print(f"Warning: '{category}' key not found in index.")
            continue

        items = index[category]
        print(f"\nValidating {category} ({len(items)} items)...")

        for name, data in items.items():
            # Ensure data is a dictionary and has 'path'
            if isinstance(data, dict) and 'path' in data:
                rel_path = data['path']
                full_path = os.path.join(base_dir, rel_path)

                # Verify file existence
                if not os.path.exists(full_path):
                    print(f"[FAIL] {name}: File not found at {full_path}")
                    fail_count += 1
                    continue

                # Verify frontmatter for Markdown files
                if full_path.endswith('.md'):
                    try:
                        with open(full_path, 'r') as f:
                            content = f.read()

                        if content.startswith('---'):
                            try:
                                parts = content.split('---', 2)
                                if len(parts) < 3:
                                    print(f"[FAIL] {name}: Invalid frontmatter format")
                                    fail_count += 1
                                    continue

                                frontmatter = parts[1]
                                meta = yaml.safe_load(frontmatter)

                                if meta is None:
                                    print(f"[FAIL] {name}: Empty frontmatter")
                                    fail_count += 1
                                    continue

                                missing_fields = []
                                if 'version' not in meta:
                                    missing_fields.append('version')
                                if 'description' not in meta:
                                    missing_fields.append('description')

                                if category == 'workflows':
                                    if 'allowed-tools' not in meta:
                                        missing_fields.append('allowed-tools')

                                if missing_fields:
                                    print(f"[FAIL] {name}: Missing metadata fields: {', '.join(missing_fields)}")
                                    fail_count += 1
                                else:
                                    pass_count += 1
                            except yaml.YAMLError as e:
                                print(f"[FAIL] {name}: Invalid YAML frontmatter: {e}")
                                fail_count += 1
                        else:
                            print(f"[FAIL] {name}: No frontmatter found")
                            fail_count += 1
                    except Exception as e:
                        print(f"[FAIL] {name}: Error reading file: {e}")
                        fail_count += 1
                else:
                    pass_count += 1
            else:
                print(f"[FAIL] {name}: Invalid entry format")
                fail_count += 1

    print("\n" + "="*30)
    print(f"Validation Complete.")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print("="*30)

if __name__ == "__main__":
    validate_index()
