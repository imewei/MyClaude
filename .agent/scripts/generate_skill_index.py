import os
import json
from pathlib import Path

def parse_frontmatter(file_path):
    """
    Manually parses YAML frontmatter without requiring PyYAML.
    Assumes standard Jekyll/Markdown format:
    ---
    key: value
    list:
      - item
    ---
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != '---':
        return None

    data = {}
    in_frontmatter = False

    # Simple state machine for parsing
    for i, line in enumerate(lines):
        line = line.rstrip()
        if i == 0:
            in_frontmatter = True
            continue

        if line == '---':
            break

        if in_frontmatter:
            if ':' in line:
                # Handle basic key: value
                parts = line.split(':', 1)
                key = parts[0].strip()
                val = parts[1].strip()

                # Handle lists (very basic implementation)
                if not val and i+1 < len(lines) and lines[i+1].strip().startswith('-'):
                    data[key] = []
                    continue

                # Remove quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]

                data[key] = val

            elif line.strip().startswith('-') and data:
                # Handle list items for the last key found
                # This is a bit fragile but works for simple lists like triggers
                last_key = list(data.keys())[-1]
                val = line.strip()[1:].strip()
                if isinstance(data[last_key], list):
                    data[last_key].append(val)

    return data

def generate_index(root_dir):
    """
    Scans the directory for SKILL.md files and generates an index.
    """
    index = {
        "generated_at": None,
        "skills": {},
        "workflows": {}
    }

    root_path = Path(root_dir)
    skills_dir = root_path / "skills"
    workflows_dir = root_path / "workflows"

    # Index Skills
    print(f"Scanning skills in {skills_dir}...")
    if skills_dir.exists():
        for skill_file in skills_dir.glob("**/SKILL.md"):
            try:
                metadata = parse_frontmatter(skill_file)
                if metadata:
                    skill_name = metadata.get('name') or skill_file.parent.name

                    # Create the index entry
                    entry = {
                        "path": str(skill_file.relative_to(root_path)),
                        "description": metadata.get('description', ''),
                        "triggers": metadata.get('triggers', []),
                        "version": metadata.get('version', '1.0.0')
                    }

                    index["skills"][skill_name] = entry
            except Exception as e:
                print(f"Skipping {skill_file}: {e}")

    # Index Workflows
    print(f"Scanning workflows in {workflows_dir}...")
    if workflows_dir.exists():
        for workflow_file in workflows_dir.glob("*.md"):
            try:
                metadata = parse_frontmatter(workflow_file)
                workflow_name = workflow_file.stem

                entry = {
                    "path": str(workflow_file.relative_to(root_path)),
                    "description": metadata.get('description', '') if metadata else '',
                    "triggers": metadata.get('triggers', []) if metadata else []
                }
                index["workflows"][workflow_name] = entry
            except Exception as e:
                print(f"Skipping {workflow_file}: {e}")

    return index

if __name__ == "__main__":
    import datetime

    ROOT_DIR = ".agent"
    OUTPUT_FILE = ".agent/skills_index.json"

    # Ensure we are running from the project root or adjust paths
    if not os.path.exists(ROOT_DIR):
        # specific logic for this environment if needed, but assuming standard layout
        pass

    index_data = generate_index(ROOT_DIR)
    index_data["generated_at"] = datetime.datetime.now().isoformat()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)

    print(f"Index generated at {OUTPUT_FILE}")
    print(f"Indexed {len(index_data['skills'])} skills and {len(index_data['workflows'])} workflows.")
