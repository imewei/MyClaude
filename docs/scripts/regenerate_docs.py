import os
import json
import re

# Calculate project root relative to this script
# Script is in docs/scripts/regenerate_docs.py
# Root is ../../ from here
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

PLUGIN_ROOT = os.path.join(PROJECT_ROOT, "plugins")
DOCS_ROOT = os.path.join(PROJECT_ROOT, "docs/suites")

SUITES = [
    "agent-core",
    "engineering-suite",
    "infrastructure-suite",
    "quality-suite",
    "science-suite"
]

def extract_metadata(path):
    """Extract metadata from YAML frontmatter in markdown files."""
    if not os.path.exists(path):
        return {"description": "No description"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            metadata = {"description": "No description"}
            # Match YAML frontmatter
            fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if fm_match:
                fm_text = fm_match.group(1)
                for line in fm_text.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip().strip("\"'")
            return metadata
    except Exception:
        return {"description": "No description"}

def generate_section(items, title, directive, suite_path, subdir):
    """Generate an RST section for agents, commands, or skills."""
    # If items is None or empty, try to find in directory
    if not items:
        items_dir = os.path.join(suite_path, subdir)
        if os.path.exists(items_dir):
            items = []
            for f in os.listdir(items_dir):
                if f.endswith(".md"):
                    items.append(f.replace(".md", ""))
                elif os.path.isdir(os.path.join(items_dir, f)) and not f.startswith("."):
                    items.append(f)
        else:
            return ""

    if not items:
        return ""

    content = f"{title}\n" + "-" * len(title) + "\n\n"
    # Sort items by name
    sorted_items = []
    for item in items:
        if isinstance(item, str):
            sorted_items.append({"name": item})
        else:
            sorted_items.append(item)

    sorted_items.sort(key=lambda x: x.get("name", ""))

    for item in sorted_items:
        name = item.get("name", "unknown")

        # Try to find metadata in .md file
        md_path = os.path.join(suite_path, subdir, f"{name}.md")
        if not os.path.exists(md_path):
            # Check for SKILL.md in subdirectory
            md_path = os.path.join(suite_path, subdir, name, "SKILL.md")

        meta = extract_metadata(md_path)

        # Priority: plugin.json item > markdown frontmatter
        description = item.get("description") or meta.get("description", "No description")
        model = item.get("model") or meta.get("model")
        version = item.get("version") or meta.get("version")

        content += f".. {directive}:: {name}\n"
        content += f"   :description: {description}\n"
        if model:
            content += f"   :model: {model}\n"
        if version:
            content += f"   :version: {version}\n"
        content += "\n"
    return content

def main():
    if not os.path.exists(DOCS_ROOT):
        os.makedirs(DOCS_ROOT)

    for suite in SUITES:
        suite_path = os.path.join(PLUGIN_ROOT, suite)
        json_path = os.path.join(suite_path, "plugin.json")
        if not os.path.exists(json_path):
            print(f"Skipping {suite}: plugin.json not found")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        display_name = data.get("displayName", data.get("name", suite).title())
        description = data.get("description", "")

        content = f"{display_name}\n" + "=" * len(display_name) + "\n\n" + description + "\n\n"

        content += generate_section(data.get("agents"), "Agents", "agent", suite_path, "agents")
        content += generate_section(data.get("commands"), "Commands", "command", suite_path, "commands")
        content += generate_section(data.get("skills"), "Skills", "skill", suite_path, "skills")

        rst_path = os.path.join(DOCS_ROOT, f"{suite}.rst")
        with open(rst_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated {rst_path}")

if __name__ == "__main__":
    main()
