
import os
import re
from pathlib import Path
import yaml

WORKFLOWS_DIR = Path(".agent/workflows")
SKILLS_DIR = Path(".agent/skills")

def optimize_workflow(path):
    with open(path, "r") as f:
        content = f.read()

    # Parse Frontmatter
    match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        print(f"Skipping {path.name}: No frontmatter")
        return

    frontmatter_str = match.group(1)
    body = content[match.end():]
    
    try:
        data = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        print(f"Skipping {path.name}: Invalid YAML")
        return

    # 1. Add Triggers if missing
    if "triggers" not in data:
        triggers = []
        # Slash Command
        triggers.append(f"/{path.stem}")
        # Intent (generic)
        if "description" in data:
            # Simple heuristic: first 3 words of description
            desc_words = data["description"].split()[:4]
            intent = " ".join(desc_words).lower().replace(":", "").replace("-", " ")
            triggers.append(intent)
        
        data["triggers"] = triggers
        print(f"[{path.name}] Added triggers: {triggers}")

    # 2. Token Density: Collapse multiple newlines
    new_body = re.sub(r"\n{3,}", "\n\n", body)
    
    # Reconstruct
    new_frontmatter = yaml.dump(data, sort_keys=False).strip()
    new_content = f"---\n{new_frontmatter}\n---{new_body}"
    
    if new_content != content:
        with open(path, "w") as f:
            f.write(new_content)
        return True
    return False

def optimize_skill(path):
    # Similar logic for skills if needed. Currently just doing density.
    with open(path, "r") as f:
        content = f.read()
        
    new_content = re.sub(r"\n{3,}", "\n\n", content)
    
    if new_content != content:
        with open(path, "w") as f:
            f.write(new_content)
        print(f"[{path.name}] Optimized density")
        return True
    return False

def main():
    # Workflows (Recursive search due to consolidation)
    print("Optimizing Workflows...")
    for path in WORKFLOWS_DIR.rglob("*.md"):
        optimize_workflow(path)

    # Skills
    print("\nOptimizing Skills...")
    for path in SKILLS_DIR.rglob("SKILL.md"):
        optimize_skill(path)

if __name__ == "__main__":
    main()
