# Sphinx Documentation Regeneration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate Sphinx documentation to reflect the new 5-suite architecture (Engineering, Infrastructure, Science, Quality, Agent Core) by automating the conversion of `plugin.json` metadata to RST files.

**Architecture:** Replace the manual/legacy 31-plugin documentation with an automated Python script (`scripts/regenerate_docs.py`) that parses the new consolidated `plugin.json` files and generates standardized RST files in `docs/suites/`. Update the main index to reflect this new structure.

**Tech Stack:** Python, Sphinx, JSON.

---

### Task 1: Cleanup Legacy Documentation

**Files:**
- Delete: `docs/plugins/` (Directory)
- Delete: `docs/categories/` (Directory)
- Create: `docs/suites/` (Directory)

**Step 1: Check existing directories**
Run: `ls -d docs/plugins docs/categories`
Expected: Directories exist

**Step 2: Remove legacy directories**
Run: `rm -rf docs/plugins docs/categories`
Expected: Directories removed

**Step 3: Create new suites directory**
Run: `mkdir -p docs/suites`
Expected: Directory created

**Step 4: Commit**
```bash
git add docs/
git commit -m "chore(docs): remove legacy plugin documentation and create suites directory"
```

### Task 2: Implement Documentation Generator

**Files:**
- Create: `scripts/regenerate_docs.py`

**Step 1: Create the generator script**

```python
import os
import json

PLUGIN_ROOT = "plugins"
DOCS_ROOT = "docs/suites"

SUITES = [
    "agent-core",
    "engineering-suite",
    "infrastructure-suite",
    "quality-suite",
    "science-suite"
]

def generate_header(name, description):
    return f"{name}\n{'=' * len(name)}\n\n{description}\n\n"

def generate_agents(agents):
    if not agents:
        return ""
    content = "Agents\n------\n\n"
    for agent in agents:
        content += f".. agent:: {agent['name']}\n"
        content += f"   :description: {agent.get('description', 'No description')}\n"
        if 'model' in agent:
            content += f"   :model: {agent['model']}\n"
        content += "\n"
    return content

def generate_commands(commands):
    if not commands:
        return ""
    content = "Commands\n--------\n\n"
    for cmd in commands:
        content += f".. command:: {cmd['name']}\n"
        content += f"   :description: {cmd.get('description', 'No description')}\n"
        content += "\n"
    return content

def generate_skills(skills):
    if not skills:
        return ""
    content = "Skills\n------\n\n"
    for skill in skills:
        content += f".. skill:: {skill['name']}\n"
        content += f"   :description: {skill.get('description', 'No description')}\n"
        content += "\n"
    return content

def main():
    if not os.path.exists(DOCS_ROOT):
        os.makedirs(DOCS_ROOT)

    for suite in SUITES:
        json_path = os.path.join(PLUGIN_ROOT, suite, "plugin.json")
        if not os.path.exists(json_path):
            print(f"Skipping {suite}: plugin.json not found")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        content = generate_header(data.get('name', suite).title(), data.get('description', ''))
        content += generate_agents(data.get('agents', []))
        content += generate_commands(data.get('commands', []))
        content += generate_skills(data.get('skills', []))

        rst_path = os.path.join(DOCS_ROOT, f"{suite}.rst")
        with open(rst_path, 'w') as f:
            f.write(content)
        print(f"Generated {rst_path}")

if __name__ == "__main__":
    main()
```

**Step 2: Run the generator**
Run: `python3 scripts/regenerate_docs.py`
Expected: Output "Generated docs/suites/..." for each suite.

**Step 3: Verify output**
Run: `ls docs/suites/`
Expected: 5 .rst files (agent-core.rst, etc.)

**Step 4: Commit**
```bash
git add scripts/regenerate_docs.py docs/suites/
git commit -m "feat(docs): add documentation generator script and initial suite docs"
```

### Task 3: Update Main Index

**Files:**
- Modify: `docs/index.rst`

**Step 1: Read existing index**
Run: `cat docs/index.rst`

**Step 2: Update index to point to suites**
Replace the old toctree with:
```rst
.. toctree::
   :maxdepth: 2
   :caption: Suites

   suites/agent-core
   suites/engineering-suite
   suites/infrastructure-suite
   suites/quality-suite
   suites/science-suite
```

**Step 3: Commit**
```bash
git add docs/index.rst
git commit -m "docs: update index to reference new suite documentation"
```

### Task 4: Verify and Build

**Files:**
- Read: `Makefile` (to check build command)

**Step 1: Run Sphinx build**
Run: `make html` (or `sphinx-build -b html docs docs/_build/html`)
Expected: Build succeeds without errors.

**Step 2: Check for warnings**
Review output for "WARNING" or "ERROR". Fix if necessary.

**Step 3: Commit fixes (if any)**
```bash
git add .
git commit -m "fix(docs): resolve build warnings"
```
