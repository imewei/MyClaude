
import os
import shutil
from pathlib import Path

WORKFLOWS_DIR = Path(".agent/workflows")

TAXONOMY = {
    # 1. AI Engineering
    "ai-assistant": "ai-engineering",
    "improve-agent": "ai-engineering",
    "langchain-agent": "ai-engineering",
    "ml-pipeline": "ai-engineering",
    "multi-agent-optimize": "ai-engineering",
    "prompt-optimize": "ai-engineering",
    "ultra-think": "ai-engineering",
    
    # 2. Scientific Computing
    "julia": "scientific-computing",
    "sciml": "scientific-computing",

    # 3. Project Scaffolding
    "c-project": "project-scaffolding",
    "component-scaffold": "project-scaffolding",
    "feature-development": "project-scaffolding",
    "full-stack-feature": "project-scaffolding",
    "multi-platform": "project-scaffolding",
    "onboard": "project-scaffolding",
    "python-scaffold": "project-scaffolding",
    "rust-project": "project-scaffolding",
    "typescript-scaffold": "project-scaffolding",

    # 4. Code Maintenance
    "adopt-code": "code-maintenance",
    "code-explain": "code-maintenance",
    "code-migrate": "code-maintenance",
    "fix-": "code-maintenance",
    "legacy-modernize": "code-maintenance",
    "refactor-clean": "code-maintenance",
    "tech-debt": "code-maintenance",
    "update-docs": "code-maintenance",

    # 5. Quality Assurance
    "double-check": "quality-assurance",
    "full-review": "quality-assurance",
    "pr-enhance": "quality-assurance",
    "profile-performance": "quality-assurance",
    "reflection": "quality-assurance",
    "run-all-tests": "quality-assurance",
    "smart-debug": "quality-assurance",
    "test-generate": "quality-assurance",

    # 6. DevOps Workflow
    "commit": "devops-workflow",
    "deps-": "devops-workflow",
    "doc-generate": "devops-workflow",
    "git-workflow": "devops-workflow",
    "merge-all": "devops-workflow",
    "monitor-setup": "devops-workflow",
    "slo-implement": "devops-workflow",
    "workflow-automate": "devops-workflow",
}

def get_category(name):
    for key, category in TAXONOMY.items():
        if key in name:
            return category
    return "uncategorized"

def main():
    if not WORKFLOWS_DIR.exists():
        print(f"Directory {WORKFLOWS_DIR} not found.")
        return

    # Create categories
    categories = set(TAXONOMY.values())
    categories.add("uncategorized")
    for category in categories:
        (WORKFLOWS_DIR / category).mkdir(exist_ok=True)

    # Move files
    moves = 0
    for item in WORKFLOWS_DIR.iterdir():
        if not item.is_file() or not item.name.endswith(".md"):
            continue
        
        category = get_category(item.stem) # stem checks filename without extension
        target = WORKFLOWS_DIR / category / item.name
        
        shutil.move(str(item), str(target))
        moves += 1

    print(f"Consolidated {moves} workflows into {len(categories)} categories.")
    
    uncategorized = list((WORKFLOWS_DIR / "uncategorized").iterdir())
    if uncategorized:
        print("\nUncategorized workflows:")
        for u in uncategorized:
            print(f"- {u.name}")
    else:
        # Cleanup
        (WORKFLOWS_DIR / "uncategorized").rmdir()

if __name__ == "__main__":
    main()
