
import os
import shutil
from pathlib import Path

# Paths
SOURCE_WORKFLOWS = Path(".agent/workflows")
SOURCE_SKILLS = Path(".agent/skills")

USER_ROOT = Path.home() / ".gemini/antigravity"
TARGET_WORKFLOWS = USER_ROOT / "global_workflows"
TARGET_SKILLS = USER_ROOT / "agent_skills"

def deploy():
    if not USER_ROOT.exists():
        print(f"Error: User root {USER_ROOT} does not exist.")
        return

    print(f"Deploying to {USER_ROOT}...")

    # 1. Deploy Workflows
    if SOURCE_WORKFLOWS.exists():
        print(f"Copying Workflows to {TARGET_WORKFLOWS}...")
        TARGET_WORKFLOWS.mkdir(exist_ok=True)
        # Copy contents using copytree with dirs_exist_ok=True
        shutil.copytree(SOURCE_WORKFLOWS, TARGET_WORKFLOWS, dirs_exist_ok=True)
        print("Workflows deployed.")
    else:
        print("No source workflows found.")

    # 2. Deploy Skills
    if SOURCE_SKILLS.exists():
        print(f"Copying Skills to {TARGET_SKILLS}...")
        TARGET_SKILLS.mkdir(exist_ok=True)
        shutil.copytree(SOURCE_SKILLS, TARGET_SKILLS, dirs_exist_ok=True)
        print("Skills deployed.")
    else:
        print("No source skills found.")

    print("\nState after deployment:")
    print(f"Workflows in {TARGET_WORKFLOWS}: {len(list(TARGET_WORKFLOWS.rglob('*.md')))}")
    print(f"Skills in {TARGET_SKILLS}: {len(list(TARGET_SKILLS.rglob('SKILL.md')))}")

if __name__ == "__main__":
    deploy()
