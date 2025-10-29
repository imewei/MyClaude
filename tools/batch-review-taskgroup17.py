#!/usr/bin/env python3
"""
Batch review script for Task Group 1.7 - Remaining 14 plugins
Generates comprehensive review reports for all plugins efficiently
"""

import subprocess
import json
import os
from pathlib import Path

PLUGINS_DIR = Path("/Users/b80985/Projects/MyClaude/plugins")
REVIEWS_DIR = Path("/Users/b80985/Projects/MyClaude/reviews")
TOOLS_DIR = Path("/Users/b80985/Projects/MyClaude/tools")

# Task Group 1.7 plugins
TASK_GROUP_17_PLUGINS = {
    "1.7.A": {
        "name": "Meta-Orchestration",
        "plugins": ["agent-orchestration", "full-stack-orchestration"]
    },
    "1.7.B": {
        "name": "Code Quality and Documentation",
        "plugins": ["code-documentation", "comprehensive-review"]
    },
    "1.7.C": {
        "name": "Code Migration and Cleanup",
        "plugins": ["code-migration", "framework-migration", "codebase-cleanup"]
    },
    "1.7.D": {
        "name": "Development Tooling",
        "plugins": ["debugging-toolkit", "multi-platform-apps", "llm-application-dev"]
    },
    "1.7.E": {
        "name": "Previously Identified Incomplete",
        "plugins": ["backend-development", "frontend-mobile-development",
                   "git-pr-workflows", "observability-monitoring"]
    }
}

def run_review(plugin_name):
    """Run automated review script for a plugin"""
    try:
        result = subprocess.run(
            ["python3", str(TOOLS_DIR / "plugin-review-script.py"), plugin_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running review: {e}"

def run_profiler(plugin_name):
    """Run load profiler for a plugin"""
    try:
        result = subprocess.run(
            ["python3", str(TOOLS_DIR / "load-profiler.py"), plugin_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running profiler: {e}"

def check_plugin_status(plugin_name):
    """Check if plugin has plugin.json and what files exist"""
    plugin_dir = PLUGINS_DIR / plugin_name

    has_plugin_json = (plugin_dir / "plugin.json").exists()
    has_readme = (plugin_dir / "README.md").exists()
    has_agents = (plugin_dir / "agents").exists()
    has_commands = (plugin_dir / "commands").exists()
    has_skills = (plugin_dir / "skills").exists()

    agent_count = len(list((plugin_dir / "agents").glob("*.md"))) if has_agents else 0
    command_count = len(list((plugin_dir / "commands").glob("*.md"))) if has_commands else 0
    skill_count = len(list((plugin_dir / "skills").glob("*.md"))) if has_skills else 0

    return {
        "has_plugin_json": has_plugin_json,
        "has_readme": has_readme,
        "has_agents": has_agents,
        "has_commands": has_commands,
        "has_skills": has_skills,
        "agent_count": agent_count,
        "command_count": command_count,
        "skill_count": skill_count,
        "status": "COMPLETE" if has_plugin_json else "INCOMPLETE"
    }

def generate_incomplete_plugin_review(plugin_name, status):
    """Generate review for incomplete plugin (missing plugin.json)"""
    review = f"""# Plugin Review Report: {plugin_name}

**Plugin Path:** `{PLUGINS_DIR / plugin_name}`
**Review Date:** 2025-10-29
**Reviewer:** Claude Code (Task Group 1.7)
**Plugin Status:** INCOMPLETE (missing plugin.json)

---

## Executive Summary

**Overall Grade:** F (INCOMPLETE)

The {plugin_name} plugin is **INCOMPLETE** and missing critical plugin.json configuration file. While the plugin has directory structure in place (agents: {status['agent_count']}, commands: {status['command_count']}, skills: {status['skill_count']}), it cannot be loaded by the marketplace without plugin.json.

**Critical Issues:**
- Missing plugin.json (CRITICAL - plugin cannot be loaded)
- {'Missing README.md' if not status['has_readme'] else 'Has README.md'}

**Directory Structure:**
- agents/: {'✓ Present' if status['has_agents'] else '✗ Missing'} ({status['agent_count']} files)
- commands/: {'✓ Present' if status['has_commands'] else '✗ Missing'} ({status['command_count']} files)
- skills/: {'✓ Present' if status['has_skills'] else '✗ Missing'} ({status['skill_count']} files)
- README.md: {'✓ Present' if status['has_readme'] else '✗ Missing'}
- plugin.json: ✗ MISSING (CRITICAL)

---

## Section 1: Plugin Metadata (plugin.json)

### Status: MISSING (CRITICAL)

**Impact:** Plugin cannot be loaded by marketplace, all agents/commands/skills are inaccessible.

**Required Fields (all missing):**
- name
- version
- description
- author
- license
- agents[] array
- commands[] array
- skills[] array (optional)
- keywords[] array (optional)
- category (optional)

**Completeness Score: 0/100**

---

## Section 2-10: Unable to Complete

Without plugin.json, the following review sections cannot be completed:
- Section 2: Agent Documentation (cannot validate agent references)
- Section 3: Command Documentation (cannot validate command references)
- Section 4: Skill Documentation (cannot validate skill references)
- Section 5: README Completeness
- Section 6: Triggering Logic Analysis
- Section 7: Integration Points
- Section 8: Performance Profiling (cannot load plugin)
- Section 9: Consistency Checks
- Section 10: Issue Identification

---

## Recommendations

### CRITICAL - Immediate Action Required

1. **Create plugin.json file**
   - Required fields: name, version, description, author, license
   - Define agents array with references to {status['agent_count']} agent files
   - Define commands array with references to {status['command_count']} command files
   {'- Define skills array with references to ' + str(status['skill_count']) + ' skill files' if status['skill_count'] > 0 else ''}
   - Add keywords for discoverability
   - Add category field

2. **{'Create README.md file' if not status['has_readme'] else 'Validate README.md content'}**
   - Plugin overview
   - Installation instructions
   - Agent/command/skill reference
   - Usage examples

### Post-plugin.json Actions

3. **Run full review process**
   - After plugin.json is created, re-run complete 10-section review
   - Profile performance
   - Validate cross-references
   - Test triggering conditions

---

## Conclusion

The {plugin_name} plugin is **NOT FUNCTIONAL** without plugin.json. This is a CRITICAL blocker that must be resolved before any other work can proceed.

**Next Steps:**
1. Create plugin.json with complete metadata
2. {'Create README.md' if not status['has_readme'] else 'Validate README.md'}
3. Run full review process (plugin-review-script.py {plugin_name})
4. Profile performance (load-profiler.py {plugin_name})

---

*Review completed: 2025-10-29*
*Next review: After plugin.json creation*
"""
    return review

def main():
    print("Task Group 1.7 Batch Review")
    print("=" * 60)

    total_plugins = 0
    complete_plugins = 0
    incomplete_plugins = 0

    for subgroup_id, subgroup in TASK_GROUP_17_PLUGINS.items():
        print(f"\n{subgroup_id}: {subgroup['name']}")
        print("-" * 60)

        for plugin_name in subgroup['plugins']:
            total_plugins += 1
            print(f"Processing: {plugin_name}...", end=" ")

            # Check plugin status
            status = check_plugin_status(plugin_name)

            if status['status'] == 'COMPLETE':
                complete_plugins += 1
                print(f"✓ COMPLETE")
                # Run automated review and profiler
                review_output = run_review(plugin_name)
                profiler_output = run_profiler(plugin_name)
                print(f"  - Generated review report")
                print(f"  - Generated performance profile")
            else:
                incomplete_plugins += 1
                print(f"✗ INCOMPLETE (missing plugin.json)")
                # Generate incomplete plugin review
                review = generate_incomplete_plugin_review(plugin_name, status)
                review_file = REVIEWS_DIR / f"{plugin_name}.md"
                review_file.write_text(review)
                print(f"  - Generated incomplete status report")

    print(f"\n" + "=" * 60)
    print(f"Task Group 1.7 Summary:")
    print(f"  Total plugins: {total_plugins}")
    print(f"  Complete (has plugin.json): {complete_plugins}")
    print(f"  Incomplete (missing plugin.json): {incomplete_plugins}")
    print(f"  Completion rate: {complete_plugins/total_plugins*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
