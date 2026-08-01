#!/usr/bin/env python3
"""PreCompact hook for agent-core plugin.

Fires before context compaction occurs. Logs priority hub skills to
stderr so the user knows which skills to reinvoke after compaction.
"""

import json
import sys

# agent-core's own hub skills only — other suites declare their own priorities.
PRIORITY_SKILLS = [
    "agent-hub",
    "agent-systems",
    "llm-engineering",
    "reasoning-and-memory",
    "thinkfirst",
]


def main() -> None:
    """Signal readiness for context compaction and log priority skills."""
    try:
        sys.stderr.write(
            f"[PreCompact] Priority skills for post-compact reload: "
            f"{', '.join(PRIORITY_SKILLS)}\n"
        )
        result = {
            "status": "success",
            "message": "PreCompact: ready for context compaction",
            "priority_skills": PRIORITY_SKILLS,
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"PreCompact hook error: {e}", file=sys.stderr)
        error_result = {
            "status": "error",
            "message": f"PreCompact hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
