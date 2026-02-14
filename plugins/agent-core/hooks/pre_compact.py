#!/usr/bin/env python3
"""PreCompact hook for agent-core plugin.

Fires before context compaction occurs, allowing the agent to save
critical state (e.g., task progress, key decisions) before older
messages are compressed.
"""

import json
import sys


def main() -> None:
    """Signal readiness for context compaction."""
    try:
        result = {
            "status": "success",
            "message": "PreCompact: ready for context compaction",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"PreCompact hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
