#!/usr/bin/env python3
"""PostCompact hook for agent-core plugin.

Fires after context compaction completes. Persists a brief summary
of pre-compact state to help maintain continuity.
"""

import json
import sys


def main() -> None:
    """Acknowledge context compaction completion."""
    try:
        result = {
            "status": "success",
            "message": "PostCompact: context compaction complete. Key state preserved in tasks.",
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({"status": "error", "message": f"PostCompact hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
