#!/usr/bin/env python3
"""Session start hook for agent-core plugin.

Provides context about Opus 4.6 capabilities and available agent teams
at the beginning of each session.
"""

import json
import sys


def get_session_context() -> dict:
    """Build session context with Opus 4.6 optimization hints."""
    context = {
        "status": "success",
        "message": "Success",
    }
    return context


def main() -> None:
    """Output session context as JSON."""
    try:
        result = get_session_context()
        json.dump(result, sys.stdout)
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"SessionStart hook error: {e}",
        }
        json.dump(error_result, sys.stdout)


if __name__ == "__main__":
    main()
