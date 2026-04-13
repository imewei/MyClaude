#!/usr/bin/env python3
"""ContextOverflow hook for agent-core plugin.

Fires when the context window is approaching capacity. Emits an urgent
warning so the agent can prioritize compaction or task completion before
context is exhausted.
"""

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Handle context overflow event."""
    try:
        usage_pct = os.environ.get("CONTEXT_USAGE_PCT", "unknown")
        tokens_used = os.environ.get("TOKENS_USED", "unknown")
        tokens_limit = os.environ.get("TOKENS_LIMIT", "unknown")

        result = {
            "status": "success",
            "additionalContext": (
                f"CONTEXT OVERFLOW WARNING: Context window at {usage_pct}% "
                f"({tokens_used}/{tokens_limit} tokens). "
                "Prioritize: (1) complete current task, (2) save progress to "
                ".claude-progress.md, (3) trigger compaction if possible. "
                "Avoid starting new multi-turn investigations."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        logger.exception("ContextOverflow hook failed")
        json.dump(
            {"status": "error", "message": f"ContextOverflow hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
