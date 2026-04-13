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
        usage_pct = os.environ.get("CONTEXT_USAGE_PCT")
        tokens_used = os.environ.get("TOKENS_USED")
        tokens_limit = os.environ.get("TOKENS_LIMIT")

        if usage_pct and tokens_used and tokens_limit:
            detail = f"at {usage_pct}% ({tokens_used}/{tokens_limit} tokens)"
        else:
            detail = "(usage details unavailable)"
            logger.warning("ContextOverflow fired without usage env vars")

        result = {
            "status": "success",
            "additionalContext": (
                f"CONTEXT OVERFLOW WARNING: Context window {detail}. "
                "Prioritize: (1) complete current task, (2) save progress to "
                ".claude-progress.md, (3) trigger compaction if possible. "
                "Avoid starting new multi-turn investigations."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        logger.exception("ContextOverflow hook failed")
        error = {"status": "error", "message": f"ContextOverflow hook error: {e}"}
        try:
            json.dump(error, sys.stdout)
        except Exception:
            sys.stderr.write(f"ContextOverflow hook fatal: {e}\n")


if __name__ == "__main__":
    main()
