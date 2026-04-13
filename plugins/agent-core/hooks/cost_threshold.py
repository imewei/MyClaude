#!/usr/bin/env python3
"""CostThreshold hook for agent-core plugin.

Fires when session cost exceeds a configurable threshold. With opus-tier
agents running multi-turn sessions, cost can spike unexpectedly. This
hook emits a warning and tracks cumulative spend.
"""

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Handle cost threshold event."""
    try:
        current_cost = os.environ.get("SESSION_COST", "unknown")
        threshold = os.environ.get("COST_THRESHOLD", "unknown")
        model = os.environ.get("MODEL_NAME", "unknown")

        result = {
            "status": "success",
            "additionalContext": (
                f"COST THRESHOLD REACHED: Session cost ${current_cost} "
                f"has exceeded threshold ${threshold} (model: {model}). "
                "Consider: (1) completing current task and stopping, "
                "(2) switching to a sonnet-tier agent for remaining work, "
                "(3) deferring non-critical subtasks to a new session."
            ),
        }
        json.dump(result, sys.stdout)
    except Exception as e:
        logger.exception("CostThreshold hook failed")
        json.dump(
            {"status": "error", "message": f"CostThreshold hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
