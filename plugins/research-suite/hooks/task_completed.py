#!/usr/bin/env python3
"""TaskCompleted hook for research-suite.

Deterministic logging of completed research tasks to .research-log.jsonl
for audit trails and pipeline stage tracking.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


LOG_FILENAME = ".research-log.jsonl"


def main() -> None:
    try:
        task_subject = os.environ.get("TASK_SUBJECT", "unknown task")
        cwd = os.environ.get("PWD", os.getcwd())
        log_path = Path(cwd) / LOG_FILENAME

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": task_subject,
        }

        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            advice = (
                f"Research task logged: '{task_subject}'. "
                f"Audit trail at {LOG_FILENAME}. "
                "If this concludes a research-spark stage, verify the stage artifact "
                "is committed before advancing."
            )
        except OSError:
            advice = (
                f"Research task completed: '{task_subject}'. "
                "Could not write audit log (non-fatal)."
            )

        json.dump({"status": "success", "additionalContext": advice}, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"TaskCompleted hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
