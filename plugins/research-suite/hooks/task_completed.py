#!/usr/bin/env python3
"""TaskCompleted hook for research-suite.

Logs completed research tasks to .research-log.jsonl inside the research-spark
workspace. Projects with no `_state.yaml` are left alone — an audit trail
dropped into an unrelated repository is pollution, not provenance.
"""

import json
import os
import sys
from datetime import UTC, datetime

import _hook_io

LOG_FILENAME = ".research-log.jsonl"


def main() -> None:
    try:
        payload = _hook_io.read_payload()
        task_subject = _hook_io.get_field(
            payload,
            "task",
            "subject",
            "description",
            "task_subject",
            env_fallback="TASK_SUBJECT",
            default="unknown task",
        )
        cwd = _hook_io.get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())

        states = _hook_io.find_state_files(cwd)
        if not states:
            json.dump(
                {
                    "status": "success",
                    "additionalContext": (
                        f"Task completed: '{task_subject}'. No research-spark "
                        "workspace here, so nothing was logged."
                    ),
                },
                sys.stdout,
            )
            return

        log_path = states[0].parent / LOG_FILENAME
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "task": task_subject,
        }

        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            advice = (
                f"Research task logged: '{task_subject}'. "
                f"Audit trail at {log_path}. "
                "If this concludes a research-spark stage, verify the stage artifact "
                "is committed before advancing."
            )
        except OSError:
            advice = (
                f"Research task completed: '{task_subject}'. "
                f"Could not write audit log at {log_path} (non-fatal)."
            )

        json.dump({"status": "success", "additionalContext": advice}, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"TaskCompleted hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
