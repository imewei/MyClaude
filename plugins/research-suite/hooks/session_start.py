#!/usr/bin/env python3
"""SessionStart hook for research-suite.

Reports the research-spark stage from `_state.yaml`, the pipeline's declared
single source of truth. Says nothing when no research-spark workspace is
present — an absence claim injected before any file is read would be a
confident guess, not evidence.
"""

import json
import os
import sys

import _hook_io


def main() -> None:
    try:
        payload = _hook_io.read_payload()
        cwd = _hook_io.get_field(payload, "cwd", env_fallback="PWD", default=os.getcwd())
        states = _hook_io.find_state_files(cwd)

        if not states:
            json.dump({"status": "success"}, sys.stdout)
            return

        if len(states) > 1:
            ctx = (
                f"Research-suite: {len(states)} _state.yaml files found "
                f"({', '.join(str(p) for p in states)}). Read the relevant one "
                "before assuming any stage."
            )
        else:
            state = states[0]
            stage = _hook_io.read_current_stage(state)
            if stage is None:
                ctx = (
                    f"Research-suite: found {state} but could not read "
                    "`current_stage` from it. Surface this to the user rather "
                    "than overwriting the file."
                )
            else:
                ctx = (
                    f"Research-suite resume: {state} reports current_stage: {stage}. "
                    "Read the file itself before acting; it is the single source of truth."
                )

        result = {"status": "success", "additionalContext": ctx}
        result.update(_hook_io.wrap_context("SessionStart", ctx))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"SessionStart hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SessionStart hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
