#!/usr/bin/env python3
"""SubagentStop hook for research-suite.

Gates on the STOPPED subagent's identity. Only fires artifact-check logic for
research-spark-orchestrator. `scientific-review` is registered as a *skill*
(.claude-plugin/plugin.json's "skills" array), not an agent — it loads via the
Skill tool in the main session, never via Task-based subagent spawn, so
SubagentStop can never see agent_type == "scientific-review". That branch used
to exist here as dead code; the review-completeness check for that track now
lives in post_tool_use.py, gated on the actual Write of the deliverable file.

All other agent types exit silently with no output.
"""

import json
import sys

import _hook_io

# Canonical artifact(s) expected once a stage NUMBER appears in
# stages_completed in _state.yaml. Stage 4-5 share theory-scaffold; 05 only
# exists once stage 5 itself is in stages_completed.
#
# NOTE: current_stage is the stage IN PROGRESS, not yet done — SKILL.md's own
# documented example pairs `current_stage: 4` with `stages_completed: [1, 2,
# 3]`, where stage 4's artifact does not exist yet. Checking current_stage's
# artifact (an earlier version of this function did) would false-flag every
# normal mid-pipeline advance. stages_completed is the actual "done" set.
STAGE_ARTIFACTS = {
    1: ["01_spark.md"],
    2: ["02_landscape.md"],
    3: ["03_claim.md"],
    4: ["04_theory.md"],
    5: ["04_theory.md", "05_formalism.tex"],
    6: ["06_prototype.md"],
    7: ["07_plan.md"],
    8: ["08_premortem.md"],
}


def check_artifacts(cwd: str) -> str | None:
    """Real filesystem check replacing self-attestation.

    Reads the actual stages_completed from each _state.yaml found under cwd
    and stats the canonical artifact path directly, instead of asking the
    model to eyeball its own transcript for a stage-completion marker.
    """
    lines = []
    for state_path in _hook_io.find_state_files(cwd):
        completed = _hook_io.read_stages_completed(state_path)
        if not completed:
            continue
        stage = max(completed)
        if stage not in STAGE_ARTIFACTS:
            continue
        artifacts_dir = state_path.parent / "artifacts"
        missing = [
            name for name in STAGE_ARTIFACTS[stage] if not (artifacts_dir / name).is_file()
        ]
        project = state_path.parent.name or str(state_path.parent)
        if missing:
            lines.append(
                f"{project}: _state.yaml lists stage {stage} as completed but "
                f"{', '.join(missing)} not found in {artifacts_dir}/ — "
                "do not report this stage complete until the artifact exists on disk."
            )
        else:
            lines.append(f"{project}: stage {stage} artifact(s) verified present on disk.")
    return "\n".join(lines) if lines else None


def main() -> None:
    try:
        payload = _hook_io.read_payload()
        agent_type = _hook_io.get_field(
            payload,
            "subagent_type",
            "agent_type",
            "agent_name",
            "matcher_input",
            default="",
        ).strip()

        if agent_type != "research-spark-orchestrator":
            sys.exit(0)

        cwd = _hook_io.get_field(payload, "cwd", default="")
        ctx = check_artifacts(cwd) if cwd else None
        if ctx:
            result = {"status": "success", "additionalContext": ctx}
            result.update(_hook_io.wrap_context("SubagentStop", ctx))
            json.dump(result, sys.stdout)
        else:
            sys.exit(0)
    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"SubagentStop hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
