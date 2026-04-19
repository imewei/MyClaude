#!/usr/bin/env python3
"""SessionStart hook for research-suite.

Detects research-spark/scientific-review artifacts in the working directory
so the orchestrator resumes where the previous session left off.
"""

import json
import os
import sys
from pathlib import Path


STAGE_MARKERS = {
    "stage_1_problem": ["problem_statement.md", "problem.md"],
    "stage_2_claim": ["falsifiable_claim.md", "claims.md"],
    "stage_3_prereg": ["pre_registration.md", "prereg.md"],
    "stage_4_plan": ["experimental_plan.md", "plan.md"],
    "stage_5_analysis": ["analysis_plan.md"],
    "stage_6_results": ["results.md", "results.ipynb"],
    "stage_7_discussion": ["discussion.md"],
    "stage_8_manuscript": ["manuscript.md", "paper.md", "paper.tex"],
}


def detect_research_artifacts(cwd: str) -> dict:
    """Detect research-spark stage artifacts present in the working tree."""
    root = Path(cwd)
    present: list[str] = []
    for stage, filenames in STAGE_MARKERS.items():
        if any((root / name).exists() or any(root.rglob(name)) for name in filenames):
            present.append(stage)
    return {"stages_present": present, "latest_stage": present[-1] if present else None}


def main() -> None:
    try:
        cwd = os.environ.get("PWD", os.getcwd())
        artifacts = detect_research_artifacts(cwd)

        if artifacts["latest_stage"]:
            ctx = (
                f"Research-suite resume: detected artifacts up to "
                f"{artifacts['latest_stage']}. Stages present: "
                f"{', '.join(artifacts['stages_present'])}."
            )
        else:
            ctx = (
                "Research-suite session start: no research-spark artifacts detected. "
                "Start at Stage 1 (problem statement) if using /research-spark."
            )

        json.dump({"status": "success", "additionalContext": ctx}, sys.stdout)
    except Exception as e:
        json.dump(
            {"status": "error", "message": f"SessionStart hook error: {e}"},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
