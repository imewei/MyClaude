#!/usr/bin/env python3
"""UserPromptSubmit hook for dev-suite.

skill-comply measurement (results/plugins-batch-clean/) found dev-suite's hub
skills consistently fail their own routing steps (classify_concern,
route_to_subskill, invoke_hub_skill...) under neutral and competing prompts —
the agent jumps straight to Bash/Write/Edit instead of checking whether a
specialized hub skill applies first. A skill description is read once at
discovery time and easy to skip under pressure; this hook re-injects the
reminder on every prompt so it can't be silently forgotten mid-session.
"""

import json
import sys

from _hook_io import wrap_context

REMINDER = (
    "dev-suite routes domain-specific work through hub skills before "
    "implementation (architecture-and-infra, backend-patterns, "
    "testing-and-quality, ci-cd-pipelines, observability-and-sre, "
    "data-and-security, dev-workflows, ai-pair/three-brain). If this request "
    "matches one of those domains, invoke the matching hub skill (or dev-hub "
    "if ambiguous) before jumping to Bash/Write/Edit."
)


def main() -> None:
    try:
        result = {"status": "success", "additionalContext": REMINDER}
        result.update(wrap_context("UserPromptSubmit", REMINDER))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"UserPromptSubmit hook error: {e}", file=sys.stderr)
        json.dump(
            {"status": "error", "message": f"UserPromptSubmit hook error: {e}"}, sys.stdout
        )


if __name__ == "__main__":
    main()
