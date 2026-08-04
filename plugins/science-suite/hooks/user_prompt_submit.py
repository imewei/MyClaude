#!/usr/bin/env python3
"""UserPromptSubmit hook for science-suite.

skill-comply measurement (results/plugins-batch-clean/) found science-suite's
hub skills consistently fail their own routing steps (classify_task,
route_to_specialized_skill, consult_routing_tree...) under neutral and
competing prompts — the agent jumps straight to Bash/Write/Edit instead of
checking whether a specialized hub skill applies first. A skill description
is read once at discovery time and easy to skip under pressure; this hook
re-injects the reminder on every prompt so it can't be silently forgotten
mid-session.
"""

import json
import sys

from _hook_io import wrap_context

REMINDER = (
    "science-suite routes domain-specific work through hub skills before "
    "implementation (deep-learning-hub, statistical-physics-hub, "
    "sciml-modern-stack, jax-computing, julia-mastery, bayesian-inference, "
    "neural-pde, simulation-and-hpc, and the rest of the science-hub tree). "
    "If this request matches one of those domains, invoke the matching hub "
    "skill (or science-hub if ambiguous) before jumping to Bash/Write/Edit."
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
