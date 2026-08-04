#!/usr/bin/env python3
"""UserPromptSubmit hook for science-suite.

A skill-comply measurement run (an external skill-comply plugin run, not
checked into this repo — see its own results dir under
~/.claude/plugins/cache/.../skill-comply/results/plugins-batch-clean/) found
science-suite's hub skills consistently fail to classify-then-route to a
specialized skill before acting, under neutral and competing prompts — the
agent jumps straight to Bash/Write/Edit instead of checking whether a
specialized hub skill applies first. (classify_task/route_to_specialized_
skill/consult_routing_tree were skill-comply's own step labels for that
behavior, not identifiers defined anywhere in this codebase.) A skill
description is read once at discovery time and easy to skip under pressure;
this hook re-injects the reminder on every prompt so it can't be silently
forgotten mid-session.
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
    "skill (science-hub if ambiguous or it spans multiple dev-suite/"
    "research-suite domains too) before jumping to Bash/Write/Edit."
)


def main() -> None:
    try:
        result = {"status": "success", "additionalContext": REMINDER}
        result.update(wrap_context("UserPromptSubmit", REMINDER))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"UserPromptSubmit hook error: {e}", file=sys.stderr)
        try:
            json.dump(
                {"status": "error", "message": f"UserPromptSubmit hook error: {e}"}, sys.stdout
            )
        except Exception:  # noqa: S110 — stdout unusable; the stderr message above is the only record
            pass


if __name__ == "__main__":
    main()
