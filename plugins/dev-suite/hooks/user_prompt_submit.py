#!/usr/bin/env python3
"""UserPromptSubmit hook for dev-suite.

A skill-comply measurement run (an external skill-comply plugin run, not
checked into this repo — see its own results dir under
~/.claude/plugins/cache/.../skill-comply/results/plugins-batch-clean/) found
dev-suite's hub skills consistently fail to classify-then-route to a
specialized skill before acting, under neutral and competing prompts — the
agent jumps straight to Bash/Write/Edit instead of checking whether a
specialized hub skill applies first. (classify_concern/route_to_subskill/
invoke_hub_skill were skill-comply's own step labels for that behavior, not
identifiers defined anywhere in this codebase.) A skill description is read
once at discovery time and easy to skip under pressure; this hook re-injects
the reminder on every prompt so it can't be silently forgotten mid-session.
"""

import json
import sys

from _hook_io import wrap_context

REMINDER = (
    "dev-suite routes domain-specific work through hub skills before "
    "implementation. The hubs are slash-invocable only "
    "(disable-model-invocation: true), so run one explicitly: "
    "/dev-suite:architecture-and-infra, /dev-suite:backend-patterns, "
    "/dev-suite:testing-and-quality, /dev-suite:ci-cd-pipelines, "
    "/dev-suite:observability-and-sre, /dev-suite:data-and-security, "
    "/dev-suite:dev-workflows, /dev-suite:three-brain. If this request "
    "matches one of those domains, run the matching hub command "
    "(/dev-suite:dev-hub if ambiguous or it spans multiple "
    "science-suite/research-suite domains too) before jumping to "
    "Bash/Write/Edit."
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
