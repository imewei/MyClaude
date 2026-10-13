#!/usr/bin/env python3
"""UserPromptSubmit hook for dev-suite: a conditional routing hint.

A skill-comply run (an external plugin, not in this repo) found models jump straight to
Bash/Write/Edit without checking whether a specialized hub skill applies. A standing reminder
fixed that but cost ~120 tokens on every prompt. This hook injects one short line only when the
prompt matches a hub's domain, names just the matching hubs, and repeats a hub's hint only after
REPEAT_AFTER prompts. Prompts that start with "/" (explicit commands) are left alone.
"""

SUITE = "dev-suite"
FALLBACK = "dev-hub"
HUBS = {
    "architecture-and-infra": r"architect|microservice|monorepo|system design|docker|kubernetes|k8s|terraform|helm|container|infrastructure|\bcli\b",
    "backend-patterns": r"\bapi\b|\brest(ful)?\b|graphql|websocket|node(\.js)?\b|express|fastapi|asyncio|\basync\b|message queue|kafka|rabbitmq|backend",
    "ci-cd-pipelines": r"ci/?cd|github actions|gitlab[- ]ci|\bpipelines?\b|workflow file|\bdeploy|release process|canary|blue-green",
    "data-and-security": r"database|\bsql\b|postgres|mysql|schema|redis|\bcach(e|ing)\b|elasticsearch|auth(entication|orization)?\b|oauth|\bjwt\b|secrets?\b|vulnerab|owasp|encrypt|security",
    "dev-workflows": r"\bgit\b|branch|commit|rebase|merge|pull request|\bprs?\b|changelog|airflow|\bdag\b|documentation|readme|debug|stack ?trace|\bbugs?\b",
    "observability-and-sre": r"prometheus|grafana|tracing|opentelemetry|\botel\b|\bslo\b|\bsli\b|observab|incident|postmortem|alerting|\balerts?\b|latency|on-?call|metrics",
    "testing-and-quality": r"\btests?\b|pytest|unit test|\be2e\b|playwright|cypress|coverage|\btdd\b|flaky|code review|\blint|\bmocks?\b",
}

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from _hook_io import get_field, read_payload, wrap_context

REPEAT_AFTER = 12  # prompts before the same hub is hinted again
MAX_HUBS = 3

PATTERNS = {hub: re.compile(rx, re.IGNORECASE) for hub, rx in HUBS.items()}


def classify(prompt: str) -> list[str]:
    """Hubs whose domain the prompt touches, best match first."""
    scores = {hub: len(rx.findall(prompt)) for hub, rx in PATTERNS.items()}
    return [h for h, n in sorted(scores.items(), key=lambda kv: -kv[1]) if n][:MAX_HUBS]


def due_hubs(matched: list[str], session_id: str) -> list[str]:
    """Matched hubs not hinted in the last REPEAT_AFTER prompts of this session (state lives in the temp dir)."""
    uid = getattr(os, "getuid", lambda: 0)()
    base = Path(os.environ.get("MYCLAUDE_HINT_DIR") or tempfile.gettempdir()) / f"myclaude-route-hints-{uid}"
    try:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        return matched  # no writable state: hint every time rather than fail the prompt
    path = base / f"{SUITE}-{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)[:80]}.json"
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        state = {}
    n = int(state.get("n", 0)) + 1
    last = state.get("last", {})
    due = [h for h in matched if n - int(last.get(h, -REPEAT_AFTER)) >= REPEAT_AFTER]
    last.update({h: n for h in due})
    try:
        path.write_text(json.dumps({"n": n, "last": last}))
    except OSError:
        pass  # state is a convenience; a failed write only means the hint may repeat
    return due


def main() -> None:
    try:
        payload = read_payload()
        prompt = payload.get("prompt")
        result: dict = {"status": "success"}
        if isinstance(prompt, str) and prompt.strip() and not prompt.lstrip().startswith("/"):
            hubs = due_hubs(classify(prompt), get_field(payload, "session_id", default="unknown"))
            if hubs:
                hint = (
                    f"Routing hint: this touches {', '.join(f'{SUITE}:{h}' for h in hubs)}. "
                    f"Load the matching hub skill before implementing ({SUITE}:{FALLBACK} if unsure)."
                )
                result["additionalContext"] = hint
                result.update(wrap_context("UserPromptSubmit", hint))
        json.dump(result, sys.stdout)
    except Exception as e:
        print(f"UserPromptSubmit hook error: {e}", file=sys.stderr)
        json.dump({"status": "error", "message": f"UserPromptSubmit hook error: {e}"}, sys.stdout)


if __name__ == "__main__":
    main()
