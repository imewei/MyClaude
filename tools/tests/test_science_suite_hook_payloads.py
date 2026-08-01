"""Science-suite hook payload tests.

Pipes realistic Claude Code hook stdin payloads into each science-suite hook
and asserts the hook actually reads them, rather than silently degrading to a
placeholder as it did when reading unset env vars.
"""

import json
import subprocess
from pathlib import Path

import pytest

HOOKS_DIR = (
    Path(__file__).parent.parent.parent / "plugins" / "science-suite" / "hooks"
)
PROGRESS_RELPATH = Path(".claude") / "progress" / "science-suite.md"


def run_hook(script: str, payload: dict, cwd: Path | None = None) -> dict:
    """Run a hook with the payload on stdin and return its parsed JSON output."""
    result = subprocess.run(
        ["python3", str(HOOKS_DIR / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    assert result.stdout, f"{script} produced no stdout (stderr: {result.stderr})"
    return json.loads(result.stdout)


class TestPostToolUse:
    """PostToolUse must read tool_response and only flag numeric NaN/Inf."""

    def test_warns_on_nan_and_inf_in_stdout(self):
        out = run_hook(
            "post_tool_use.py",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": "python3 train.py"},
                "tool_response": {"stdout": "loss = nan\ngrad = inf"},
            },
        )
        context = out.get("additionalContext", "")
        assert "nan" in context.lower(), f"no NaN warning in {out}"
        assert "inf" in context.lower(), f"no Inf warning in {out}"

    @pytest.mark.parametrize(
        "stdout",
        [
            "grep -rn nan src/\ntools/tests/test_nan_handling.py",
            "commit a1b2c3d fix nan handling in the solver",
            "collected 12 items\ntest_nan_guard.py .... [100%]",
        ],
    )
    def test_no_warning_on_unrelated_text(self, stdout):
        out = run_hook(
            "post_tool_use.py",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_response": {"stdout": stdout},
            },
        )
        context = out.get("additionalContext", "")
        assert "verify the computation" not in context, (
            f"false positive on {stdout!r}: {out}"
        )

    def test_reports_clean_check_on_normal_output(self):
        out = run_hook(
            "post_tool_use.py",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_response": {"stdout": "loss = 0.031\naccuracy = 0.98"},
            },
        )
        assert "no NaN/Inf" in out.get("additionalContext", "")


class TestSubagentStop:
    """SubagentStop must resolve the agent name and gate on relevance."""

    @pytest.mark.parametrize("key", ["agent_name", "subagent_type"])
    def test_resolves_numerical_agent_name(self, key):
        out = run_hook(
            "subagent_stop.py", {"hook_event_name": "SubagentStop", key: "jax-pro"}
        )
        context = out.get("additionalContext", "")
        assert "jax-pro" in context, f"agent name not resolved: {out}"
        assert "unknown" not in context

    def test_silent_for_non_numerical_agent(self):
        out = run_hook(
            "subagent_stop.py",
            {"hook_event_name": "SubagentStop", "agent_name": "sci-workflow-engineer"},
        )
        assert "additionalContext" not in out, f"nudge fired on advisory agent: {out}"


class TestSessionRoundTrip:
    """SessionEnd writes a namespaced summary that SessionStart reads back."""

    def test_progress_round_trip_keeps_timestamp(self, tmp_path):
        end = run_hook(
            "session_end.py",
            {"hook_event_name": "SessionEnd", "reason": "clear", "cwd": str(tmp_path)},
        )
        assert "clear" in end.get("message", ""), f"reason not read: {end}"

        progress_file = tmp_path / PROGRESS_RELPATH
        assert progress_file.exists(), "SessionEnd did not write the namespaced path"

        start = run_hook(
            "session_start.py",
            {"hook_event_name": "SessionStart", "source": "startup", "cwd": str(tmp_path)},
        )
        context = start.get("additionalContext", "")
        assert "## Session ended:" in context, f"timestamp truncated away: {context}"
        assert "Science compute env:" in context

    def test_stale_progress_is_not_injected(self, tmp_path):
        progress_file = tmp_path / PROGRESS_RELPATH
        progress_file.parent.mkdir(parents=True)
        progress_file.write_text(
            "## Session ended: 2020-01-01 00:00 UTC\nReason: clear\n", encoding="utf-8"
        )
        start = run_hook(
            "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(tmp_path)}
        )
        assert "## Session ended:" not in start.get("additionalContext", "")

    def test_unparseable_progress_is_not_injected(self, tmp_path):
        progress_file = tmp_path / PROGRESS_RELPATH
        progress_file.parent.mkdir(parents=True)
        progress_file.write_text("?? some/stale/git/status.py\n", encoding="utf-8")
        start = run_hook(
            "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(tmp_path)}
        )
        assert "stale/git/status" not in start.get("additionalContext", "")


class TestNoEnvVarDependency:
    """The env vars nothing sets must not be the only source of truth."""

    def test_hooks_do_not_read_unset_env_vars(self):
        for script in ("post_tool_use.py", "subagent_stop.py"):
            source = (HOOKS_DIR / script).read_text(encoding="utf-8")
            assert "TOOL_OUTPUT" not in source
            assert "AGENT_NAME" not in source
