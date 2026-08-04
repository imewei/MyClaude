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


def hook_context(out: dict) -> str:
    """Pull additionalContext from the shape Claude Code actually consumes.

    A top-level "additionalContext" key is silently ignored by Claude Code —
    only hookSpecificOutput.additionalContext reaches the model, so that is
    the shape tests must assert against, not the hook's own convention.
    """
    return out.get("hookSpecificOutput", {}).get("additionalContext", "")


class TestUserPromptSubmit:
    """An external skill-comply plugin run (not checked into this repo) found
    science-suite hub skills fail to classify-then-route to a specialized
    skill before acting, under neutral/competing prompts. (classify_task/
    route_to_specialized_skill/consult_routing_tree were skill-comply's own
    step labels for that behavior, not identifiers defined in this codebase.)
    This hook re-injects a routing reminder every turn, regardless of what
    the prompt says, so it can't be silently skipped mid-session."""

    def test_emits_routing_reminder(self):
        out = run_hook(
            "user_prompt_submit.py",
            {"hook_event_name": "UserPromptSubmit", "prompt": "train a small CNN on MNIST"},
        )
        assert out["status"] == "success"
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit", (
            f"wrong hookEventName, a typo here would silently disable delivery: {out}"
        )
        context = hook_context(out)
        assert "hub skill" in context
        assert "science-hub" in context

    def test_fires_regardless_of_prompt_content(self):
        out = run_hook(
            "user_prompt_submit.py",
            {"hook_event_name": "UserPromptSubmit", "prompt": "what time is it"},
        )
        assert out["status"] == "success"
        assert "hub skill" in hook_context(out)

    def test_output_is_identical_regardless_of_prompt_content(self):
        """The reminder is a static, input-independent nudge — assert exact
        equality across unrelated prompts, not just a shared substring, so
        prompt text leaking into the context or partial truncation would fail."""
        first = run_hook(
            "user_prompt_submit.py",
            {"hook_event_name": "UserPromptSubmit", "prompt": "optimize this MCMC sampler"},
        )
        second = run_hook(
            "user_prompt_submit.py",
            {"hook_event_name": "UserPromptSubmit", "prompt": "what time is it"},
        )
        assert hook_context(first) == hook_context(second)

    def test_survives_absent_stdin(self, tmp_path):
        """No stdin must not hang or crash — the timeout below is the assertion."""
        result = subprocess.run(
            ["python3", str(HOOKS_DIR / "user_prompt_submit.py")],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(tmp_path),
            check=False,
        )
        assert result.returncode == 0, f"crashed on absent stdin: {result.stderr}"
        assert json.loads(result.stdout)["status"] == "success"

    def test_survives_malformed_stdin(self, tmp_path):
        """Malformed JSON on stdin must degrade to defaults, not crash — this
        hook never reads its payload, but nothing currently pins that down."""
        result = subprocess.run(
            ["python3", str(HOOKS_DIR / "user_prompt_submit.py")],
            input="{not valid json",
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(tmp_path),
            check=False,
        )
        assert result.returncode == 0, f"crashed on malformed stdin: {result.stderr}"
        assert json.loads(result.stdout)["status"] == "success"


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
        context = hook_context(out)
        assert "nan" in context.lower(), f"no NaN warning in {out}"
        assert "inf" in context.lower(), f"no Inf warning in {out}"

    @pytest.mark.parametrize(
        "stdout",
        [
            "[ 1.  nan  2.]",
            "3-element Vector{Float64}:\n 1.0\n  nan\n 2.0",
            "mean     nan\nstd      nan",
            "100   nan    inf",
            "r_hat  nan",
            "nan",
            "energy=nan",
        ],
        ids=[
            "numpy-1d-repr",
            "julia-vector-display",
            "pandas-describe",
            "whitespace-table",
            "arviz-rhat",
            "bare-nan-line",
            "key-equals-nan",
        ],
    )
    def test_warns_on_real_diverged_output_formats(self, stdout):
        """Regression guard: the regex must catch the formats this suite actually emits."""
        out = run_hook(
            "post_tool_use.py",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_response": {"stdout": stdout},
            },
        )
        context = hook_context(out)
        assert "nan" in context.lower() or "inf" in context.lower(), (
            f"missed real diverged-output format {stdout!r}: {out}"
        )

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
        context = hook_context(out)
        assert "verify the computation" not in context, (
            f"false positive on {stdout!r}: {out}"
        )

    def test_no_context_on_clean_output(self):
        """A regex scan can warn on a hit; it must not assert cleanliness on a miss."""
        out = run_hook(
            "post_tool_use.py",
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Bash",
                "tool_response": {"stdout": "loss = 0.031\naccuracy = 0.98"},
            },
        )
        assert "additionalContext" not in out
        assert "hookSpecificOutput" not in out


class TestSubagentStop:
    """SubagentStop must resolve the agent name and gate on relevance."""

    @pytest.mark.parametrize("key", ["agent_name", "subagent_type"])
    def test_resolves_numerical_agent_name(self, key):
        out = run_hook(
            "subagent_stop.py", {"hook_event_name": "SubagentStop", key: "jax-pro"}
        )
        context = hook_context(out)
        assert "jax-pro" in context, f"agent name not resolved: {out}"
        assert "unknown" not in context

    def test_silent_for_non_numerical_agent(self):
        out = run_hook(
            "subagent_stop.py",
            {"hook_event_name": "SubagentStop", "agent_name": "sci-workflow-engineer"},
        )
        assert "additionalContext" not in out, f"nudge fired on advisory agent: {out}"
        assert "hookSpecificOutput" not in out, f"nudge fired on advisory agent: {out}"

    def test_resolves_continuum_mechanics_engineer(self):
        """continuum-mechanics-engineer (FEM/FEA, constitutive modeling) is
        clearly numerical but was missing from NUMERICAL_AGENTS — allowlist
        gap fixed alongside the audit's other findings."""
        out = run_hook(
            "subagent_stop.py",
            {"hook_event_name": "SubagentStop", "agent_name": "continuum-mechanics-engineer"},
        )
        context = hook_context(out)
        assert "continuum-mechanics-engineer" in context, f"agent name not resolved: {out}"


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
        context = hook_context(start)
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
        assert "## Session ended:" not in hook_context(start)

    def test_unparseable_progress_is_not_injected(self, tmp_path):
        progress_file = tmp_path / PROGRESS_RELPATH
        progress_file.parent.mkdir(parents=True)
        progress_file.write_text("?? some/stale/git/status.py\n", encoding="utf-8")
        start = run_hook(
            "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(tmp_path)}
        )
        assert "stale/git/status" not in hook_context(start)


class TestNoEnvVarDependency:
    """The env vars nothing sets must not be the only source of truth."""

    def test_hooks_do_not_read_unset_env_vars(self):
        for script in ("post_tool_use.py", "subagent_stop.py"):
            source = (HOOKS_DIR / script).read_text(encoding="utf-8")
            assert "TOOL_OUTPUT" not in source
            assert "AGENT_NAME" not in source
