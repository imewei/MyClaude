"""Execute dev-suite hooks against representative payloads.

dev-suite is the largest suite (49 sub-skills) but had no hook-payload test
file — unlike agent-core and science-suite — which is plausibly why
session_end.py's raw `json.load(sys.stdin)` (bypassing the shared
`_hook_io.read_payload()` stdin-first pattern every other hook here uses)
survived the original architecture-audit fix pass. These tests pipe
realistic JSON into each hook and assert it resolves real payload fields,
survives malformed/absent stdin, and — for context-injecting hooks — emits
the nested `hookSpecificOutput.additionalContext` shape Claude Code actually
reads, not just the hook's own top-level convention.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent.parent / "plugins" / "dev-suite" / "hooks"

ALL_HOOKS = [
    "session_start.py",
    "post_tool_use.py",
    "subagent_stop.py",
    "task_completed.py",
    "session_end.py",
    "stop_failure.py",
]

# (script, stdin payload, substring the output must contain)
PAYLOAD_CASES = [
    (
        "post_tool_use.py",
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Edit",
            "tool_input": {"file_path": "/repo/module.py"},
        },
        "/repo/module.py",
    ),
    (
        "subagent_stop.py",
        {"hook_event_name": "SubagentStop", "agent_name": "debugger-pro"},
        "debugger-pro",
    ),
    (
        "stop_failure.py",
        {"hook_event_name": "StopFailure", "error_message": "rate limited"},
        "rate limited",
    ),
]


def run_hook(script, payload=None, cwd=None, stdin_devnull=False):
    """Run a hook and return its parsed JSON stdout."""
    env = {k: v for k, v in os.environ.items() if k not in ("TOOL_INPUT", "AGENT_NAME")}
    if cwd is not None:
        env["PWD"] = str(cwd)
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input=None if stdin_devnull else json.dumps(payload or {}),
        stdin=subprocess.DEVNULL if stdin_devnull else None,
        capture_output=True,
        text=True,
        timeout=15,
        cwd=cwd or HOOKS_DIR,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{script} exited {result.returncode}: {result.stderr}"
    return json.loads(result.stdout)


def hook_context(out: dict) -> str:
    """Pull additionalContext from the shape Claude Code actually consumes."""
    return out.get("hookSpecificOutput", {}).get("additionalContext", "")


@pytest.mark.parametrize("script,payload,expected", PAYLOAD_CASES, ids=[c[0] for c in PAYLOAD_CASES])
def test_hook_resolves_payload_fields(script, payload, expected):
    """A hook given a real payload must reflect it, not fall back to a placeholder."""
    output = run_hook(script, payload)
    text = json.dumps(output)
    assert output.get("status") == "success", output
    assert expected in text, f"{script} did not resolve payload field: {text}"
    # A top-level "additionalContext" key is silently ignored by Claude Code —
    # only hookSpecificOutput.additionalContext actually reaches the model.
    if "additionalContext" in output:
        nested = output.get("hookSpecificOutput", {})
        assert nested.get("additionalContext") == output["additionalContext"], (
            f"{script} emits additionalContext but not the nested shape Claude "
            f"Code actually reads: {text}"
        )
        assert nested.get("hookEventName"), f"{script} nested output missing hookEventName: {text}"


@pytest.mark.parametrize("script", ALL_HOOKS)
def test_hook_survives_absent_stdin(script, tmp_path):
    """No stdin must not hang or crash — the timeout in run_hook is the assertion."""
    output = run_hook(script, stdin_devnull=True, cwd=tmp_path)
    assert output.get("status") == "success", output


@pytest.mark.parametrize("script", ALL_HOOKS)
def test_hook_survives_malformed_stdin(script, tmp_path):
    """Malformed JSON on stdin must degrade to defaults, not crash."""
    env = dict(os.environ)
    env["PWD"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / script)],
        input="{not valid json",
        capture_output=True,
        text=True,
        timeout=15,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"{script} crashed on malformed stdin: {result.stderr}"
    output = json.loads(result.stdout)
    assert output.get("status") == "success", output


def test_session_end_resolves_real_reason_not_matcher_input_only(tmp_path):
    """Regression guard: session_end.py used to bypass _hook_io and read only
    the matcher_input fallback key, so every progress file recorded Reason:
    unknown regardless of what the SessionEnd payload actually said."""
    output = run_hook("session_end.py", {"reason": "clear"}, cwd=tmp_path)
    assert output["status"] == "success"
    assert "clear" in output.get("message", ""), f"reason not resolved: {output}"

    progress = tmp_path / ".claude" / "progress" / "dev-suite.md"
    assert progress.exists(), "progress file not written"
    text = progress.read_text()
    assert "Reason: clear" in text, f"reason not persisted correctly: {text}"
    assert "Reason: unknown" not in text


def test_session_roundtrip_is_namespaced(tmp_path):
    """session_end writes a namespaced file that session_start reads back."""
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)

    end = run_hook("session_end.py", {"reason": "clear"}, cwd=tmp_path)
    assert end["status"] == "success"

    progress = tmp_path / ".claude" / "progress" / "dev-suite.md"
    assert progress.exists(), "progress file not written to the namespaced path"

    start = run_hook("session_start.py", cwd=tmp_path)
    context = hook_context(start)
    assert "## Session ended: " in context, "timestamp did not survive round trip"


def test_session_start_reads_payload_cwd_not_just_pwd(tmp_path):
    """Regression guard: session_start.py used to read only the PWD env var,
    ignoring the payload's own cwd field — breaking under isolation: worktree,
    which several dev-suite agents declare."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")

    # PWD points somewhere else entirely; only the payload cwd is correct.
    env = dict(os.environ)
    env["PWD"] = "/nonexistent/wrong/directory"
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / "session_start.py")],
        input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(tmp_path)}),
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(tmp_path),
        env=env,
        check=False,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    context = hook_context(output)
    assert "python" in context.lower(), f"did not detect stack from payload cwd: {context}"


# --- ai-pair fabrication check (subagent_stop.py) -------------------------


def test_subagent_stop_catches_real_reviewer_heading(tmp_path):
    """The reviewer subagent's OWN transcript uses 'Codex Code Review' /
    'Codex Content Review' (agent-prompts.md), never the bare 'Codex Review'
    heading — that only appears in the Team Lead's later consolidation."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"assistant","content":"## Codex Code Review\\nFabricated, no CLI call."}\n'
    )
    out = run_hook(
        "subagent_stop.py",
        {"agent_name": "codex-reviewer", "transcript_path": str(transcript)},
    )
    assert "integrity check" in hook_context(out)


def test_subagent_stop_case_insensitive_heading_match(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("## CODEX CODE REVIEW\nno bash call to codex anywhere\n")
    out = run_hook(
        "subagent_stop.py",
        {"agent_name": "x", "transcript_path": str(transcript)},
    )
    assert "integrity check" in hook_context(out)


def test_subagent_stop_does_not_false_flag_on_escaped_quotes(tmp_path):
    """A real codex invocation preceded by a quoted echo/arg must not trip
    the CLI-invocation regex just because an escaped quote appears first."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"input": {"command": "echo \\"note\\" && codex exec --prompt \\"review\\""}}\n'
        "## Codex Code Review\nfindings here\n"
    )
    out = run_hook(
        "subagent_stop.py",
        {"agent_name": "codex-reviewer", "transcript_path": str(transcript)},
    )
    assert "integrity check" not in hook_context(out)


def test_subagent_stop_clean_when_cli_actually_invoked(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"input": {"command": "codex exec --full-auto review.txt"}}\n'
        "## Codex Code Review\nfindings here\n"
    )
    out = run_hook(
        "subagent_stop.py",
        {"agent_name": "codex-reviewer", "transcript_path": str(transcript)},
    )
    assert "integrity check" not in hook_context(out)


def test_subagent_stop_logs_when_transcript_path_missing(tmp_path):
    """A stale/wrong transcript_path must not silently no-op with zero trail."""
    result = subprocess.run(
        [sys.executable, str(HOOKS_DIR / "subagent_stop.py")],
        input=json.dumps(
            {"agent_name": "x", "transcript_path": str(tmp_path / "does-not-exist.jsonl")}
        ),
        capture_output=True,
        text=True,
        timeout=15,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0
    assert "transcript_path not found" in result.stderr


# --- HEAD staleness (session_end.py -> session_start.py) ------------------


def test_stale_head_marker_appears_after_new_commit(tmp_path):
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", "a"], cwd=tmp_path, check=True
    )

    end = run_hook("session_end.py", {"reason": "clear"}, cwd=tmp_path)
    assert end["status"] == "success"

    subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", "b"], cwd=tmp_path, check=True
    )

    start = run_hook("session_start.py", cwd=tmp_path)
    assert "STALE" in hook_context(start)


def test_head_unchanged_no_stale_marker(tmp_path):
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", "a"], cwd=tmp_path, check=True
    )

    end = run_hook("session_end.py", {"reason": "clear"}, cwd=tmp_path)
    assert end["status"] == "success"

    start = run_hook("session_start.py", cwd=tmp_path)
    assert "STALE" not in hook_context(start)


def test_head_check_fails_open_when_not_a_git_repo(tmp_path):
    """Deliberate design choice, pinned down: if HEAD can't be verified (no
    git repo here), show the recorded progress unmodified rather than
    guessing staleness — see the comment in session_start.py."""
    from datetime import UTC, datetime

    progress_dir = tmp_path / ".claude" / "progress"
    progress_dir.mkdir(parents=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    (progress_dir / "dev-suite.md").write_text(
        f"## Session ended: {ts} UTC\nReason: clear\nHEAD: abc1234\n\n### Recent commits\nx\n"
    )
    # tmp_path is NOT a git repository at this point.

    start = run_hook("session_start.py", cwd=tmp_path)
    assert "STALE" not in hook_context(start)
    assert "HEAD: abc1234" in hook_context(start)


def test_progress_file_without_head_line_degrades_gracefully(tmp_path):
    """Backward compatibility: progress files written before this HEAD-check
    feature existed have no HEAD: line — must not crash or false-flag."""
    from datetime import UTC, datetime

    progress_dir = tmp_path / ".claude" / "progress"
    progress_dir.mkdir(parents=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    (progress_dir / "dev-suite.md").write_text(
        f"## Session ended: {ts} UTC\nReason: clear\n\n### Recent commits\nx\n"
    )

    start = run_hook("session_start.py", cwd=tmp_path)
    assert start["status"] == "success"
    assert "STALE" not in hook_context(start)
    assert "Reason: clear" in hook_context(start)
