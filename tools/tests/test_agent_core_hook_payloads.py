"""Execute agent-core hooks against representative payloads.

The hooks are only ever exercised by the Claude Code runtime, so nothing in the
test suite used to run them. That let several hooks read environment variables
nothing sets, silently producing no output. These tests pipe realistic JSON into
each hook and assert it resolves real values rather than "unknown" placeholders.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent.parent / "plugins" / "agent-core" / "hooks"

# (script, stdin payload, substring the output must contain)
PAYLOAD_CASES = [
    (
        "pre_task_use.py",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Task",
            "tool_input": {"subagent_type": "orchestrator"},
        },
        "orchestrator",
    ),
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
        "permission_denied.py",
        {"hook_event_name": "PermissionDenied", "tool_name": "Bash"},
        "Bash",
    ),
    (
        "subagent_stop.py",
        {"hook_event_name": "SubagentStop", "agent_name": "quality-specialist"},
        "quality-specialist",
    ),
    (
        "subagent_start.py",
        {"hook_event_name": "SubagentStart", "agent_type": "Explore"},
        "Explore",
    ),
    (
        "stop_failure.py",
        {"hook_event_name": "StopFailure", "error_type": "rate_limit"},
        "rate_limit",
    ),
]

ALL_HOOKS = [
    "pre_task_use.py",
    "post_tool_use.py",
    "permission_denied.py",
    "subagent_stop.py",
    "subagent_start.py",
    "stop_failure.py",
    "session_start.py",
    "session_end.py",
    "pre_compact.py",
    "post_compact.py",
    "task_created.py",
    "task_completed.py",
]


def run_hook(script, payload=None, cwd=None, stdin_devnull=False):
    """Run a hook and return its parsed JSON stdout."""
    env = {k: v for k, v in os.environ.items() if k not in ("TOOL_INPUT", "TOOL_NAME", "AGENT_NAME")}
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


@pytest.mark.parametrize("script,payload,expected", PAYLOAD_CASES, ids=[c[0] for c in PAYLOAD_CASES])
def test_hook_resolves_payload_fields(script, payload, expected):
    """A hook given a real payload must reflect it, not fall back to a placeholder."""
    output = run_hook(script, payload)
    text = json.dumps(output)
    assert output.get("status") == "success", output
    assert expected in text, f"{script} did not resolve payload field: {text}"
    assert "unknown" not in text, f"{script} emitted a placeholder despite a valid payload: {text}"


@pytest.mark.parametrize("script", ALL_HOOKS)
def test_hook_survives_absent_stdin(script, tmp_path):
    """No stdin must not hang or crash — the timeout in run_hook is the assertion."""
    # tmp_path, not the hooks dir: session_end writes a progress file under cwd.
    output = run_hook(script, stdin_devnull=True, cwd=tmp_path)
    assert output.get("status") == "success", output


def test_pre_task_use_covers_agents_from_every_suite():
    """Capabilities are derived from frontmatter, so cross-suite agents must resolve."""
    for agent in ("julia-ml-hpc", "research-spark-orchestrator", "pinn-engineer"):
        output = run_hook("pre_task_use.py", {"tool_input": {"subagent_type": agent}})
        assert agent in output.get("additionalContext", ""), f"{agent} not resolved"


def test_session_roundtrip_is_namespaced_and_dated(tmp_path):
    """session_end writes a capped, namespaced file that session_start reads head-first."""
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    for i in range(300):
        (tmp_path / f"padding_file_with_a_long_name_{i}.txt").write_text("x")

    end = run_hook("session_end.py", {"reason": "clear"}, cwd=tmp_path)
    assert end["status"] == "success"

    progress = tmp_path / ".claude" / "progress" / "agent-core.md"
    assert progress.exists(), "progress file not written to the namespaced path"
    assert not (tmp_path / ".claude-progress.md").exists(), "wrote the old unnamespaced path"

    text = progress.read_text()
    assert text.startswith("## Session ended: "), "timestamp header must be first"
    assert len(text) < 4000, f"progress file not capped: {len(text)} chars"

    context = run_hook("session_start.py", cwd=tmp_path)["additionalContext"]
    assert "## Session ended: " in context, "timestamp did not survive truncation"
    assert "Working tree:" not in context, "duplicated the uncommitted count"


def test_session_start_skips_stale_progress(tmp_path):
    """A progress file older than the freshness window must not be injected."""
    progress = tmp_path / ".claude" / "progress" / "agent-core.md"
    progress.parent.mkdir(parents=True)
    progress.write_text("## Session ended: 2020-01-01 00:00 UTC\nReason: clear\n")

    context = run_hook("session_start.py", cwd=tmp_path)["additionalContext"]
    assert "2020-01-01" not in context, "injected a stale progress file"


def test_session_start_skips_undated_progress(tmp_path):
    """An unparsable header means unknown age, which must be treated as stale."""
    progress = tmp_path / ".claude" / "progress" / "agent-core.md"
    progress.parent.mkdir(parents=True)
    progress.write_text("no timestamp here\nsome leftover content\n")

    context = run_hook("session_start.py", cwd=tmp_path)["additionalContext"]
    assert "leftover content" not in context, "injected an undated progress file"


def test_orchestrator_routing_table_matches_agent_roster():
    """The hook derives capabilities from frontmatter; this table is still hand-synced."""
    plugins_root = HOOKS_DIR.parent.parent
    actual = set()
    for agent_file in plugins_root.glob("*/agents/*.md"):
        for line in agent_file.read_text(encoding="utf-8").splitlines()[:15]:
            if line.startswith("name:"):
                actual.add(line[5:].strip())
                break

    table_text = (HOOKS_DIR.parent / "agents" / "orchestrator.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\| .*?`([a-z0-9-]+)`", table_text, re.MULTILINE))

    assert not listed - actual, f"routing table lists non-existent agents: {listed - actual}"
    assert not actual - listed, f"agents missing from routing table: {actual - listed}"


def test_pre_compact_claims_only_agent_core_skills():
    """This hook has no business asserting priority for another suite's skills."""
    skills = run_hook("pre_compact.py", {})["priority_skills"]
    own_skills = {p.name for p in (HOOKS_DIR.parent / "skills").iterdir() if p.is_dir()}
    assert set(skills) <= own_skills, f"claims skills agent-core does not own: {set(skills) - own_skills}"
