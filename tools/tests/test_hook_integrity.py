"""Hook integrity tests across all plugin suites.

Validates hooks.json structure and handler script existence.
"""

import json
import pytest
from pathlib import Path

PLUGINS_ROOT = Path(__file__).parent.parent.parent / "plugins"

VALID_HOOK_EVENTS = {
    "SessionStart",
    "SessionEnd",
    "PreToolUse",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "PreSubagentUse",
    "SubagentStart",
    "SubagentStop",
    "PermissionDenied",
    "PermissionPrompt",
    "PermissionApproved",
    "TaskCreated",
    "TaskCompleted",
    "Notification",
    "IdlePrompt",
    "AuthSuccess",
    "ElicitationDialog",
    "StopFailure",
    "ExecutionError",
    "ContextOverflow",
    "CostThreshold",
}

VALID_HANDLER_TYPES = {"command", "http", "prompt", "agent"}

SUITES_WITH_HOOKS = ["agent-core", "dev-suite", "science-suite"]


@pytest.fixture
def hooks_data():
    """Load all hooks.json files."""
    data = {}
    for suite in SUITES_WITH_HOOKS:
        hooks_file = PLUGINS_ROOT / suite / "hooks" / "hooks.json"
        if hooks_file.exists():
            with open(hooks_file, "r", encoding="utf-8") as f:
                data[suite] = json.load(f)
    return data


class TestHookStructure:
    """Test hooks.json structure for all suites."""

    @pytest.mark.parametrize("suite", SUITES_WITH_HOOKS)
    def test_hooks_json_exists(self, suite):
        hooks_file = PLUGINS_ROOT / suite / "hooks" / "hooks.json"
        assert hooks_file.exists(), f"{suite} missing hooks/hooks.json"

    @pytest.mark.parametrize("suite", SUITES_WITH_HOOKS)
    def test_hooks_json_valid_json(self, suite):
        hooks_file = PLUGINS_ROOT / suite / "hooks" / "hooks.json"
        if not hooks_file.exists():
            pytest.skip(f"{suite} has no hooks.json")
        with open(hooks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "hooks" in data, f"{suite} hooks.json missing 'hooks' key"
        assert "description" in data, f"{suite} hooks.json missing 'description'"

    @pytest.mark.parametrize("suite", SUITES_WITH_HOOKS)
    def test_hook_events_are_valid(self, suite):
        hooks_file = PLUGINS_ROOT / suite / "hooks" / "hooks.json"
        if not hooks_file.exists():
            pytest.skip(f"{suite} has no hooks.json")
        with open(hooks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for event_name in data.get("hooks", {}):
            assert event_name in VALID_HOOK_EVENTS, (
                f"{suite}: Unknown hook event '{event_name}'"
            )

    @pytest.mark.parametrize("suite", SUITES_WITH_HOOKS)
    def test_handler_types_are_valid(self, suite):
        hooks_file = PLUGINS_ROOT / suite / "hooks" / "hooks.json"
        if not hooks_file.exists():
            pytest.skip(f"{suite} has no hooks.json")
        with open(hooks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for event_name, handlers_list in data.get("hooks", {}).items():
            for handler_group in handlers_list:
                for hook in handler_group.get("hooks", []):
                    assert hook.get("type") in VALID_HANDLER_TYPES, (
                        f"{suite}/{event_name}: Invalid handler type '{hook.get('type')}'"
                    )


class TestHandlerScripts:
    """Test that command-type hooks reference existing scripts."""

    @pytest.mark.parametrize("suite", SUITES_WITH_HOOKS)
    def test_command_scripts_exist(self, suite):
        hooks_file = PLUGINS_ROOT / suite / "hooks" / "hooks.json"
        if not hooks_file.exists():
            pytest.skip(f"{suite} has no hooks.json")
        with open(hooks_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        hooks_dir = PLUGINS_ROOT / suite / "hooks"

        for event_name, handlers_list in data.get("hooks", {}).items():
            for handler_group in handlers_list:
                for hook in handler_group.get("hooks", []):
                    if hook.get("type") == "command":
                        cmd = hook.get("command", "")
                        # Extract script path from command like
                        # "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/script.py"
                        if "${CLAUDE_PLUGIN_ROOT}/hooks/" in cmd:
                            script_name = cmd.split("/hooks/")[-1]
                            script_path = hooks_dir / script_name
                            assert script_path.exists(), (
                                f"{suite}/{event_name}: Script missing: {script_name}"
                            )
