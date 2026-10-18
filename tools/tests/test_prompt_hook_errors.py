"""Prompt routing failures must still emit Codex-compatible JSON."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("suite,prompt", [
    ("dev-suite", "debug this bug"),
    ("science-suite", "train a transformer with pytorch"),
])
def test_invalid_routing_state_reports_error_without_invalid_json(suite, prompt, tmp_path):
    state_dir = tmp_path / f"myclaude-route-hints-{getattr(os, 'getuid', lambda: 0)()}"
    state_dir.mkdir()
    (state_dir / f"{suite}-error-test.json").write_text('{"n": "invalid"}')
    script = Path(__file__).resolve().parents[2] / "plugins" / suite / "hooks" / "user_prompt_submit.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps({"session_id": "error-test", "prompt": prompt}),
        text=True,
        capture_output=True,
        env={**os.environ, "MYCLAUDE_HINT_DIR": str(tmp_path)},
        timeout=15,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {}
    assert "UserPromptSubmit hook error:" in result.stderr
