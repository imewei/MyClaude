"""Safety-gate tests for the dev-suite iterative error resolution engine.

Focused on the negative space: the mutations that must NOT happen.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ENGINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "dev-suite"
    / "skills"
    / "iterative-error-resolution"
    / "engine.py"
)


def _load_engine() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ier_engine", ENGINE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ier_engine"] = module
    spec.loader.exec_module(module)
    return module


engine = _load_engine()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty repo-shaped cwd so the engine's relative paths stay sandboxed."""
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def no_subprocess(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every subprocess invocation instead of running it."""
    calls: list[list[str]] = []

    def record(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        calls.append(cmd)
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(engine.subprocess, "run", record)
    return calls


def make_engine(**kwargs: Any) -> Any:
    return engine.IterativeFixEngine(repo="o/r", workflow="ci.yml", **kwargs)


def analysis(error_type: str, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = {
        "category": "test",
        "error_type": error_type,
        "pattern": "x",
        "confidence": 0.9,
        "suggested_fix": "fix",
        "priority": 1,
        "context": "",
    }
    defaults.update(kwargs)
    return engine.ErrorAnalysis(**defaults)


class TestSuppressionGate:
    """C1: snapshot regeneration must never auto-report success."""

    def test_snapshot_fix_refuses_without_optin(
        self, repo: Path, no_subprocess: list[list[str]]
    ) -> None:
        eng = make_engine()
        result = eng.fix_snapshot_errors()
        assert result is engine.FixResult.NO_FIX_AVAILABLE
        assert no_subprocess == [], "must not shell out when suppression is disabled"
        assert eng.suppression_applied is False

    def test_snapshot_fix_never_returns_success_even_with_optin(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: None)
        eng = make_engine(allow_suppression=True)
        result = eng.fix_snapshot_errors()
        assert result is not engine.FixResult.SUCCESS
        assert eng.suppression_applied is True

    def test_suppression_type_is_planned_as_manual(self, repo: Path) -> None:
        actionable, skipped = make_engine().plan_fixes([analysis("test_failure")])
        assert actionable == []
        assert "--allow-suppression" in skipped[0][1]


class TestConfidenceGate:
    """H1: the documented threshold is actually enforced."""

    def test_low_confidence_is_not_dispatched(self, repo: Path) -> None:
        low = analysis("npm_eresolve", confidence=0.3)
        actionable, skipped = make_engine().plan_fixes([low])
        assert actionable == []
        assert "below the" in skipped[0][1]

    def test_high_confidence_passes(self, repo: Path) -> None:
        high = analysis("npm_eresolve", confidence=0.95)
        actionable, skipped = make_engine().plan_fixes([high])
        assert actionable == [high]
        assert skipped == []


class TestDependencySafety:
    """H3: no automated dependency removal, no unmapped installs."""

    def test_npm_404_never_uninstalls(
        self, repo: Path, no_subprocess: list[list[str]]
    ) -> None:
        err = analysis("npm_404", context="npm ERR! 404 'left-pad'")
        result = make_engine().fix_npm_404(err)
        assert result is engine.FixResult.NO_FIX_AVAILABLE
        assert no_subprocess == []

    def test_npm_404_is_planned_as_manual(self, repo: Path) -> None:
        actionable, skipped = make_engine().plan_fixes([analysis("npm_404")])
        assert actionable == []
        assert "manual confirmation" in skipped[0][1]

    def test_unmapped_module_is_not_installed(
        self, repo: Path, no_subprocess: list[list[str]]
    ) -> None:
        err = analysis(
            "python_import", context="ModuleNotFoundError: No module named 'reqeusts'"
        )
        result = make_engine().fix_python_import(err)
        assert result is engine.FixResult.NO_FIX_AVAILABLE
        assert no_subprocess == []
        assert not (repo / "requirements.txt").exists()

    def test_mapped_module_installs_the_mapped_name(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            engine.subprocess, "run", lambda cmd, **k: calls.append(cmd)
        )
        err = analysis(
            "python_import", context="ModuleNotFoundError: No module named 'cv2'"
        )
        assert make_engine().fix_python_import(err) is engine.FixResult.SUCCESS
        assert calls == [["pip", "install", "opencv-python"]]


class TestTimeoutOnlyIncreases:
    """H2: a deliberate high timeout must survive; no-op must not count as a fix."""

    def test_higher_existing_timeout_is_preserved(self, repo: Path) -> None:
        wf = repo / ".github" / "workflows" / "ci.yml"
        wf.write_text("jobs:\n  a:\n    timeout-minutes: 180\n    runs-on: ubuntu\n")
        result = make_engine().fix_timeout_error()
        assert "timeout-minutes: 180" in wf.read_text()
        assert result is engine.FixResult.NO_FIX_AVAILABLE

    def test_lower_existing_timeout_is_raised(self, repo: Path) -> None:
        wf = repo / ".github" / "workflows" / "ci.yml"
        wf.write_text("jobs:\n  a:\n    timeout-minutes: 5\n    runs-on: ubuntu\n")
        assert make_engine().fix_timeout_error() is engine.FixResult.SUCCESS
        assert "timeout-minutes: 60" in wf.read_text()

    def test_unchanged_file_is_not_rewritten(self, repo: Path) -> None:
        wf = repo / ".github" / "workflows" / "ci.yml"
        wf.write_text("jobs:\n  a:\n    timeout-minutes: 90\n    runs-on: ubuntu\n")
        before = wf.stat().st_mtime_ns
        make_engine().fix_timeout_error()
        assert wf.stat().st_mtime_ns == before


class TestOomYamlStaysValid:
    """M2: the OOM fix must not emit YAML that fails to parse."""

    def test_step_level_env_does_not_corrupt_yaml(self, repo: Path) -> None:
        wf = repo / ".github" / "workflows" / "ci.yml"
        wf.write_text(
            "name: CI\n"
            "on: push\n"
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu\n"
            "    steps:\n"
            "      - run: make\n"
            "        env:\n"
            "          FOO: bar\n"
        )
        assert make_engine().fix_oom_error() is engine.FixResult.SUCCESS
        parsed = engine.yaml.safe_load(wf.read_text())
        assert parsed["env"]["NODE_OPTIONS"] == "--max-old-space-size=4096"
        assert parsed["jobs"]["build"]["steps"][0]["env"] == {"FOO": "bar"}

    def test_existing_top_level_env_keeps_its_keys(self, repo: Path) -> None:
        wf = repo / ".github" / "workflows" / "ci.yml"
        wf.write_text(
            "name: CI\n"
            "on: push\n"
            "env:\n"
            "  EXISTING: keepme\n"
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu\n"
        )
        assert make_engine().fix_oom_error() is engine.FixResult.SUCCESS
        parsed = engine.yaml.safe_load(wf.read_text())
        assert parsed["env"]["NODE_OPTIONS"] == "--max-old-space-size=4096"
        assert parsed["env"]["EXISTING"] == "keepme"
        assert parsed["jobs"]["build"]["runs-on"] == "ubuntu"

    def test_already_invalid_workflow_is_left_alone(self, repo: Path) -> None:
        wf = repo / ".github" / "workflows" / "ci.yml"
        broken = "jobs:\n  a:\n   - [unclosed\n"
        wf.write_text(broken)
        assert make_engine().fix_oom_error() is engine.FixResult.NO_FIX_AVAILABLE
        assert wf.read_text() == broken


class TestDryRunGate:
    """C2: without --auto-commit nothing is mutated at all."""

    def test_plan_gate_exits_before_any_fix_runs(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eng = make_engine()
        monkeypatch.setattr(
            eng,
            "analyze_run",
            lambda _rid: [
                {
                    "type": "npm_eresolve",
                    "pattern": "p",
                    "match": "npm ERR! code ERESOLVE",
                    "context": "",
                }
            ],
        )

        def boom(_err: Any) -> Any:
            raise AssertionError("apply_fix ran before the dry-run gate")

        monkeypatch.setattr(eng, "apply_fix", boom)
        monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: None)

        with pytest.raises(SystemExit) as exc:
            eng.run("1")
        assert exc.value.code == 0


class TestErroredIsDistinct:
    """L1: a crashing strategy is not the same as no strategy existing."""

    def test_exception_maps_to_errored(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eng = make_engine()

        def blow_up() -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(eng, "fix_npm_eresolve", blow_up)
        result = eng.apply_fix(analysis("npm_eresolve", category="dependency"))
        assert result is engine.FixResult.ERRORED


class TestKnowledgeBaseSampleFloor:
    """M3: one lucky result must not pin a strategy at 100%."""

    def test_single_success_is_not_trusted(self, repo: Path) -> None:
        kb = engine.KnowledgeBase()
        kb.record_fix("npm fix", True)
        assert kb.get_success_rate("npm_eresolve") == engine.NEUTRAL_CONFIDENCE
        assert kb.get_confidence("npm_eresolve") == engine.NEUTRAL_CONFIDENCE

    def test_rate_is_trusted_once_sampled(self, repo: Path) -> None:
        kb = engine.KnowledgeBase()
        for _ in range(engine.MIN_SAMPLES):
            kb.record_fix("npm fix", True)
        assert kb.get_success_rate("npm_eresolve") == 1.0
