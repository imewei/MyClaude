"""Research-suite script and hook behavior tests.

These cover the failure modes where the tooling produced a wrong answer while
reporting success: silent bibliography truncation, fabricated symbol
mismatches, and confidently-wrong stage claims injected at session start.

Scripts and hooks live under directories with hyphens, so they are exercised
through subprocess rather than imported.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SUITE = Path(__file__).parent.parent.parent / "plugins" / "research-suite"
HOOKS = SUITE / "hooks"
COMMONS = SUITE / "skills" / "_research-commons" / "scripts"
DEDUPE = SUITE / "skills" / "landscape-scanner" / "scripts" / "dedupe_refs.py"


def run_script(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def run_hook(name: str, payload: dict) -> dict:
    """Run a hook with a JSON payload on stdin; return parsed stdout ({} if empty)."""
    result = subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(HOOKS),
    )
    assert result.returncode == 0, f"{name} exited {result.returncode}: {result.stderr}"
    return json.loads(result.stdout) if result.stdout.strip() else {}


# --- #1 dedupe_refs.py: silent data destruction ---------------------------

AT_IN_FIELD = """@article{Alpha2020,
  author = {Doe, Jane},
  title = {First Paper},
  year = {2020},
  note = {Contact jane@example.com for data}
}

@article{Beta2021,
  author = {Roe, Rick},
  title = {Second Paper},
  year = {2021}
}
"""

BRACE_ON_SAME_LINE = """@article{Alpha2020,
  author = {Doe, Jane},
  title = {First Paper},
  doi = {10.1/a}}

@article{Beta2021,
  author = {Roe, Rick},
  title = {Second Paper},
  doi = {10.1/b}}
"""


@pytest.mark.parametrize("bib_text", [AT_IN_FIELD, BRACE_ON_SAME_LINE])
def test_dedupe_keeps_both_entries(tmp_path, bib_text):
    """Neither an `@` inside a field nor `}}` on the last line may drop entries."""
    src = tmp_path / "refs.bib"
    src.write_text(bib_text, encoding="utf-8")
    out = tmp_path / "out.bib"

    result = run_script(DEDUPE, str(src), str(out))

    assert result.returncode == 0, result.stderr
    assert "Read 2/2 entries" in result.stderr
    written = out.read_text(encoding="utf-8")
    assert "Alpha2020" in written
    assert "Beta2021" in written


def test_dedupe_still_drops_real_duplicates(tmp_path):
    src = tmp_path / "refs.bib"
    src.write_text(
        "@article{A,\n  doi = {10.1/x}\n}\n\n@article{B,\n  doi = {10.1/X}\n}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.bib"

    result = run_script(DEDUPE, str(src), str(out))

    assert result.returncode == 0
    assert "dropped 1 duplicate(s)" in result.stderr


def test_dedupe_aborts_without_writing_on_unparseable_input(tmp_path):
    """An unaccountable parse must fail loudly and leave the output untouched."""
    src = tmp_path / "refs.bib"
    src.write_text("@article{Broken,\n  title = {Never closed}\n", encoding="utf-8")
    out = tmp_path / "out.bib"
    out.write_text("PRE-EXISTING\n", encoding="utf-8")

    result = run_script(DEDUPE, str(src), str(out))

    assert result.returncode != 0
    assert "not verified" in result.stderr
    assert out.read_text(encoding="utf-8") == "PRE-EXISTING\n"


def test_dedupe_aborts_on_mid_line_entry_start(tmp_path):
    """An entry opened mid-line must abort, not vanish from both counters."""
    src = tmp_path / "refs.bib"
    src.write_text(
        "@article{A,\n  title = {One}\n} @article{B,\n  title = {Two}\n}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.bib"

    result = run_script(DEDUPE, str(src), str(out))

    assert result.returncode != 0
    assert "not at line start" in result.stderr
    assert not out.exists()


# --- #3 formalism_code_reconcile.py: false mismatches ---------------------


def test_subscripted_symbol_reconciles(tmp_path):
    """D_{eff} in LaTeX must match D_eff in code, not appear on both sides."""
    tex = tmp_path / "05_formalism.tex"
    tex.write_text(r"The effective diffusivity is $D_{eff} = \frac{k}{6}$.", encoding="utf-8")
    code = tmp_path / "src"
    code.mkdir()
    (code / "model.py").write_text("D_eff = 1.0\n", encoding="utf-8")

    result = run_script(
        COMMONS / "formalism_code_reconcile.py", str(tex), str(code)
    )

    assert result.returncode == 0, result.stderr
    assert "All symbols reconcile." in result.stdout
    assert "D_" not in result.stdout.replace("D_eff", "")


def test_latex_stdlib_filtered_from_identifier_pass(tmp_path):
    """\\frac and \\sqrt are syntax, not unimplemented physics symbols."""
    tex = tmp_path / "f.tex"
    tex.write_text(r"$\frac{\sqrt{x}}{2}$", encoding="utf-8")
    code = tmp_path / "src"
    code.mkdir()
    (code / "m.py").write_text("x = 1\n", encoding="utf-8")

    result = run_script(COMMONS / "formalism_code_reconcile.py", str(tex), str(code))

    assert "frac" not in result.stdout
    assert "sqrt" not in result.stdout


def test_genuine_missing_symbol_still_reported(tmp_path):
    """The fix must not silence real mismatches — that would rubber-stamp the gate."""
    tex = tmp_path / "f.tex"
    tex.write_text(r"$D_{eff} + Pe_{crit}$", encoding="utf-8")
    code = tmp_path / "src"
    code.mkdir()
    (code / "m.py").write_text("D_eff = 1.0\n", encoding="utf-8")

    result = run_script(COMMONS / "formalism_code_reconcile.py", str(tex), str(code))

    assert "In LaTeX but not found in code" in result.stdout
    assert "Pe_crit" in result.stdout


# --- #6B style_lint.py: honest denominator --------------------------------


def test_ignored_file_excluded_from_denominator(tmp_path):
    (tmp_path / "a.md").write_text("This is innovative work.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text(
        "<!-- style_lint:ignore-file -->\nAlso innovative.\n", encoding="utf-8"
    )

    result = run_script(COMMONS / "style_lint.py", str(tmp_path))

    assert "1 issue(s) across 1 linted file(s)" in result.stderr
    assert "skipped via" in result.stdout


# --- #7 latex_sanity.py: not-checked is not a failure ---------------------


def test_missing_bib_path_is_an_error(tmp_path):
    tex = tmp_path / "p.tex"
    tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")

    result = run_script(
        COMMONS / "latex_sanity.py", str(tex), "--bib", str(tmp_path / "nope.bib")
    )

    assert result.returncode == 2
    assert "not found" in result.stderr


def test_absent_pdflatex_reports_skipped_not_fail(tmp_path, monkeypatch):
    tex = tmp_path / "p.tex"
    tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")

    env_result = subprocess.run(
        [sys.executable, str(COMMONS / "latex_sanity.py"), str(tex), "--compile", "--strict"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/nonexistent", "HOME": str(tmp_path)},
    )

    assert "[compile:skipped]" in env_result.stdout
    assert env_result.returncode == 0, "a missing TeX install must not fail --strict"


def load_latex_sanity():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "latex_sanity", COMMONS / "latex_sanity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("compile_result", "expected_total"),
    [(("ok", "x", 2), 2), (("skipped", "x", 0), 0), (("fail", "x", 0), 1)],
)
def test_compile_state_scoring(compile_result, expected_total):
    """Undefined refs must reach `total`; a skipped compile must not."""
    mod = load_latex_sanity()
    empty_refs = {"undefined": [], "unused": [], "duplicates": [], "n_labels": 0, "n_refs": 0}
    empty_cites = {"orphan": [], "unused_entries": [], "n_cited": 0, "n_bib": 0, "bib_checked": False}
    _, total = mod.format_report(
        Path("p.tex"),
        empty_refs,
        empty_cites,
        {"mismatched": []},
        {"unbalanced_double_dollar": False, "unbalanced_single_dollar": False},
        [],
        compile_result,
    )
    assert total == expected_total


# --- #2 / #8 / #9 hooks ---------------------------------------------------

STATE_FILE = "project: demo\ncurrent_stage: 7\n"


def test_session_start_reads_state_file(tmp_path):
    (tmp_path / "_state.yaml").write_text(STATE_FILE, encoding="utf-8")
    (tmp_path / "artifacts").mkdir()

    out = run_hook("session_start.py", {"cwd": str(tmp_path)})

    assert "current_stage: 7" in out["additionalContext"]
    assert "no research-spark artifacts" not in out.get("additionalContext", "")


def test_session_start_finds_nested_workspace(tmp_path):
    proj = tmp_path / "my-idea"
    proj.mkdir()
    (proj / "_state.yaml").write_text(STATE_FILE, encoding="utf-8")

    out = run_hook("session_start.py", {"cwd": str(tmp_path)})

    assert "current_stage: 7" in out["additionalContext"]


def test_session_start_silent_on_unrelated_repo(tmp_path):
    """Incidental docs/plan.md must not produce a stage claim."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "plan.md").write_text("roadmap\n", encoding="utf-8")
    (docs / "discussion.md").write_text("notes\n", encoding="utf-8")

    out = run_hook("session_start.py", {"cwd": str(tmp_path)})

    assert "additionalContext" not in out


def test_session_start_declines_to_guess_with_multiple_states(tmp_path):
    (tmp_path / "_state.yaml").write_text(STATE_FILE, encoding="utf-8")
    nested = tmp_path / "other-idea"
    nested.mkdir()
    (nested / "_state.yaml").write_text("current_stage: 2\n", encoding="utf-8")

    out = run_hook("session_start.py", {"cwd": str(tmp_path)})

    assert "current_stage: 7" not in out["additionalContext"]
    assert "2 _state.yaml files found" in out["additionalContext"]


def test_session_start_flags_unparseable_state(tmp_path):
    (tmp_path / "_state.yaml").write_text("project: demo\n", encoding="utf-8")

    out = run_hook("session_start.py", {"cwd": str(tmp_path)})

    assert "could not read" in out["additionalContext"]


def test_task_completed_reads_task_from_stdin(tmp_path):
    (tmp_path / "_state.yaml").write_text(STATE_FILE, encoding="utf-8")

    out = run_hook("task_completed.py", {"cwd": str(tmp_path), "task": "Stage 7 plan"})

    assert "Stage 7 plan" in out["additionalContext"]
    logged = (tmp_path / ".research-log.jsonl").read_text(encoding="utf-8")
    assert json.loads(logged)["task"] == "Stage 7 plan"


def test_task_completed_does_not_pollute_non_research_repo(tmp_path):
    out = run_hook("task_completed.py", {"cwd": str(tmp_path), "task": "unrelated"})

    assert not (tmp_path / ".research-log.jsonl").exists()
    assert "No research-spark workspace" in out["additionalContext"]


# --- #4 subagent_stop registration ---------------------------------------


def test_subagent_stop_is_registered():
    hooks = json.loads((HOOKS / "hooks.json").read_text(encoding="utf-8"))
    events = hooks["hooks"]
    assert "SubagentStop" in events
    cmd = events["SubagentStop"][0]["hooks"][0]["command"]
    assert cmd.endswith("subagent_stop.py")


def test_subagent_stop_flags_missing_artifact_on_disk(tmp_path):
    """Real filesystem check, not self-attestation: the stage artifact the
    orchestrator claims to have produced must actually exist on disk."""
    proj = tmp_path / "my-idea"
    proj.mkdir()
    (proj / "_state.yaml").write_text("current_stage: 3\n", encoding="utf-8")
    (proj / "artifacts").mkdir()

    out = run_hook(
        "subagent_stop.py",
        {"subagent_type": "research-spark-orchestrator", "cwd": str(tmp_path)},
    )

    assert "03_claim.md" in out["additionalContext"]
    assert "not found" in out["additionalContext"]


def test_subagent_stop_confirms_artifact_present_on_disk(tmp_path):
    proj = tmp_path / "my-idea"
    (proj / "artifacts").mkdir(parents=True)
    (proj / "_state.yaml").write_text("current_stage: 3\n", encoding="utf-8")
    (proj / "artifacts" / "03_claim.md").write_text("claim\n", encoding="utf-8")

    out = run_hook(
        "subagent_stop.py",
        {"subagent_type": "research-spark-orchestrator", "cwd": str(tmp_path)},
    )

    assert "verified present" in out["additionalContext"]


def test_subagent_stop_silent_for_other_agents():
    assert run_hook("subagent_stop.py", {"subagent_type": "code-reviewer"}) == {}


def test_subagent_stop_silent_for_scientific_review():
    """scientific-review is a skill, not an agent — it never spawns a Task
    subagent, so SubagentStop must never claim to gate it (that branch used
    to be dead code; removed rather than left unreachable)."""
    assert run_hook("subagent_stop.py", {"subagent_type": "scientific-review"}) == {}


# --- post_tool_use: scientific-review deliverable completeness ------------


def test_post_tool_use_is_registered():
    hooks = json.loads((HOOKS / "hooks.json").read_text(encoding="utf-8"))
    events = hooks["hooks"]
    assert "PostToolUse" in events
    cmd = events["PostToolUse"][0]["hooks"][0]["command"]
    assert cmd.endswith("post_tool_use.py")


def test_post_tool_use_flags_incomplete_review(tmp_path):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    review = reviews / "paper.md"
    review.write_text("# Notes\nLooks fine.\n", encoding="utf-8")

    out = run_hook("post_tool_use.py", {"tool_input": {"file_path": str(review)}})

    assert "missing required section" in out["additionalContext"]
    assert "summary" in out["additionalContext"]
    assert "recommendation" in out["additionalContext"]


def test_post_tool_use_silent_for_complete_review(tmp_path):
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    review = reviews / "paper.md"
    review.write_text("# Summary\nGood.\n# Recommendation\nAccept.\n", encoding="utf-8")

    assert run_hook("post_tool_use.py", {"tool_input": {"file_path": str(review)}}) == {}


def test_post_tool_use_silent_for_non_review_writes(tmp_path):
    other = tmp_path / "notes.md"
    other.write_text("# Summary\n# Recommendation\n", encoding="utf-8")

    assert run_hook("post_tool_use.py", {"tool_input": {"file_path": str(other)}}) == {}


def test_subagent_stop_writes_no_debug_log():
    source = (HOOKS / "subagent_stop.py").read_text(encoding="utf-8")
    assert "jsonl" not in source
    assert "expanduser" not in source
