"""The marketplace and routing validators: clean on this repo, and they catch what they claim."""

import json
import shutil
from pathlib import Path

from tools.validation import marketplace_validator, routing_checker

REPO_ROOT = Path(__file__).parent.parent.parent


def test_repo_marketplace_is_clean():
    assert marketplace_validator.validate(REPO_ROOT) == []


def test_repo_routing_is_clean():
    assert routing_checker.check(REPO_ROOT) == []


def _copy_marketplace(tmp_path: Path) -> Path:
    for rel in (".claude-plugin", "plugins/research-suite/.claude-plugin", "plugins/research-suite/agents", "plugins/research-suite/commands", "plugins/research-suite/skills"):
        shutil.copytree(REPO_ROOT / rel, tmp_path / rel, dirs_exist_ok=True)
    market = json.loads((tmp_path / ".claude-plugin/marketplace.json").read_text())
    market["plugins"] = [p for p in market["plugins"] if p["name"] == "research-suite"]
    (tmp_path / ".claude-plugin/marketplace.json").write_text(json.dumps(market))
    return tmp_path


def test_marketplace_catches_version_and_unlisted_agent(tmp_path: Path):
    root = _copy_marketplace(tmp_path)
    assert marketplace_validator.validate(root) == []
    path = root / ".claude-plugin/marketplace.json"
    market = json.loads(path.read_text())
    entry = market["plugins"][0]
    entry["version"] = "0.0.1"
    entry["agents"] = entry["agents"][:1]
    path.write_text(json.dumps(market))
    errors = " ".join(marketplace_validator.validate(root))
    assert "version" in errors and "on disk but not listed" in errors


def test_routing_catches_hidden_reference_and_bad_yaml(tmp_path: Path):
    skills = tmp_path / "plugins/demo/skills"
    for name, front in {
        "hub": "name: hub\ndescription: Route to sub-skill for details",
        "sub-skill": "name: sub-skill\ndisable-model-invocation: true\ndescription: x",
        "broken": "name: broken\ndescription: Use when: colons break plain scalars",
    }.items():
        (skills / name).mkdir(parents=True)
        (skills / name / "SKILL.md").write_text(f"---\n{front}\n---\nbody\n")
    problems = " ".join(routing_checker.check(tmp_path))
    assert "names 'sub-skill'" in problems
    assert "broken/SKILL.md: frontmatter is not valid YAML" in problems


def test_routing_catches_cross_suite_hidden_target_and_missing_load_note(tmp_path: Path):
    def skill(suite: str, name: str, front: str, body: str = "body") -> None:
        d = tmp_path / "plugins" / suite / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n{front}\n---\n{body}\n")

    skill("dev-suite", "secret-sub", "disable-model-invocation: true\ndescription: x")
    skill("science-suite", "local-sub", "disable-model-invocation: true\ndescription: x")
    skill("science-suite", "hub", "description: hub", "See dev-suite:secret-sub and [x](../local-sub/SKILL.md)")
    problems = " ".join(routing_checker.check(tmp_path))
    assert "dev-suite:secret-sub, a hidden skill in another suite" in problems
    assert "without the 'Loading a routing target' note" in problems


def test_routing_catches_hubs_that_route_to_each_other(tmp_path: Path):
    for name, other in (("hub-a", "hub-b"), ("hub-b", "hub-a")):
        d = tmp_path / "plugins/demo/skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: hub\n---\nSee science-suite:{other}\n")
    (tmp_path / ".claude-plugin").mkdir()
    problems = " ".join(routing_checker.check(tmp_path))
    assert "hubs hub-a and hub-b route to each other" in problems
