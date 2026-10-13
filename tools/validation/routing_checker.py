#!/usr/bin/env python3
"""
Routing Checker

Descriptions are the router, so this checks what decides triggering, statically:
frontmatter that parses as YAML, descriptions within the 1,536-char listing cap,
sub-skills opted out of the session listing, agents that do not preload skills,
descriptions that send the model to a skill it cannot invoke, and look-alike entries
that do not name each other. It cannot tell whether routing is *good*; measure that
with a prompt eval.

Usage:
    python3 tools/validation/routing_checker.py
    python3 tools/validation/routing_checker.py --root /path/to/repo
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

DESCRIPTION_CAP = 1536
# Names that are also ordinary topic words, not routing instructions.
TOPIC_WORDS = {"bioinformatics", "extreme-value-statistics"}
LOOKALIKES = [
    ("julia-language", "julia-mastery"),
    ("deep-learning", "deep-learning-hub"),
    ("statistical-physics", "statistical-physics-hub"),
    ("machine-learning", "ml-and-data-science"),
    ("python-development", "research-and-domains"),
    ("julia-pro", "julia-implementer"),
    ("julia-ml-hpc", "julia-implementer"),
    ("code-review", "review-pr"),
]


def _frontmatter(path: Path) -> dict | None:
    match = re.match(r"---\n(.*?)\n---", path.read_text(encoding="utf-8"), re.DOTALL)
    try:
        data = yaml.safe_load(match.group(1)) if match else None
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def check(root: Path) -> list[str]:
    plugins = root / "plugins"
    problems: list[str] = []
    visible: dict[str, str] = {}
    skills_visible: set[str] = set()
    hidden: set[str] = set()

    for path in sorted(plugins.glob("*/skills/*/SKILL.md")):
        rel, name = path.relative_to(root).as_posix(), path.parent.name
        data = _frontmatter(path)
        if data is None:
            problems.append(f"{rel}: frontmatter is not valid YAML")
            continue
        if data.get("disable-model-invocation") is True:
            hidden.add(name)
        else:
            visible[name] = str(data.get("description", ""))
            skills_visible.add(name)
    for kind in ("agents", "commands"):
        for path in sorted(plugins.glob(f"*/{kind}/*.md")):
            rel = path.relative_to(root).as_posix()
            data = _frontmatter(path)
            if data is None:
                problems.append(f"{rel}: frontmatter is not valid YAML")
                continue
            if kind == "agents" and data.get("skills"):
                problems.append(f"{rel}: agent preloads skills (full text injected per dispatch)")
            if data.get("disable-model-invocation") is not True:
                visible[path.stem] = str(data.get("description", ""))

    for name, desc in visible.items():
        if len(desc) > DESCRIPTION_CAP:
            problems.append(f"{name}: description is {len(desc)} chars (cap {DESCRIPTION_CAP})")
        for h in sorted(hidden - TOPIC_WORDS):
            if re.search(rf"(?<![\w/-]){re.escape(h)}(?![\w-])", desc):
                problems.append(f"{name}: description names '{h}', which the model cannot invoke")

    hubs = {
        Path(ref).name
        for manifest in plugins.glob("*/.claude-plugin/plugin.json")
        for ref in _plugin_skills(manifest)
    }
    for name in sorted(skills_visible - hubs):
        problems.append(f"skill {name}: model-invocable but not a registered hub (listed every session)")

    problems += _handoff_problems(root)
    problems += _hub_cycle_problems(root, skills_visible)

    for left, right in LOOKALIKES:
        for a, b in ((left, right), (right, left)):
            if a in visible and not re.search(rf"(?<![\w-]){re.escape(b)}(?![\w-])", visible[a]):
                problems.append(f"{a}: description should name its look-alike {b}")
    return problems


def _handoff_problems(root: Path) -> list[str]:
    """Hand-offs between entries. A sibling suite's hidden skill cannot be read by path (installs
    sit in separate directories), and a skill that routes to hidden skills must say how to load them."""
    plugins = root / "plugins"
    hidden: dict[str, set[str]] = {}
    for path in plugins.glob("*/skills/*/SKILL.md"):
        if (_frontmatter(path) or {}).get("disable-model-invocation") is True:
            hidden.setdefault(path.parts[-4], set()).add(path.parent.name)
    problems: list[str] = []
    for path in sorted([*plugins.glob("*/skills/*/SKILL.md"), *plugins.glob("*/agents/*.md"), *plugins.glob("*/commands/*.md")]):
        suite = path.relative_to(plugins).parts[0]
        rel = path.relative_to(root).as_posix()
        body = path.read_text(encoding="utf-8")
        for other, name in re.findall(r"\b(dev-suite|research-suite|science-suite):([a-z0-9][a-z0-9-]+)", body):
            if other != suite and name in hidden.get(other, set()):
                problems.append(f"{rel}: routes to {other}:{name}, a hidden skill in another suite (route to that suite's hub)")
        if path.name == "SKILL.md" and path.parent.name not in hidden.get(suite, set()):
            links = set(re.findall(r"\]\(\.\./([a-z0-9_-]+)/SKILL\.md\)", body))
            if links & hidden.get(suite, set()) and "Loading a routing target" not in body:
                problems.append(f"{rel}: routes to hidden skills without the 'Loading a routing target' note")
    return problems


def _hub_cycle_problems(root: Path, hubs: set[str]) -> list[str]:
    """Two hubs that route to each other can bounce a request between them. The top-level routers
    (`*-hub`) are excluded: every hub may say "if ambiguous, use the router", and the router lists every hub."""
    plugins = root / "plugins"
    routers = {h for h in hubs if h in {"dev-hub", "science-hub", "research-hub"}}
    edges: dict[str, set[str]] = {}
    for h in hubs - routers:
        for path in plugins.glob(f"*/skills/{h}/SKILL.md"):
            body = path.read_text(encoding="utf-8")
            targets = set(re.findall(r"\]\(\.\./([a-z0-9_-]+)/SKILL\.md\)", body))
            targets |= {n for _, n in re.findall(r"\b(dev-suite|research-suite|science-suite):([a-z0-9][a-z0-9-]+)", body)}
            edges[h] = (targets & hubs) - routers - {h}
    return [
        f"hubs {a} and {b} route to each other"
        for a in sorted(edges)
        for b in sorted(edges[a])
        if a < b and a in edges.get(b, set())
    ]


def _plugin_skills(manifest: Path) -> list[str]:
    import json

    try:
        refs = json.loads(manifest.read_text(encoding="utf-8")).get("skills", [])
    except (OSError, json.JSONDecodeError):
        return []
    return [r for r in refs if isinstance(r, str)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Static checks on skill/agent/command routing")
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    args = parser.parse_args()
    problems = check(Path(args.root).resolve())
    for p in problems:
        print(f"✗ {p}")
    print(f"Routing: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
