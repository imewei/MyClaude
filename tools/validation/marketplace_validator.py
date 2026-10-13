#!/usr/bin/env python3
"""
Marketplace Validator

Checks .claude-plugin/marketplace.json against the plugins it lists: required fields,
unique names, source directories, versions (marketplace entry == plugin.json ==
metadata.version), listed agent/command/skill paths that exist, agents and commands on
disk that no entry lists.

Usage:
    python3 tools/validation/marketplace_validator.py
    python3 tools/validation/marketplace_validator.py --root /path/to/repo
"""

import argparse
import json
import sys
from pathlib import Path


def _json(path: Path, errors: list[str]) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _listed(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    market = _json(root / ".claude-plugin" / "marketplace.json", errors)
    if not market:
        return errors or ["marketplace.json is empty"]

    for key in ("name", "owner", "plugins"):
        if key not in market:
            errors.append(f"marketplace.json missing '{key}'")
    meta = market.get("metadata", {})
    entries = market.get("plugins", [])

    names = [e.get("name") for e in entries]
    if len(set(names)) != len(names):
        errors.append(f"duplicate plugin names: {names}")

    for entry in entries:
        name = entry.get("name", "?")
        for key in ("name", "source", "description", "version"):
            if not entry.get(key):
                errors.append(f"{name}: missing '{key}'")
        src = root / str(entry.get("source", ""))
        manifest = src / ".claude-plugin" / "plugin.json"
        if not manifest.is_file():
            errors.append(f"{name}: source {entry.get('source')} has no .claude-plugin/plugin.json")
            continue
        plugin = _json(manifest, errors)
        if plugin.get("name") != name:
            errors.append(f"{name}: plugin.json name is '{plugin.get('name')}'")
        for label, version in (("plugin.json", plugin.get("version")), ("metadata", meta.get("version"))):
            if version != entry.get("version"):
                errors.append(f"{name}: entry version {entry.get('version')} != {label} version {version}")

        for kind in ("agents", "commands"):
            listed = _listed(entry.get(kind))
            for rel in listed:
                if not (src / rel).is_file():
                    errors.append(f"{name}: {kind} entry {rel} does not exist")
            on_disk = {f"./{kind}/{p.name}" for p in (src / kind).glob("*.md")}
            if missing := sorted(on_disk - set(listed)):
                errors.append(f"{name}: {kind} on disk but not listed: {missing}")
            if listed != _listed(plugin.get(kind)):
                errors.append(f"{name}: {kind} list differs between marketplace entry and plugin.json")
        for rel in _listed(entry.get("skills")):
            if not (src / rel / "SKILL.md").is_file():
                errors.append(f"{name}: skill {rel} has no SKILL.md")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate marketplace.json against its plugins")
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    args = parser.parse_args()
    errors = validate(Path(args.root).resolve())
    for err in errors:
        print(f"✗ {err}")
    print(f"Marketplace: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
