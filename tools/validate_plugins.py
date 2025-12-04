#!/usr/bin/env python3
"""
Comprehensive Plugin Validator
Validates plugin structure, syntax, and cross-references
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class PluginValidator:
    def __init__(self, plugins_dir: str, fix: bool = False):
        self.plugins_dir = Path(plugins_dir)
        self.fix = fix
        self.errors = []
        self.warnings = []
        self.fixes_applied = []
        self.stats = {
            'plugins_scanned': 0,
            'files_scanned': 0,
            'agent_refs_checked': 0,
            'skill_refs_checked': 0,
        }

        # Build map of all available agents and skills
        self.available_agents = {}  # {plugin: [agent_names]}
        self.available_skills = {}  # {plugin: [skill_names]}

    def log_error(self, file_path: str, line_num: int, message: str, suggestion: str = ""):
        """Log a validation error"""
        self.errors.append({
            'file': str(file_path),
            'line': line_num,
            'message': message,
            'suggestion': suggestion
        })

    def log_warning(self, file_path: str, line_num: int, message: str, suggestion: str = ""):
        """Log a validation warning"""
        self.warnings.append({
            'file': str(file_path),
            'line': line_num,
            'message': message,
            'suggestion': suggestion
        })

    def log_fix(self, file_path: str, line_num: int, old: str, new: str):
        """Log an applied fix"""
        self.fixes_applied.append({
            'file': str(file_path),
            'line': line_num,
            'old': old,
            'new': new
        })

    def scan_plugin_structure(self):
        """Scan all plugins and build available agents/skills map"""
        print("🔍 Scanning plugin structure...")

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir() or plugin_dir.name.startswith('.'):
                continue

            plugin_name = plugin_dir.name
            self.stats['plugins_scanned'] += 1

            # Scan for agents
            agents_dir = plugin_dir / 'agents'
            if agents_dir.exists():
                agents = []
                for agent_file in agents_dir.glob('*.md'):
                    if agent_file.name != 'README.md':
                        agent_name = agent_file.stem
                        agents.append(agent_name)
                self.available_agents[plugin_name] = agents

            # Scan for skills
            skills_dir = plugin_dir / 'skills'
            if skills_dir.exists():
                skills = []
                for skill_dir in skills_dir.iterdir():
                    if skill_dir.is_dir() and (skill_dir / 'SKILL.md').exists():
                        skill_name = skill_dir.name
                        skills.append(skill_name)
                self.available_skills[plugin_name] = skills

        print(f"   Found {len(self.available_agents)} plugins with agents")
        print(f"   Found {len(self.available_skills)} plugins with skills")

    def validate_plugin_json(self, plugin_dir: Path) -> bool:
        """Validate plugin.json structure"""
        plugin_json = plugin_dir / 'plugin.json'

        if not plugin_json.exists():
            self.log_warning(plugin_json, 0, f"plugin.json not found in {plugin_dir.name}")
            return False

        try:
            with open(plugin_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check required fields
            required_fields = ['name', 'version', 'description']
            for field in required_fields:
                if field not in data:
                    self.log_error(plugin_json, 0, f"Missing required field: {field}")

            # Validate agent listings match actual files
            if 'agents' in data:
                for agent_entry in data['agents']:
                    # Use 'file' field if present (preferred), otherwise try 'id' (kebab-case)
                    if 'file' in agent_entry:
                        agent_file = plugin_dir / agent_entry['file']
                        if not agent_file.exists():
                            self.log_error(plugin_json, 0,
                                          f"Agent '{agent_entry.get('name', 'unknown')}' file not found: {agent_file}",
                                          f"Check 'file' path in plugin.json or create the file")
                    elif 'id' in agent_entry:
                        agent_file = plugin_dir / 'agents' / f"{agent_entry['id']}.md"
                        if not agent_file.exists():
                            self.log_error(plugin_json, 0,
                                          f"Agent '{agent_entry.get('name', agent_entry['id'])}' file not found: {agent_file}",
                                          f"Create {agent_file} or fix 'id' in plugin.json")

            return True

        except json.JSONDecodeError as e:
            self.log_error(plugin_json, e.lineno, f"Invalid JSON: {e.msg}")
            return False
        except Exception as e:
            self.log_error(plugin_json, 0, f"Error reading plugin.json: {e}")
            return False

    def validate_agent_references(self, file_path: Path, content: str) -> List[Tuple[int, str, str]]:
        """
        Validate agent references in a file
        Returns list of (line_number, original, fixed) tuples
        """
        fixes = []
        lines = content.split('\n')

        # Pattern for agent references: plugin::agent or plugin:agent or bare agent
        # Look for: subagent_type="...", Task tool with subagent_type=..., etc.
        patterns = [
            r'subagent_type\s*[=:]\s*["\']([^"\']+)["\']',
            r'@([a-z][a-z0-9-]*)',  # @agent-name mentions
            r'`([a-z][a-z0-9-]*:[a-z][a-z0-9-]*)`',  # `plugin:agent` in backticks
        ]

        for line_num, line in enumerate(lines, 1):
            self.stats['agent_refs_checked'] += 1

            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    ref = match.group(1)

                    # Check for double colon (wrong syntax)
                    if '::' in ref:
                        fixed_ref = ref.replace('::', ':')
                        self.log_error(file_path, line_num,
                                      f"Double colon in agent reference: '{ref}'",
                                      f"Change to: {fixed_ref}")
                        if self.fix:
                            fixes.append((line_num, ref, fixed_ref))
                            self.log_fix(file_path, line_num, ref, fixed_ref)

                    # Check if reference includes plugin namespace
                    if ':' in ref:
                        plugin_name, agent_name = ref.split(':', 1)

                        # Check if plugin exists
                        if plugin_name not in self.available_agents:
                            self.log_error(file_path, line_num,
                                          f"Plugin not found: '{plugin_name}' in reference '{ref}'",
                                          f"Available plugins: {', '.join(sorted(self.available_agents.keys())[:5])}...")

                        # Check if agent exists in that plugin
                        elif agent_name not in self.available_agents.get(plugin_name, []):
                            self.log_error(file_path, line_num,
                                          f"Agent '{agent_name}' not found in plugin '{plugin_name}'",
                                          f"Available agents: {', '.join(self.available_agents[plugin_name])}")

        return fixes

    def apply_fixes(self, file_path: Path, fixes: List[Tuple[int, str, str]]):
        """Apply fixes to a file"""
        if not fixes:
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply all fixes
        for _, old, new in fixes:
            content = content.replace(old, new)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def validate_file(self, file_path: Path):
        """Validate a single file"""
        self.stats['files_scanned'] += 1

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Validate agent references
            fixes = self.validate_agent_references(file_path, content)

            # Apply fixes if --fix flag is set
            if self.fix and fixes:
                self.apply_fixes(file_path, fixes)

        except Exception as e:
            self.log_error(file_path, 0, f"Error reading file: {e}")

    def validate_all_plugins(self):
        """Validate all plugins"""
        print(f"\n🔍 Validating all plugins in {self.plugins_dir}/")
        if self.fix:
            print("🔧 Auto-fix mode enabled\n")

        # First scan structure
        self.scan_plugin_structure()

        # Validate each plugin
        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir() or plugin_dir.name.startswith('.'):
                continue

            # Validate plugin.json
            self.validate_plugin_json(plugin_dir)

            # Validate all markdown files
            for md_file in plugin_dir.rglob('*.md'):
                if md_file.name != 'README.md':  # Skip READMEs
                    self.validate_file(md_file)

    def print_report(self):
        """Print validation report"""
        print("\n" + "="*80)
        print("PLUGIN VALIDATION REPORT")
        print("="*80)

        print(f"\n📊 Statistics:")
        print(f"   Plugins scanned:      {self.stats['plugins_scanned']}")
        print(f"   Files scanned:        {self.stats['files_scanned']}")
        print(f"   Agent refs checked:   {self.stats['agent_refs_checked']}")

        print(f"\n📈 Results:")
        print(f"   🔴 Errors:   {len(self.errors)}")
        print(f"   🟡 Warnings: {len(self.warnings)}")

        if self.fix and self.fixes_applied:
            print(f"   🔧 Fixes applied: {len(self.fixes_applied)}")

        # Print errors
        if self.errors:
            print(f"\n{'─'*80}")
            print("🔴 ERRORS (Must Fix)")
            print(f"{'─'*80}\n")

            for error in self.errors[:20]:  # Limit to first 20
                print(f"  [ERROR] {error['file']}:{error['line']}")
                print(f"  {error['message']}")
                if error['suggestion']:
                    print(f"  💡 Suggestion: {error['suggestion']}")
                print()

            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors\n")

        # Print warnings
        if self.warnings:
            print(f"\n{'─'*80}")
            print("🟡 WARNINGS")
            print(f"{'─'*80}\n")

            for warning in self.warnings[:10]:  # Limit to first 10
                print(f"  [WARNING] {warning['file']}:{warning['line']}")
                print(f"  {warning['message']}")
                if warning['suggestion']:
                    print(f"  💡 Suggestion: {warning['suggestion']}")
                print()

        # Print fixes if applied
        if self.fix and self.fixes_applied:
            print(f"\n{'─'*80}")
            print("✅ FIXES APPLIED")
            print(f"{'─'*80}\n")

            for fix in self.fixes_applied[:20]:  # Limit to first 20
                print(f"  {fix['file']}:{fix['line']}")
                print(f"  - {fix['old']}")
                print(f"  + {fix['new']}")
                print()

        # Summary
        print(f"\n{'─'*80}")
        if self.errors:
            print(f"⚠️  Found {len(self.errors)} error(s) that must be fixed.")
            if not self.fix:
                print("💡 Run with --fix to auto-correct syntax errors")
            return 1
        else:
            print("✅ All validations passed!")
            return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate plugin syntax and structure')
    parser.add_argument('--plugins-dir', default='plugins', help='Plugins directory')
    parser.add_argument('--fix', action='store_true', help='Auto-fix syntax errors')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    validator = PluginValidator(args.plugins_dir, fix=args.fix)
    validator.validate_all_plugins()
    exit_code = validator.print_report()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
