# agent-core Retirement + dev-suite Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the `agent-core` plugin, trim `dev-suite`'s command/agent/skill surface to what's actually used and not duplicated by the installed plugin ecosystem, and reposition the 6 surviving agents around scientific-computing specificity.

**Architecture:** This is a plugin-content repo, not an application — this work itself has no new unit tests to write. But the repo DOES have a real 20-file `pytest` suite under `tools/tests/` that asserts structural facts about the plugin content (file existence, registered-command counts, version strings) — 5 of those files hardcode `agent-core` paths/counts or the pre-bump version and must be updated as part of Tasks 1 and 7 respectively, or `uv run pytest` breaks. Beyond that suite, "tests" are the repo's own validation tooling (`make validate`, `context_budget_checker.py`, `xref_validator.py`) plus exact `grep`/`git status` checks with expected output. Every task ends with a concrete, runnable verification command and its expected result.

**Tech Stack:** Markdown (agent/command/skill files), JSON (`plugin.json` manifests), the repo's Python validation tooling under `tools/validation/`.

## Global Constraints

- Dedupe baseline is the actual installed plugin set on this machine (ecc, ruflo-*, superpowers, ccg, pr-review-toolkit, code-modernization, plugin-dev, caveman, ponytail), not a hypothetical minimal install. (Spec §1)
- Only edit `plugins/agent-core/`, `plugins/dev-suite/`, and cross-reference sites outside them (docs, `.claude-plugin/marketplace.json`, science-suite skill files that reference agent-core). Do not touch `plugins/science-suite/` or `plugins/research-suite/` content otherwise — those are separate plans. (Spec §2)
- `uv.lock`'s pre-existing modification (present before this work started) must stay out of every commit in this plan — never `git add uv.lock`. (Spec §8)
- **Task 8 in this plan has a cross-plan dependency**: it must not run until the science-suite plan's Python-consolidation task (which reads content from the 5 directories Task 8 deletes) has been committed. See Task 8's header for the exact check to run first.
- Version bump is repo-wide (all 3 remaining `plugin.json` files, post-`agent-core`-deletion, + `pyproject.toml` share one synced version, enforced by `make validate`), and this is a breaking removal, so it's a **major** bump, not minor. (Spec §8)
- `uv run pytest` is a real, currently-passing 20-file suite, not a no-op — 5 files (`test_agent_core_integrity.py`, `test_cross_suite_invariants.py`, `test_hook_integrity.py`, `test_readme_safeguards.py`, `test_scicomp_redesign.py`) hardcode `agent-core` paths or the pre-bump version string and will fail the moment Task 1/Task 7 land unless updated in the same commit. See Task 1 Step 8 and Task 7 Step 2.

---

### Task 1: Delete agent-core and fix its cross-references

**Files:**
- Delete: `plugins/agent-core/` (entire directory — 3 agents, 6 commands, 18 skill directories, `.claude-plugin/plugin.json`, `README.md`, `output-styles/`)
- Modify: `.claude-plugin/marketplace.json` — remove the `agent-core` entry
- Modify: `plugins/science-suite/skills/research-and-domains/SKILL.md` — remove the dangling `agent-core:reasoning-and-memory` reference
- Modify: `plugins/science-suite/skills/llm-application-dev/SKILL.md`, `plugins/science-suite/skills/llm-and-ai/SKILL.md` — check for and remove/update agent-core references
- Modify: `README.md`, `CHANGELOG.md`, `docs/index.rst`, `docs/integration-map.rst`, `docs/claude-code-spec-compliance.md`, `docs/guides/integration-patterns.rst`, `docs/reference/agents.md`, `docs/agent-teams-guide.md`, `docs/reference/cheatsheet.md`, `docs/categories/core.rst`, `docs/changelog.rst`, `docs/suites/science-suite.rst`, `docs/guides/scientific-workflows.rst`, `docs/reference/commands.md`, `tools/README.md`, `plugins/dev-suite/README.md`, `plugins/science-suite/README.md` — remove/update agent-core mentions
- Modify: `tools/validation/metadata_validator.py` — remove the stale `"thinkfirst"` entry (with its `# agent-core: ...` comment) from `_TIER2_STANDALONE_WHITELIST`
- Modify: `tools/tests/test_agent_core_integrity.py` (delete — entire file targets the deleted plugin), `tools/tests/test_cross_suite_invariants.py`, `tools/tests/test_hook_integrity.py`, `tools/tests/test_readme_safeguards.py`, `tools/tests/test_scicomp_redesign.py` (the `agent-core`-specific parametrize entry only — its hardcoded `3.5.2` version assertion is Task 7's concern, not this task's) — see Step 8

**Interfaces:**
- Produces: `plugins/agent-core/` no longer exists on disk; zero repo-wide matches for the string `agent-core` outside `docs/superpowers/specs/`, `docs/superpowers/plans/` (historical planning docs are allowed to keep the reference), and `CHANGELOG.md`/`docs/changelog.rst` (historical release notes about *past* releases are never rewritten — see Step 6); `uv run pytest` passes with no test referencing `plugins/agent-core` or `agent-core`'s registered-command count.

- [ ] **Step 1: Confirm the exact current cross-reference list**

Run:
```bash
cd /home/wei/Documents/GitHub/MyClaude
git grep -l "agent-core" -- '*.md' '*.json' '*.rst' \
  | grep -v "^plugins/agent-core/" \
  | grep -v "docs/superpowers/specs/" \
  | grep -v "docs/superpowers/plans/" \
  | grep -v "graphify-out/"
```
Note: this uses `git grep` (tracked files only), not a plain recursive `grep`, deliberately — this repo has untracked, gitignored local files (root `CLAUDE.md`, `reports/`) that also happen to contain the string `agent-core` (confirmed via `git ls-files` / `git log -- CLAUDE.md`, both intentionally untracked per commit `94df6fa1`). They are not part of the shipped marketplace content this task edits, are never staged by `git add -A -- ':!uv.lock'` in Step 9's commit, and must NOT be added to this task's file list — a plain `grep -r` (or a shell alias that ignores `.gitignore`) will over-report them. If your shell's `grep` is aliased to something that already respects `.gitignore` (e.g. via `--ignore-files`), plain `grep -r` with the same excludes gives the same 21-file result as `git grep`.

Expected: 21 files — the 17 doc files listed above (README.md, CHANGELOG.md, the 12 docs/* files, tools/README.md, plugins/dev-suite/README.md, plugins/science-suite/README.md), the 3 science-suite skill files, and `.claude-plugin/marketplace.json` (handled separately in Step 3, not part of the doc-sweep in Step 6). If the list differs from this, use the actual current output as the authoritative list for the rest of this task — the repo may have changed since this plan was written.

- [ ] **Step 2: Delete the plugin directory**

```bash
git rm -r plugins/agent-core
```
Expected: `rm 'plugins/agent-core/...'` lines for every file in the directory, no errors.

- [ ] **Step 3: Remove agent-core from the marketplace manifest**

Read `.claude-plugin/marketplace.json`, find the JSON object/entry for `"agent-core"` (matching the pattern used for the other 3 plugin entries in the same file), and delete that entire entry, keeping the JSON valid (correct comma placement on the entry before/after it).

Also update the file's summary metadata, which currently says (confirmed current content):
```
"description": "4-suite marketplace for agent orchestration, scientific research, software development, and scientific computing",
...
"total_plugins": 4,
"note": "v3.5.2: Full marketplace consistency sync across all 4 plugins, documentation, and project metadata. ..."
```
Change `"description"` to a 3-suite description dropping "agent orchestration" (that was agent-core's role), `"total_plugins"` to `3`, and prepend a new note entry for this change (e.g. `"v4.0.0: agent-core plugin retired (see CHANGELOG); marketplace reduced to 3 plugins. "` before the existing note text) rather than rewriting the historical part of the note string.

Verify:
```bash
python3 -c "import json; d = json.load(open('.claude-plugin/marketplace.json')); assert 'agent-core' not in json.dumps(d), 'agent-core still present'; print('OK:', len(d.get('plugins', d)), 'plugins remain')"
grep -n '"total_plugins"\|"description"' .claude-plugin/marketplace.json
```
Expected: `OK: 3 plugins remain` (or however the manifest structures the count — the assertion passing with no `AssertionError` is what matters); `total_plugins` shows `3`; `description` no longer says "4-suite".

- [ ] **Step 4: Fix the dangling `agent-core` references in research-and-domains**

This file has **two** hits, not one — confirmed via `grep -c "agent-core"` = `2` against current content. Read `plugins/science-suite/skills/research-and-domains/SKILL.md` and fix both:

1. The frontmatter `description:` field (line 3) ends with: `...for operational agent self-improvement loops with persistent prompt/policy updates, use agent-core reasoning-and-memory; for general study design, literature synthesis, or paper methodology, use research-suite research-practice.` Delete the clause `for operational agent self-improvement loops with persistent prompt/policy updates, use agent-core reasoning-and-memory; ` (keep the `for general study design...` clause and the sentence's grammar/punctuation intact).
2. The routing-tree line (under the `self-improving-ai` branch):
```
|   (for persistent agent prompt/policy optimization, use agent-core:reasoning-and-memory)
```
Delete this line entirely (it's a parenthetical aside — removing it doesn't break the branch's structure, just drops the now-invalid pointer).

Verify:
```bash
grep -c "agent-core" plugins/science-suite/skills/research-and-domains/SKILL.md
```
Expected: `0`

- [ ] **Step 5: Check and fix the other 2 flagged science-suite skill files**

Run:
```bash
grep -n "agent-core" plugins/science-suite/skills/llm-application-dev/SKILL.md plugins/science-suite/skills/llm-and-ai/SKILL.md
```
For each matching line, read enough surrounding context to determine whether it's a functional cross-reference (names agent-core as a place to route to or delegate to) or an incidental mention. Remove or rephrase functional references the same way as Step 4 (delete the specific line/clause, don't restructure the surrounding routing tree). If a whole routing-tree branch's destination was agent-core with no fallback, redirect it to the nearest still-existing equivalent named in the trim spec's Objective section (`ruflo-core`, `ruflo-swarm`, `superpowers`, or ecc's `agent-*` skills) rather than deleting the branch outright.

Verify:
```bash
grep -c "agent-core" plugins/science-suite/skills/llm-application-dev/SKILL.md plugins/science-suite/skills/llm-and-ai/SKILL.md
```
Expected: `0` for both files.

- [ ] **Step 6: Fix the remaining 14 documentation files**

For each file in the Step 1 list not already handled (README.md, CHANGELOG.md, the docs/*.rst and docs/*.md files, tools/README.md, plugins/dev-suite/README.md, plugins/science-suite/README.md):

1. Run `grep -n "agent-core" <file>` to find the exact lines.
2. Read enough context around each hit to classify it: (a) a table row or list entry naming agent-core as one of the 4 suites — delete that row/entry and adjust any "4 suites" / "4 plugins" count in the same file to 3; (b) a narrative sentence mentioning agent-core's capabilities — remove the agent-core-specific clause, keeping the sentence grammatical; (c) a changelog entry describing agent-core's historical addition — leave changelog entries about *past* releases untouched (history shouldn't be rewritten), only fix forward-looking reference docs (agents.md, commands.md, cheatsheet.md, agent-teams-guide.md, the suite `.rst` files, integration-map.rst, index.rst, claude-code-spec-compliance.md, integration-patterns.rst, scientific-workflows.rst, the two plugin README.md files, tools/README.md).
3. Apply the fix with Edit.

Verify:
```bash
git grep -l "agent-core" -- '*.md' '*.json' '*.rst' \
  | grep -v "^plugins/agent-core/" \
  | grep -v "docs/superpowers/specs/" \
  | grep -v "docs/superpowers/plans/" \
  | grep -v "graphify-out/" \
  | grep -v "CHANGELOG.md" \
  | grep -v "docs/changelog.rst"
```
Expected: empty output (CHANGELOG.md and docs/changelog.rst are intentionally excluded per the historical-record rule in this step).

- [ ] **Step 7: Remove the stale `thinkfirst` whitelist entry from metadata_validator.py**

`thinkfirst` was an agent-core skill (registered hub with no routing tree, hence the whitelist entry). Read `tools/validation/metadata_validator.py`, find `_TIER2_STANDALONE_WHITELIST` (currently at line 572-585), and delete this line from the set literal:
```python
        "thinkfirst",       # agent-core: user-invoked prompt optimizer
```

Verify:
```bash
grep -c "agent-core" tools/validation/metadata_validator.py
```
Expected: `0`

- [ ] **Step 8: Fix the pytest suite for the agent-core deletion**

Confirmed via direct inspection: 5 files under `tools/tests/` hardcode `plugins/agent-core` paths or its registered-command count and will fail `uv run pytest` the moment Step 2's `git rm -r plugins/agent-core` lands. Fix each in this same task/commit (not deferred to Task 7) so no task ever leaves the suite red:

1. **`tools/tests/test_agent_core_integrity.py`** — the entire file (all tests) targets `plugins/agent-core/.claude-plugin/plugin.json`, its `agents/`, and its `skills/`. Delete the file: `git rm tools/tests/test_agent_core_integrity.py`.

2. **`tools/tests/test_cross_suite_invariants.py`** — line 22: `SUITES = ["agent-core", "dev-suite", "science-suite"]` → remove `"agent-core"` from the list. Line 28: `EXPECTED_REGISTERED_COMMANDS` has an `"agent-core": 2` entry → delete that dict entry. (Confirm exact line numbers with `grep -n "agent-core" tools/tests/test_cross_suite_invariants.py` before editing — they may have shifted.)

3. **`tools/tests/test_hook_integrity.py`** — line 39: `SUITES_WITH_HOOKS = ["agent-core", "dev-suite", "science-suite"]` → remove `"agent-core"` from the list.

4. **`tools/tests/test_readme_safeguards.py`** — this file is mostly about `tools.common.readme_sanitizer` (keep that; do not delete the file). Only its `TestTeamAssembleSafeguardsPresent` class is agent-core-specific (doc-drift guard for `plugins/agent-core/commands/team-assemble.md`, which no longer exists after Step 2). Remove:
   - The `TEAM_ASSEMBLE_PATH = REPO_ROOT / "plugins/agent-core/commands/team-assemble.md"` constant (currently line 37).
   - The `# Group 4 — Doc-drift regression guards for team-assemble.md` comment block and the entire `TestTeamAssembleSafeguardsPresent` class (currently lines 406-487 inclusive — verify boundaries with `grep -n "Group 4\|^class Test\|^def test_sanitizing_a_wrapped" tools/tests/test_readme_safeguards.py` since the smoke test at the end of the file must survive).
   - The module docstring's paragraph 2 (the "Doc-drift regression guards" bullet and the "See also" line referencing `team-assemble.md`), keeping paragraph 1 (the sanitizer description).

5. **`tools/tests/test_scicomp_redesign.py`** — only the `agent-core` parametrize entry belongs to this task; its hardcoded `3.5.2` version assertion is Task 7's concern (that constant is correct until Task 7's bump). In `TestManifests.test_version_is_351` (currently lines 177-184), change:
```python
    @pytest.mark.parametrize("suite_dir", [
        AGENT_CORE, DEV_SUITE, RESEARCH, SCIENCE
    ], ids=["agent-core", "dev-suite", "research-suite", "science-suite"])
```
to:
```python
    @pytest.mark.parametrize("suite_dir", [
        DEV_SUITE, RESEARCH, SCIENCE
    ], ids=["dev-suite", "research-suite", "science-suite"])
```
Leave the `AGENT_CORE = PLUGINS / "agent-core"` module-level constant and its use at line 236 (`(AGENT_CORE / "hooks/pre_compact.py")`, inside a different test) alone for now if any other test in the file still references it after this edit — re-check with `grep -n "AGENT_CORE" tools/tests/test_scicomp_redesign.py` and remove the constant only if nothing else uses it.

Verify:
```bash
cd /home/wei/Documents/GitHub/MyClaude
uv run pytest tools/tests/test_cross_suite_invariants.py tools/tests/test_hook_integrity.py tools/tests/test_readme_safeguards.py tools/tests/test_scicomp_redesign.py -v 2>&1 | tail -30
```
Expected: all pass, no `agent-core`/`AGENT_CORE` related failures. (`test_agent_core_integrity.py` no longer exists so it won't appear.)

- [ ] **Step 9: Commit**

```bash
git add -A -- ':!uv.lock'
git status --short
```
Confirm `uv.lock` does NOT appear in the staged list (if it does, run `git restore --staged uv.lock`).

```bash
git commit -m "$(cat <<'EOF'
chore: retire agent-core plugin

Fully covered by ruflo-core, ruflo-swarm, superpowers, and ecc's agent-*
skill family on this install. Removes the plugin directory, its
marketplace.json entry, dangling cross-references from science-suite
skills and top-level docs, the stale metadata_validator.py whitelist
entry, and the 5 pytest files that hardcoded agent-core paths/counts.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Delete the 15 unregistered dev-suite commands

**Files:**
- Delete: `plugins/dev-suite/commands/adopt-code.md`, `c-project.md`, `code-analyze.md`, `code-explain.md`, `deps.md`, `fix-imports.md`, `github-assist.md`, `monitor-setup.md`, `multi-platform.md`, `onboard.md`, `profile-performance.md`, `rust-project.md`, `scaffold.md`, `slo-implement.md`, `tech-debt.md`

**Interfaces:**
- Consumes: none (these files are never referenced in `plugin.json`, confirmed in research prior to this plan — zero-risk deletion).
- Produces: `plugins/dev-suite/commands/` contains exactly the 10 files listed in `plugin.json`'s `commands` array.

- [ ] **Step 1: Confirm these 15 are actually unregistered before deleting**

```bash
cd /home/wei/Documents/GitHub/MyClaude
comm -23 \
  <(ls plugins/dev-suite/commands/*.md | xargs -n1 basename | sort) \
  <(python3 -c "import json; print('\n'.join(sorted(p.split('/')[-1] for p in json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['commands'])))")
```
Expected: exactly these 15 filenames, one per line: `adopt-code.md`, `c-project.md`, `code-analyze.md`, `code-explain.md`, `deps.md`, `fix-imports.md`, `github-assist.md`, `monitor-setup.md`, `multi-platform.md`, `onboard.md`, `profile-performance.md`, `rust-project.md`, `scaffold.md`, `slo-implement.md`, `tech-debt.md`. If the output differs, stop and reconcile against the current `plugin.json` before proceeding — do not delete a file that has since been registered.

- [ ] **Step 2: Delete the 15 files**

```bash
git rm plugins/dev-suite/commands/adopt-code.md plugins/dev-suite/commands/c-project.md \
  plugins/dev-suite/commands/code-analyze.md plugins/dev-suite/commands/code-explain.md \
  plugins/dev-suite/commands/deps.md plugins/dev-suite/commands/fix-imports.md \
  plugins/dev-suite/commands/github-assist.md plugins/dev-suite/commands/monitor-setup.md \
  plugins/dev-suite/commands/multi-platform.md plugins/dev-suite/commands/onboard.md \
  plugins/dev-suite/commands/profile-performance.md plugins/dev-suite/commands/rust-project.md \
  plugins/dev-suite/commands/scaffold.md plugins/dev-suite/commands/slo-implement.md \
  plugins/dev-suite/commands/tech-debt.md
```
Expected: 15 `rm '...'` lines, no errors.

- [ ] **Step 3: Verify the command directory now matches the manifest exactly**

```bash
diff <(ls plugins/dev-suite/commands/*.md | xargs -n1 basename | sort) \
     <(python3 -c "import json; print('\n'.join(sorted(p.split('/')[-1] for p in json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['commands'])))")
```
Expected: no output (empty diff).

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore(dev-suite): delete 15 unregistered commands

Never reachable via plugin.json — each has a direct installed-plugin
equivalent (ecc's per-language build-resolvers, api-connector-builder,
github-ops, dashboard-builder, database-migrations, pm2).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Delete 3 unused dev-suite agents and update the manifest

**Files:**
- Delete: `plugins/dev-suite/agents/debugger-pro.md`, `devops-architect.md`, `systems-engineer.md`
- Modify: `plugins/dev-suite/.claude-plugin/plugin.json` (the `agents` array)
- Modify: `plugins/dev-suite/skills/dev-hub/SKILL.md` — remove the 3 agents from the "Expert Agents" bullet list (full dev-hub rewrite happens in Task 6; this step only removes the 3 dead bullets so the file isn't briefly wrong between commits)
- Modify: `plugins/dev-suite/agents/app-developer.md`, `automation-engineer.md`, `software-architect.md`, `quality-specialist.md`, `sre-expert.md`, `documentation-expert.md` — remove dangling "Delegation Strategy" table rows that route to the 3 deleted agents (confirmed present in all 6 surviving agents by direct grep; Task 4's later rewrite only touches the frontmatter `description` and opening paragraph, not these tables, so this cleanup must happen here)
- Modify: `plugins/dev-suite/hooks/subagent_stop.py` — fix the docstring/comment mentioning `debugger-pro`

**Interfaces:**
- Produces: `plugins/dev-suite/.claude-plugin/plugin.json`'s `agents` array contains exactly 6 entries: `app-developer.md`, `automation-engineer.md`, `documentation-expert.md`, `quality-specialist.md`, `software-architect.md`, `sre-expert.md`; zero matches for `debugger-pro|devops-architect|systems-engineer` anywhere under `plugins/dev-suite/` outside historical CHANGELOG-style content (there is none in this suite).

- [ ] **Step 1: Confirm zero references from the 10 kept commands**

```bash
cd /home/wei/Documents/GitHub/MyClaude/plugins/dev-suite/commands
grep -l -E "debugger-pro|devops-architect|systems-engineer" smart-debug.md double-check.md run-all-tests.md test-generate.md docs.md modernize.md workflow-automate.md fix-commit-errors.md merge-all.md eng-feature-dev.md
```
Expected: no output (no matches in any of the 10 kept commands).

- [ ] **Step 2: Delete the 3 agent files**

```bash
cd /home/wei/Documents/GitHub/MyClaude
git rm plugins/dev-suite/agents/debugger-pro.md plugins/dev-suite/agents/devops-architect.md plugins/dev-suite/agents/systems-engineer.md
```
Expected: 3 `rm '...'` lines.

- [ ] **Step 3: Update plugin.json's agents array**

Read `plugins/dev-suite/.claude-plugin/plugin.json`. Edit the `agents` array from:
```json
  "agents": [
    "./agents/app-developer.md",
    "./agents/automation-engineer.md",
    "./agents/debugger-pro.md",
    "./agents/devops-architect.md",
    "./agents/documentation-expert.md",
    "./agents/quality-specialist.md",
    "./agents/software-architect.md",
    "./agents/sre-expert.md",
    "./agents/systems-engineer.md"
  ],
```
to:
```json
  "agents": [
    "./agents/app-developer.md",
    "./agents/automation-engineer.md",
    "./agents/documentation-expert.md",
    "./agents/quality-specialist.md",
    "./agents/software-architect.md",
    "./agents/sre-expert.md"
  ],
```

Verify:
```bash
python3 -c "import json; a = json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['agents']; assert len(a) == 6, a; assert not any('debugger-pro' in x or 'devops-architect' in x or 'systems-engineer' in x for x in a); print('OK: 6 agents,', a)"
```
Expected: `OK: 6 agents, [...]` listing the 6 kept files.

- [ ] **Step 4: Remove the 3 dead bullets from dev-hub's Expert Agents list**

Read `plugins/dev-suite/skills/dev-hub/SKILL.md`. In the "## Expert Agents" section, delete these 3 lines:
```
- **`debugger-pro`** — Root-cause analysis and systematic bug resolution
- **`devops-architect`** — Infrastructure, Kubernetes, Terraform, and cloud design
- **`systems-engineer`** — Low-level systems, performance, and OS-level concerns
```

Verify:
```bash
grep -c -E "debugger-pro|devops-architect|systems-engineer" plugins/dev-suite/skills/dev-hub/SKILL.md
```
Expected: `0`

- [ ] **Step 5: Fix dangling references to the 3 deleted agents in the surviving agents and the hook**

Confirmed via direct grep: all 6 surviving agents have a "Delegation Strategy" markdown table with a row routing to one or more of the 3 deleted agents, and `plugins/dev-suite/hooks/subagent_stop.py`'s docstring mentions `debugger-pro`. Task 4 does not cover this (it only touches each agent's frontmatter `description` and opening paragraph). Fix each:

- `app-developer.md`: delete the `| debugger-pro | Complex bug resolution and root cause analysis |` and `| systems-engineer | Native modules requiring low-level C/C++ code |` rows.
- `automation-engineer.md`: delete the `| devops-architect | Infrastructure provisioning strategies |` and `| systems-engineer | Build tool (CLI) development |` rows.
- `software-architect.md`: delete the `| systems-engineer | Low-level optimization, kernel/embedded work |` and `| devops-architect | Infrastructure provisioning, Kubernetes, Cloud |` rows.
- `quality-specialist.md`: delete the `| debugger-pro | Root cause analysis of complex bugs |` and `| devops-architect | Infrastructure security and pipeline implementation |` rows.
- `sre-expert.md`: delete the `| devops-architect | Requesting platform-level infrastructure changes from the Platform Owner |` row.
- `documentation-expert.md`: delete the `| devops-architect | Documenting infrastructure and deployment processes |` row.
- `plugins/dev-suite/hooks/subagent_stop.py`: the docstring reads `Collects test/review results when debugger-pro or quality-specialist finish.` — remove the `debugger-pro or ` clause, leaving `Collects test/review results when quality-specialist finishes.` (adjust the verb to match; check the rest of the docstring/file for any other logic keyed on the literal string `debugger-pro` before assuming this is comment-only).

Do not restructure the surrounding Delegation Strategy tables beyond removing these specific rows — other rows (e.g. `software-architect`, `quality-specialist`, `ml-expert (science-suite)`) stay as-is.

Verify:
```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -rc -E "debugger-pro|devops-architect|systems-engineer" plugins/dev-suite/agents/*.md plugins/dev-suite/hooks/subagent_stop.py
```
Expected: `0` for every file listed.

- [ ] **Step 6: Run the plugin metadata validator**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/dev-suite
```
Expected: exits 0, no errors about the `agents` array or missing files.

- [ ] **Step 7: Commit**

```bash
git add plugins/dev-suite/agents/ plugins/dev-suite/.claude-plugin/plugin.json plugins/dev-suite/skills/dev-hub/SKILL.md plugins/dev-suite/hooks/subagent_stop.py
git commit -m "$(cat <<'EOF'
chore(dev-suite): delete 3 agents with zero command references

debugger-pro, devops-architect, systems-engineer aren't wired into any of
the 10 kept commands. Generic ground already covered by
mattpocock-skills:diagnosing-bugs, ecc:build-fix family, and
ecc:homelab-*/ecc:kubernetes-patterns. Also cleans up dangling
Delegation Strategy table rows in the 6 surviving agents and the
subagent_stop.py hook that referenced the 3 deleted agents.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rewrite the 6 surviving agents to a scientific-computing scope

**Files:**
- Modify: `plugins/dev-suite/agents/documentation-expert.md`, `software-architect.md`, `app-developer.md`, `automation-engineer.md`, `quality-specialist.md`, `sre-expert.md` (frontmatter `description:` field and the first paragraph of body text only)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: each agent's `description:` field matches the exact text below (later tasks / other plans that quote these descriptions, if any, should match this text verbatim).

For each of the 6 files, the pattern is identical:

1. Read the file.
2. Locate the YAML frontmatter `description:` field (the line starts with `description:`, may be a single quoted line or a `>-` block — check which form this file uses before editing).
3. Replace its entire value with the exact new description below.
4. Read the first paragraph of the body (the text immediately after the `# <Agent Name>` heading, before any `## Examples` or `## Core Responsibilities` section) and rewrite it to open with the same scientific-computing framing as the new description — one to three sentences, matching the file's existing tone/length. Do not touch anything below that opening paragraph (Examples, tables, checklists stay as-is per the spec: "not a full rewrite of technical content").

**New descriptions (exact text for the frontmatter field):**

- `documentation-expert`: `Docs for numerical/ML/SciML codebases — API specs for JAX/Julia interfaces, Sphinx integration, notebook-to-doc pipelines. For general documentation, defer to ecc:update-docs.`
- `software-architect`: `Numerical/ML/simulation system architecture — JAX pipeline boundaries, SciML module design, data/compute separation for scientific workloads. For general system architecture, defer to ecc's architecture skills.`
- `app-developer`: `Scientific application development — PyQt/PySide6 scientific GUIs, JAX/Julia app integration. For general app development, defer to ecc per-framework reviewers.`
- `automation-engineer`: `Scientific workflow automation — experiment pipelines, Airflow/data-pipeline orchestration for numerical workloads. For general CI/CD automation, defer to ecc:deployment-patterns/ecc:docker-patterns.`
- `quality-specialist`: `Scientific-computing validation — numerical precision, property-based mathematical invariants, reproducibility checks. For general test coverage, defer to ecc:test-coverage.`
- `sre-expert`: `Reliability for long-running scientific workloads — HPC job monitoring, GPU/cluster observability, simulation checkpoint/resume. For general SRE, defer to ruflo-observability:observe.`

(Preserve whatever quoting style — `"..."` vs unquoted vs `>-` block — the file already uses for that field; only the text content changes.)

- [ ] **Step 1: Rewrite documentation-expert.md**

Read `plugins/dev-suite/agents/documentation-expert.md`, apply the `description:` replacement and opening-paragraph rewrite per the pattern above using the `documentation-expert` text.

Verify:
```bash
grep -A1 "^description:" plugins/dev-suite/agents/documentation-expert.md | head -2
```
Expected: shows the new description text (or its first line, if the field wraps).

- [ ] **Step 2: Rewrite software-architect.md**

Same pattern with the `software-architect` text.

Verify: `grep -A1 "^description:" plugins/dev-suite/agents/software-architect.md | head -2`

- [ ] **Step 3: Rewrite app-developer.md**

Same pattern with the `app-developer` text.

Verify: `grep -A1 "^description:" plugins/dev-suite/agents/app-developer.md | head -2`

- [ ] **Step 4: Rewrite automation-engineer.md**

Same pattern with the `automation-engineer` text.

Verify: `grep -A1 "^description:" plugins/dev-suite/agents/automation-engineer.md | head -2`

- [ ] **Step 5: Rewrite quality-specialist.md**

Same pattern with the `quality-specialist` text.

Verify: `grep -A1 "^description:" plugins/dev-suite/agents/quality-specialist.md | head -2`

- [ ] **Step 6: Rewrite sre-expert.md**

Same pattern with the `sre-expert` text.

Verify: `grep -A1 "^description:" plugins/dev-suite/agents/sre-expert.md | head -2`

- [ ] **Step 7: Run the metadata validator**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/dev-suite
```
Expected: exits 0 with no frontmatter errors on the 6 modified files.

- [ ] **Step 8: Commit**

```bash
git add plugins/dev-suite/agents/documentation-expert.md plugins/dev-suite/agents/software-architect.md \
  plugins/dev-suite/agents/app-developer.md plugins/dev-suite/agents/automation-engineer.md \
  plugins/dev-suite/agents/quality-specialist.md plugins/dev-suite/agents/sre-expert.md
git commit -m "$(cat <<'EOF'
refactor(dev-suite): reposition 6 surviving agents to scientific-computing scope

Matches the "for general X use ecosystem-plugin Y, this handles
scientific-computing-specific Z" framing already used by smart-debug,
test-generate, and eng-feature-dev. Description + opening paragraph only;
technical content (Examples, tables, checklists) unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Delete 10 duplicate/stack-specific dev-suite skills

**Files:**
- Delete: `plugins/dev-suite/skills/frontend-and-mobile/`, `frontend-mobile-engineering/`, `graphql-patterns/`, `mobile-testing-patterns/`, `modern-javascript-patterns/`, `nodejs-backend-patterns/`, `typescript-advanced-types/`, `typescript-project-scaffolding/`, `websocket-patterns/`, `plugin-syntax-validator/` (10 directories)
- Modify: `plugins/dev-suite/.claude-plugin/plugin.json` — remove `./skills/frontend-and-mobile` from the `skills` array (the only one of these 10 that's a registered hub)
- Modify: `plugins/dev-suite/agents/app-developer.md` — remove `frontend-and-mobile` from the frontmatter `skills:` list (confirmed present: `skills:` block currently lists `frontend-and-mobile` and `backend-patterns`; no other agent's `skills:` frontmatter references any of the 10 cut skills)

**Interfaces:**
- Produces: `plugins/dev-suite/.claude-plugin/plugin.json`'s `skills` array shrinks from 12 to 11 entries (only `frontend-and-mobile` was registered; the other 9 are unregistered sub-skills reachable only through hub routing, so removing their directories requires no manifest edit); `app-developer.md`'s `skills:` frontmatter no longer lists a deleted skill.

- [ ] **Step 1: Confirm which of the 10 are registered hubs**

```bash
cd /home/wei/Documents/GitHub/MyClaude
python3 -c "
import json
skills = json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['skills']
cut = {'frontend-and-mobile','frontend-mobile-engineering','graphql-patterns','mobile-testing-patterns','modern-javascript-patterns','nodejs-backend-patterns','typescript-advanced-types','typescript-project-scaffolding','websocket-patterns','plugin-syntax-validator'}
registered = {s.split('/')[-1] for s in skills}
print('registered cut hubs:', sorted(cut & registered))
"
```
Expected: `registered cut hubs: ['frontend-and-mobile']` — confirms only this one needs a manifest edit.

- [ ] **Step 2: Delete the 10 skill directories**

```bash
git rm -r plugins/dev-suite/skills/frontend-and-mobile plugins/dev-suite/skills/frontend-mobile-engineering \
  plugins/dev-suite/skills/graphql-patterns plugins/dev-suite/skills/mobile-testing-patterns \
  plugins/dev-suite/skills/modern-javascript-patterns plugins/dev-suite/skills/nodejs-backend-patterns \
  plugins/dev-suite/skills/typescript-advanced-types plugins/dev-suite/skills/typescript-project-scaffolding \
  plugins/dev-suite/skills/websocket-patterns plugins/dev-suite/skills/plugin-syntax-validator
```
Expected: `rm '...'` lines for every file under each of the 10 directories.

- [ ] **Step 3: Remove frontend-and-mobile from plugin.json's skills array**

Read `plugins/dev-suite/.claude-plugin/plugin.json`. Edit the `skills` array, removing the line `"./skills/frontend-and-mobile",`.

Verify:
```bash
python3 -c "
import json
skills = json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['skills']
assert len(skills) == 11, skills
assert 'frontend-and-mobile' not in ' '.join(skills)
print('OK: 11 skills remain')
"
```
Expected: `OK: 11 skills remain`

- [ ] **Step 4: Fix cross-references to the deleted skills from files that survive**

The earlier repo sweep found these skills reference the cut names: `plugins/dev-suite/skills/dev-hub/SKILL.md`, `plugins/dev-suite/skills/backend-patterns/SKILL.md`, and (for `plugin-syntax-validator`) `plugins/dev-suite/skills/testing-and-quality/SKILL.md`. `dev-hub`'s full rewrite happens in Task 6 — skip it here. For `backend-patterns/SKILL.md` and `testing-and-quality/SKILL.md`:

```bash
grep -n -E "frontend-and-mobile|frontend-mobile-engineering|graphql-patterns|mobile-testing-patterns|modern-javascript-patterns|nodejs-backend-patterns|typescript-advanced-types|typescript-project-scaffolding|websocket-patterns|plugin-syntax-validator" \
  plugins/dev-suite/skills/backend-patterns/SKILL.md plugins/dev-suite/skills/testing-and-quality/SKILL.md
```
For each hit, read the surrounding context and remove the specific routing/reference line or table row pointing at the deleted skill (same judgment pattern as Task 1 Step 5 — don't restructure the rest of the file).

Verify:
```bash
grep -c -E "frontend-and-mobile|frontend-mobile-engineering|graphql-patterns|mobile-testing-patterns|modern-javascript-patterns|nodejs-backend-patterns|typescript-advanced-types|typescript-project-scaffolding|websocket-patterns|plugin-syntax-validator" \
  plugins/dev-suite/skills/backend-patterns/SKILL.md plugins/dev-suite/skills/testing-and-quality/SKILL.md
```
Expected: `0` for both files.

- [ ] **Step 5: Remove `frontend-and-mobile` from app-developer.md's `skills:` frontmatter**

`plugins/dev-suite/agents/app-developer.md`'s frontmatter has (confirmed current content):
```yaml
skills:
  - frontend-and-mobile
  - backend-patterns
```
Delete the `- frontend-and-mobile` line, leaving only `- backend-patterns`. (Its `description:` field's own `frontend-and-mobile` mention is separately superseded by Task 4's full description rewrite — if Task 4 hasn't run yet when this step executes, this step only touches the `skills:` list, not `description:`.)

Verify:
```bash
grep -c "frontend-and-mobile" plugins/dev-suite/agents/app-developer.md
```
Expected: `0` once Task 4 has also landed (Task 4's description rewrite drops the other mention); `0` in the `skills:` block regardless of Task 4's status.

- [ ] **Step 6: Run validators**

```bash
cd /home/wei/Documents/GitHub/MyClaude
PYTHONPATH=. python3 tools/validation/metadata_validator.py plugins/dev-suite
PYTHONPATH=. python3 tools/validation/xref_validator.py 2>&1 | tail -30
```
Expected: `metadata_validator.py` exits 0. `xref_validator.py` shows no new dangling references introduced by this task (compare against the baseline count noted in the science-suite spec's validation section — 564 valid references — a drop is expected since references into deleted files are gone; new *dangling* references are the failure signal, not a lower total).

- [ ] **Step 7: Commit**

```bash
git add -A -- plugins/dev-suite/skills/ plugins/dev-suite/.claude-plugin/plugin.json plugins/dev-suite/agents/app-developer.md
git commit -m "$(cat <<'EOF'
chore(dev-suite): delete 10 skills duplicated by ecc/plugin-dev

9 stack-specific skills (frontend/mobile/GraphQL/Node/TypeScript/WebSocket)
fully covered by ecc's per-language reviewers and patterns; this user's
actual stack is Python/JAX + Julia, not general web/mobile.
plugin-syntax-validator duplicates plugin-dev:plugin-validator. Also
drops app-developer.md's now-dangling frontend-and-mobile skills-list
entry.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Rewrite dev-hub to match the trimmed surface

**Files:**
- Modify: `plugins/dev-suite/skills/dev-hub/SKILL.md` (frontmatter `description`, "Expert Agents" list, "Hub Skills" list, routing decision tree, routing table)

**Interfaces:**
- Consumes: the final agent list from Task 3/4 (6 agents) and skill list from Task 5 (11 registered hubs, `frontend-and-mobile` gone).
- Produces: `dev-hub`'s trigger surface (description + routing table) contains no keywords for GraphQL, Node.js/Express/Fastify, generic frontend/mobile, or any of the 3 cut agents.

- [ ] **Step 1: Rewrite the frontmatter description**

Read `plugins/dev-suite/skills/dev-hub/SKILL.md`. Replace the `description:` block:

Old (current):
```yaml
description: >-
  Top-level router for all software development lifecycle topics. Use for: REST APIs/Node.js/Express/Fastify/FastAPI/asyncio/GraphQL/WebSockets/message queues; frontend accessibility/scientific GUIs/PyQt/cross-platform testing; system architecture/microservices/monorepo/containers/cloud/CLI tools/Terraform/K8s; test automation/TDD/E2E/coverage/code review/plugin validation; GitHub Actions/GitLab CI/deployment pipelines/security scanning/CI errors; Prometheus/Grafana/distributed tracing/SLOs/monitoring/observability/incident response; Python packaging/uv/ruff/mypy/performance profiling/error handling/legacy migration; database schema/SQL optimization/caching/search/authentication/secrets management; Git workflow/technical documentation/Airflow data pipelines/systematic debugging; AI pair programming/multi-model dev team/Codex+Gemini review pipeline/content team/team-stop/ai-pair/start dev team/start content team/three-model collaboration/dual-model review/ongoing iterative review.
```

New:
```yaml
description: >-
  Top-level router for scientific-computing software development lifecycle topics. Use for: FastAPI/asyncio scientific services; scientific GUIs (PyQt/PySide6); system architecture for numerical/ML/simulation systems, microservices/containers/cloud/CLI tools/Terraform/K8s; test automation/TDD/E2E/coverage/code review/plugin validation for scientific codebases; GitHub Actions/GitLab CI/deployment pipelines/security scanning/CI errors; Prometheus/Grafana/distributed tracing/SLOs/monitoring/observability/incident response for long-running scientific workloads; database schema/SQL optimization/caching/search/authentication/secrets management; Git workflow/technical documentation for numerical/ML codebases/Airflow data pipelines/systematic debugging for scientific computing; AI pair programming/multi-model dev team/Codex+Gemini review pipeline/content team/team-stop/ai-pair/start dev team/start content team/three-model collaboration/dual-model review/ongoing iterative review.
```

- [ ] **Step 2: Rewrite the Expert Agents list**

Replace the "## Expert Agents" section body:

Old:
```markdown
- **`app-developer`** — Scientific GUI and data-interface implementation
- **`automation-engineer`** — CI/CD pipelines, scripting, and workflow automation
- **`debugger-pro`** — Root-cause analysis and systematic bug resolution
- **`devops-architect`** — Infrastructure, Kubernetes, Terraform, and cloud design
- **`documentation-expert`** — API docs, READMEs, and technical writing
- **`quality-specialist`** — Test strategy, coverage, and code quality gates
- **`software-architect`** — System design, service decomposition, and contracts
- **`sre-expert`** — SLOs, incident response, and reliability engineering
- **`systems-engineer`** — Low-level systems, performance, and OS-level concerns
```

New:
```markdown
- **`app-developer`** — Scientific application development: PyQt/PySide6 GUIs, JAX/Julia app integration
- **`automation-engineer`** — Scientific workflow automation: experiment pipelines, Airflow/data-pipeline orchestration
- **`documentation-expert`** — Docs for numerical/ML/SciML codebases: API specs, Sphinx, notebook-to-doc pipelines
- **`quality-specialist`** — Scientific-computing validation: numerical precision, property-based invariants, reproducibility
- **`software-architect`** — Numerical/ML/simulation system architecture: JAX pipeline boundaries, SciML module design
- **`sre-expert`** — Reliability for long-running scientific workloads: HPC job monitoring, GPU/cluster observability
```

(Note this is already consistent with Task 4's rewritten descriptions — if Task 4 hasn't run yet when this step executes, use the same text anyway since it's the target state either way.)

- [ ] **Step 3: Rewrite the Hub Skills list**

Replace the "## Hub Skills" section body, removing the `frontend-and-mobile` and `python-toolchain` lines (python-toolchain moves to science-suite — its removal from this list is a Task 8 concern once the directory is actually gone, but list it as removed now since Task 8 is guaranteed to land before this suite ships; if Task 8 hasn't run yet, the hub bullet pointing to a not-yet-deleted directory is harmless, just stale for the interim):

Old:
```markdown
- [backend-patterns](../backend-patterns/SKILL.md) — Node.js, async Python, REST/GraphQL/WebSocket, message queues
- [frontend-and-mobile](../frontend-and-mobile/SKILL.md) — React/Vue/Svelte, React Native/Flutter, UI patterns
- [architecture-and-infra](../architecture-and-infra/SKILL.md) — System design, microservices, clean architecture, Terraform/K8s
- [testing-and-quality](../testing-and-quality/SKILL.md) — TDD, test automation, e2e testing, code quality
- [ci-cd-pipelines](../ci-cd-pipelines/SKILL.md) — GitHub Actions, GitLab CI, deployment pipelines
- [observability-and-sre](../observability-and-sre/SKILL.md) — Prometheus, Grafana, SLOs, distributed tracing, incident response
- [python-toolchain](../python-toolchain/SKILL.md) — uv/ruff/mypy, packaging, performance optimization, type hints
- [data-and-security](../data-and-security/SKILL.md) — Databases, SQL optimization, secrets management, auth patterns
- [dev-workflows](../dev-workflows/SKILL.md) — Git workflow, documentation, debugging, Airflow pipelines
- [three-brain](../three-brain/SKILL.md) — Second-opinion review via Codex/Gemini, multimodal analysis, long-context scan (one-shot)
- [ai-pair](../ai-pair/SKILL.md) — AI pair programming patterns
```

New:
```markdown
- [backend-patterns](../backend-patterns/SKILL.md) — Async Python/FastAPI services for scientific workflows, REST, message queues
- [architecture-and-infra](../architecture-and-infra/SKILL.md) — System design, microservices, clean architecture, Terraform/K8s
- [testing-and-quality](../testing-and-quality/SKILL.md) — TDD, test automation, e2e testing, code quality
- [ci-cd-pipelines](../ci-cd-pipelines/SKILL.md) — GitHub Actions, GitLab CI, deployment pipelines
- [observability-and-sre](../observability-and-sre/SKILL.md) — Prometheus, Grafana, SLOs, distributed tracing, incident response
- [data-and-security](../data-and-security/SKILL.md) — Databases, SQL optimization, secrets management, auth patterns
- [dev-workflows](../dev-workflows/SKILL.md) — Git workflow, documentation, debugging, Airflow pipelines
- [three-brain](../three-brain/SKILL.md) — Second-opinion review via Codex/Gemini, multimodal analysis, long-context scan (one-shot)
- [ai-pair](../ai-pair/SKILL.md) — AI pair programming patterns
```

- [ ] **Step 4: Rewrite the routing decision tree**

Remove the two branches:
```
+-- UI components / mobile app / frontend framework / TypeScript project / WCAG / accessibility?
|   --> dev-suite:frontend-and-mobile
|
```
and:
```
+-- Python packaging / uv / typing / profiling / error handling / legacy migration?
|   --> dev-suite:python-toolchain
|
```
from the decision tree (leaving the surrounding branches' `+--`/`|` structure intact — just delete these two blocks).

- [ ] **Step 5: Rewrite the routing table**

Remove these two rows from the "## Routing Table" markdown table:
```
| React, Vue, Svelte, React Native, Flutter, TypeScript, tsconfig, WCAG, ARIA, accessibility, mobile testing | dev-suite:frontend-and-mobile |
```
```
| uv, ruff, mypy, packaging, type hints, pyproject.toml, PyPI, profiling, Cython, try/except, retry, Python migration | dev-suite:python-toolchain |
```
And edit the `backend-patterns` row to drop the now-irrelevant Node.js/GraphQL framing:

Old:
```
| REST, GraphQL, Node.js, FastAPI, queues | dev-suite:backend-patterns |
```
New:
```
| REST APIs, FastAPI, async Python services, message queues | dev-suite:backend-patterns |
```

- [ ] **Step 6: Verify no stale keywords remain**

```bash
cd /home/wei/Documents/GitHub/MyClaude
grep -c -E "GraphQL|Node\.js|Express|Fastify|React|Vue|Svelte|Flutter|WCAG|debugger-pro|devops-architect|systems-engineer|frontend-and-mobile|python-toolchain" plugins/dev-suite/skills/dev-hub/SKILL.md
```
Expected: `0`

- [ ] **Step 7: Run the skill validator**

```bash
PYTHONPATH=. python3 tools/validation/skill_validator.py plugins/dev-suite 2>&1 | grep -A5 "dev-hub"
```
Expected: no frontmatter/description-length errors for `dev-hub`.

- [ ] **Step 8: Commit**

```bash
git add plugins/dev-suite/skills/dev-hub/SKILL.md
git commit -m "$(cat <<'EOF'
refactor(dev-suite): rewrite dev-hub routing to match trimmed surface

Drops GraphQL/Node.js/frontend/mobile/Python-toolchain keywords and the 3
cut agents so dev-hub's trigger surface matches what dev-suite actually
still contains.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Version bump and validation checkpoint

**Note on "final":** this task is a validation *checkpoint*, not the last word — Task 8 is blocked on an external cross-plan dependency and may land well after this task, and Task 8 Step 6 re-runs the exact same full validation suite as its own final gate once its deletions land. Treat this task's validation as "everything through Task 7 is clean," not "the repo's final state is clean."

**Files:**
- Modify: `plugins/dev-suite/.claude-plugin/plugin.json`, `plugins/science-suite/.claude-plugin/plugin.json`, `plugins/research-suite/.claude-plugin/plugin.json`, `pyproject.toml` — bump `version` field
- Modify: `docs/conf.py` (Sphinx `version`/`release` vars, currently `"3.5"`/`"3.5.2"`), `Makefile` (2 occurrences of the literal string `3.5.2`, currently at the top-of-file header comment and in a `make version`-style echo target), `tools/tests/test_scicomp_redesign.py` (the `test_version_is_351` assertion and its docstring, currently hardcoding `3.5.2`) — same version bump, different file formats, easy to miss since they're outside `plugin.json`/`pyproject.toml`

**Interfaces:**
- Consumes: nothing new.
- Produces: all 3 remaining `plugin.json` files, `pyproject.toml`, `docs/conf.py`, `Makefile`, and `test_scicomp_redesign.py`'s assertion all agree on one new version string (major bump from `3.5.2` — e.g. `4.0.0`, confirm the exact next value follows this repo's semver convention by checking `CHANGELOG.md`'s most recent entries before picking the number).

- [ ] **Step 1: Determine the exact next version**

```bash
cd /home/wei/Documents/GitHub/MyClaude
head -20 CHANGELOG.md
grep '"version"' plugins/dev-suite/.claude-plugin/plugin.json
grep '^version' pyproject.toml
```
Note: `pyproject.toml` is TOML, not JSON — its line reads `version = "3.5.2"` (unquoted key, no colon), so `grep '"version"'` never matches it. Use `grep '^version'` for `pyproject.toml` specifically (as above); the two files need different grep patterns.

Confirm current version is `3.5.2` everywhere and that a major bump means `4.0.0` per standard semver (breaking removal of agents/commands/skills = major). If `CHANGELOG.md` shows a different convention in use (e.g. suite-specific major numbers), follow that instead — this step exists to catch drift between this plan (written before implementation) and the actual repo state.

- [ ] **Step 2: Bump all 3 plugin.json files, pyproject.toml, docs/conf.py, Makefile, and test_scicomp_redesign.py**

Read and Edit `plugins/dev-suite/.claude-plugin/plugin.json`, `plugins/science-suite/.claude-plugin/plugin.json`, `plugins/research-suite/.claude-plugin/plugin.json`: change `"version": "3.5.2"` to `"version": "4.0.0"` (or the version confirmed in Step 1) in each. Read and Edit `pyproject.toml`'s `version = "3.5.2"` line the same way.

Also bump the 3 files confirmed to hardcode `3.5.2`/`3.5` outside the manifests:
- `docs/conf.py`: `version = "3.5"` → `version = "4.0"`; `release = "3.5.2"` → `release = "4.0.0"`.
- `Makefile`: 2 occurrences of the literal `3.5.2` (a header comment and a `make`-target echo string) → `4.0.0`.
- `tools/tests/test_scicomp_redesign.py`: the module docstring's `(v3.5.2)` and the `test_version_is_351` assertion `assert plugin["version"] == "3.5.2"` (plus its f-string message) → `4.0.0`. (The test's *name* references the old version number and reads oddly post-bump; renaming it is optional polish, not required for correctness — the assertion body is what pytest checks.)

Note: science-suite and research-suite plugin.json will get further content changes from their own plans — this version bump can land now and those plans should NOT bump the version again, just add their content changes on top of `4.0.0`.

- [ ] **Step 3: Verify version sync**

```bash
grep -h '"version"' plugins/*/.claude-plugin/plugin.json
grep '^version\|^release' pyproject.toml docs/conf.py
grep '3\.5\.2' Makefile tools/tests/test_scicomp_redesign.py
```
Expected: every `plugin.json`/`pyproject.toml`/`docs/conf.py` line shows the same new version string (`4.0.0` or whatever was confirmed in Step 1, `4.0` for `docs/conf.py`'s short `version` var per Sphinx convention); the `Makefile`/`test_scicomp_redesign.py` grep for the old string returns empty.

- [ ] **Step 4: Run the full validation suite**

```bash
cd /home/wei/Documents/GitHub/MyClaude
make validate
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | tail -20
PYTHONPATH=. python3 tools/validation/xref_validator.py 2>&1 | tail -20
uv run pytest 2>&1 | tail -20
```
Expected: `make validate` exits 0 (warnings acceptable, no errors). `context_budget_checker.py` shows a lower total skill count than the pre-trim baseline (223 across all plugins) reflecting Tasks 1/5's deletions — Task 8's 5 further deletions are NOT yet reflected here if Task 8 is still blocked on its cross-plan dependency, so don't treat this count as final (see the note at the top of this task). `xref_validator.py` shows no dangling references. `uv run pytest` passes.

- [ ] **Step 5: Commit**

```bash
git add plugins/dev-suite/.claude-plugin/plugin.json plugins/science-suite/.claude-plugin/plugin.json \
  plugins/research-suite/.claude-plugin/plugin.json pyproject.toml docs/conf.py Makefile \
  tools/tests/test_scicomp_redesign.py
git commit -m "$(cat <<'EOF'
chore: bump version to 4.0.0 for breaking agent-core/dev-suite trim

agent-core removed; dev-suite's command/agent/skill surface reduced. All
3 remaining plugin manifests, pyproject.toml, docs/conf.py, and the
Makefile's version string stay synced per this repo's make validate
drift check; test_scicomp_redesign.py's hardcoded version assertion
updated to match.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Delete the 5 moved Python-tooling skills (cross-plan dependency)

**⚠️ DO NOT START THIS TASK until the science-suite plan's Python-consolidation task is committed.** That task reads content from the 5 directories this task deletes — running this task first destroys the source content before it's been folded into science-suite.

Check before starting:
```bash
cd /home/wei/Documents/GitHub/MyClaude
git log --oneline --all | grep -i "python-tooling\|python consolidation" | head -5
grep -rn "async-python-patterns\|python-toolchain\|python-packaging\|uv-package-manager\|python-performance-optimization" plugins/science-suite/skills/python-development/SKILL.md plugins/science-suite/skills/python-packaging-advanced/SKILL.md 2>/dev/null | head -5
```
If science-suite's `python-development`/`python-packaging-advanced` files show no evidence of newly-folded content (no commit found, no trace of the incoming skills' distinctive content), stop and run the science-suite plan's Python-consolidation task first.

**Files:**
- Delete: `plugins/dev-suite/skills/async-python-patterns/`, `python-packaging/`, `python-performance-optimization/`, `python-toolchain/`, `uv-package-manager/` (5 directories)
- Modify: `plugins/dev-suite/.claude-plugin/plugin.json` — remove `./skills/python-toolchain` from the `skills` array
- Modify: `plugins/dev-suite/skills/backend-patterns/SKILL.md` — fix 3 dangling links/routes to the deleted `async-python-patterns` (confirmed present at lines 21, 45, 69; not covered by Task 6, which only rewrites `dev-hub`)

**Known out-of-scope dangling reference (not fixed by this task):** `plugins/science-suite/commands/benchmark.md` (lines 10, 33) and `plugins/science-suite/agents/simulation-expert.md` (line 76) reference `systems-engineer`/`devops-architect` (dev-suite agents deleted in Task 3, not this task's skills — flagged here since it's adjacent). Confirmed these are bare markdown-table/backtick mentions that `xref_validator.py`'s `_extract_agent_references` regexes (`agent:\s*name`, `@name`, `` agent `name` ``) do NOT match, so `xref_validator.py` will NOT flag them and this task's/Task 7's "zero dangling references" expectation is not contradicted by leaving them. Editing `plugins/science-suite/` is out of scope for this plan (Global Constraints, spec §2) — pass this file+line list to the science-suite expand plan as a known cleanup item.

**Interfaces:**
- Produces: `plugins/dev-suite/.claude-plugin/plugin.json`'s `skills` array shrinks from 11 (after Task 5) to 10 entries; `backend-patterns/SKILL.md` has zero remaining links to `async-python-patterns`.

- [ ] **Step 1: Confirm which of the 5 is the registered hub**

```bash
cd /home/wei/Documents/GitHub/MyClaude
python3 -c "
import json
skills = json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['skills']
cut = {'async-python-patterns','python-packaging','python-performance-optimization','python-toolchain','uv-package-manager'}
registered = {s.split('/')[-1] for s in skills}
print('registered cut hubs:', sorted(cut & registered))
"
```
Expected: `registered cut hubs: ['python-toolchain']`

- [ ] **Step 2: Delete the 5 skill directories**

```bash
git rm -r plugins/dev-suite/skills/async-python-patterns plugins/dev-suite/skills/python-packaging \
  plugins/dev-suite/skills/python-performance-optimization plugins/dev-suite/skills/python-toolchain \
  plugins/dev-suite/skills/uv-package-manager
```
Expected: `rm '...'` lines for all files under each of the 5 directories.

- [ ] **Step 3: Remove python-toolchain from plugin.json's skills array**

Read `plugins/dev-suite/.claude-plugin/plugin.json`. Remove the line `"./skills/python-toolchain",` from the `skills` array.

Verify:
```bash
python3 -c "
import json
skills = json.load(open('plugins/dev-suite/.claude-plugin/plugin.json'))['skills']
assert len(skills) == 10, skills
assert 'python-toolchain' not in ' '.join(skills)
print('OK: 10 skills remain:', skills)
"
```
Expected: `OK: 10 skills remain: [...]`

- [ ] **Step 4: Confirm dev-hub no longer references the deleted directory (should already be clean from Task 6)**

```bash
grep -c -E "async-python-patterns|python-packaging|python-performance-optimization|python-toolchain|uv-package-manager" plugins/dev-suite/skills/dev-hub/SKILL.md
```
Expected: `0`. If non-zero, apply the same fix pattern as Task 6 Steps 3-5 now.

- [ ] **Step 5: Fix backend-patterns' dangling links to async-python-patterns**

Confirmed present, not touched by any earlier task:
```bash
grep -n "async-python-patterns" plugins/dev-suite/skills/backend-patterns/SKILL.md
```
Expected before fix: 3 hits — a `### [Async Python Patterns](../async-python-patterns/SKILL.md)` section heading/link (~line 21), a routing-tree line `--> dev-suite:async-python-patterns` (~line 45), and a routing-table row `| FastAPI, asyncio, aiohttp | dev-suite:async-python-patterns |` (~line 69). Read the file and: delete the `### [Async Python Patterns](...)` subsection heading and its body content (the async-Python content it pointed to no longer exists in dev-suite — it moved to science-suite per spec §6, out of this task's scope to re-target); delete the routing-tree line; delete the routing-table row (or fold the "FastAPI, asyncio, aiohttp" keywords into the FastAPI-related row that survives, if one exists in the same table, rather than dropping FastAPI coverage from `backend-patterns` entirely — check the table's other rows before deciding).

Verify:
```bash
grep -c "async-python-patterns" plugins/dev-suite/skills/backend-patterns/SKILL.md
```
Expected: `0`

- [ ] **Step 6: Full validation pass**

```bash
cd /home/wei/Documents/GitHub/MyClaude
make validate
PYTHONPATH=. python3 tools/validation/context_budget_checker.py 2>&1 | tail -10
PYTHONPATH=. python3 tools/validation/xref_validator.py 2>&1 | tail -10
uv run pytest 2>&1 | tail -20
```
Expected: all pass, zero dangling references, `dev-suite`'s skill directory count on disk now matches `61 - 9 - 1 - 5 = 46`.

```bash
ls -d plugins/dev-suite/skills/*/ | wc -l
```
Expected: `46`

- [ ] **Step 7: Commit**

```bash
git add -A -- plugins/dev-suite/skills/ plugins/dev-suite/.claude-plugin/plugin.json
git commit -m "$(cat <<'EOF'
chore(dev-suite): delete 5 Python-tooling skills absorbed by science-suite

Content already folded into science-suite's python-development and
python-packaging-advanced (see the science-suite expand plan's Python
consolidation task). This is the deletion half of that move. Also fixes
backend-patterns/SKILL.md's now-dangling links to async-python-patterns.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** §3 (agent-core retire) → Task 1 (including the pytest suite and `metadata_validator.py` whitelist cleanup Step 1's "remove all references... from any xref_validator target lists" language implies but didn't originally enumerate). §4 (command surface) → Task 2. §5 (agent surface) → Tasks 3-4 (Task 3 also sweeps the dangling Delegation Strategy table references Task 4's narrower scope can't reach). §6 (skill surface) → Tasks 5, 6, 8 (Task 5 also fixes app-developer.md's frontmatter; Task 8 also fixes backend-patterns.md). §7 (cross-reference check) → folded into Tasks 1, 3, 5, 8 (checked before/after each deletion rather than as one giant end-of-plan task, since each task's deletions create their own dangling-reference risk); one known exception left for the science-suite plan to pick up (Task 8's out-of-scope note) since it isn't reachable via `xref_validator.py`'s matched patterns and isn't in this plan's edit scope. §8 (validation criteria, including "uv run pytest passes unchanged") → Task 1 Step 8 (agent-core test fixes) + Task 7 (version bump touching all version-bearing files including docs/conf.py, Makefile, and the one test file with a hardcoded version) + per-task validator runs. §9 (out of scope) → not actioned, correctly.
- **Placeholder scan:** every deletion lists exact paths; the frontmatter/table/manifest rewrites give exact before/after text. A few steps deliberately use bounded judgment rather than a byte-exact script, because the target text is either open-ended prose or requires reading context that varies per file: Task 1 Step 5 (classify llm-application-dev/llm-and-ai hits as functional vs. incidental), Task 1 Step 6 (classify each of the 14 doc-sweep hits per the 3-category rule), Task 4 Step 4 (each agent's 1-3 sentence opening-paragraph rewrite — the frontmatter `description` itself IS exact), and Task 8 Step 5 (backend-patterns' `async-python-patterns` section removal). Each of these gives an exact verification command (`grep -c ... ` → `0`) so a wrong judgment call is caught immediately even though the prose itself isn't pre-written.
- **Type consistency:** N/A (no code interfaces cross task boundaries in this plan — all interfaces are file/directory existence and `plugin.json` array contents, verified by the exact `python3 -c` assertions in each task).
