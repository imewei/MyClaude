---
name: git-branch
description: Full branch lifecycle — finish a branch end-to-end (review, commit, push, merge direct or via PR/MR, sync, cleanup), sweep merged/stale branches, roll back to a prior revision, or manage worktrees
argument-hint: "[finish (default)|clean|rollback|worktree] [action-specific options]"
allowed-tools: [Bash, Read, Task, AskUserQuestion]
---

# Git Branch

Routes to `automation-engineer` via `dev-suite:dev-workflows` → `git-workflow`; `finish` invokes `/code-review` (no-PR/MR path) or `/review-pr` (PR/MR path) as its review gate.

One command for a branch's full lifecycle. The first argument selects the action; everything after it is that action's own flags. If the first token starts with `-` or is absent, there is no action word — treat it as `finish` and pass every argument through as `finish`'s own flags (e.g. bare `/git-branch --dry-run` runs `finish --dry-run`).

**Arguments:** $ARGUMENTS

## Actions

| Action | Default? | Purpose |
|--------|----------|---------|
| `finish` | Yes — used when the first argument isn't `clean`, `rollback`, or `worktree` | Review, commit, push, and merge the current (or `--all`) branch, then sync main and clean up |
| `clean` | | Sweep merged/stale local and remote branches, dry-run by default |
| `rollback` | | Interactively reset or revert a branch to a prior commit or tag, dry-run by default |
| `worktree` | | Manage Git worktrees: `add`/`list`/`remove`/`prune`/`migrate` |

---

## Action: Finish

Review the branch, commit what's outstanding, push, and merge it — directly into main if no PR/MR is open for it, or through review, rebase, and a platform merge if one is. Then sync main and clean up the worktree and branches this leaves behind.

### Flags

| Flag | Effect |
|------|--------|
| `--all` | Process every local branch (except main/master), not just the current one |
| `--skip-commit` | Fail if uncommitted changes exist, instead of committing them |
| `--no-delete` | Keep the branch, its remote, and its worktree after merging |
| `--force` | Skip confirmation prompts — never silently overrides a Critical review finding (§3a, §5b.1) or the draft-PR/MR warning (§3) |
| `--dry-run` | Show the plan only, make no changes |
| `--no-review` | Skip the review gate (`/code-review` or `/review-pr`), still rebases and merges |
| `--platform=github\|gitlab` | Force platform instead of auto-detecting from the `origin` remote |

With `--all`, confirmations still happen per branch (the isDraft warning at step 3, the merge-plan confirmation at 5a.3, an ambiguous merge method at 5b.3) unless `--force` is also given — `--all` alone does not imply non-interactive. Combine `--all --force` for a fully unattended sweep.

### Execution

#### 1. Detect Platform and Repository State

Single command:
```bash
git remote get-url origin 2>/dev/null && git rev-parse --abbrev-ref HEAD && git status -s && git fetch --all --prune 2>&1 | tail -5 && git for-each-ref --format='%(refname:short)' refs/heads/ | grep -E '^(main|master)$' | head -1
```

**Platform detection:** `origin` URL containing `github.com` → GitHub (`gh` CLI); containing a GitLab host → GitLab (`glab` CLI); anything else → honor `--platform` if given, otherwise treat every branch as having no PR/MR (direct path only) and note that platform detection was skipped.

**Abort conditions:**
- No main/master branch → "Create main branch first: `git checkout -b main`"
- Detached HEAD → "Checkout a branch first"
- `--skip-commit` with dirty state → "Commit first or remove --skip-commit"
- Default scope, already on main/master → "Nothing to finish — already on the main branch"

#### 2. Determine Scope

- **Default:** the current branch only.
- **`--all`:** every local branch except main/master. Process each branch through steps 3-7 in turn; a failure on one branch (conflict, CI-blocked merge, etc.) is reported and skipped, not fatal to the sweep.

#### 3. Classify Each Branch: PR/MR or Direct

For each branch in scope, if a platform was detected:
```bash
# GitHub
gh pr view <branch> --json number,state,mergeable,isDraft,mergeStateStatus 2>/dev/null
# GitLab
glab mr view <branch> --output json 2>/dev/null
```

- No result, or `state` is not open → **direct path** (step 5a).
- Open result → **PR/MR path** (step 5b).
- `isDraft: true` → warn and ask before proceeding down the PR/MR path, even under `--force`; a draft signals the author doesn't consider it ready.
- **`gh`/`glab` not installed**: the command above fails outright rather than returning "no result" — check for the binary first (`command -v gh` / `command -v glab`); if missing, treat platform detection as skipped (every branch takes the direct path) and say so, same as an undetected platform. Don't let a missing CLI masquerade as "no PR/MR open".
- `glab`'s exact JSON flag varies by version (`--output json` vs `-F json` across releases) — if the command above errors on flag parsing rather than returning no data, retry with `-F json` before falling back to "platform detection skipped".

#### 3a. Review Gate — Direct Path Only (skip if `--no-review`)

The PR/MR path already gets a review at step 5b.1 via `/review-pr`. Branches with no open PR/MR get no review otherwise, since they never pass through a PR — so run `/code-review` here, before anything is committed, pushed, or merged. `/code-review` covers uncommitted local changes directly (no PR needed), so it fits this path where `/review-pr` (PR-only) cannot.

Surface Critical/Important findings. On any Critical finding, stop this branch and ask whether to proceed anyway (default: no) — `--force` does not silently override a Critical finding.

#### 4. Commit Outstanding Work (if dirty and no `--skip-commit`)

```bash
git add -u && git diff --cached --stat
```

Stages only already-tracked modifications. If untracked files exist (`git status --short` shows `??` entries), list them and ask before adding any — even under `--force`, which skips confirmation prompts but must never silently sweep up `.env` files, keys, or build artifacts that were never meant to be committed.

Generate a conventional commit message from the changed files.

#### 5. Push

```bash
git push -u origin <branch>
```

On rejection (non-fast-forward): `git pull --rebase`, retry once; if it still fails, report and skip this branch (don't force-push without going through step 5b's explicit rebase-and-merge flow).

Push runs for the direct path too, not just PR/MR branches: it's a remote backup of the branch before the merge attempt in step 5a, which matters most under `--all` — if a later branch in the sweep hits an unresolvable conflict and the run is aborted, every branch already pushed is recoverable from its remote ref even if its local copy gets left mid-merge.

#### 5a. Direct Path — No PR/MR Open

1. Show the merge plan: branch, commit count ahead of main.
2. `--dry-run`: stop here.
3. Confirm unless `--force`: AskUserQuestion "Merge `<branch>` into main?" with options "Proceed", "Abort", "Show details".
4. Execute:
   ```bash
   git checkout main
   git merge <branch> --no-ff -m "Merge branch '<branch>'"
   ```
   On conflict: ask the user to resolve, skip this branch, or abort the whole run. Track the result.
5. Push main: `git push origin main`.

#### 5b. PR/MR Path — Review, Rebase, Platform Merge

1. **Review gate** (skip if `--no-review`): invoke `/review-pr <PR-or-MR-number>`. Surface Critical/Important findings. On any Critical finding, stop this branch and ask whether to proceed anyway (default: no) — `--force` does not silently override a Critical finding.
2. **Rebase check:**
   ```bash
   git fetch origin main && git rev-list --left-right --count HEAD...origin/main
   ```
   If behind main: `git rebase origin/main`. On conflict, ask the user to resolve or abort — never auto-resolve. If any conflict required a manual, non-mechanical resolution (not just a clean replay), re-run the review gate on the rebased result before merging — the review in step 1 covered the pre-rebase diff, and conflict resolution can change code the review never saw. A clean rebase (no conflicts) doesn't change the diff, so the original review still stands. After a successful rebase, force-push with lease:
   ```bash
   git push --force-with-lease origin <branch>
   ```
3. **Merge via platform CLI.** `--squash` below is the default, not a fixed invariant — prefer the repo's configured default merge method when detectable (branch protection / `mergeStateStatus`); if the method is ambiguous and `--force` was not given, ask before proceeding instead of assuming squash:
   ```bash
   # GitHub
   gh pr merge <number> --squash --delete-branch
   # GitLab
   glab mr merge <number> --squash --remove-source-branch
   ```
4. `--dry-run` for this path: stop after step 1 (review) and report what steps 2-3 would do, without rebasing or merging.

#### 6. Sync Main

```bash
git checkout main
git pull --ff-only origin main
```

#### 7. Clean Up (skip if `--no-delete`)

- **Local branch:** delete each successfully merged branch (`git branch -d`; already handled by `--delete-branch`/`--remove-source-branch` for platform merges, so only needed for the direct path or if the platform flag failed).
- **Local worktree:** if the branch had a worktree (`git worktree list --porcelain`), remove it (see Action: Worktree's `remove`) and run `git worktree prune`.
- **Remote branch:** delete the remote ref if the platform merge didn't already remove it (`git push origin --delete <branch>`).

#### 8. Report

```markdown
## Finish Report

| Branch | Path | Review | Rebase | Merge | Cleanup |
|--------|------|--------|--------|-------|---------|
| feature/x | PR #42 | ✅ no criticals | rebased, 3 commits | squash-merged | branch + worktree + remote removed |
| chore/y | direct | — | — | merged --no-ff | branch removed |

Skipped: <branch> (merge conflict — resolve manually and re-run)

Next steps: none | manual conflict resolution needed on <branch>
```

### Examples

```bash
# Finish the current branch (default action, no action word needed)
/git-branch
/git-branch --dry-run

# Finish a branch with an open PR without a review gate
/git-branch finish --no-review

# Sweep every local branch, no prompts
/git-branch finish --all --force

# Finish, but never delete anything afterward
/git-branch finish --no-delete
```

### Rollback (Finish)

- **Direct-path merge:** `git reset --hard HEAD~1` only undoes the single most recent merge and discards any uncommitted work — after a multi-branch `--all` run, find the commit `main` was at before this run via `git reflog show main` and `git reset --hard <that-hash>` instead.
- **PR/MR-path merge:** already pushed to the remote default branch; a local reset does not undo it. Use `/git-branch rollback --mode revert --target <merge-commit>` (or `git revert -m 1 <merge-commit>` directly for a squash/no-ff merge) and push the revert.
- **Restore a deleted branch:** `git checkout -b <name> <hash>` (find the hash via `git reflog`).
- **Abort a merge still in conflict:** `git merge --abort`.
- **Abort a rebase still in conflict:** `git rebase --abort`.

---

## Action: Clean

Identify and remove branches that are already merged or have gone stale, without touching anything still in flight.

### Flags

| Flag | Effect |
|------|--------|
| `--base <branch>` | Base branch to compare against (default: `main`/`master`, auto-detected) |
| `--stale <days>` | Also flag branches with no commits in the last N days |
| `--remote` | Also clean matching remote-tracking branches |
| `--dry-run` | Preview only, make no changes (**default**) |
| `--yes` | Skip confirmation and delete |
| `--force-unmerged` | Force-delete branches with unmerged commits (`git branch -D`) — deliberately not named `--force`: unlike `finish`'s `--force` (skip prompts only), this one can discard commits |

### Execution

1. **Preflight:** `git fetch --all --prune`; read protected-branch config (below); resolve the base branch.
2. **Identify candidates:**
   - Merged: fully merged into `--base` (`git branch --merged <base>`, excluding `--base` itself).
   - Stale (only if `--stale <days>` given): last commit older than N days.
   - Exclude any branch matching the protected-branch list, regardless of merge/stale status.
3. **Report preview:**
   ```markdown
   ## Branches to delete

   ### Merged
   - feature/old-feature (merged 3 days ago)

   ### Stale
   - experiment/old-test (last commit 90 days ago)
   ```
   Stop here for `--dry-run`.
4. **Confirm:** unless `--yes`, AskUserQuestion "Delete N branches?" with options "Proceed", "Abort", "Show details".
5. **Execute:**
   ```bash
   git branch -d <branch>                    # local
   git push origin --delete <branch>         # remote, if --remote
   git branch -D <branch>                    # unmerged, if --force-unmerged
   ```
   Track and report any deletion failures.

### Protected Branches

```bash
git config --add branch.cleanup.protected develop
git config --add branch.cleanup.protected 'release/*'
git config --get-all branch.cleanup.protected
```

`--base`, `main`, `master`, and `production` are always implicitly protected (matching Rollback's hardcoded protection below — a branch named `production` deserves the same floor of protection whether it's being deleted or rolled back).

### Examples

```bash
/git-branch clean --dry-run
/git-branch clean --stale 90
/git-branch clean --base release/v2.1 --remote --yes
```

---

## Action: Rollback

Safely roll a branch back to a prior version, interactively when arguments are omitted.

### Flags

| Flag | Effect |
|------|--------|
| `--branch <branch>` | Branch to roll back |
| `--target <rev>` | Target commit, tag, or reflog entry |
| `--mode reset\|revert` | Rollback mode |
| `--depth <n>` | Number of recent revisions to list (default 20) |
| `--dry-run` | Preview only, make no changes (**default**) |
| `--yes` | Skip confirmation and execute |

### Execution

1. **Sync remote:** `git fetch --all --prune`.
2. **Select branch:** list local/remote branches, filter protected ones (Action: Clean's config), use `--branch` or ask.
3. **Select target revision:** show the last N revisions (`git log --oneline -<depth>`) and tags reachable from the branch (`git tag --merged`); use `--target` or ask.
4. **Select mode:**

   | Mode | Effect | Push method |
   |------|--------|-------------|
   | `reset` | Hard rollback, rewrites history | `--force-with-lease` |
   | `revert` | Generates inverse commit(s), preserves history | normal push |

   Default to `revert` unless `--mode reset` is explicit.
5. **Final confirmation:** show the exact command sequence. Wait unless `--yes`.
6. **Execute:**
   ```bash
   # reset mode
   git switch <branch>
   git reset --hard <target>

   # revert mode
   git switch <branch>
   git revert --no-edit <target>..HEAD
   ```

### Safety Guardrails

1. **Automatic backup** — the pre-rollback HEAD lands in reflog automatically; note its hash in the report for `git reset --hard <hash>` recovery.
2. **Protected branches** — `main`/`master`/`production` (same set Clean implicitly protects) require explicit extra confirmation, even with `--yes`.
3. **`--dry-run` by default.**
4. **No `--force`** — force-pushing after `reset` mode needs a manual, deliberate command.

### Examples

```bash
/git-branch rollback
/git-branch rollback --branch dev
/git-branch rollback --branch main --target v1.2.0 --mode reset --yes
/git-branch rollback --branch release/v2.1 --target v2.0.5 --mode revert
```

### Notes

- **`reset` vs `revert`**: `reset` rewrites history and needs a force-push; `revert` is the safer default and preserves history.
- **LFS / submodules**: verify consistency before rolling back.
- **CI**: a rollback push may re-trigger pipelines.

---

## Action: Worktree

Manage Git worktrees in a structured directory, with smart defaults and IDE integration.

### Subcommands

| Subcommand | Effect |
|------------|--------|
| `add <path>` | Create a new worktree |
| `list` | List all worktrees |
| `remove <path>` | Remove a worktree |
| `prune` | Clean up stale worktree references |
| `migrate <target>` | Migrate content into a target worktree |

### Flags

| Flag | Effect |
|------|--------|
| `-b <branch>` | Create a new branch for the worktree |
| `-o, --open` | Open in an IDE after creation |
| `--from <source>` | Migration source path |
| `--stash` | Migrate stash entries |
| `--track` | Track a remote branch |
| `--detach` | Detached HEAD |
| `--lock` | Lock the worktree against pruning |

### Directory Layout

```text
parent-directory/
├── your-project/            # main checkout
│   ├── .git/
│   └── src/
└── .worktrees/               # worktree management directory
    └── your-project/
        ├── feature-ui/       # feature branch
        ├── hotfix/           # fix branch
        └── debug/            # scratch worktree
```

### Execution

- **Add:** verify the current directory is a Git repository; compute the path `../.worktrees/<project-name>/<path>`; create it (`git worktree add`, with `-b` if a new branch was requested); copy git-ignored `.env*` files from the source checkout; optionally open in an IDE.
- **List:** `git worktree list`.
- **Remove:** `git worktree remove <path>`.
- **Prune:** `git worktree prune` — clears stale administrative files for worktrees whose directory was deleted outside Git.
- **Migrate:** verify the source has uncommitted content and the target is clean; show the changes about to move; migrate (`git stash` in source → `git stash pop` in target, or `--stash` for stash-entry migration); confirm the result.

### Examples

```bash
/git-branch worktree add feature-ui
/git-branch worktree add feature-ui -o
/git-branch worktree add hotfix -b fix/login -o
/git-branch worktree migrate feature-ui --from main
/git-branch worktree migrate feature-ui --stash
/git-branch worktree list
/git-branch worktree remove feature-ui
/git-branch worktree prune
```

### Smart Defaults

1. **Path-derived branch name** — when no branch is specified, derive it from the path.
2. **IDE detection** — auto-detect VS Code / Cursor / WebStorm.
3. **Env files** — automatically copy `.gitignore`-matched `.env*` files.
4. **Absolute paths** — always resolve to absolute paths to avoid nesting issues.
5. **Branch protection** — verify a branch isn't already checked out elsewhere before reusing it.

### Notes

- Worktrees share one `.git` directory — cheap on disk relative to a full clone.
- `migrate` only moves uncommitted content; use `git cherry-pick` for committed work.
- Cross-platform: Windows, macOS, Linux.

---

## Related

- `/code-review` — the review gate `finish` invokes for the no-PR/MR direct path; run it standalone for a review without merging.
- `/review-pr` — the review gate `finish` invokes for PR/MR branches; run it standalone for a review without merging.
