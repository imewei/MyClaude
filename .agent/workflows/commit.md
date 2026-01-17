---
description: Intelligent git commit with automated analysis, quality validation, and
  atomic commit enforcement
triggers:
- /commit
- workflow for commit
version: 1.0.7
allowed-tools: Bash(git:*), Read, Grep
argument-hint: '[commit-message] [--quick] [--split] [--amend] [--no-verify]'
color: green
agents:
  primary:
  - code-reviewer
  conditional:
  - agent: security-auditor
    trigger: files "*.env|secrets|credentials|keys"
  orchestrated: false
---


# Smart Commit Command

Expert Git commit assistant for creating high-quality, atomic commits with conventional commit format.

## Context

$ARGUMENTS

## Critical Requirements

**NEVER include in commit messages:**
- AI/assistant attribution ("Generated with Claude", "Co-Authored-By: Claude", etc.)
- Marketing language ("amazing", "revolutionary", "game-changing")
- Vague terms ("improves things", "enhances experience")

**ALWAYS use:**
- Technical, factual language
- Specific descriptions ("reduces query time by 40%", not "improves performance")
- Professional tone indistinguishable from human-written commits

---

## Workflow

### Phase 1: Context Gathering

Run single optimized command:
```bash
git status --porcelain=v2 && echo "---STATS---" && git diff --cached --numstat && echo "---LOG---" && git log --oneline -5 && echo "---BRANCH---" && git branch --show-current
```

Extract: staged files, change stats (additions/deletions), recent commit style, branch name.

### Phase 2: Type & Scope Detection

**Commit Type** (from file patterns):

| Pattern | Type |
|---------|------|
| `test/`, `*.test.*`, `*.spec.*` | test |
| `docs/`, `*.md`, `README` | docs |
| `.github/`, `Jenkinsfile`, `.circleci/` | ci |
| `Dockerfile`, `Makefile`, `*.gradle` | build |
| `*.css`, `*.scss`, `styles/` | style |
| Branch contains `fix/`, `bug` | fix |
| Branch contains `feat/`, new files | feat |
| Renames, moves, structure changes | refactor |

**Scope**: Extract from common directory of changed files (e.g., `auth`, `api`, `ui`). Match against recent commit scopes for consistency.

### Phase 3: Breaking Change Detection

Check diff for:
- Removed exports: `-export (function|const|class)`
- Changed signatures: function/method parameter changes
- Config key removals in `.env`, `config.*` files
- `DROP TABLE|COLUMN` in migrations

If breaking changes found → require `BREAKING CHANGE:` footer.

### Phase 4: Atomicity Validation

**Atomic commit criteria:**
- Cohesion ≥80% (files in related directories)
- Size ≤300 lines changed
- ≤10 files
- Single logical change

If not atomic, suggest `--split` flag and show file groupings.

### Phase 5: Message Generation

**Format:**
```
type(scope): imperative description (≤50 chars)

Body explaining WHY (wrapped at 72 chars).
Focus on motivation, not what (code shows that).

[BREAKING CHANGE: description if applicable]
[Fixes #123 if from branch name]
```

**Subject rules:**
- Imperative mood ("add", not "added")
- Lowercase after colon
- No period at end
- ≤50 characters

### Phase 6: Quality Scoring

| Criteria | Points |
|----------|--------|
| Conventional format match | 10 |
| Subject ≤50 chars | 10 |
| No trailing period | 5 |
| Lowercase after colon | 5 |
| Imperative mood | 10 |
| Specific (not vague) | 10 |
| Body explains WHY | 10 |
| No AI attribution | 5 |
| No marketing language | 5 |
| Reasonable size (≤300 lines) | 15 |
| Reasonable file count (≤5) | 10 |
| Has staged files | 5 |
| **Total** | **100** |

**Grades:** ≥90 Excellent, ≥80 Good, ≥70 Acceptable, <70 Needs Improvement

---

## Command Flags

| Flag | Behavior |
|------|----------|
| `--quick` | Skip validation, use defaults |
| `--split` | Show split recommendations for large commits |
| `--amend` | Amend last commit (only if unpushed, created this session) |
| `--no-verify` | Skip pre-commit hooks (emergency only) |

---

## Output Format

```
📊 Analysis Complete:
- X files staged (Y additions, Z deletions)
- Branch: feature/name
- Cohesion: XX/100 ✅|⚠️

🤖 Suggested Message (Quality: XX/100):

type(scope): description

Body text here.

[Issues found - if any]

✅ Ready to commit | ⚠️ Issues to address
```

---

## Success Criteria

- ✅ Quality Score ≥70
- ✅ Conventional commit format
- ✅ Atomic (cohesion ≥80, ≤300 lines)
- ✅ Subject ≤50 chars with explanatory body
- ✅ Zero AI attribution
- ✅ Professional, technical language
