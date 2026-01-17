---
description: Complete Git workflow orchestration from code review through PR creation
triggers:
- /git-workflow
- workflow for git workflow
version: 1.0.7
allowed-tools: Bash(git:*), Read, Grep, Task
argument-hint: '[target-branch] [--skip-tests] [--draft-pr] [--no-push] [--squash]
  [--conventional]'
color: cyan
---


# Git Workflow Orchestration

End-to-end git workflow: code review → testing → commit → PR creation.

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Configuration

**Target**: $ARGUMENTS (default: 'main')

**Flags:**
- `--skip-tests`: Skip test execution
- `--draft-pr`: Create draft PR
- `--no-push`: Local-only validation
- `--squash`: Squash before push
- `--conventional`: Enforce Conventional Commits

## Phase 1: Review & Validation (Parallel Execution)

> **Orchestration Note**: Run static analysis and automated tests concurrently.

**Quality Check**: Review uncommitted changes for style, security, performance, error handling

**Test Execution**: Run unit, integration, e2e tests; generate coverage report

**Breaking Changes**: Analyze dependencies, API changes, schema modifications, compatibility

**Gap Analysis**: Identify missing scenarios, edge cases, integration points

## Phase 2: Commit Preparation (Sequential)

### Requirements
- ❌ NO AI/assistant attribution (Claude, GPT, "Generated with", "Co-Authored-By: Claude")
- ❌ NO marketing language ("amazing", "revolutionary", "drastically improves")
- ✅ Human-authored, professional, factual language
- ✅ Specific metrics ("reduces query time by 40%" not "improves performance")

### Steps
1. **Categorize**: Determine type (feat/fix/docs/refactor/perf/test/ci/chore), scope, atomic vs multi-commit
2. **Generate**: Create Conventional Commits format: `<type>(<scope>): <subject>` with body explaining what/why, footer with BREAKING CHANGE if needed

## Phase 3: Pre-Push (Sequential)

1. **Branch Strategy**: Verify branch follows pattern, check conflicts with target
2. **Validation**: CI checks, no secrets, signatures, protection rules, review status

## Phase 4: PR Creation (Sequential)

1. **Description**: Summary (what/why), type checklist, testing, deployment notes, breaking changes, reviewer checklist
2. **Metadata**: Reviewers (CODEOWNERS), labels, linked issues, milestone, merge strategy, auto-merge config

## Success Criteria

- Critical/high issues resolved
- Test coverage ≥80%, all tests pass
- Conventional Commits format
- No merge conflicts
- PR description complete
- Branch protection satisfied
- No critical security issues
- Documentation updated for API changes

## Rollback

1. Revert PR: `git revert <hash>`
2. Feature flag disable (if applicable)
3. Hotfix branch from main
4. Notify team
5. Document in postmortem

## Best Practices

- Commit early/often, atomically
- Branch: `(feature|bugfix|hotfix)/<ticket>-<desc>`
- PRs <400 lines
- Address reviews within 24h
- Squash feature branches, merge release branches
- ≥2 approvals for main
