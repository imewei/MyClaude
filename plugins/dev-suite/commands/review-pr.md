---
description: Comprehensive PR review using specialized agents
argument-hint: "[PR-number-or-URL] [--focus=comments|tests|errors|types|code|simplify]"
---

Run a comprehensive multi-perspective review of a pull request.

## Usage

`/review-pr [PR-number-or-URL] [--focus=comments|tests|errors|types|code|simplify]`

If no PR is specified, review the current branch's PR. If no focus is specified, run the full review stack.

## Steps

1. Identify the PR:
   - use `gh pr view` to get PR details, changed files, and diff
2. Find project guidance:
   - look for `CLAUDE.md`, lint config, TypeScript config, repo conventions
3. Run specialized review agents:
   - `code-reviewer`
   - `comment-analyzer`
   - `pr-test-analyzer`
   - `silent-failure-hunter`
   - `type-design-analyzer`
   - `code-simplifier`
4. Aggregate results:
   - dedupe overlapping findings
   - rank by severity
5. Report findings grouped by severity

## Confidence Rule

Only report issues with confidence >= 80:

- Critical: bugs, security, data loss
- Important: missing tests, quality problems, style violations
- Advisory: suggestions only when explicitly requested

---

## dev-suite Integration

Everything above this line is the unmodified upstream command (`ecc:review-pr`), except one added frontmatter field: `argument-hint` (upstream has none — added for `/help` autocomplete and command discovery, copied verbatim from the `## Usage` line already present in the body). The following is dev-suite-specific wiring, added without altering the copied text above.

- The 6 fan-out agents (`code-reviewer`, `comment-analyzer`, `pr-test-analyzer`, `silent-failure-hunter`, `type-design-analyzer`, `code-simplifier`) intentionally run on `model: sonnet` as lightweight parallel passes — a deliberate design choice, not an unnoticed deviation from CLAUDE.md §7's Opus-for-review routing. Contrast with `quality-specialist` (`model: opus`), dev-suite's general-purpose review/audit agent, used by `/code-review` and `dev-suite:code-review`.
- See also `/code-review` — single-pass, covers local diffs too, can publish the review to GitHub (`gh pr review`, inline comments). This command is multi-agent, PR-only, and report-only (no GitHub publish).
