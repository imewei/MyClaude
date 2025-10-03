---
description: Perform a comprehensive code review of recent changes
allowed-tools: Bash(git diff:*), Bash(git log:*)
argument-hint: [PR-number|commit-range]
color: yellow
agents:
  primary:
    - code-quality-master
  conditional:
    - agent: devops-security-engineer
      trigger: pattern "security|auth|crypto|secret"
    - agent: systems-architect
      trigger: files > 10 OR complexity > 12
  orchestrated: false
---

## Context

- Current git status: !`git status`
- Recent changes: !`git diff HEAD~1`
- Recent commits: !`git log --oneline -5`
- Current branch: !`git branch --show-current`

## Your task

Perform a comprehensive code review focusing on:

1. **Code Quality**: Check for readability, maintainability, and adherence to best practices
2. **Security**: Look for potential vulnerabilities or security issues
3. **Performance**: Identify potential performance bottlenecks
4. **Testing**: Assess test coverage and quality
5. **Documentation**: Check if code is properly documented

Provide specific, actionable feedback with line-by-line comments where appropriate.