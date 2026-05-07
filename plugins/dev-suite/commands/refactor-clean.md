---
name: refactor-clean
description: Retired — use code-simplifier:code-simplifier or the global simplify skill
allowed-tools: [Read, Edit]
---

## Migration

> **Retired.** Superseded by dedicated simplification tools (installed via `claude-plugins-official`).

```bash
# Code simplification / SOLID refactoring
/code-simplifier:code-simplifier

# Quick cleanup of recently changed code
/simplify

# PR-scoped simplification
/pr-review-toolkit:code-simplifier
```
