---
description: Create high-quality pull requests with comprehensive descriptions, automated
  review, and best practices
triggers:
- /pr-enhance
- create high quality pull requests
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `['--mode=basic|enhanced|enterprise']`
The agent should parse these arguments from the user's request.

# Pull Request Enhancement

Generate high-quality PRs with comprehensive descriptions, automated review checks, and context-aware best practices.

<!-- SYSTEM: Use .agent/skills_index.json for O(1) skill discovery. Do not scan directories. -->

## Context

$ARGUMENTS

---

## Mode Selection

| Mode | Time | Content |
|------|------|---------|
| `--mode=basic` | 5-10 min | Summary, changes, checklist |
| enhanced (default) | 10-20 min | + Automated checks, risk assessment |
| `--mode=enterprise` | 20-40 min | + Coverage, diagrams, split suggestions |

---

## 1. Analyze Changes (Analysis)

```bash
git diff --name-status main...HEAD
git diff --shortstat main...HEAD
git log --oneline main..HEAD
```

**File Categories:**

| Pattern | Category | Icon |
|---------|----------|------|
| `.js`, `.ts`, `.py`, `.go`, `.rs` | source | 🔧 |
| `test`, `spec`, `.test.` | test | ✅ |
| `.json`, `.yml`, `config` | config | ⚙️ |
| `.md`, `README`, `CHANGELOG` | docs | 📝 |
| `.css`, `.scss` | styles | 🎨 |
| `Dockerfile`, `Makefile` | build | 🏗️ |

---

## 2. Generate Content (Parallel Execution)

> **Orchestration Note**: Execute description generation, automated checks, and risk assessment concurrently.

**PR Description Generation:**
- Summary (what/why)
- Impact analysis
- Type of change classification
- Testing approach documentation

**Automated Checks:**
Scan diff for issues:
- Console logs, TODOs, Large functions
- Hardcoded secrets, SQL injection, XSS

**Risk Assessment:**
Calculate risk score (0-10) based on:
- Size, Files, Coverage
- Dependencies, Security impact

---

## 3. Review & Output (Sequential)

**Context-Aware Checklist:**
Generate specific items for Source, Test, Config, and Security files.

**Large PR Detection:**
Flag if >20 files or >1000 lines and suggest splits.

**Final Output Generation:**
Assemble the PR body using the selected template (Basic/Enhanced/Enterprise).

---

## Output Format

**Basic:** Summary, Description, Checklist
**Enhanced:** + Automated findings, Risk assessment, Size recommendations
**Enterprise:** + Coverage report, Architecture diagrams, Response templates

---

**Focus**: Create PRs that are easy to review with all necessary context.
