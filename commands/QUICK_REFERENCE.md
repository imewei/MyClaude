# Slash Commands Quick Reference

> **Fast user guide** for slash commands. For technical details and implementation, see `AGENT_SYSTEM.md`.

---

## 🚀 Most Used Commands

```bash
/commit                    # Smart conventional commit
/quality <path>            # Code quality suite (audit/optimize/refactor)
/fix "error"               # Systematic error resolution
/analyze-codebase          # Comprehensive project analysis
/ultra-think "question"    # Deep multi-agent reasoning
```

---

## 📋 Commands by Category

### 🟢 Workflow & Git
| Command | Purpose |
|---------|---------|
| `/commit [msg]` | Conventional commits with pre-commit checks |
| `/code-review [PR]` | Review changes with multi-agent analysis |
| `/double-check [task]` | Multi-angle validation |

### 🔍 Analysis & Understanding
| Command | Purpose |
|---------|---------|
| `/analyze-codebase [path]` | Full project analysis with context detection |
| `/explain-code <target>` | Detailed code explanation |
| `/ultra-think <question>` | Deep reasoning with meta-analysis |

### ✅ Code Quality (Orchestrated)
| Command | Purpose |
|---------|---------|
| `/quality <path> [flags]` | **Unified quality suite** |
| `/fix <error>` | Systematic debugging |

**Quality Flags:**
- `--audit` → Security audit
- `--optimize` → Performance optimization
- `--refactor` → Code refactoring

### ⚙️ Setup & Configuration
| Command | Purpose |
|---------|---------|
| `/ci-setup [platform]` | CI/CD pipeline automation |
| `/create-hook [type]` | Git hooks with validation |
| `/command-creator` | Generate new slash commands |

### 📝 Maintenance
| Command | Purpose |
|---------|---------|
| `/update-claudemd` | Auto-update CLAUDE.md from git |

---

## 🤖 Agent System Overview

All commands feature **intelligent multi-agent integration**. Agents auto-activate based on context.

**Quick Facts:**
- **12 specialized agents** across 3 tiers (2 orchestrators, 4 core, 6 specialists)
- **Auto-trigger** at relevance score >0.7
- **4 orchestration modes** (single, parallel, sequential, orchestrated)
- **Performance:** <50ms agent selection, <500ms parallel execution

**Trigger Types:**
- Patterns: `numpy|scipy`, `torch|tensorflow`, `jax|flax`
- Files: `*.ipynb`, `Dockerfile`, `package.json`
- Complexity: `files > 50`, `cyclomatic > 15`
- Flags: `--audit`, `--optimize`

**Example:** `/fix "JAX gradient NaN"` auto-triggers `jax-pro` agent (score: 0.95)

👉 **Full technical details:** See `AGENT_SYSTEM.md`

---

## 🎯 Common Workflows

### 1. Scientific Code Quality
```bash
/quality research/ --optimize
# Auto-triggers: code-quality-master, scientific-computing-master, systems-architect

/double-check "numerical accuracy preserved"
/commit "feat: optimize numerical computations"
```

### 2. Bug Fixing with AI
```bash
/fix "TypeError in model training"
# Auto-detects framework → triggers specialist agents

/commit "fix: resolve training type error"
```

### 3. Full-Stack Project Setup
```bash
/analyze-codebase .
# Detects: package.json, React, Express
# Triggers: systems-architect, fullstack-developer, devops-security-engineer

/ci-setup github-actions
/create-hook pre-commit
```

### 4. Research Project Analysis
```bash
/explain-code paper_implementation.py
# Detects: NumPy, research patterns
# Triggers: research-intelligence + scientific-computing

/ultra-think "Is the algorithm implementation correct?"
```

---

## 🏷️ Arguments & Flags

### Argument Types
- **Optional:** `[arg]` - Can omit
- **Required:** `<arg>` - Must provide

### Common Flags
```bash
--audit          # Security audit
--optimize       # Performance optimization
--refactor       # Code refactoring
--trace          # Detailed stack trace
--force          # Override safety checks
--no-verify      # Skip pre-commit hooks
```

---

## ⚠️ Deprecated Commands

Use unified commands instead:

```bash
/audit       → /quality --audit
/optimize    → /quality --optimize
/refactor    → /quality --refactor
```

---

## 💡 Pro Tips

### 1. Let Agents Auto-Detect Context
```bash
/quality research_code/
# Auto-detects NumPy → scientific-computing-master
# Auto-detects complexity → systems-architect
```

### 2. Use Orchestrated Commands for Complex Tasks
```bash
/ultra-think "Analyze performance bottlenecks"
# Orchestrator coordinates multiple specialist agents
```

### 3. Leverage Conditional Agents
```bash
/fix error-in-dockerfile
# Primary: code-quality-master
# Auto-triggered: devops-security-engineer (pattern: docker)
```

---

## 📈 Quick Command Selector

**Need to...** → **Use this command**

- Fix an error → `/fix "error message"`
- Review code quality → `/quality path/`
- Audit security → `/quality path/ --audit`
- Optimize performance → `/quality path/ --optimize`
- Understand code → `/explain-code file.py`
- Analyze project → `/analyze-codebase`
- Deep analysis → `/ultra-think "question"`
- Make commit → `/commit`
- Validate work → `/double-check`
- Setup CI/CD → `/ci-setup`
- Create hooks → `/create-hook`
- Update docs → `/update-claudemd`

---

## 🔗 Documentation Links

- **`AGENT_SYSTEM.md`** - Complete technical reference (agent specs, triggers, implementation)
- **`CHANGELOG.md`** - Historical optimization and integration record
- **Command files** - Individual `.md` files in this directory

---

**Last Updated:** 2025-10-03
**Version:** 4.0 (Consolidated)
**Status:** ✅ Production Ready

For technical documentation, agent specifications, trigger system details, and implementation guidance, see **`AGENT_SYSTEM.md`**.
