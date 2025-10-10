# Command Registry

## Core Commands

### Development Workflow
- **commit** - Smart git commit helper with conventional commits
- **code-review** - Perform comprehensive code review (use `quality` for full suite)
- **double-check** - Validate completeness and multi-angle analysis

### Analysis & Understanding
- **analyze-codebase** - Generate comprehensive codebase documentation
- **explain-code** - Detailed code explanation and documentation
- **ultra-think** - Deep analysis with multi-dimensional reasoning

### Quality & Maintenance
- **quality** 🆕 - Unified code quality suite (audit, optimize, refactor, review)
- **fix** 🆕 - Systematic error debugging and resolution
- **clean-codebase** - Advanced codebase cleanup with AST-based unused import removal
- **run-all-tests** - Iteratively run all tests to fix failures until 100% pass rate
- **update-claudemd** - Update CLAUDE.md with project changes
- **update-docs** 🆕 - Comprehensive Sphinx/README/API documentation update with AST analysis
- **generate-tests** - Generate comprehensive test suites with numerical validation

### Multi-Agent & Workflow
- **reflection** - Advanced reflection engine for AI reasoning and session analysis
- **multi-agent-optimize** - Coordinate multiple specialized agents for optimization tasks
- **fix-commit-errors** - Automatically analyze and fix GitHub Actions failures

### Code Adoption & Integration
- **adopt-code** - Analyze, integrate, and optimize scientific computing codebases

### Setup & Configuration
- **ci-setup** - CI/CD pipeline setup and configuration
- **create-hook** - Create git hooks and automation
- **command-creator** - Generate new slash commands

## Deprecated/Consolidated

The following commands have been consolidated into `quality`:
- ~~audit~~ → `quality --audit`
- ~~optimize~~ → `quality --optimize`
- ~~refactor~~ → `quality --refactor`

The following commands have been merged into `fix`:
- ~~debug-error~~ → `fix`
- ~~fix-issue~~ → `fix`

## Usage Patterns

### Quick Quality Check
```bash
/quality src/ --audit
```

### Deep Analysis
```bash
/ultra-think "How can we improve architecture?"
/analyze-codebase .
/quality . --optimize --refactor
```

### Documentation Update
```bash
/update-docs --full               # Complete documentation overhaul
/update-docs --sphinx             # Update Sphinx docs only
/update-docs --readme             # Update README only
```

### Error Resolution
```bash
/fix "TypeError: Cannot read property 'x' of undefined" --trace
```

### Workflow
```bash
/code-review PR-123
/double-check "implementation complete"
/commit "feat: add feature"
```

## Command Selection Guide

**When to use what:**
- Bug/error? → `/fix`
- Code quality? → `/quality`
- Codebase cleanup? → `/clean-codebase`
- Test failures? → `/run-all-tests` or `/fix`
- Understanding code? → `/explain-code` or `/analyze-codebase`
- Deep thinking needed? → `/ultra-think` or `/reflection`
- Git workflow? → `/commit` or `/fix-commit-errors`
- Setup/config? → `/ci-setup`, `/create-hook`
- Documentation update? → `/update-docs` or `/update-claudemd`
- Generate tests? → `/generate-tests`
- Multi-agent coordination? → `/multi-agent-optimize`
- Adopt external code? → `/adopt-code`

## Tips

1. **Chain commands**: `/quality . && /double-check && /commit`
2. **Use arguments**: Most commands accept paths or descriptions
3. **Check context**: Commands auto-detect project type
4. **Combine modes**: `/quality --audit --optimize` runs multiple checks
