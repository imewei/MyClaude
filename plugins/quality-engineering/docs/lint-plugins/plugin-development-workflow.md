# Plugin Development Workflow

Best practices for Claude Code plugin development with automated validation, CI/CD integration, and pre-commit hooks.

## Overview

This guide covers the complete development workflow for creating, validating, and maintaining Claude Code plugins with quality automation at every stage.

---

## 1. Development Environment Setup

### Prerequisites

```bash
# Required tools
- Python 3.12+
- Git
- Code editor (VS Code, Vim, etc.)

# Optional but recommended
- pre-commit framework
- GitHub CLI (gh)
```

### Clone Plugin Repository

```bash
# Clone plugins repository
git clone https://github.com/your-org/claude-plugins.git
cd claude-plugins

# Install development dependencies
uv uv pip install pre-commit pyyaml jsonschema

# Initialize pre-commit hooks
pre-commit install
```

---

## 2. Plugin Creation

### Plugin Structure

```
plugins/my-new-plugin/
├── plugin.json           # Plugin metadata (required)
├── README.md             # Plugin documentation
├── CHANGELOG.md          # Version history
├── agents/               # Agent definitions
│   ├── primary-agent.md
│   └── specialist-agent.md
├── commands/             # Slash commands
│   ├── main-command.md
│   └── helper-command.md
├── skills/               # Reusable skills
│   └── my-skill/
│       └── SKILL.md
└── docs/                 # External documentation
    └── guides/
        └── advanced-usage.md
```

### Create plugin.json

```json
{
  "name": "my-new-plugin",
  "version": "1.0.0",
  "description": "Comprehensive plugin for X, Y, and Z workflows",
  "agents": [
    {
      "name": "primary-agent",
      "description": "Main agent for core workflows",
      "status": "active",
      "tags": ["core", "automation"]
    }
  ],
  "commands": [
    {
      "name": "/my-command",
      "description": "Execute primary workflow"
    }
  ],
  "keywords": [
    "automation",
    "workflow",
    "productivity"
  ],
  "dependencies": {
    "comprehensive-review": ">=1.0.0"
  }
}
```

### Create Agent File

```markdown
<!-- agents/primary-agent.md -->
# Primary Agent

Expert in X, Y, and Z with deep knowledge of best practices and patterns.

## Expertise

- Domain expertise in X
- Design patterns for Y
- Optimization strategies for Z

## Responsibilities

1. Analyze requirements and constraints
2. Design solutions following best practices
3. Implement with modern tooling
4. Validate quality and performance

## Approach

### Phase 1: Analysis
- Review existing code
- Identify patterns and anti-patterns
- Document findings

### Phase 2: Design
- Propose solution architecture
- Consider alternatives
- Evaluate trade-offs

### Phase 3: Implementation
- Write clean, maintainable code
- Follow project conventions
- Add comprehensive tests

### Phase 4: Validation
- Run automated tests
- Perform code review
- Document changes

## Output Format

Deliver:
1. Solution design document
2. Implementation code
3. Tests with >80% coverage
4. Documentation updates
```

### Create Command File

```markdown
<!-- commands/my-command.md -->
---
description: Execute primary workflow with X, Y, and Z
argument-hint: [target-path] [--option]
color: blue
agents:
  primary:
    - my-new-plugin:primary-agent
  conditional:
    - agent: comprehensive-review:code-reviewer
      trigger: pattern "review|quality"
---

# My Command

Execute the primary workflow with comprehensive automation.

## Arguments

```bash
# Basic usage
/my-command target/path

# With options
/my-command target/path --option value

# Help
/my-command --help
```

## Workflow

### Phase 1: Preparation
1. Validate inputs
2. Load configuration
3. Initialize environment

### Phase 2: Execution
1. Execute primary logic
2. Handle errors gracefully
3. Log progress

### Phase 3: Validation
1. Verify outputs
2. Run tests
3. Generate report

## Output

Provides:
- Execution summary
- Results and metrics
- Next steps and recommendations
```

---

## 3. Validation During Development

### Manual Validation

```bash
# Validate syntax
/lint-plugins --plugin=my-new-plugin

# Auto-fix issues
/lint-plugins --plugin=my-new-plugin --fix

# Generate detailed report
/lint-plugins --plugin=my-new-plugin --report
```

### Validation Script Usage

```bash
# Run validation script directly
python .agent/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins \
  --plugin my-new-plugin \
  --verbose

# Check for specific issues
python validate_plugin_syntax.py \
  --plugins-dir plugins \
  --check-cross-references \
  --check-metadata
```

### Common Issues and Fixes

#### Issue 1: Agent Reference Format

**Error**:
```
[SYNTAX_001] Double colon in agent reference
commands/my-command.md:10: 'comprehensive-review::code-reviewer'
```

**Fix**:
```bash
# Auto-fix
/lint-plugins --plugin=my-new-plugin --fix

# Manual fix (if needed)
sed -i 's/::/:/g' commands/my-command.md
```

#### Issue 2: Missing Namespace

**Error**:
```
[SYNTAX_002] Bare agent name without namespace
commands/my-command.md:15: 'code-reviewer'
```

**Fix**:
```markdown
# Before
agents:
  - code-reviewer

# After
agents:
  - comprehensive-review:code-reviewer
```

#### Issue 3: Non-Existent Agent

**Error**:
```
[REFERENCE_001] Agent file not found
Expected: plugins/my-new-plugin/agents/specialist-agent.md
```

**Fix**:
```bash
# Create missing agent file
touch plugins/my-new-plugin/agents/specialist-agent.md

# Or remove from plugin.json
```

---

## 4. Pre-Commit Hook Setup

### Install pre-commit Framework

```bash
# Install pre-commit
uv uv pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: lint-plugins
        name: Validate Plugin Syntax
        entry: python .agent/scripts/validate_plugin_syntax.py
        language: system
        pass_filenames: false
        files: '^plugins/.*\.(md|json)$'
        stages: [commit]

      - id: validate-json
        name: Validate JSON Files
        entry: python -m json.tool
        language: system
        files: '^plugins/.*/plugin\.json$'

      - id: trailing-whitespace
        name: Remove Trailing Whitespace
        entry: trailing-whitespace-fixer
        language: system
        files: '\.(md|json)$'
EOF

# Install hooks
pre-commit install
```

### Pre-Commit Hook Behavior

```bash
# On commit, hooks run automatically
git add plugins/my-new-plugin/
git commit -m "Add new plugin"

# Output:
# Validate Plugin Syntax.....................................Passed
# Validate JSON Files........................................Passed
# Remove Trailing Whitespace.................................Passed
# [main 1234567] Add new plugin
```

### Skip Hooks (Emergency Only)

```bash
# Skip all hooks (not recommended)
git commit --no-verify -m "Emergency fix"

# Skip specific hook
SKIP=lint-plugins git commit -m "WIP: Draft changes"
```

---

## 5. CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/lint-plugins.yml`:

```yaml
name: Lint Plugins

on:
  push:
    branches: [main, develop]
    paths:
      - 'plugins/**/*.md'
      - 'plugins/**/plugin.json'
  pull_request:
    branches: [main, develop]
    paths:
      - 'plugins/**/*.md'
      - 'plugins/**/plugin.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          uv uv pip install pyyaml jsonschema

      - name: Validate plugin syntax
        run: |
          python .agent/scripts/validate_plugin_syntax.py \
            --plugins-dir plugins \
            --verbose

      - name: Generate validation report
        if: failure()
        run: |
          python .agent/scripts/validate_plugin_syntax.py \
            --plugins-dir plugins \
            --report \
            --output validation-report.md

      - name: Upload report
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: validation-report.md

      - name: Comment on PR
        if: failure() && github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('validation-report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Plugin Validation Failed\n\n${report}`
            });
```

### GitLab CI Configuration

Create `.gitlab-ci.yml`:

```yaml
stages:
  - validate

validate-plugins:
  stage: validate
  image: python:3.12
  before_script:
    - uv uv pip install pyyaml jsonschema
  script:
    - python .agent/scripts/validate_plugin_syntax.py
        --plugins-dir plugins
        --verbose
  only:
    changes:
      - plugins/**/*.md
      - plugins/**/plugin.json
  artifacts:
    when: on_failure
    paths:
      - validation-report.md
    expire_in: 1 week
```

---

## 6. Development Best Practices

### Plugin Versioning

Follow semantic versioning (semver):

```
MAJOR.MINOR.PATCH

1.0.0 → Initial release
1.0.1 → Bug fix (backward compatible)
1.1.0 → New feature (backward compatible)
1.0.2 → Breaking change
```

**Update CHANGELOG.md**:
```markdown
# Changelog

## [1.1.0] - 2025-01-15

### Added
- New command `/my-command` for X workflow
- Agent `specialist-agent` for Y operations

### Changed
- Improved performance of `primary-agent`
- Updated documentation with examples

### Fixed
- Agent reference syntax in commands

## [1.0.0] - 2025-01-01

### Added
- Initial release
```

### Documentation Standards

**README.md structure**:
```markdown
# My New Plugin

Brief description of plugin purpose.

## Features

- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Commands

### /my-command

Execute primary workflow.

**Usage**:
```bash
/my-command target/path
```

**Options**:
- `--option`: Description

## Agents

### primary-agent

Expert in X, Y, and Z.

**Use cases**:
- Use case 1
- Use case 2

## Installation

Plugins are automatically available in Claude Code.

## Examples

### Example 1: Basic Usage

```bash
/my-command src/
```

### Example 2: Advanced Usage

```bash
/my-command src/ --option value
```

## Troubleshooting

### Issue: X doesn't work

**Solution**: Y

## Contributing

See CONTRIBUTING.md

## License

MIT
```

### Testing Plugins

```bash
# Test command manually
/my-command test-data/

# Test with different options
/my-command test-data/ --verbose
/my-command test-data/ --dry-run

# Test error handling
/my-command invalid-path/
```

---

## 7. Troubleshooting Common Issues

### Issue: Pre-commit Hook Fails

**Error**:
```
Validate Plugin Syntax.....................................Failed
- hook id: lint-plugins
- exit code: 1
```

**Solution**:
```bash
# View detailed errors
pre-commit run --verbose

# Fix issues
/lint-plugins --fix

# Retry commit
git add -u
git commit -m "Fix validation issues"
```

### Issue: CI Pipeline Fails

**Error**: GitHub Actions validation fails

**Solution**:
```bash
# Run validation locally first
python .agent/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins

# Fix all errors
/lint-plugins --fix

# Push fixes
git add -u
git commit -m "Fix CI validation"
git push
```

### Issue: Validation Too Slow

**Problem**: Validation takes >1 minute

**Solution**:
```bash
# Validate only changed plugin
/lint-plugins --plugin=my-new-plugin

# Skip unchanged files in CI
# (Already configured in GitHub Actions with paths filter)
```

---

## 8. Release Workflow

### Prepare Release

```bash
# 1. Update version in plugin.json
jq '.version = "1.1.0"' plugins/my-new-plugin/plugin.json > tmp.json
mv tmp.json plugins/my-new-plugin/plugin.json

# 2. Update CHANGELOG.md
# (Add new version entry)

# 3. Validate everything
/lint-plugins --plugin=my-new-plugin

# 4. Commit changes
git add plugins/my-new-plugin/
git commit -m "Release v1.1.0: Add new features"

# 5. Create tag
git tag -a v1.1.0 -m "Version 1.1.0"

# 6. Push
git push origin main --tags
```

### Release Checklist

- [ ] Version bumped in plugin.json
- [ ] CHANGELOG.md updated
- [ ] All validation passes
- [ ] Documentation updated
- [ ] Examples tested
- [ ] CI pipeline green
- [ ] Git tag created
- [ ] Release notes published

---

## 9. Continuous Improvement

### Regular Maintenance

**Weekly**:
- Review validation errors/warnings
- Update documentation
- Test commands with latest Claude Code

**Monthly**:
- Review and update dependencies
- Check for unused agents
- Optimize command performance

**Quarterly**:
- Major version planning
- Architecture review
- Breaking changes assessment

### Metrics to Track

```bash
# Validation pass rate
/lint-plugins --report

# Plugin usage (if analytics available)
# - Command invocation count
# - Agent usage frequency
# - Error rates

# Code quality
# - Documentation coverage
# - Test coverage (if applicable)
# - Maintenance burden
```

---

## Summary

**Quick Reference**:

```bash
# Create new plugin
mkdir -p plugins/my-plugin/{agents,commands,docs}
touch plugins/my-plugin/plugin.json

# Validate during development
/lint-plugins --plugin=my-plugin --fix

# Setup pre-commit hooks
pre-commit install

# Commit changes
git add plugins/my-plugin/
git commit -m "Add my-plugin"
# (Pre-commit hooks run automatically)

# Release
# 1. Update version
# 2. Update CHANGELOG
# 3. Tag and push
git tag -a v1.0.0 -m "Initial release"
git push --tags
```

**Key Principles**:
1. Validate early and often
2. Automate validation (pre-commit + CI)
3. Follow naming conventions
4. Document thoroughly
5. Version semantically
6. Test before release
7. Monitor and improve

---

This workflow ensures high-quality plugins with automated validation at every stage of development.
