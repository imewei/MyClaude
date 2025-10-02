# Frequently Asked Questions (FAQ)

> Answers to common questions about the Claude Code Command Executor Framework

---

## General Questions

### What is the Claude Code Command Executor Framework?

It's a production-ready AI-powered development automation system that provides 14 specialized commands, 23 AI agents, workflow automation, and extensibility through plugins. It helps developers automate repetitive tasks, improve code quality, optimize performance, and streamline development workflows.

### Who is it for?

- **Developers** - Individual developers looking to boost productivity
- **Teams** - Development teams wanting standardized workflows and quality gates
- **Researchers** - Scientists and researchers working with scientific computing code
- **Enterprises** - Organizations needing compliance, security, and scalability

### Is it free?

The framework itself is open source and free to use. It runs within Claude Code CLI, which requires a Claude subscription.

### What languages are supported?

Primary support:
- Python (full support)
- Julia (full support)
- JAX (full support)
- JavaScript/TypeScript (full support)

Additional support:
- Fortran (adoption and conversion)
- C/C++ (analysis and integration)
- Java (analysis and refactoring)
- Go, Rust, and others (basic support)

---

## Installation & Setup

### How do I install it?

The commands are already available in your Claude Code CLI. Simply verify by running:
```bash
# Commands are available as slash commands
/think-ultra --help
```

### Do I need to install anything else?

For basic usage, no additional installation required. For specific features:
```bash
# Python dependencies
pip install -r requirements.txt

# Scientific computing support
pip install numpy scipy jax

# GPU support
pip install jax[cuda11]
```

### How do I update to the latest version?

```bash
# Update through Claude Code CLI
claude-commands update

# Or pull latest from repository
cd ~/.claude/commands
git pull
```

---

## Commands & Usage

### What are the 14 commands?

1. **/think-ultra** - Advanced analytical thinking
2. **/reflection** - Reflection and session analysis
3. **/double-check** - Verification and auto-completion
4. **/check-code-quality** - Code quality analysis
5. **/refactor-clean** - AI-powered refactoring
6. **/clean-codebase** - Codebase cleanup
7. **/generate-tests** - Test generation
8. **/run-all-tests** - Test execution with auto-fix
9. **/debug** - Scientific debugging
10. **/optimize** - Performance optimization
11. **/commit** - Smart git commits
12. **/fix-commit-errors** - GitHub Actions fixes
13. **/fix-github-issue** - GitHub issue resolution
14. **/ci-setup** - CI/CD pipeline setup

Plus: **/multi-agent-optimize**, **/adopt-code**, **/explain-code**, **/update-docs**

### How do I use commands?

Commands use slash format in Claude Code CLI:
```bash
/command-name [--flags] [arguments]

# Examples
/check-code-quality --auto-fix .
/generate-tests --coverage=90 src/
/optimize --implement --category=all
```

### Can I combine commands?

Yes! Combine commands for complex workflows:
```bash
/check-code-quality --auto-fix && \
/generate-tests --coverage=90 && \
/run-all-tests --auto-fix && \
/double-check "all improvements complete"
```

### What does --auto-fix do?

The `--auto-fix` flag enables automatic fixing of identified issues:
- Code quality issues
- Failing tests
- Performance problems
- Security vulnerabilities

Not all issues can be auto-fixed - some require manual intervention.

---

## 23-Agent System

### What are the 23 agents?

Specialized AI agents with expertise in different domains:
- **Core** (3): Orchestrator, Quality Assurance, DevOps
- **Scientific** (4): Scientific Computing, Performance Engineer, GPU Specialist, Research Scientist
- **AI/ML** (3): AI/ML Engineer, JAX Specialist, Model Optimization
- **Engineering** (5): Backend, Frontend, Security, Database, Cloud
- **Domain** (8): Language experts, Documentation, Testing, Refactoring, Quantum

### How are agents selected?

Three selection strategies:

1. **Automatic (default)**: System selects appropriate agents based on task
```bash
/command-name --agents=auto
```

2. **Explicit**: You specify which agents to use
```bash
/optimize --agents=scientific
/multi-agent-optimize --agents=core
```

3. **Intelligent**: Advanced ML-based selection
```bash
/multi-agent-optimize --intelligent
```

### Can I use all agents at once?

Yes:
```bash
/multi-agent-optimize --agents=all --orchestrate
```

This uses all 23 agents coordinated by the Orchestrator. Resource-intensive but comprehensive.

### Do agents work together?

Yes! The Orchestrator agent coordinates multiple agents:
```bash
/multi-agent-optimize --orchestrate
```

Agents can work:
- **Sequentially**: One after another
- **Parallel**: Simultaneously
- **Coordinated**: Orchestrator manages execution

---

## Workflows

### What are workflows?

Pre-built multi-step automations combining multiple commands. Examples:
- Quality gate workflow
- Performance optimization workflow
- CI/CD setup workflow
- Research workflow

### How do I use workflows?

```bash
# Use pre-built workflow
/multi-agent-optimize --mode=review --focus=quality --implement

# Or create custom workflow YAML
# See workflows/ directory
```

### Can I create custom workflows?

Yes! Create YAML workflow definitions:
```yaml
name: My Workflow
steps:
  - name: step1
    command: check-code-quality
    args: {auto_fix: true}
  - name: step2
    command: run-all-tests
    depends_on: [step1]
```

See [Workflow Guide](../../workflows/INDEX.md) for details.

### Where are workflows stored?

```
~/.claude/commands/workflows/
├── definitions/      # Pre-built workflows
├── templates/        # Workflow templates
└── custom/          # Your custom workflows
```

---

## Plugins

### What are plugins?

Extensions that add new functionality:
- Custom commands
- Custom agents
- Custom workflows
- Tool integrations

### How do I install plugins?

```bash
# From registry
claude-commands install plugin-name

# From file
claude-commands install ./my-plugin.zip

# From git
claude-commands install github:user/plugin-repo
```

### Where can I find plugins?

- **[Plugin Index](../../plugins/PLUGIN_INDEX.md)** - Browse available plugins
- **Plugin Registry** - `claude-commands search plugins`
- **GitHub** - Search for "claude-commands-plugin"

### How do I create plugins?

See **[Plugin Development Guide](../../plugins/docs/PLUGIN_DEVELOPMENT_GUIDE.md)** for complete guide.

Basic structure:
```python
from claude_commands.plugin import Plugin, command

class MyPlugin(Plugin):
    name = "my-plugin"

    @command("my-command")
    def my_command(self, args):
        return {"status": "success"}
```

---

## Performance

### Why is command execution slow?

Common causes and solutions:

1. **Large codebase**: Use `--target` to limit scope
```bash
/check-code-quality --target=src/ --exclude=tests/
```

2. **No caching**: Enable cache
```bash
export CLAUDE_COMMANDS_CACHE=true
```

3. **Sequential execution**: Use parallel
```bash
/multi-agent-optimize --parallel
```

4. **Resource constraints**: Increase limits
```bash
export CLAUDE_COMMANDS_MEMORY_LIMIT=8G
export CLAUDE_COMMANDS_TIMEOUT=600
```

### How can I improve performance?

Performance optimization tips:

1. **Enable caching**
```bash
export CLAUDE_COMMANDS_CACHE=true
```

2. **Use parallel execution**
```bash
/command-name --parallel
```

3. **Limit scope**
```bash
/command-name --target=specific/directory
```

4. **Use appropriate agents**
```bash
# Don't use all agents unless needed
/optimize --agents=scientific  # Instead of --agents=all
```

5. **Incremental processing**
```bash
/command-name --incremental
```

### Does the system cache results?

Yes! Intelligent caching is enabled by default:
- Analysis results cached
- Agent results cached
- Build artifacts cached
- Cache automatically invalidated on file changes

```bash
# Check cache status
claude-commands cache-status

# Clear cache
claude-commands cache-clear
```

---

## Testing

### Can it generate tests automatically?

Yes!
```bash
# Generate unit tests with 90% coverage
/generate-tests --type=unit --coverage=90

# Generate all test types
/generate-tests --type=all --coverage=95

# Scientific tests
/generate-tests --type=scientific --gpu
```

### Does it handle test failures?

Yes, with `--auto-fix`:
```bash
/run-all-tests --auto-fix
```

This automatically fixes:
- Import errors
- Assertion errors (when fix is obvious)
- Configuration issues
- Missing fixtures

### What test frameworks are supported?

- **Python**: pytest, unittest
- **Julia**: Test.jl
- **JavaScript**: Jest, Mocha, Jasmine
- **Scientific**: Custom scientific test frameworks

---

## Code Quality

### What quality issues can it detect?

- **Style**: PEP 8, linting issues
- **Complexity**: High cyclomatic complexity
- **Type hints**: Missing or incorrect types
- **Security**: Security vulnerabilities
- **Performance**: Performance anti-patterns
- **Maintainability**: Code smells
- **Documentation**: Missing docstrings

### Can it automatically fix quality issues?

Yes, many issues can be auto-fixed:
```bash
/check-code-quality --auto-fix
```

Auto-fixable issues:
- Style violations
- Missing imports
- Unused variables
- Simple refactorings
- Documentation

Some issues require manual intervention:
- Complex refactoring
- Algorithmic changes
- Architecture changes

### How is code quality scored?

Quality score (0-100) based on:
- **Style** (20%): Code style compliance
- **Complexity** (20%): Cyclomatic complexity
- **Types** (15%): Type hint coverage
- **Documentation** (15%): Docstring coverage
- **Security** (15%): Security issues
- **Maintainability** (15%): Code smells

---

## Scientific Computing

### Can it work with Fortran code?

Yes! Use `/adopt-code`:
```bash
# Analyze Fortran code
/adopt-code --analyze --language=fortran legacy/

# Convert to Python
/adopt-code --integrate --language=fortran --target=python

# Preserve parallel structure
/adopt-code --language=fortran --target=python --parallel=mpi
```

### Does it support GPU computing?

Yes, full GPU support:
```bash
# GPU optimization
/optimize --gpu --agents=scientific

# GPU debugging
/debug --gpu --profile

# GPU tests
/generate-tests --type=gpu --framework=pytest
```

Supported frameworks:
- CUDA
- JAX
- PyTorch
- TensorFlow

### Can it ensure reproducibility?

Yes:
```bash
# Reproducible tests
/generate-tests --type=scientific --reproducible

# Set random seeds
/run-all-tests --scientific --reproducible

# Pin dependencies
/ci-setup --reproducible
```

---

## CI/CD

### Can it setup CI/CD pipelines?

Yes:
```bash
# GitHub Actions
/ci-setup --platform=github --type=enterprise

# GitLab CI
/ci-setup --platform=gitlab --type=basic

# With security scanning
/ci-setup --platform=github --security --monitoring
```

### Can it fix CI/CD failures?

Yes:
```bash
# Auto-fix GitHub Actions errors
/fix-commit-errors --auto-fix <commit-hash>

# Emergency fix
/fix-commit-errors --emergency --auto-fix

# Learn from patterns
/fix-commit-errors --learn --batch
```

### Does it work with pre-commit hooks?

Yes, integrates with pre-commit:
```bash
# Setup pre-commit
pre-commit install

# Auto-fix before commit
/commit --validate --auto-fix

# Quality gate in pre-commit
# Add to .pre-commit-config.yaml:
- repo: local
  hooks:
    - id: claude-quality
      name: Claude Quality Check
      entry: /check-code-quality --auto-fix
```

---

## Enterprise

### Is it suitable for enterprise use?

Yes! Enterprise features:
- **Security**: Security scanning and vulnerability detection
- **Compliance**: Audit trails and compliance checking
- **Scalability**: Handles large codebases
- **Team workflows**: Standardized team processes
- **CI/CD**: Full CI/CD integration
- **Monitoring**: Performance and quality monitoring

### Can it enforce team standards?

Yes:
```bash
# Quality gates in CI/CD
/ci-setup --type=enterprise --security

# Pre-commit enforcement
# Configure in .pre-commit-config.yaml

# Custom quality rules
# Configure in .claude-commands.yml
quality:
  min_coverage: 90
  style_guide: pep8
  security_scan: true
```

### Does it provide audit trails?

Yes:
```bash
# Enable audit logging
export CLAUDE_COMMANDS_AUDIT=true

# View audit log
claude-commands audit-log

# Export audit report
claude-commands audit-export --format=json
```

---

## Customization

### Can I customize agent behavior?

Yes, through configuration:
```yaml
# .claude-commands.yml
agents:
  selection_strategy: intelligent
  preferred: [scientific, quality]
  exclude: []

quality:
  min_coverage: 90
  strict_mode: true
```

### Can I create custom commands?

Yes, through plugins:
```python
from claude_commands.plugin import Plugin, command

class MyPlugin(Plugin):
    @command("my-custom-command")
    def my_command(self, args):
        # Implementation
        return result
```

### Can I customize workflows?

Yes, create custom workflow YAML:
```yaml
name: My Custom Workflow
steps:
  - name: analyze
    command: check-code-quality
  - name: test
    command: run-all-tests
    depends_on: [analyze]
```

---

## Troubleshooting

### Command not working?

1. Check command format: `/command-name` (slash required)
2. Verify installation: `claude-commands status`
3. Check logs: `claude-commands logs`
4. Run diagnostics: `claude-commands diagnostics`
5. See [Troubleshooting Guide](TROUBLESHOOTING.md)

### How do I debug issues?

```bash
# Enable debug mode
/command-name --debug

# Verbose output
/command-name --verbose

# Check system status
claude-commands diagnostics

# View logs
claude-commands logs --tail=100
```

### Where can I get help?

1. **Documentation** - Check [User Guide](USER_GUIDE.md)
2. **Troubleshooting** - See [Troubleshooting Guide](TROUBLESHOOTING.md)
3. **FAQ** - This document
4. **GitHub Issues** - Report bugs or request features
5. **Community** - Join discussions

---

## Integration

### Does it work with IDEs?

Yes! Works with:
- VS Code (through Claude Code integration)
- JetBrains IDEs
- Vim/Neovim
- Emacs
- Any editor with terminal access

Use commands directly in terminal within your IDE.

### Can it integrate with Git?

Yes, full Git integration:
```bash
# Smart commits
/commit --ai-message --validate

# Fix commit errors
/fix-commit-errors --auto-fix

# GitHub issue resolution
/fix-github-issue --auto-fix <issue-number>
```

### Does it work with Docker?

Yes:
```bash
# In Dockerfile
RUN pip install claude-commands

# Run in container
docker run -v $(pwd):/code myimage /check-code-quality
```

---

## Licensing & Contribution

### What license is it under?

MIT License - free to use, modify, and distribute.

### Can I contribute?

Yes! Contributions welcome:
1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

See [Contributing Guide](../CONTRIBUTING.md) for details.

### How do I report bugs?

1. Check existing issues on GitHub
2. Run diagnostics: `claude-commands diagnostics`
3. Create GitHub issue with:
   - Error message
   - Command used
   - System info
   - Debug output

---

## Advanced Questions

### Can it handle monorepos?

Yes:
```bash
# Analyze specific package
/check-code-quality --target=packages/package1

# Cross-package optimization
/multi-agent-optimize --scope=monorepo

# Package-specific workflows
/multi-agent-optimize --package=package1
```

### Does it support microservices?

Yes:
```bash
# Per-service analysis
/check-code-quality --target=services/service1

# Service orchestration
/multi-agent-optimize --mode=optimize --focus=architecture

# API contract testing
/generate-tests --type=integration --focus=api
```

### Can it help with refactoring?

Yes:
```bash
# Modern patterns
/refactor-clean --patterns=modern --implement

# Performance refactoring
/refactor-clean --patterns=performance --implement

# Security refactoring
/refactor-clean --patterns=security --implement
```

### Does it support quantum computing?

Yes, through Quantum Computing agent:
```bash
# Quantum code optimization
/optimize --agents=quantum --category=algorithm

# Quantum circuit analysis
/multi-agent-optimize --agents=quantum --mode=optimize
```

---

## Best Practices

### What's the recommended workflow?

1. **Quality check**
```bash
/check-code-quality --auto-fix
```

2. **Generate tests**
```bash
/generate-tests --coverage=90
```

3. **Run tests**
```bash
/run-all-tests --auto-fix
```

4. **Optimize**
```bash
/optimize --implement
```

5. **Verify**
```bash
/double-check "all improvements complete"
```

6. **Commit**
```bash
/commit --ai-message --validate
```

### How often should I run commands?

**Continuous**:
- Pre-commit: `/check-code-quality --auto-fix`
- CI/CD: All quality checks

**Regular** (daily/weekly):
- `/clean-codebase` - Weekly
- `/refactor-clean` - Monthly
- `/optimize` - As needed

**On-demand**:
- `/debug` - When issues arise
- `/fix-commit-errors` - When CI fails
- `/adopt-code` - For legacy code

### What's the best agent selection strategy?

- **Default tasks**: `--agents=auto` (default)
- **Scientific code**: `--agents=scientific`
- **Web applications**: `--agents=engineering`
- **Comprehensive**: `--agents=all --orchestrate`
- **Performance-critical**: `--agents=scientific --intelligent`

---

## Future Plans

### What features are planned?

See [Roadmap](../ROADMAP.md) for complete plans.

Upcoming:
- More language support
- Enhanced AI capabilities
- Advanced workflows
- More plugins
- IDE integrations
- Cloud deployment

### How can I request features?

1. Check [Roadmap](../ROADMAP.md)
2. Search existing GitHub issues
3. Create feature request issue
4. Join community discussions

---

**Version**: 1.0.0 | **Last Updated**: September 2025

**More Questions?**
- **[User Guide](USER_GUIDE.md)** - Complete documentation
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Problem solving
- **[Tutorials](../tutorials/)** - Hands-on learning
- **GitHub Issues** - Ask questions