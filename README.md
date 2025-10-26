# Scientific Computing Workflows Marketplace

Custom Claude Code marketplace combining selected plugins from claude-code-workflows with specialized scientific computing agents for HPC, JAX, ML, and numerical methods.

## Overview

This marketplace provides:
- **16 curated plugins** from claude-code-workflows for general development
- **Custom scientific-computing plugin** with specialized agents and commands
- **Independent modification** without affecting source plugins
- **Git-based version control** for all customizations
- **Conflict resolution** for custom agents/commands (renamed to avoid overlaps)

## Plugins Included

### From claude-code-workflows (Modified)
1. `comprehensive-review` - Multi-perspective code analysis
2. `code-documentation` - Code explanation and documentation generation
3. `agent-orchestration` - Multi-agent system optimization
4. `unit-testing` - Test generation and automation
5. `codebase-cleanup` - Technical debt reduction and refactoring
6. `incident-response` - Smart debugging and incident management
7. `machine-learning-ops` - ML pipelines and MLOps
8. `data-engineering` - ETL pipelines and data workflows
9. `framework-migration` - Legacy modernization and migration
10. `observability-monitoring` - Metrics, logging, and tracing
11. `backend-development` - Backend API design and architecture
12. `frontend-mobile-development` - UI/UX and mobile development
13. `multi-platform-apps` - Cross-platform application development
14. `cicd-automation` - CI/CD pipeline configuration
15. `debugging-toolkit` - Interactive debugging and DX optimization
16. `git-pr-workflows` - Git workflow automation

### Custom Plugin
17. **custom-scientific-computing** - Specialized for HPC/JAX/ML workflows
    - **12 Specialized Agents**:
      - `hpc-numerical-coordinator` - HPC and numerical methods specialist
      - `jax-pro` - JAX optimization expert
      - `jax-scientific-domains` - JAX physics/quantum/CFD/MD specialist
      - `neural-architecture-engineer` - Deep learning architecture design
      - `correlation-function-expert` - Statistical physics specialist
      - `simulation-expert` - Molecular dynamics and simulation
      - `scientific-code-adoptor` - Legacy scientific code migration
      - `visualization-interface-master` - Scientific visualization + AR/VR
      - `research-intelligence-master` - Research methodology expert
      - `command-systems-engineer` - CLI tool design specialist
      - `database-workflow-engineer` - Database + scientific workflows
      - `multi-agent-orchestrator` - Custom agent orchestration

    - **14 Custom Commands**:
      - `/adopt-code` - Scientific codebase adoption
      - `/sci-multi-optimize` - Multi-agent optimization (renamed to avoid conflict)
      - `/explain-scientific` - Scientific code explanation (renamed)
      - `/sci-test-gen` - Scientific test generation (renamed)
      - `/sci-ci-setup` - CI with security + lock-check (renamed)
      - `/double-check` - Ultrathink validation
      - `/fix-commit-errors` - GitHub Actions failure analysis
      - `/fix-imports` - Import resolution sessions
      - `/lint-plugins` - Plugin validation tool
      - `/run-all-tests` - Iterative test fixing
      - `/ultra-think` - Multi-dimensional reasoning
      - `/reflection` - Meta-cognitive analysis
      - `/update-claudemd` - CLAUDE.md automation
      - `/update-docs` - Documentation updates

## Installation

### Prerequisites
- Claude Code installed
- Git installed
- `jq` installed (for metadata generation)
  ```bash
  # macOS
  brew install jq

  # Linux
  sudo apt-get install jq
  ```

### Quick Setup

```bash
# 1. Navigate to project directory
cd /Users/b80985/Projects/MyClaude

# 2. Run setup script
./setup-marketplace.sh

# 3. Generate marketplace metadata
./generate-metadata.sh

# 4. Restart Claude Code

# 5. Verify installation
# In Claude Code, run:
/plugin list
```

### Manual Verification

```bash
# Check marketplace structure
ls -la /Users/b80985/Projects/MyClaude/

# Verify symlink
ls -la ~/.claude/plugins/marketplaces/scientific-computing-workflows

# Check plugin count
ls /Users/b80985/Projects/MyClaude/plugins/ | wc -l
# Should show 17 (16 + custom-scientific-computing)
```

## Usage

### Installing Plugins

```bash
# List available plugins
/plugin list

# Install custom scientific computing plugin
/plugin install custom-scientific-computing

# Install other plugins
/plugin install comprehensive-review
/plugin install machine-learning-ops
```

### Using Custom Commands

```bash
# Use renamed commands (no conflicts)
/sci-multi-optimize path/to/code --focus=performance
/explain-scientific path/to/scientific/code.py
/sci-test-gen path/to/module.py --coverage

# Use unique custom commands
/ultra-think "Complex problem to solve" --depth=deep
/double-check recent-changes --deep
/fix-commit-errors workflow-id --auto-fix
```

### Using Custom Agents

Custom agents are invoked via the Task tool:

```python
# Use JAX optimization agent
Task(
    subagent_type="custom-scientific-computing:jax-pro",
    prompt="Optimize this JAX code for GPU performance..."
)

# Use HPC coordination agent
Task(
    subagent_type="custom-scientific-computing:hpc-numerical-coordinator",
    prompt="Design parallel workflow for molecular dynamics simulation..."
)
```

## Customization

### Adding New Agents

1. Create agent file: `plugins/custom-scientific-computing/agents/my-agent.md`
2. Regenerate metadata: `./generate-metadata.sh`
3. Restart Claude Code

### Adding New Commands

1. Create command file: `plugins/custom-scientific-computing/commands/my-command.md`
2. Regenerate metadata: `./generate-metadata.sh`
3. Restart Claude Code

### Modifying Existing Plugins

All plugins are independent copies - modify freely without affecting source:

```bash
# Edit plugin files directly
vim plugins/comprehensive-review/agents/code-reviewer.md

# Changes are git-tracked
git add plugins/comprehensive-review/
git commit -m "Customize code-reviewer agent for scientific code"
```

## Hybrid Migration Plan

This marketplace supports the hybrid approach from the overlap analysis:

### Conflicts Resolved
- ✅ `/multi-agent-optimize` → `/sci-multi-optimize` (renamed)
- ✅ `/explain-code` → `/explain-scientific` (renamed)

### High-Overlap Items Removed
- ❌ `clean-codebase` → Use `codebase-cleanup:refactor-clean` plugin
- ❌ `clean-project` → Use `codebase-cleanup` plugin
- ❌ `code-review` → Use `comprehensive-review:code-reviewer` plugin
- ❌ `fix` → Use `incident-response:smart-fix` plugin

### Unique Value Preserved
- ✅ 12 specialized scientific computing agents (no plugin equivalents)
- ✅ 14 custom commands for scientific workflows
- ✅ HPC/JAX/ML/numerical methods expertise

## Maintenance

### Updating from Upstream

```bash
# Check upstream changes
cd ~/.claude/plugins/marketplaces/claude-code-workflows
git pull

# Copy updated plugin manually
cp -R ~/.claude/plugins/marketplaces/claude-code-workflows/plugins/comprehensive-review \
      /Users/b80985/Projects/MyClaude/plugins/

# Or use update script (if created)
./scripts/update-plugin.sh comprehensive-review
```

### Version Control

```bash
# Commit changes
git add plugins/
git commit -m "Update plugin customizations"

# Create version tags
git tag -a v0.1.0 -m "Initial release"

# Push to remote (if configured)
git push origin main --tags
```

## Troubleshooting

### Marketplace not recognized

```bash
# Verify symlink
ls -la ~/.claude/plugins/marketplaces/scientific-computing-workflows

# Re-create symlink
rm ~/.claude/plugins/marketplaces/scientific-computing-workflows
ln -s /Users/b80985/Projects/MyClaude ~/.claude/plugins/marketplaces/scientific-computing-workflows

# Restart Claude Code
```

### Plugins not loading

```bash
# Validate marketplace.json
jq '.' /Users/b80985/Projects/MyClaude/.claude-plugin/marketplace.json

# Regenerate metadata
./generate-metadata.sh

# Check plugin structure
./scripts/validate-plugins.sh  # (if created)
```

### Command conflicts

If you still see conflicts:
1. Check which marketplace is active: `/plugin list`
2. Uninstall conflicting plugin: `/plugin uninstall <plugin-name>`
3. Verify custom commands use `sci-` prefix

## Directory Structure

```
/Users/b80985/Projects/MyClaude/
├── .claude-plugin/
│   └── marketplace.json          # Marketplace metadata
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── LICENSE                        # MIT License
├── setup-marketplace.sh           # Initial setup script
├── generate-metadata.sh           # Metadata generation script
├── docs/                          # Documentation (future)
│   ├── SETUP.md
│   ├── MIGRATION.md
│   └── CUSTOMIZATION.md
├── plugins/                       # All plugins
│   ├── comprehensive-review/
│   ├── code-documentation/
│   ├── agent-orchestration/
│   ├── ... (13 more plugins)
│   └── custom-scientific-computing/
│       ├── agents/                # 12 custom agents
│       └── commands/              # 14 custom commands
└── scripts/                       # Utility scripts
    └── (future scripts)
```

## Contributing

### To This Marketplace

1. Fork and create a branch
2. Make changes
3. Test thoroughly
4. Submit PR with description

### To Upstream (claude-code-workflows)

If you develop features that could benefit the broader community:
1. Extract feature into standalone plugin
2. Test with source marketplace format
3. Submit PR to https://github.com/wshobson/agents

## License

MIT License (see [LICENSE](LICENSE))

## Acknowledgments

- Based on [claude-code-workflows](https://github.com/wshobson/agents) by Seth Hobson
- Custom scientific computing agents developed for HPC/JAX/ML workflows
- Hybrid migration plan designed with ultra-think analysis

## Version History

### v0.1.0 (Initial Release)
- 16 selected plugins from claude-code-workflows v1.2.2
- Custom scientific-computing plugin with 12 agents and 14 commands
- Renamed conflicting commands (multi-agent-optimize, explain-code, etc.)
- Git-based version control
- Automated setup scripts

## Contact

For questions, issues, or suggestions:
- Create an issue in your repository
- Or contact: $(git config user.email 2>/dev/null || echo 'your.email@example.com')
