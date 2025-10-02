# Claude Code Agent Installation Guide

**Version**: 1.0
**Last Updated**: 2025-09-29
**Compatibility**: Claude Code CLI v1.0+

---

## Quick Start

### 1. Verify Directory Structure

```bash
# Check if .claude directory exists
ls ~/.claude/agents/

# If not, create it
mkdir -p ~/.claude/agents/
```

### 2. Install Agent Definitions

**Option A: Clone from Repository** (recommended)
```bash
# Navigate to .claude directory
cd ~/.claude/

# Clone agent repository
git clone [repository-url] agents/

# Verify installation
ls agents/*.md | wc -l
# Expected output: 23 agent files + documentation
```

**Option B: Manual Installation**
```bash
# Copy all agent files to .claude/agents/
cp /path/to/agent-files/*.md ~/.claude/agents/

# Copy validation script
cp /path/to/validate-agents.sh ~/.claude/agents/
chmod +x ~/.claude/agents/validate-agents.sh
```

### 3. Validate Installation

```bash
# Navigate to agents directory
cd ~/.claude/agents/

# Run validation script
./validate-agents.sh

# Expected output:
# ====================================
#   PASSED WITH 1 WARNINGS ⚠
# ====================================
```

---

## What Gets Installed

### Core Agent Definitions (23 files)

**Engineering Core** (7 agents):
- systems-architect.md
- fullstack-developer.md
- code-quality-master.md
- command-systems-engineer.md
- devops-security-engineer.md
- database-workflow-engineer.md
- documentation-architect.md

**AI/ML Core** (3 agents):
- ai-ml-specialist.md
- ai-systems-architect.md
- neural-networks-master.md

**Scientific Computing** (3 agents):
- scientific-computing-master.md
- jax-pro.md
- jax-scientific-domains.md

**Domain Specialists** (7 agents):
- advanced-quantum-computing-expert.md
- correlation-function-expert.md
- neutron-soft-matter-expert.md
- xray-soft-matter-expert.md
- nonequilibrium-stochastic-expert.md
- scientific-code-adoptor.md

**Support Specialists** (3 agents):
- data-professional.md
- visualization-interface-master.md
- research-intelligence-master.md
- multi-agent-orchestrator.md

### Meta-Documentation (7 files)

**Foundation**:
- AGENT_TEMPLATE.md - Standardized agent structure
- README.md - Comprehensive project documentation

**Guides**:
- AGENT_CATEGORIES.md - Agent taxonomy and selection guide
- AGENT_COMPATIBILITY_MATRIX.md - Multi-agent workflow patterns
- QUICK_REFERENCE.md - Quick agent selection reference
- INSTALLATION_GUIDE.md - This file

### Validation Tooling

- **validate-agents.sh** - Automated quality validation script

---

## Verification

### Test 1: Agent Count
```bash
cd ~/.claude/agents/
ls *.md | grep -v "PHASE\|OPTIMIZATION\|TEMPLATE\|CATEGORIES\|COMPATIBILITY\|INSTALLATION" | wc -l
# Expected: 23
```

### Test 2: Validation Script
```bash
cd ~/.claude/agents/
./validate-agents.sh
# Expected: PASSED (with 1 acceptable warning)
```

### Test 3: Agent Invocation (via Claude Code)
```bash
# Test agent selection
claude-code "I need help with database design"
# Should suggest: database-workflow-engineer or systems-architect
```

---

## Directory Structure

```
~/.claude/
└── agents/
    ├── ai-ml-specialist.md
    ├── ai-systems-architect.md
    ├── [... 21 more agent files ...]
    ├── AGENT_CATEGORIES.md
    ├── AGENT_COMPATIBILITY_MATRIX.md
    ├── AGENT_TEMPLATE.md
    ├── CHANGELOG.md
    ├── INSTALLATION_GUIDE.md
    ├── QUICK_REFERENCE.md
    ├── README.md
    └── validate-agents.sh
```

---

## Usage

### Selecting the Right Agent

**By Category** (see AGENT_CATEGORIES.md):
```markdown
- Engineering Core: Software development, architecture
- AI/ML Core: Machine learning, AI systems
- Scientific Computing: Numerical computing, HPC
- Domain Specialists: Physics, chemistry, specialized science
- Support Specialists: Data, documentation, orchestration
```

**By Task** (see "When to Invoke" sections in each agent):
- Read agent's "When to Invoke This Agent" section
- Check "Differentiation from similar agents" for clarity
- Review AGENT_COMPATIBILITY_MATRIX.md for multi-agent workflows

**By Workflow** (see AGENT_COMPATIBILITY_MATRIX.md):
- Pattern 1: Strategic Planning → Implementation (architect → developer)
- Pattern 2: AI/ML Development Pipeline (data → model → infrastructure)
- Pattern 3: Scientific Computing Workflow (classical → JAX → domain)
- Pattern 4: Database-Centric Application (schema → backend → frontend)
- Pattern 5: Research to Production (research → implementation → docs)

### Multi-Agent Workflows

**Simple (1-2 agents)**: Invoke agents directly
```bash
# Architecture then implementation
claude-code --agent=systems-architect "Design API architecture"
claude-code --agent=fullstack-developer "Implement API from design"
```

**Complex (3-5 agents)**: Use sequential invocation
```bash
# See AGENT_COMPATIBILITY_MATRIX.md for common workflows
```

**Large-Scale (5+ agents)**: Use multi-agent-orchestrator
```bash
claude-code --agent=multi-agent-orchestrator "Build complete ML platform"
```

---

## Maintenance

### Weekly Maintenance

```bash
# Run validation after any agent modifications
cd ~/.claude/agents/
./validate-agents.sh
```

### Monthly Maintenance

```bash
# Check for updates
cd ~/.claude/agents/
git pull  # If using git

# Verify integrity
./validate-agents.sh

# Review any new agents or documentation
```

### Quarterly Maintenance

See maintenance guidelines in README.md (section: Maintenance and Evolution)

---

## Troubleshooting

### Issue: Validation Script Not Executable

```bash
chmod +x ~/.claude/agents/validate-agents.sh
```

### Issue: Agent Not Found

```bash
# List all available agents
cd ~/.claude/agents/
ls *.md | grep -v "PHASE\|OPTIMIZATION\|TEMPLATE\|CATEGORIES\|COMPATIBILITY\|INSTALLATION"

# Check specific agent exists
ls [agent-name].md
```

### Issue: Validation Fails

```bash
# Run validation with verbose output
cd ~/.claude/agents/
./validate-agents.sh

# Review specific failures listed in output
# Most common: Missing sections, line count warnings
```

### Issue: Unclear Agent Selection

```bash
# Read category guide
cat ~/.claude/agents/AGENT_CATEGORIES.md

# Read compatibility matrix
cat ~/.claude/agents/AGENT_COMPATIBILITY_MATRIX.md

# Check agent differentiation
grep "Differentiation from similar agents" ~/.claude/agents/[agent-name].md
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Validate Agents

on:
  push:
    paths:
      - '.claude/agents/*.md'
  pull_request:
    paths:
      - '.claude/agents/*.md'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate Agent Definitions
        run: |
          cd .claude/agents
          chmod +x validate-agents.sh
          ./validate-agents.sh
```

### GitLab CI Example

```yaml
validate-agents:
  stage: test
  script:
    - cd .claude/agents
    - chmod +x validate-agents.sh
    - ./validate-agents.sh
  only:
    changes:
      - .claude/agents/*.md
```

### Pre-Commit Hook Example

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Check if agent files changed
if git diff --cached --name-only | grep -q ".claude/agents/.*\.md"; then
    echo "Validating agent definitions..."
    cd .claude/agents
    ./validate-agents.sh

    if [ $? -ne 0 ]; then
        echo "Agent validation failed. Commit aborted."
        exit 1
    fi
fi

exit 0
```

---

## Uninstallation

### Remove Agent Definitions

```bash
# Backup first (optional)
cp -r ~/.claude/agents/ ~/agent-backup/

# Remove agents directory
rm -rf ~/.claude/agents/
```

### Restore from Backup

```bash
# Restore from backup
cp -r ~/agent-backup/ ~/.claude/agents/

# Validate restoration
cd ~/.claude/agents/
./validate-agents.sh
```

---

## Support & Resources

### Documentation
- **AGENT_CATEGORIES.md** - Agent taxonomy and selection guide
- **AGENT_COMPATIBILITY_MATRIX.md** - Multi-agent workflow patterns
- **AGENT_TEMPLATE.md** - Agent structure reference
- **Phase Reports** - Implementation history and rationale

### Validation
- **validate-agents.sh** - Quality assurance tool
- Run after any modifications to agents
- 6 comprehensive test suites

### Updates
- Check repository for new agents or improvements
- Follow maintenance guidelines in phase reports
- Run validation after updates

---

## Version History

**v1.0** (2025-09-29)
- Initial installation guide
- 23 agent definitions
- Automated validation framework
- Comprehensive documentation suite

---

**Installation Complete**

You now have access to 23 specialized Claude Code agents organized across 5 categories with comprehensive documentation and automated quality validation.

**Next Steps**:
1. Review AGENT_CATEGORIES.md for agent overview
2. Explore AGENT_COMPATIBILITY_MATRIX.md for workflow patterns
3. Start with agent "When to Invoke" sections for specific needs
4. Run validate-agents.sh periodically to ensure quality