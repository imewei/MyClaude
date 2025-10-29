# Scientific Computing Workflows Marketplace

Comprehensive Claude Code marketplace with 32 specialized plugins for scientific computing, software development, and research workflows.

## Overview

This marketplace provides:
- **32 specialized plugins** for scientific computing and software development workflows
- **Comprehensive coverage** across scientific domains, languages, and development practices
- **Independent modification** without affecting source plugins
- **Git-based version control** for all customizations
- **Integrated ecosystem** with cross-plugin collaboration

## Plugins Included (32 Total)

### Scientific Computing & Research (13 plugins)
1. **julia-development** ✨ NEW - Julia programming, SciML ecosystem, Bayesian inference (Turing.jl)
   - Agents: julia-pro, julia-developer, sciml-pro, turing-pro
   - Commands: /sciml-setup, /julia-optimize, /julia-scaffold, /julia-package-ci

2. **jax-implementation** - JAX optimization, physics-informed ML, scientific domains
3. **python-development** - Python programming, async patterns, packaging
4. **hpc-computing** - High-performance computing, parallel computing, numerical methods
5. **deep-learning** - Neural architectures, training workflows, model optimization
6. **machine-learning** - ML pipelines, MLOps, model serving
7. **molecular-simulation** - Molecular dynamics, atomistic modeling, force fields
8. **statistical-physics** - Correlation functions, statistical mechanics, complex systems
9. **research-methodology** - Research intelligence, literature analysis, trend forecasting
10. **data-visualization** - Scientific visualization, D3.js, Plotly, AR/VR
11. **llm-application-dev** - LLM applications, RAG systems, intelligent agents
12. **ai-reasoning** - Multi-dimensional reasoning, meta-cognitive analysis
13. **systems-programming** - Low-level systems, performance optimization, memory management

### Language-Specific Development (2 plugins)
14. **javascript-typescript** - Modern JS/TS patterns, Node.js, testing
15. **cli-tool-design** - Command-line tool design, developer automation

### Full-Stack & Application Development (5 plugins)
16. **backend-development** - Backend API design, microservices, distributed systems
17. **frontend-mobile-development** - UI/UX, mobile development, responsive design
18. **multi-platform-apps** - Cross-platform development (React Native, Flutter)
19. **full-stack-orchestration** - End-to-end full-stack workflows
20. **developer-essentials** - Core development workflows and patterns

### Quality & Testing (3 plugins)
21. **unit-testing** - Test generation, automation, TDD practices
22. **quality-engineering** - Code quality analysis, refactoring, technical debt
23. **debugging-toolkit** - Interactive debugging, DX optimization

### DevOps & Infrastructure (2 plugins)
24. **cicd-automation** - CI/CD pipelines, GitHub Actions, deployment automation
25. **observability-monitoring** - Metrics, logging, tracing, performance monitoring

### Code Management & Migration (4 plugins)
26. **code-documentation** - Documentation generation, code explanation
27. **codebase-cleanup** - Technical debt reduction, import fixing
28. **code-migration** - Framework migration, legacy modernization
29. **framework-migration** - Cross-framework migration strategies

### Collaboration & Review (3 plugins)
30. **comprehensive-review** - Multi-perspective code analysis, security audits
31. **git-pr-workflows** - Git workflow automation, PR management
32. **agent-orchestration** - Multi-agent system optimization, workflow coordination

## Featured: julia-development Plugin

The newly implemented **julia-development** plugin provides comprehensive support for Julia programming:

### 4 Specialized Agents
- **julia-pro**: General Julia expert (HPC, simulations, data analysis, ML, JuMP.jl)
- **julia-developer**: Package development specialist (testing, CI/CD, web dev)
- **sciml-pro**: SciML ecosystem expert (DifferentialEquations.jl, ModelingToolkit.jl, full ecosystem)
- **turing-pro**: Bayesian inference expert (MCMC, variational inference, model comparison)

### 4 Priority Commands
1. **/sciml-setup** - Auto-detect problem type (ODE/PDE/SDE/optimization) and generate templates
2. **/julia-optimize** - Performance profiling with type stability and allocation analysis
3. **/julia-scaffold** - Bootstrap Julia packages with proper structure
4. **/julia-package-ci** - Generate GitHub Actions workflows for Julia

### 21 Focused Skills
Distributed across agents covering: core Julia patterns, JuMP optimization, SciML ecosystem, differential equations, Turing.jl workflows, package development, testing, CI/CD, visualization, interoperability, and more.

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

# 2. Run setup script (if available)
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
# Should show 32
```

## Usage

### Installing Plugins

```bash
# List available plugins
/plugin list

# Install Julia development plugin
/plugin install julia-development

# Install other plugins
/plugin install comprehensive-review
/plugin install machine-learning
/plugin install hpc-computing
```

### Using Julia Development Commands

```bash
# Set up SciML project with auto-detection
/sciml-setup "coupled oscillator system"

# Optimize Julia code for performance
/julia-optimize path/to/code.jl

# Create new Julia package
/julia-scaffold MyAwesomePackage

# Generate CI/CD workflows
/julia-package-ci
```

### Using Agents

Agents are invoked via the Task tool:

```python
# Use Julia pro for general Julia questions
Task(
    subagent_type="julia-development:julia-pro",
    prompt="How do I optimize this Julia function for type stability?"
)

# Use SciML expert for differential equations
Task(
    subagent_type="julia-development:sciml-pro",
    prompt="Set up a stochastic differential equation with callbacks..."
)

# Use Turing expert for Bayesian inference
Task(
    subagent_type="julia-development:turing-pro",
    prompt="Design a hierarchical Bayesian model for parameter estimation..."
)
```

## Cross-Plugin Integration

The marketplace plugins are designed to work together:

### Scientific Computing Workflows
- **julia-development** + **jax-implementation**: Hybrid Julia/JAX workflows
- **julia-development** + **python-development**: Multi-language scientific computing
- **julia-development** + **hpc-computing**: Large-scale HPC simulations
- **julia-development** + **deep-learning**: Neural differential equations

### Development Workflows
- **julia-development** + **unit-testing**: Comprehensive Julia testing
- **julia-development** + **cicd-automation**: Automated Julia CI/CD
- **julia-development** + **code-documentation**: Julia package documentation

### Research Workflows
- **julia-development** + **research-methodology**: Research project management
- **julia-development** + **data-visualization**: Scientific visualization
- **julia-development** + **llm-application-dev**: AI-powered research tools

## Customization

### Adding New Agents

1. Create agent file: `plugins/[plugin-name]/agents/my-agent.md`
2. Regenerate metadata: `./generate-metadata.sh`
3. Restart Claude Code

### Adding New Commands

1. Create command file: `plugins/[plugin-name]/commands/my-command.md`
2. Regenerate metadata: `./generate-metadata.sh`
3. Restart Claude Code

### Modifying Existing Plugins

All plugins are independent - modify freely:

```bash
# Edit plugin files directly
vim plugins/julia-development/agents/julia-pro.md

# Changes are git-tracked
git add plugins/julia-development/
git commit -m "Customize julia-pro agent"
```

## Maintenance

### Updating Plugins

```bash
# Commit changes
git add plugins/
git commit -m "Update plugin customizations"

# Create version tags
git tag -a v1.0.0 -m "Julia development plugin release"

# Push to remote (if configured)
git push origin main --tags
```

### Version Control

```bash
# Check plugin status
git status plugins/

# View plugin changes
git diff plugins/julia-development/

# Rollback changes if needed
git checkout -- plugins/julia-development/
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
ls -R plugins/julia-development/
```

### Command not found

```bash
# Verify plugin is installed
/plugin list

# Install plugin if needed
/plugin install julia-development

# Restart Claude Code
```

## Directory Structure

```
/Users/b80985/Projects/MyClaude/
├── .claude-plugin/
│   └── marketplace.json          # Marketplace metadata
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── LICENSE                        # MIT License
├── agent-os/                      # Product planning and specs
│   ├── product/                   # Product documentation
│   │   ├── mission.md
│   │   ├── roadmap.md
│   │   └── tech-stack.md
│   ├── specs/                     # Feature specifications
│   │   └── 2025-10-28-julia-development/
│   │       ├── spec.md
│   │       ├── tasks.md
│   │       ├── planning/
│   │       └── verifications/
│   └── standards/                 # Development standards
│       ├── global/
│       ├── frontend/
│       ├── backend/
│       └── testing/
├── plugins/                       # All 32 plugins
│   ├── julia-development/         # NEW: Julia ecosystem
│   │   ├── plugin.json
│   │   ├── README.md
│   │   ├── agents/                # 4 agents
│   │   ├── commands/              # 4 commands
│   │   └── skills/                # 21 skills
│   ├── jax-implementation/
│   ├── python-development/
│   ├── hpc-computing/
│   ├── deep-learning/
│   ├── machine-learning/
│   ├── ... (26 more plugins)
│   └── agent-orchestration/
└── scripts/                       # Utility scripts
    └── (setup and maintenance scripts)
```

## Contributing

### To This Marketplace

1. Fork and create a branch
2. Make changes to plugins
3. Test thoroughly with Claude Code
4. Submit PR with detailed description

### Creating New Plugins

1. Use existing plugin as template
2. Follow agent-os standards
3. Document agents, commands, and skills
4. Test integration with related plugins
5. Update README.md with plugin description

## License

MIT License (see [LICENSE](LICENSE))

## Acknowledgments

- Based on [claude-code-workflows](https://github.com/wshobson/agents) patterns
- julia-development plugin: Comprehensive Julia ecosystem support
- Integration with scientific computing community best practices
- Agent-os framework for systematic plugin development

## Version History

### v1.1.0 (Current) - Julia Development Release
- ✨ NEW: julia-development plugin with 4 agents, 4 commands, 21 skills
- 32 total plugins covering scientific computing to web development
- Full SciML ecosystem support (DifferentialEquations.jl, ModelingToolkit.jl, etc.)
- Bayesian inference with Turing.jl
- Julia package development lifecycle automation
- Cross-plugin integration patterns documented

### v1.0.0 (Previous)
- Initial marketplace with 31 plugins
- Scientific computing focus with JAX, Python, HPC
- Full-stack development support
- Quality engineering and testing tools

## Roadmap

See `agent-os/product/roadmap.md` for detailed development plans.

**Near-term priorities:**
- Enhance cross-plugin workflows
- Add more domain-specific skills
- Improve documentation and tutorials
- Community feedback integration

## Contact

For questions, issues, or suggestions:
- Create an issue in your repository
- Email: $(git config user.email 2>/dev/null || echo 'b80985@users.noreply.github.com')
