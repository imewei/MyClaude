# Scientific Computing Workflows Marketplace

Comprehensive Claude Code marketplace with 31 specialized plugins for scientific computing, software development, and research workflows.

## Full Documentation

**[View Complete Plugin Documentation →](https://docs.example.com)**

For comprehensive documentation including detailed plugin guides, integration patterns, quick-start tutorials, and API references, visit the full Sphinx documentation.

To build documentation locally:

```bash
cd docs/
make html
open _build/html/index.html
```

## Overview

This marketplace provides:
- **31 specialized plugins** across 9 categories
- **Comprehensive coverage** for scientific computing and software development
- **Independent modification** without affecting source plugins
- **Git-based version control** for all customizations
- **Integrated ecosystem** with cross-plugin collaboration

## Statistics

- **Total Plugins:** 31
- **Categories:** 9 (Scientific Computing, Development, DevOps, AI/ML, Tools, Orchestration, Quality)
- **Total Agents:** 60+ | **Commands:** 40+ | **Skills:** 100+

## Categories

- **Scientific Computing (2)**: HPC, molecular simulation, numerical methods
- **Development (7)**: Full-stack, backend, frontend/mobile, Python, JavaScript/TypeScript, CLI
- **DevOps (2)**: CI/CD automation, observability & monitoring
- **AI/ML (2)**: Deep learning, machine learning pipelines
- **Tools (14)**: Code quality, documentation, migration, testing, debugging
- **Orchestration (1)**: Multi-agent workflow coordination
- **Quality (1)**: Quality engineering and testing

## Featured Plugins

- **julia-development**: Comprehensive Julia ecosystem support with SciML
- **jax-implementation**: JAX optimization and physics-informed ML
- **python-development**: Modern Python patterns and async workflows
- **comprehensive-review**: Multi-perspective code analysis
- **agent-orchestration**: Multi-agent system optimization

[Browse all plugins →](https://docs.example.com/plugins/)

## Quick Start

### Installation

```bash
# Navigate to project directory
cd /Users/b80985/Projects/MyClaude

# Verify marketplace (should show 31 plugins)
ls plugins/ | wc -l
```

### Using Plugins

```bash
# List available plugins
/plugin list

# Install a plugin
/plugin install python-development

# Use an agent
@python-architect "Design a REST API..."

# Run a command
/ultra-think "Analyze this architecture..."
```

### Integration Examples

See [full documentation](https://docs.example.com/guides/) for detailed workflow guides:
- Scientific computing (Julia + HPC + GPU)
- Development (Python + API + Testing)
- DevOps (Docker + Kubernetes + CI/CD)
- Infrastructure (Terraform + Cloud + Monitoring)

## Customization

```bash
# Add new agent
vim plugins/[plugin-name]/agents/my-agent.md
./generate-metadata.sh

# Add new command
vim plugins/[plugin-name]/commands/my-command.md
./generate-metadata.sh
```

## Maintenance

```bash
# Commit changes
git add plugins/
git commit -m "Update plugin customizations"

# Create version tag
git tag -a v1.1.0 -m "Release v1.1.0"
```

## Contributing

1. Fork and create a branch
2. Make changes to plugins
3. Test thoroughly with Claude Code
4. Submit PR with detailed description

See [contribution guidelines](https://docs.example.com/contributing.html) for details.

## Documentation Links

- **Full Documentation:** [docs.example.com](https://docs.example.com)
- **Plugin Guides:** [docs.example.com/plugins/](https://docs.example.com/plugins/)
- **Quick-Start Guides:** [docs.example.com/guides/](https://docs.example.com/guides/)
- **Integration Patterns:** [docs.example.com/integration-map.html](https://docs.example.com/integration-map.html)
- **Technical Glossary:** [docs.example.com/glossary.html](https://docs.example.com/glossary.html)

## Version History

### v1.1.0 (Current) - Julia Development Release
- 31 total plugins covering scientific computing to web development
- Full SciML ecosystem support
- Bayesian inference with Turing.jl
- Cross-plugin integration patterns

### v1.0.0 (Previous)
- Initial marketplace with 31 plugins

See [changelog](https://docs.example.com/changelog.html) for detailed version history.

## License

MIT License (see [LICENSE](LICENSE))
