# Scientific Computing Workflows Marketplace

Comprehensive Claude Code marketplace with 31 specialized plugins for scientific computing, software development, and research workflows.

## Full Documentation

**[View Complete Plugin Documentation →](https://myclaude.readthedocs.io/en/latest/)**

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

[Browse all plugins →](https://myclaude.readthedocs.io/en/latest/plugins/)

## Quick Start

### Installation

1. Clone or download this repository to your local machine
2. Configure Claude Code to use this marketplace as your plugin source
3. All 31 plugins will be automatically available in Claude Code

```bash
# Verify plugin count
ls plugins/ | wc -l
# Should show: 31
```

### Using Plugins

Once installed, plugins provide agents, commands, and skills that are automatically available in Claude Code:

```bash
# Use an agent from a plugin
Ask Claude to use specialized agents like @python-pro or @rust-pro

# Run a command
/ultra-think "Analyze this architecture..."
/double-check "Validate this implementation"

# Access skills
Skills are automatically loaded and available to Claude
```

### Integration Examples

See [full documentation](https://myclaude.readthedocs.io/en/latest/guides/) for detailed workflow guides:
- Scientific computing (Julia + HPC + GPU)
- Development (Python + API + Testing)
- DevOps (Docker + Kubernetes + CI/CD)
- Infrastructure (Terraform + Cloud + Monitoring)

## Maintenance

```bash
# Commit changes
git add .
git commit -m "Update plugins and configurations"

# Create version tag
git tag -a v1.0.0 -m "Release v1.0.0"
```

## Contributing

1. Fork and create a branch
2. Make changes to plugins
3. Test thoroughly with Claude Code
4. Submit PR with detailed description

See [contribution guidelines](https://myclaude.readthedocs.io/en/latest/contributing.html) for details.

## Documentation Links

- **Full Documentation:** [myclaude.readthedocs.io](https://myclaude.readthedocs.io/en/latest/)
- **Plugin Guides:** [myclaude.readthedocs.io/plugins/](https://myclaude.readthedocs.io/en/latest/plugins/)
- **Quick-Start Guides:** [myclaude.readthedocs.io/guides/](https://myclaude.readthedocs.io/en/latest/guides/)
- **Integration Patterns:** [myclaude.readthedocs.io/integration-map.html](https://myclaude.readthedocs.io/en/latest/integration-map.html)
- **Technical Glossary:** [myclaude.readthedocs.io/glossary.html](https://myclaude.readthedocs.io/en/latest/glossary.html)

## Version History

### v1.0.0 (Current)
- Initial release of customized plugin marketplace
- 31 specialized plugins across 9 categories
- Comprehensive coverage for scientific computing and software development
- Includes HPC computing, Julia/JAX development, deep learning, and full-stack development
- Standardized documentation and author information
- 60+ agents, 40+ commands, 100+ skills

See [changelog](https://myclaude.readthedocs.io/en/latest/changelog.html) for detailed version history.

## License

MIT License (see [LICENSE](LICENSE))
