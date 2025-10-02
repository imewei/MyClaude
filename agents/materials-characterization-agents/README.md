# Materials Science Agent System

[![Status](https://img.shields.io/badge/status-Phase%202-blue)](docs/history/progress/PHASE2_PROGRESS.md)
[![Agents](https://img.shields.io/badge/agents-9%2F12-green)](#agent-status)
[![Tests](https://img.shields.io/badge/tests-374%2B%20passing-success)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](#testing)

A comprehensive multi-agent platform for materials characterization integrating experimental and computational workflows. Designed to accelerate materials discovery through systematic cross-validation and AI-driven optimization.

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run your first analysis
python -c "
from light_scattering_agent import LightScatteringAgent
agent = LightScatteringAgent()
result = agent.execute({'technique': 'DLS', 'parameters': {'temperature': 298, 'angle': 90}})
print(f'Particle size: {result.data[\"size_distribution\"][\"mean_diameter_nm\"]:.1f} nm')
"
```

📖 **New to the system?** Start with the [Quick Start Guide](docs/getting-started/quickstart.md)

## 📚 Documentation

### Getting Started
- **[Installation](docs/getting-started/installation.md)** - Set up your environment
- **[Quick Start](docs/getting-started/quickstart.md)** - 5-minute tutorial
- **[First Analysis](docs/getting-started/first-analysis.md)** - Step-by-step guide

### User Guides
- **[User Guide](docs/guides/user-guide.md)** - Comprehensive documentation
- **[Common Workflows](docs/guides/workflows.md)** - Practical examples
- **[CLI Reference](docs/guides/cli-reference.md)** - Command-line interface

### Architecture & Development
- **[Architecture Overview](docs/architecture/overview.md)** - System design
- **[Agent System](docs/architecture/agent-system.md)** - Agent patterns
- **[Contributing](docs/development/contributing.md)** - Join development
- **[Adding Agents](docs/development/adding-agents.md)** - Create new agents

### Project History
- **[Progress Reports](docs/history/progress/)** - Development timeline
- **[Implementation Details](docs/history/implementation/)** - Technical summaries
- **[Verification Reports](docs/history/verification/)** - Quality assurance

## 🤖 Agent Status

### ✅ Phase 1: Core Agents (Complete)
1. **Light Scattering & Optical Expert** - DLS, SLS, Raman, 3D-DLS, multi-speckle
2. **Rheologist** - Mechanical properties, viscoelasticity, extensional rheology
3. **Simulation Expert** - MD, MLFFs, HOOMD-blue, DPD
4. **DFT Expert** - Electronic structure, phonons, AIMD, high-throughput
5. **Electron Microscopy** - TEM/SEM/STEM, EELS, 4D-STEM, cryo-EM

### ✅ Phase 1.5: Soft Matter Focus (Complete)
6. **X-ray Expert** - SAXS/WAXS/GISAXS/RSoXS/XPCS/XAS
7. **Neutron Expert** - SANS/NSE/QENS/NR/INS

### ✅ Phase 2: Enhancement (9/10 Complete)
8. **Spectroscopy Expert** - FTIR/NMR/EPR, BDS, EIS, THz
9. **Crystallography Expert** - XRD powder/single crystal, PDF, Rietveld
10. **Characterization Master** - 🔧 Multi-technique coordinator (in progress)

### 📋 Phase 3: Advanced (Planned)
11. **Materials Informatics** - GNNs, active learning, Bayesian optimization
12. **Surface Science** - QCM-D, SPR, contact angle, adsorption

## 💡 Key Features

- **🔄 Cross-Validation** - Validate results across multiple techniques
- **🧩 Modular Design** - Easy to extend with new agents
- **⚡ Smart Caching** - Avoid redundant calculations
- **📊 Rich Metadata** - Full provenance tracking
- **🎯 Resource-Aware** - Intelligent HPC/GPU allocation
- **🔧 CLI & API** - Flexible interfaces for all users

## 🔬 Example Workflows

### Nanoparticle Size Analysis
```python
from characterization_master import CharacterizationMaster

master = CharacterizationMaster()
result = master.execute({
    'workflow_type': 'nanoparticle_analysis',
    'sample_info': {'type': 'gold_nanoparticles'}
})
```

### Polymer Rheology
```python
from rheologist_agent import RheologistAgent

agent = RheologistAgent()
result = agent.execute({
    'mode': 'oscillatory',
    'frequency_range': [0.1, 100],
    'strain': 0.01
})
```

See [Common Workflows](docs/guides/workflows.md) for more examples.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ --cov=. --cov-report=html

# Run specific agent
pytest tests/test_light_scattering_agent.py -v

# Current status
✅ 374+ tests passing (100%)
✅ 97% materials characterization coverage
```

## 📈 Project Status

**Version**: 1.2.0-beta
**Phase**: 2 (Enhancement) - Week 12
**Agents**: 9/12 operational
**Tests**: 374+ passing (100%)
**Coverage**: 97% characterization capability
**Next**: Characterization Master Agent completion

## 🤝 Contributing

We welcome contributions! See [Contributing Guide](docs/development/contributing.md) for:
- How to add new agents
- Code standards and testing requirements
- Development workflow

## 📄 License

[Your License Here - e.g., MIT, Apache 2.0]

## 📞 Contact & Support

- **Issues**: Report bugs at [repository-url]/issues
- **Discussions**: Ask questions at [repository-url]/discussions
- **Email**: your-email@example.com

## 🙏 Acknowledgments

Built on established materials characterization methods and community contributions.

---

**Last Updated**: 2025-09-30 | [View Change History](docs/history/progress/)
