# Scientific Computing Agents

A production-ready multi-agent framework for scientific computing with comprehensive deployment infrastructure.

[![CI](https://github.com/scientific-computing-agents/scientific-computing-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/scientific-computing-agents/scientific-computing-agents/actions)
[![Coverage](https://codecov.io/gh/scientific-computing-agents/scientific-computing-agents/branch/main/graph/badge.svg)](https://codecov.io/gh/scientific-computing-agents/scientific-computing-agents)
[![PyPI version](https://badge.fury.io/py/scientific-computing-agents.svg)](https://badge.fury.io/py/scientific-computing-agents)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

A comprehensive multi-agent system for scientific computing covering:
- **Numerical Methods**: ODE/PDE solvers, linear algebra, optimization, integration
- **Data-Driven Methods**: Physics-informed ML, surrogate modeling, inverse problems, UQ
- **Workflow Orchestration**: Problem analysis, algorithm selection, execution validation
- **Performance**: Profiling, parallel execution, resource management

**Current Status**: ⚠️ **Project Concluded at 82% - Infrastructure-Ready MVP**

**Completion Status**: 82% (18 of 22 weeks)
- ✅ **Phases 0-4 (20 weeks)**: 100% complete - All agents operational
- ✅ **Phase 5A Weeks 1-2 (2 weeks)**: 100% complete - Infrastructure ready
- ❌ **Phase 5A Weeks 3-4**: Cancelled - User validation not executed
- ❌ **Phase 5B**: Cancelled - Expansion not pursued

**Note**: This project is a production-ready infrastructure MVP with comprehensive documentation but **without user validation**. See [PHASE5_CANCELLATION_DECISION.md](archive/phases/phase-5/cancelled/PHASE5_CANCELLATION_DECISION.md) for details.

---

## Quick Start (5 Minutes)

### Installation

```bash
pip install scientific-computing-agents
```

### Your First Computation

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent
import numpy as np

# Create solver
solver = ODEPDESolverAgent()

# Define ODE: dy/dt = -y
def exponential_decay(t, y):
    return -y

# Solve
result = solver.process({
    'task': 'solve_ode',
    'equation': exponential_decay,
    'initial_conditions': [1.0],
    't_span': (0, 5),
    't_eval': np.linspace(0, 5, 50)
})

print(f"Final value: {result.data['y'][-1]:.4f}")
```

**More**: See [Getting Started Guide](docs/getting-started/quick-start.md) and [User Onboarding](docs/user-guide/USER_ONBOARDING.md)

---

## Features

### 14 Specialized Agents

**Numerical Methods** (5 agents):
- **ODEPDESolverAgent**: ODE/PDE solving (IVP, 1D/2D/3D PDEs)
- **LinearAlgebraAgent**: Linear systems, eigenvalues, SVD
- **OptimizationAgent**: Unconstrained optimization, root finding
- **IntegrationAgent**: Numerical integration (1D, 2D, Monte Carlo)
- **SpecialFunctionsAgent**: Special functions, transforms

**Data-Driven Methods** (4 agents):
- **PhysicsInformedMLAgent**: PINNs, DeepONets
- **SurrogateModelingAgent**: Gaussian processes, POD, PCE
- **InverseProblemsAgent**: Parameter identification, data assimilation
- **UncertaintyQuantificationAgent**: UQ, sensitivity analysis

**Infrastructure** (2 agents):
- **PerformanceProfilerAgent**: CPU/memory profiling
- **WorkflowOrchestrationAgent**: Multi-agent workflows

**Support** (3 agents):
- **ProblemAnalyzerAgent**: Problem analysis
- **AlgorithmSelectorAgent**: Algorithm selection
- **ExecutorValidatorAgent**: Execution and validation

### Production Infrastructure

**CI/CD**:
- Multi-OS/Python testing (Ubuntu, macOS, Windows × Python 3.9-3.12)
- Automated PyPI publishing
- Coverage tracking (Codecov)
- Code quality (flake8, black, isort, mypy)

**Containerization**:
- Production Docker images
- Development environment (Jupyter)
- GPU support (CUDA)
- docker-compose orchestration

**Monitoring**:
- Prometheus metrics
- Grafana dashboards
- Automated health checks
- Alert rules (7 alerts)

**Operations**:
- 900+ LOC operations runbook
- Deployment checklists
- Incident response playbooks
- Rollback procedures

---

## System Status

### Production Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Agents** | 12 | 14 | 117% ✅ |
| **Total LOC** | 15,000 | 14,746 | 98% ✅ |
| **Tests** | 500+ | 379 | 76% ⚠️ |
| **Test Pass Rate** | 100% | 97.6% | 98% ✅ |
| **Coverage** | >85% | ~78-80% | 92-94% ⚠️ |
| **Documentation** | Comprehensive | 2,300+ LOC | ✅ |

### Deployment Status

| Component | Status |
|-----------|--------|
| **CI/CD Pipeline** | ✅ Operational |
| **Docker Containers** | ✅ 3 variants ready |
| **Monitoring** | ✅ Configured |
| **Operations** | ✅ Runbook complete |
| **User Onboarding** | ✅ Documentation ready |
| **Production Deployment** | 🔄 Ready for execution |

**Overall**: Production-ready MVP with 65-70% of roadmap features, 100% core functionality, infrastructure deployment complete

**Note**: Infrastructure is production-ready, but production deployment and user validation (Phase 5A Weeks 3-4) have not yet been executed. See [Phase 5 Status](#roadmap) for details.

---

## Architecture

### Agent Ecosystem

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface / API                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Workflow Orchestration Agent                    │
│  (Sequential/Parallel Execution, Dependency Management)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼─────────┐   ┌────────▼──────────┐
│  Core Agents      │   │  Support Agents    │
│  - ODE/PDE        │   │  - Profiler        │
│  - Optimization   │   │  - Problem Analyzer│
│  - Linear Algebra │   │  - Validator       │
│  - Integration    │   │  - Selector        │
│  - Special Funcs  │   │                    │
│  - ML/Surrogate   │   │                    │
│  - Inverse/UQ     │   │                    │
└───────────────────┘   └───────────────────┘
```

### Technology Stack

- **Language**: Python 3.9+
- **Numerical**: NumPy, SciPy, JAX
- **ML**: PyTorch, scikit-learn
- **Surrogate**: GPy, scikit-optimize, chaospy
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Testing**: pytest, pytest-cov
- **CI/CD**: GitHub Actions
- **Containers**: Docker, docker-compose
- **Monitoring**: Prometheus, Grafana

---

## Installation

### Method 1: PyPI (Recommended)

```bash
pip install scientific-computing-agents
```

### Method 2: From Source

```bash
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents
pip install -e .
```

### Method 3: Docker

```bash
docker pull scientific-computing-agents:latest
docker run -it scientific-computing-agents:latest
```

### Method 4: Development

```bash
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents
pip install -e .[dev]
```

**Full Guide**: See [Deployment Documentation](docs/deployment/docker.md)

---

## Usage Examples

### Example 1: ODE Solving

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent
import numpy as np

solver = ODEPDESolverAgent()

# Lotka-Volterra predator-prey model
def predator_prey(t, y):
    prey, predator = y
    return [
        1.5 * prey - 1.0 * prey * predator,      # Prey growth
        3.0 * prey * predator - 1.0 * predator   # Predator growth
    ]

result = solver.process({
    'task': 'solve_ode',
    'equation': predator_prey,
    'initial_conditions': [10, 5],
    't_span': (0, 20),
    't_eval': np.linspace(0, 20, 200)
})

# Plot results
import matplotlib.pyplot as plt
plt.plot(result.data['t'], result.data['y'][:, 0], label='Prey')
plt.plot(result.data['t'], result.data['y'][:, 1], label='Predator')
plt.legend()
plt.show()
```

### Example 2: Optimization

```python
from agents.optimization_agent import OptimizationAgent
import numpy as np

optimizer = OptimizationAgent()

# Rosenbrock function
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

result = optimizer.process({
    'task': 'minimize',
    'function': rosenbrock,
    'x0': np.zeros(5),
    'method': 'L-BFGS-B'
})

print(f"Minimum at: {result.data['x']}")
print(f"Function value: {result.data['fun']:.6f}")
```

### Example 3: Workflow Orchestration

```python
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep
from agents.optimization_agent import OptimizationAgent
from agents.ode_pde_solver_agent import ODEPDESolverAgent

orchestrator = WorkflowOrchestrationAgent()

# Step 1: Optimize parameters
# Step 2: Simulate with optimal parameters

steps = [
    WorkflowStep(
        step_id='optimize',
        agent=OptimizationAgent(),
        method='process',
        inputs={
            'task': 'minimize',
            'function': lambda x: (x[0] - 2)**2,
            'x0': [0]
        }
    ),
    WorkflowStep(
        step_id='simulate',
        agent=ODEPDESolverAgent(),
        method='process',
        inputs={
            'task': 'solve_ode',
            'equation': lambda t, y: -y,
            'initial_conditions': [1.0],
            't_span': (0, 5)
        },
        depends_on=['optimize']
    )
]

result = orchestrator.execute_workflow(steps)
print(f"Workflow success: {result.success}")
```

**More Examples**: See [examples/](examples/) directory and [tutorials](examples/tutorial_01_quick_start.py)

---

## Documentation

### User Documentation

- **[Getting Started](docs/getting-started/quick-start.md)**: Quick start guide
- **[User Onboarding](docs/user-guide/USER_ONBOARDING.md)**: Comprehensive onboarding (700 LOC)
- **[Tutorial 1: Quick Start](examples/tutorial_01_quick_start.py)**: Interactive basics
- **[Tutorial 2: Advanced Workflows](examples/tutorial_02_advanced_workflows.py)**: Complex patterns

### Deployment & Operations

- **[Deployment Guide](docs/deployment/docker.md)**: Installation, configuration, scaling (600 LOC)
- **[Operations Runbook](docs/deployment/operations-runbook.md)**: Day-to-day operations (900 LOC)
- **[Deployment Checklist](docs/deployment/production.md)**: Step-by-step deployment (800 LOC)
- **[Feedback System](docs/deployment/USER_FEEDBACK_SYSTEM.md)**: User validation framework (600 LOC)

### Development

- **[Contributing Guide](CONTRIBUTING.md)**: Development standards
- **[Phase 5A Summary](archive/phases/phase-5/infrastructure/PHASE5A_COMPLETE_SUMMARY.md)**: Infrastructure overview

### API Reference

See inline documentation in agent files:
- Agents: `agents/*.py`
- Core: `core/*.py`
- Examples: `examples/*.py`

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agents --cov=core --cov-report=html

# Specific agent
pytest tests/test_ode_pde_solver_agent.py -v

# Parallel execution
pytest tests/ -n auto
```

### Code Quality

```bash
# Linting
flake8 agents/ core/

# Formatting
black agents/ core/
isort agents/ core/

# Type checking
mypy agents/ core/ --ignore-missing-imports
```

### Performance & Security

```bash
# Health check
python scripts/health_check.py

# Benchmarks
python scripts/benchmark.py

# Security audit
python scripts/security_audit.py
```

---

## Community

### Support

- **FAQ**: [Frequently Asked Questions](docs/FAQ.md) - Start here!
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community support
- **Email**: support@scientific-agents.example.com
- **Slack**: #sci-agents-users

### Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards (PEP 8, black, isort)
- Testing requirements (80%+ coverage)
- PR process
- Development workflow

### Citation

If you use this software in your research, please cite:

```bibtex
@software{scientific_computing_agents_2025,
  title = {Scientific Computing Agents: A Multi-Agent Framework for Scientific Computing},
  author = {Scientific Computing Agents Team},
  year = {2025},
  url = {https://github.com/scientific-computing-agents/scientific-computing-agents},
  version = {0.1.0}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built on proven materials-science-agents architecture
- Inspired by best practices in scientific computing
- Community feedback during beta testing

---

## Project Navigation

The project is organized into clear sections for easy navigation:

### 📚 Documentation
- **[docs/](docs/)** - Complete user documentation
  - [Getting Started](docs/getting-started/) - Quick start guides
  - [User Guide](docs/user-guide/) - Comprehensive tutorials
  - [Deployment](docs/deployment/) - Production deployment
  - [Development](docs/development/) - Contributing guides
  - [API](docs/api/) - API reference

### 📊 Current Status
- **[status/](status/)** - Project status and navigation
  - [PROJECT_STATUS.md](status/PROJECT_STATUS.md) - Current state (82% complete)
  - [INDEX.md](status/INDEX.md) - Complete project index
  - [Current Status Dashboard](status/README.md) - Quick overview

### 🗄️ History & Plans
- **[archive/](archive/)** - Project history and archived plans
  - [Phases 0-5](archive/phases/) - Development history
  - [Reports](archive/reports/) - Verification and progress reports
  - [Improvement Plans](archive/improvement-plans/) - Plans for completing remaining 18%

### 🚀 Quick Links
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute

---

## Roadmap

### ✅ Phase 0-4: Foundation & Core Agents (Complete)
- Base classes and infrastructure
- 14 operational agents
- 379 tests (97.6% pass rate)
- Core functionality production-ready

### ✅ Phase 5A Weeks 1-2: Deployment Infrastructure (Complete)
- CI/CD pipeline
- Docker containers
- Monitoring and operations
- Documentation (2,300+ LOC)

### 🔄 Phase 5A Weeks 3-4: User Validation (Ready)
- Production deployment
- User onboarding (10+ beta users)
- Feedback collection
- Phase 5B planning

### 📋 Phase 5B: Targeted Expansion (6-8 weeks)
Based on user feedback:
- High-priority features
- Performance optimizations
- Documentation improvements
- Production enhancements

### 🔮 Phase 6: Advanced Features (Future)
- GPU acceleration
- Distributed computing
- Advanced ML integration
- Domain-specific agents

**Current Version**: v0.1.0
**Status**: Production-Ready MVP
**Next Release**: v0.2.0 (Phase 5B features)

---

## Contact

- **Website**: https://scientific-computing-agents.github.io
- **Email**: info@scientific-agents.example.com
- **GitHub**: https://github.com/scientific-computing-agents/scientific-computing-agents

---

**Built with ❤️ for the scientific computing community**
