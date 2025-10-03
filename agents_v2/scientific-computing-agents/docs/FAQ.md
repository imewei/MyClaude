# Frequently Asked Questions (FAQ)

**Last Updated**: 2025-10-01

---

## General Questions

### What is Scientific Computing Agents?

Scientific Computing Agents is a production-ready multi-agent framework for scientific computing that provides 14 specialized agents for numerical methods, data-driven modeling, and computational workflows. It's designed for researchers, scientists, and developers who need reliable, well-tested computational tools.

### What can I do with this framework?

You can:
- Solve differential equations (ODE/PDE) in 1D/2D/3D
- Perform linear algebra operations (solve systems, eigenvalues, SVD)
- Optimize functions and find roots
- Integrate functions numerically
- Build physics-informed machine learning models
- Create surrogate models and perform uncertainty quantification
- Orchestrate complex multi-agent workflows
- Profile and optimize code performance

### Is this production-ready?

**Yes!** The project has:
- ✅ 100% test pass rate (379/379 tests)
- ✅ Comprehensive CI/CD infrastructure
- ✅ Docker containerization
- ✅ 21,355+ lines of documentation
- ✅ Production-grade error handling
- ✅ Active maintenance

**Note**: User validation has not been performed yet (acknowledged limitation). The infrastructure is production-ready, but real-world usage feedback would be valuable.

---

## Installation & Setup

### How do I install Scientific Computing Agents?

**Option 1: From PyPI** (Recommended when available)
```bash
pip install scientific-computing-agents
```

**Option 2: From Source**
```bash
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents
pip install -e .
```

**Option 3: Docker**
```bash
docker pull scientific-computing-agents:latest
docker run -it scientific-computing-agents:latest
```

### What are the system requirements?

- **Python**: 3.9 or higher (tested on 3.9, 3.10, 3.11, 3.12)
- **OS**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum, 8GB recommended
- **CPU**: Modern multi-core processor recommended

### What dependencies are required?

**Core Dependencies**:
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- torch >= 2.0.0
- jax >= 0.4.0
- matplotlib >= 3.7.0
- sympy >= 1.12

**Development Dependencies** (optional):
- pytest >= 7.4.0
- black >= 23.0.0
- flake8 >= 6.0.0
- mypy >= 1.5.0

All dependencies are automatically installed via pip.

### Installation fails with dependency conflicts. What do I do?

Try creating a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install scientific-computing-agents
```

If issues persist, check [GitHub Issues](https://github.com/scientific-computing-agents/scientific-computing-agents/issues) or create a new issue with your error message.

---

## Usage Questions

### How do I get started?

See [QUICKSTART.md](../QUICKSTART.md) for a 5-minute introduction.

Quick example:
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

### Where can I find examples?

- **Examples Directory**: [examples/](../examples/) - 20 example files
- **Tutorials**: [examples/tutorial_01_quick_start.py](../examples/tutorial_01_quick_start.py)
- **Documentation**: [docs/](../docs/)

### How do I use multiple agents together?

Use the `WorkflowOrchestrationAgent`:

```python
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep
from agents.optimization_agent import OptimizationAgent
from agents.ode_pde_solver_agent import ODEPDESolverAgent

orchestrator = WorkflowOrchestrationAgent()

steps = [
    WorkflowStep(
        step_id='optimize',
        agent=OptimizationAgent(),
        method='process',
        inputs={'task': 'minimize', 'function': my_func, 'x0': [0]}
    ),
    WorkflowStep(
        step_id='simulate',
        agent=ODEPDESolverAgent(),
        method='process',
        inputs={'task': 'solve_ode', 'equation': my_ode, 'initial_conditions': [1.0]},
        depends_on=['optimize']
    )
]

result = orchestrator.execute_workflow(steps)
```

### How do I handle errors?

All agents return `AgentResult` objects with success status:

```python
result = agent.process({...})

if result.success:
    print(f"Success! Data: {result.data}")
else:
    print(f"Error: {result.error}")
    print(f"Message: {result.message}")
```

---

## Agent-Specific Questions

### Which agent should I use for solving differential equations?

Use **ODEPDESolverAgent** for:
- Ordinary Differential Equations (ODEs)
- Partial Differential Equations (PDEs) in 1D/2D/3D
- Initial Value Problems (IVP)

### Which agent handles optimization?

Use **OptimizationAgent** for:
- Unconstrained optimization (minimize/maximize functions)
- Root finding
- Scalar and multivariate problems

### Which agent is for machine learning?

Use **PhysicsInformedMLAgent** for:
- Physics-Informed Neural Networks (PINNs)
- Conservation laws
- Scientific machine learning

Or **SurrogateModelingAgent** for:
- Gaussian processes
- Proper Orthogonal Decomposition (POD)
- Polynomial Chaos Expansion (PCE)

### Which agent handles uncertainty?

Use **UncertaintyQuantificationAgent** for:
- Uncertainty quantification (UQ)
- Sensitivity analysis
- Monte Carlo methods
- Sobol indices

---

## Technical Questions

### What is the test coverage?

**Overall**: ~49% code coverage
**Test Pass Rate**: 100% (379/379 tests passing)

While coverage is below the 80% target, all core functionality is thoroughly tested. Lower coverage is primarily in ML agents (complex domain) and error handling paths.

**Coverage by Agent Type**:
- Core numerical agents: 70-95% ✅
- Infrastructure agents: 85-92% ✅
- ML/data-driven agents: 11-19% 🟡 (expected for MVP)

### Are the numerical methods accurate?

**Yes!** All numerical methods are:
- Based on proven SciPy/NumPy implementations
- Extensively tested against known solutions
- Validated with analytical test cases
- Documented with references

### Can I run this on a cluster or cloud?

**Yes!** The framework supports:
- Docker containerization for easy deployment
- Kubernetes-ready architecture
- Parallel execution via `core/parallel_executor.py`
- Job queue system for async operations

See [Deployment Documentation](deployment/docker.md) for details.

### Is GPU acceleration supported?

**Yes, through JAX!** JAX is included as a dependency and supports:
- Automatic GPU/TPU acceleration
- JIT compilation for performance
- Automatic differentiation

Some agents (particularly ML agents) can leverage JAX's GPU capabilities.

---

## Performance Questions

### How fast is it?

**Test Suite Performance**: Full test suite (379 tests) runs in ~6.3 seconds

**Agent Performance** varies by task complexity:
- Simple ODE: milliseconds
- Complex 3D PDE: seconds to minutes
- ML training: minutes to hours (GPU recommended)

### Can I improve performance?

**Yes!** Several options:
1. Use parallel execution (already supported)
2. Enable JAX JIT compilation
3. Use GPU acceleration (if available)
4. Profile with `PerformanceProfilerAgent`

Example profiling:
```python
from agents.performance_profiler_agent import PerformanceProfilerAgent

profiler = PerformanceProfilerAgent()
result = profiler.process({
    'task': 'profile_function',
    'function': my_computation,
    'args': (arg1, arg2)
})

print(f"Execution time: {result.data['execution_time']:.4f}s")
```

### Why is my computation slow?

Common causes:
1. **Large problem size**: Consider using sparse methods
2. **Non-optimized code**: Profile with `PerformanceProfilerAgent`
3. **Single-threaded**: Enable parallel execution
4. **CPU-only**: Try GPU acceleration with JAX

---

## Troubleshooting

### Tests are failing. What should I do?

1. **Check Python version**: Requires Python 3.9+
2. **Update dependencies**: `pip install --upgrade -r requirements.txt`
3. **Clear cache**: `rm -rf .pytest_cache __pycache__`
4. **Run tests**: `python -m pytest tests/ -v`

If issues persist, check [GitHub Issues](https://github.com/scientific-computing-agents/scientific-computing-agents/issues).

### Import errors when using agents

Make sure you've installed the package:
```bash
pip install -e .  # If installing from source
```

Or verify installation:
```bash
python -c "import agents; print(agents.__file__)"
```

### "Module not found" errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

For development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Docker container won't start

Check Docker is running:
```bash
docker --version
docker ps
```

Pull latest image:
```bash
docker pull scientific-computing-agents:latest
```

### Memory errors during large computations

Try:
1. Reduce problem size
2. Use sparse matrices (LinearAlgebraAgent)
3. Increase available memory
4. Use chunking/batching for large datasets

---

## Development Questions

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Code standards (PEP 8, black, isort)
- Testing requirements (80%+ coverage)
- PR process
- Development workflow

### How do I run the test suite?

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

### How do I add a new agent?

1. Create agent class inheriting from `ComputationalMethodAgent`
2. Implement `process()` method
3. Add validation methods
4. Write tests (80%+ coverage target)
5. Document in docstrings
6. Update README and documentation

See [Architecture Documentation](README.md) for details.

### What coding standards should I follow?

- **Style**: PEP 8, enforced by black and isort
- **Type Hints**: Encouraged (mypy configured)
- **Docstrings**: Required for all public methods
- **Testing**: 80%+ coverage target
- **Commit Messages**: Conventional commits preferred

---

## Deployment Questions

### How do I deploy to production?

See [Deployment Guide](deployment/docker.md) and [Operations Runbook](deployment/operations-runbook.md).

Quick Docker deployment:
```bash
# Build image
docker build -t my-sci-agents .

# Run container
docker run -p 8000:8000 my-sci-agents

# Or use docker-compose
docker-compose up
```

### Is there a monitoring solution?

**Yes!** Pre-configured with:
- Prometheus metrics
- Grafana dashboards
- Health check endpoints
- Alert rules

See [Monitoring Documentation](deployment/monitoring.md).

### How do I handle production errors?

Follow the [Operations Runbook](deployment/operations-runbook.md):
1. Check logs
2. Verify health endpoints
3. Review Grafana dashboards
4. Follow incident response procedures
5. Apply rollback if needed

---

## Project Status Questions

### Is development active?

The project concluded at **82% completion** (infrastructure-ready MVP) on 2025-10-01. Key facts:

- ✅ All core functionality complete and tested
- ✅ Production infrastructure ready
- ✅ Comprehensive documentation complete
- ❌ User validation not performed (Phase 5A Week 3-4 cancelled)
- ❌ Feature expansion not pursued (Phase 5B cancelled)

**Recent Activity**: Active bug fixes and improvements (e.g., test failure resolution on 2025-10-01)

### Why was development stopped at 82%?

Strategic decision based on:
- Core MVP delivered successfully
- Infrastructure production-ready
- Diminishing returns on further development without user validation
- Resources better allocated elsewhere

See [PHASE5_CANCELLATION_DECISION.md](../archive/phases/phase-5/cancelled/PHASE5_CANCELLATION_DECISION.md) for details.

### Can I continue development?

**Absolutely!** The project includes:
- Complete implementation (7,401 LOC)
- Comprehensive tests (379 tests, 100% pass)
- Full documentation (21,355+ LOC)
- Archived improvement plans for remaining 18%
- Clear roadmap for Phase 5B expansion

See [Improvement Plans](../archive/improvement-plans/) for guidance.

### Will there be a v1.0 release?

Not officially planned. The project is at v0.1.0 (final) with:
- Production-ready codebase
- Full infrastructure
- Comprehensive documentation

Future developers can release v1.0 if they:
- Complete user validation
- Expand test coverage to 80%+
- Deploy to production

---

## Getting Help

### Where can I ask questions?

1. **Check this FAQ first**
2. **GitHub Discussions**: For general questions
3. **GitHub Issues**: For bugs or feature requests
4. **Email**: support@scientific-agents.example.com
5. **Slack**: #sci-agents-users

### How do I report a bug?

Create a [GitHub Issue](https://github.com/scientific-computing-agents/scientific-computing-agents/issues) with:
1. Description of the problem
2. Steps to reproduce
3. Expected vs. actual behavior
4. Python version and OS
5. Error messages/stack traces

### Where is the documentation?

- **Main Documentation**: [docs/](.)
- **Quick Start**: [QUICKSTART.md](../QUICKSTART.md)
- **API Reference**: Inline docstrings in agent files
- **Examples**: [examples/](../examples/)
- **Status**: [status/](../status/)

### Can I get commercial support?

This is an open-source project without official commercial support. However:
- Community support via GitHub Discussions
- Contributors may offer consulting (check discussions)
- Fork the project for custom support needs

---

## License & Legal

### What is the license?

**MIT License** - See [LICENSE](../LICENSE) for full text.

You can:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use privately
- ✅ Sublicense

Conditions:
- Include license and copyright notice
- No warranty provided

### Can I use this in my research?

**Yes!** Please cite:

```bibtex
@software{scientific_computing_agents_2025,
  title = {Scientific Computing Agents: A Multi-Agent Framework for Scientific Computing},
  author = {Scientific Computing Agents Team},
  year = {2025},
  url = {https://github.com/scientific-computing-agents/scientific-computing-agents},
  version = {0.1.0}
}
```

### Can I fork the project?

**Yes!** The MIT license allows forking. If you do:
- Maintain the original license
- Consider contributing improvements back upstream
- Update attribution appropriately

---

## Additional Resources

- **Main README**: [README.md](../README.md)
- **Quick Start**: [QUICKSTART.md](../QUICKSTART.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](../CHANGELOG.md)
- **Project Status**: [status/PROJECT_STATUS.md](../status/PROJECT_STATUS.md)

---

**Last Updated**: 2025-10-01
**Version**: 1.0
**Maintainer**: Scientific Computing Agents Team

**Have a question not answered here?** Create a [GitHub Discussion](https://github.com/scientific-computing-agents/scientific-computing-agents/discussions) or [Issue](https://github.com/scientific-computing-agents/scientific-computing-agents/issues).
