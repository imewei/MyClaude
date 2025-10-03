# Quick Start Guide

Get started with Scientific Computing Agents in 5 minutes!

---

## Installation (1 minute)

```bash
# Install from PyPI
pip install scientific-computing-agents

# Or install from source
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents
pip install -e .
```

---

## Your First Computation (3 minutes)

### Example 1: Solve an ODE

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent
import numpy as np

# Create solver
solver = ODEPDESolverAgent()

# Define ODE: dy/dt = -y (exponential decay)
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

print(f"Final value: {result.data['y'][-1]:.4f}")  # Expected: ~0.0067
```

### Example 2: Optimize a Function

```python
from agents.optimization_agent import OptimizationAgent
import numpy as np

# Create optimizer
optimizer = OptimizationAgent()

# Define function to minimize: f(x) = (x - 2)^2
def quadratic(x):
    return (x[0] - 2)**2

# Minimize
result = optimizer.process({
    'task': 'minimize',
    'function': quadratic,
    'x0': [0.0]
})

print(f"Minimum at x = {result.data['x'][0]:.4f}")  # Expected: 2.0000
```

### Example 3: Linear Algebra

```python
from agents.linear_algebra_agent import LinearAlgebraAgent
import numpy as np

# Create agent
linalg = LinearAlgebraAgent()

# Solve linear system Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

result = linalg.process({
    'task': 'solve_linear_system',
    'A': A,
    'b': b
})

print(f"Solution: x = {result.data['x']}")  # Expected: [2., 3.]
```

---

## What's Next? (1 minute)

### Learn More
- **[User Guide](docs/user-guide/)** - Learn about all 14 agents
- **[Examples](examples/)** - 40+ working examples
- **[Tutorials](examples/tutorial_01_quick_start.py)** - Interactive learning

### Deploy to Production
- **[Docker Guide](docs/deployment/docker.md)** - Containerize your application
- **[Production Guide](docs/deployment/production.md)** - Deploy for real
- **[Operations](docs/deployment/operations-runbook.md)** - Run in production

### Contribute
- **[Contributing Guide](CONTRIBUTING.md)** - Join the project
- **[Architecture](docs/README.md)** - Understand the design
- **[Complete the Project](archive/improvement-plans/)** - Help finish the remaining 18%

---

## Available Agents

**Numerical Methods** (5):
- `ODEPDESolverAgent` - Solve differential equations
- `LinearAlgebraAgent` - Linear systems, eigenvalues, SVD
- `OptimizationAgent` - Minimize/maximize functions
- `IntegrationAgent` - Numerical integration
- `SpecialFunctionsAgent` - Special functions, transforms

**Data-Driven** (4):
- `PhysicsInformedMLAgent` - Physics-informed neural networks
- `SurrogateModelingAgent` - Gaussian processes, POD
- `InverseProblemsAgent` - Parameter identification
- `UncertaintyQuantificationAgent` - UQ, sensitivity analysis

**Infrastructure** (2):
- `PerformanceProfilerAgent` - Profile CPU/memory
- `WorkflowOrchestrationAgent` - Multi-agent workflows

**Support** (3):
- `ProblemAnalyzerAgent` - Analyze problems
- `AlgorithmSelectorAgent` - Select algorithms
- `ExecutorValidatorAgent` - Validate results

---

## Common Patterns

### Pattern 1: Workflow Orchestration

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

### Pattern 2: Error Handling

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent

solver = ODEPDESolverAgent()
result = solver.process({...})

if result.success:
    print(f"Success! Result: {result.data}")
else:
    print(f"Failed: {result.error}")
    print(f"Message: {result.message}")
```

### Pattern 3: Performance Profiling

```python
from agents.performance_profiler_agent import PerformanceProfilerAgent

profiler = PerformanceProfilerAgent()

# Profile a computation
result = profiler.process({
    'task': 'profile_function',
    'function': my_computation,
    'args': (arg1, arg2)
})

print(f"Execution time: {result.data['execution_time']:.4f}s")
print(f"Memory used: {result.data['memory_used_mb']:.2f} MB")
```

---

## Project Status

⚠️ **Note**: This project is 82% complete (infrastructure-ready MVP, not user-validated)

**What Works**:
- ✅ All 14 agents operational
- ✅ 379 tests (97.6% pass rate)
- ✅ Complete CI/CD infrastructure
- ✅ Production-ready containers
- ✅ Comprehensive documentation

**What's Missing**:
- ❌ User validation (0 real users tested it)
- ❌ Production deployment (not hosted anywhere)
- ❌ Real-world use cases (no feedback data)

**Can I Use It?** YES - if you're comfortable self-deploying. All infrastructure is ready.

**See**: [Project Status](status/PROJECT_STATUS.md) for details

---

## Getting Help

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Email**: support@scientific-agents.example.com

---

## License

MIT License - See [LICENSE](LICENSE)

---

**Ready to dive deeper?** See the [User Guide](docs/user-guide/) for comprehensive tutorials!
