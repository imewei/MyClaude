# Getting Started with Scientific Computing Agents

**Version**: 1.0
**Last Updated**: 2025-09-30

Welcome! This guide will help you get started with the Scientific Computing Agents system in under 10 minutes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start (5 minutes)](#quick-start)
4. [Your First Workflow](#your-first-workflow)
5. [What's Next](#whats-next)

---

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **OS**: Linux, macOS, or Windows
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB for installation

### Knowledge Prerequisites
- Basic Python programming
- Familiarity with scientific computing concepts (helpful but not required)
- Understanding of numerical methods (optional)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/scientific-computing-agents.git
cd scientific-computing-agents
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, matplotlib; print('Dependencies OK')"
```

### Step 4: Verify Setup

```bash
# Run basic tests
python -m pytest tests/test_linear_algebra_agent.py -v

# Should see tests passing
```

---

## Quick Start (5 minutes)

Let's solve a simple problem with the `LinearAlgebraAgent`.

### Example 1: Solve a Linear System

Create a file `quick_start.py`:

```python
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.linear_algebra_agent import LinearAlgebraAgent

# Create agent
agent = LinearAlgebraAgent()

# Define a linear system: Ax = b
A = np.array([[3, 2], [1, 4]])
b = np.array([7, 6])

# Solve using the agent
result = agent.solve_linear_system({
    'A': A,
    'b': b,
    'method': 'direct'
})

# Display results
print(f"Solution: x = {result.data['solution']}")
print(f"Residual norm: {result.data['residual_norm']:.2e}")
print(f"Status: {result.status.name}")
```

Run it:

```bash
python quick_start.py
```

**Output**:
```
Solution: x = [1. 1.]
Residual norm: 0.00e+00
Status: SUCCESS
```

### Example 2: Solve an ODE

Let's solve a simple ODE: dy/dt = -2y, y(0) = 1

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent

# Create agent
agent = ODEPDESolverAgent()

# Define ODE problem
def dydt(t, y):
    return -2 * y

# Solve
result = agent.solve_ode_ivp({
    'f': dydt,
    'y0': [1.0],
    't_span': (0, 2),
    't_eval': np.linspace(0, 2, 50),
    'method': 'RK45'
})

# Get solution
t = result.data['t']
y = result.data['y']

print(f"y(0) = {y[0, 0]:.4f}")
print(f"y(2) = {y[0, -1]:.4f}")
print(f"Analytical y(2) = {np.exp(-4):.4f}")
```

### Example 3: Optimize a Function

Minimize f(x) = (x-2)² + (y-3)²

```python
from agents.optimization_agent import OptimizationAgent

# Create agent
agent = OptimizationAgent()

# Define objective function
def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

# Optimize
result = agent.optimize({
    'objective': objective,
    'x0': [0, 0],
    'method': 'Nelder-Mead'
})

# Display results
x_opt = result.data['x']
f_opt = result.data['fun']

print(f"Optimal point: x = {x_opt}")
print(f"Optimal value: f(x) = {f_opt:.6f}")
print(f"Expected: x = [2, 3], f = 0")
```

---

## Your First Workflow

Let's create a multi-agent workflow that combines optimization and validation.

### Problem: Fit a Model to Data

```python
import numpy as np
import matplotlib.pyplot as plt
from agents.optimization_agent import OptimizationAgent
from agents.executor_validator_agent import ExecutorValidatorAgent

# Generate synthetic data
np.random.seed(42)
x_data = np.linspace(0, 10, 50)
y_true = 2.5 * x_data + 1.5
y_data = y_true + np.random.normal(0, 2, size=x_data.shape)

# Define model: y = a*x + b
def model(params, x):
    a, b = params
    return a * x + b

# Define loss function
def loss(params):
    y_pred = model(params, x_data)
    return np.sum((y_data - y_pred)**2)

# Step 1: Optimize parameters
opt_agent = OptimizationAgent()
opt_result = opt_agent.optimize({
    'objective': loss,
    'x0': [1.0, 1.0],
    'method': 'Nelder-Mead'
})

# Get fitted parameters
a_fit, b_fit = opt_result.data['x']

# Step 2: Validate results
validator = ExecutorValidatorAgent()

def validation_func():
    y_pred = model([a_fit, b_fit], x_data)
    r_squared = 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)
    rmse = np.sqrt(np.mean((y_data - y_pred)**2))
    return {'r_squared': r_squared, 'rmse': rmse}

validation_result = validator.validate_execution({
    'execution_function': validation_func,
    'validation_criteria': {
        'r_squared': {'min': 0.7},
        'rmse': {'max': 5.0}
    }
})

# Display results
print(f"Fitted model: y = {a_fit:.2f}*x + {b_fit:.2f}")
print(f"True model: y = 2.50*x + 1.50")
print(f"R² = {validation_result.data['metrics']['r_squared']:.3f}")
print(f"RMSE = {validation_result.data['metrics']['rmse']:.3f}")
print(f"Validation: {validation_result.data['validation_passed']}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', alpha=0.5)
plt.plot(x_data, y_true, 'g--', label='True model', linewidth=2)
plt.plot(x_data, model([a_fit, b_fit], x_data), 'r-', label='Fitted model', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Model Fitting with Optimization + Validation')
plt.grid(True, alpha=0.3)
plt.savefig('first_workflow.png', dpi=150, bbox_inches='tight')
print("Plot saved to first_workflow.png")
```

**What This Demonstrates**:
1. ✅ Multi-agent composition (optimization + validation)
2. ✅ Real-world problem (curve fitting)
3. ✅ Result validation
4. ✅ Visualization

---

## Understanding Agent Results

All agents return `AgentResult` objects with consistent structure:

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent

agent = ODEPDESolverAgent()
result = agent.solve_ode_ivp({...})

# Check if successful
if result.status.name == 'SUCCESS':
    # Access data
    solution = result.data['y']
    times = result.data['t']

    # Access metadata
    method = result.metadata.get('method')

    # Check provenance
    print(f"Agent: {result.provenance.agent_name}")
    print(f"Version: {result.provenance.agent_version}")
    print(f"Time: {result.provenance.execution_time_sec:.3f}s")
else:
    # Handle errors
    print("Errors:", result.errors)
```

**Key Fields**:
- `status`: SUCCESS, FAILED, PENDING, etc.
- `data`: Dictionary with results
- `metadata`: Additional information
- `provenance`: Execution tracking
- `errors`: List of error messages

---

## Available Agents

The system includes 12 specialized agents:

### Phase 1: Core Scientific Computing
1. **LinearAlgebraAgent**: Solve linear systems, eigenvalues, decompositions
2. **ODEPDESolverAgent**: Solve ODEs and PDEs (1D, 2D, 3D)
3. **IntegrationAgent**: Numerical integration and quadrature
4. **OptimizationAgent**: Minimize/maximize functions
5. **SpecialFunctionsAgent**: Bessel, gamma, elliptic functions

### Phase 2: Advanced Methods
6. **UncertaintyQuantificationAgent**: Monte Carlo, sensitivity analysis
7. **InverseProblemsAgent**: Parameter estimation
8. **SurrogateModelingAgent**: Build fast approximations
9. **PhysicsInformedMLAgent**: Physics-informed neural networks

### Phase 3: Orchestration
10. **ProblemAnalyzerAgent**: Analyze and classify problems
11. **AlgorithmSelectorAgent**: Recommend best methods
12. **ExecutorValidatorAgent**: Execute and validate solutions

### Phase 4: Performance
13. **PerformanceProfilerAgent**: Profile and optimize code
14. **WorkflowOrchestrationAgent**: Coordinate multi-agent workflows

---

## Common Patterns

### Pattern 1: Simple Agent Usage

```python
from agents.some_agent import SomeAgent

agent = SomeAgent()
result = agent.method_name(input_data)

if result.success:
    use_result(result.data)
else:
    handle_errors(result.errors)
```

### Pattern 2: Multi-Agent Pipeline

```python
# Step 1: Analyze
analyzer = ProblemAnalyzerAgent()
analysis = analyzer.analyze(problem_description)

# Step 2: Select algorithm
selector = AlgorithmSelectorAgent()
recommendation = selector.recommend(analysis.data)

# Step 3: Solve
solver = get_solver_for(recommendation)
solution = solver.solve(problem_data)

# Step 4: Validate
validator = ExecutorValidatorAgent()
validation = validator.validate(solution, criteria)
```

### Pattern 3: Parallel Execution

```python
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent
from core.parallel_executor import ParallelMode

orchestrator = WorkflowOrchestrationAgent(
    parallel_mode=ParallelMode.THREADS,
    max_workers=4
)

# Solve multiple problems in parallel
results = orchestrator.execute_agents_parallel(
    agents=[solver] * 4,
    method_name='solve',
    inputs_list=[problem1, problem2, problem3, problem4]
)

# 3x faster than serial execution
```

---

## Running Examples

The `examples/` directory contains comprehensive demonstrations:

### Workflow Examples
```bash
# Complete optimization pipeline
python examples/workflow_01_optimization_pipeline.py

# Multi-physics simulation
python examples/workflow_02_multi_physics.py

# Inverse problem solving
python examples/workflow_03_inverse_problem.py

# ML-enhanced computing
python examples/workflow_04_ml_enhanced.py
```

### PDE Examples
```bash
# 2D heat equation
python examples/example_2d_heat.py

# 2D Poisson equation (electrostatics)
python examples/example_2d_poisson.py

# 3D Poisson equation
python examples/example_3d_poisson.py

# 2D wave equation
python examples/example_2d_wave.py
```

### Performance Examples
```bash
# Profile PDE solvers
python examples/example_profiling_pde.py

# Parallel PDE solving
python examples/example_parallel_pde.py
```

---

## What's Next?

### Learn More

1. **User Guide** (`docs/USER_GUIDE.md`)
   - Detailed agent documentation
   - Advanced workflows
   - Best practices

2. **API Reference** (`docs/API_REFERENCE.md`)
   - Complete API documentation
   - Parameter reference
   - Return types

3. **Optimization Guide** (`docs/OPTIMIZATION_GUIDE.md`)
   - Performance tuning
   - Profiling techniques
   - Parallel execution

### Explore Examples

- Browse `examples/` directory
- Start with simple examples
- Build your own workflows

### Get Help

- Check `docs/TROUBLESHOOTING.md` for common issues
- Review test files in `tests/` for usage patterns
- Read agent docstrings for detailed documentation

### Contribute

- See `CONTRIBUTING.md` for guidelines
- Report issues on GitHub
- Submit pull requests

---

## Quick Reference

### Import Agents
```python
from agents.linear_algebra_agent import LinearAlgebraAgent
from agents.ode_pde_solver_agent import ODEPDESolverAgent
from agents.optimization_agent import OptimizationAgent
from agents.uncertainty_quantification_agent import UncertaintyQuantificationAgent
```

### Solve Linear System
```python
agent = LinearAlgebraAgent()
result = agent.solve_linear_system({'A': A, 'b': b})
x = result.data['solution']
```

### Solve ODE
```python
agent = ODEPDESolverAgent()
result = agent.solve_ode_ivp({
    'f': dydt,
    'y0': y0,
    't_span': (t0, tf),
    't_eval': t_points
})
y = result.data['y']
```

### Optimize Function
```python
agent = OptimizationAgent()
result = agent.optimize({
    'objective': f,
    'x0': initial_guess,
    'method': 'Nelder-Mead'
})
x_opt = result.data['x']
```

### Profile Performance
```python
from utils.profiling import profile_performance

@profile_performance()
def my_function():
    # ... code ...
    pass
```

### Parallel Execution
```python
orchestrator = WorkflowOrchestrationAgent(parallel_mode=ParallelMode.THREADS)
results = orchestrator.execute_agents_parallel(agents, method, inputs)
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'agents'`

**Solution**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Dependency Errors

**Problem**: Missing packages

**Solution**:
```bash
pip install -r requirements.txt
```

### Test Failures

**Problem**: Tests fail on setup

**Solution**:
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Run specific test
python -m pytest tests/test_agent_name.py -v
```

---

## Summary

You've learned:
- ✅ How to install the system
- ✅ How to use individual agents
- ✅ How to create multi-agent workflows
- ✅ How to run examples
- ✅ Where to find more information

**Next steps**: Explore the `examples/` directory and read the User Guide!

---

**Questions?** See `docs/TROUBLESHOOTING.md` or check the examples in `examples/`.

**Want to contribute?** Read `CONTRIBUTING.md` to get started!
