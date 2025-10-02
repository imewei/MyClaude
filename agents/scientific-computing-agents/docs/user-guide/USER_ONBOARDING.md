# User Onboarding Guide

**Scientific Computing Agents System**
**Version**: 1.0
**Last Updated**: 2025-10-01

---

## Welcome! 🎉

Thank you for joining the Scientific Computing Agents community! This guide will help you get started quickly and make the most of the system.

---

## Table of Contents

1. [What You'll Learn](#what-youll-learn)
2. [Prerequisites](#prerequisites)
3. [Quick Start (5 Minutes)](#quick-start-5-minutes)
4. [Your First Workflow](#your-first-workflow)
5. [Common Use Cases](#common-use-cases)
6. [Learning Path](#learning-path)
7. [Getting Help](#getting-help)
8. [Next Steps](#next-steps)

---

## What You'll Learn

By the end of this guide, you'll be able to:

- ✅ Install and configure the system
- ✅ Run your first scientific computation
- ✅ Create multi-agent workflows
- ✅ Profile and optimize performance
- ✅ Access help and resources

**Time Required**: 15-30 minutes

---

## Prerequisites

### Required Knowledge

- **Python**: Basic to intermediate (functions, classes, imports)
- **NumPy**: Basic array operations
- **Scientific Computing**: Understanding of ODEs, optimization, or linear algebra

### System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Operating System**: Linux, macOS, or Windows

### Recommended Tools

- Code editor (VS Code, PyCharm, Jupyter)
- Terminal/command line
- Git (for examples)

---

## Quick Start (5 Minutes)

### Step 1: Installation

Choose one installation method:

**Option A: PyPI (Recommended)**
```bash
pip install scientific-computing-agents
```

**Option B: From Source**
```bash
git clone https://github.com/scientific-computing-agents/scientific-computing-agents.git
cd scientific-computing-agents
pip install -e .
```

**Verify Installation**:
```bash
python -c "from agents import *; print('Installation successful!')"
```

### Step 2: Your First Agent

Create a file `hello_agents.py`:

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent
import numpy as np

# Create an agent
solver = ODEPDESolverAgent()

# Define a simple ODE: dy/dt = -y
def exponential_decay(t, y):
    return -y

# Solve the ODE
result = solver.process({
    'task': 'solve_ode',
    'equation': exponential_decay,
    'initial_conditions': [1.0],
    't_span': (0, 5),
    't_eval': np.linspace(0, 5, 50)
})

# Check results
if result.success:
    print("✓ Solution computed successfully!")
    print(f"Final value: {result.data['y'][-1]:.4f}")
else:
    print(f"✗ Error: {result.errors}")
```

**Run it**:
```bash
python hello_agents.py
```

**Expected Output**:
```
✓ Solution computed successfully!
Final value: 0.0067
```

**Congratulations!** You've run your first scientific computation! 🎊

---

## Your First Workflow

Let's create a more complex workflow that uses multiple agents.

### Example: Optimization with Profiling

Create `optimization_workflow.py`:

```python
from agents.optimization_agent import OptimizationAgent
from agents.performance_profiler_agent import PerformanceProfilerAgent
import numpy as np

# Define the Rosenbrock function (classic optimization test)
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Step 1: Optimize the function
print("Step 1: Optimizing Rosenbrock function...")
optimizer = OptimizationAgent()

result = optimizer.process({
    'task': 'minimize',
    'function': rosenbrock,
    'x0': [0, 0],  # Starting point
    'method': 'L-BFGS-B'
})

if result.success:
    print(f"✓ Optimization successful!")
    print(f"  Minimum at: {result.data['x']}")
    print(f"  Function value: {result.data['fun']:.6f}")
    print(f"  Iterations: {result.data['nit']}")
else:
    print(f"✗ Optimization failed: {result.errors}")

# Step 2: Profile the optimization
print("\nStep 2: Profiling optimization performance...")
profiler = PerformanceProfilerAgent()

def run_optimization():
    optimizer.process({
        'task': 'minimize',
        'function': rosenbrock,
        'x0': [0, 0],
        'method': 'L-BFGS-B'
    })

profile_result = profiler.process({
    'task': 'profile_function',
    'function': run_optimization
})

if profile_result.success:
    print(f"✓ Profiling complete!")
    print(f"  Execution time: {profile_result.data['total_time']:.4f}s")
    print("\nTop functions by time:")
    print(profile_result.data['report'][:500])  # First 500 chars
else:
    print(f"✗ Profiling failed: {profile_result.errors}")
```

**Run it**:
```bash
python optimization_workflow.py
```

**What's Happening**:
1. OptimizationAgent finds the minimum of the Rosenbrock function
2. PerformanceProfilerAgent measures how long it takes
3. You get both the result AND performance insights!

---

## Common Use Cases

### Use Case 1: Solving Differential Equations

**When to use**: Physical simulations, population dynamics, chemical kinetics

```python
from agents.ode_pde_solver_agent import ODEPDESolverAgent
import numpy as np

agent = ODEPDESolverAgent()

# Example: Lotka-Volterra predator-prey model
def predator_prey(t, y):
    prey, predator = y
    alpha, beta, delta, gamma = 1.5, 1.0, 3.0, 1.0
    dprey = alpha * prey - beta * prey * predator
    dpredator = delta * prey * predator - gamma * predator
    return [dprey, dpredator]

result = agent.process({
    'task': 'solve_ode',
    'equation': predator_prey,
    'initial_conditions': [10, 5],  # Initial prey and predator populations
    't_span': (0, 20),
    't_eval': np.linspace(0, 20, 200)
})

# Plot results (requires matplotlib)
import matplotlib.pyplot as plt
plt.plot(result.data['t'], result.data['y'][:, 0], label='Prey')
plt.plot(result.data['t'], result.data['y'][:, 1], label='Predator')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Dynamics')
plt.show()
```

### Use Case 2: Multi-Dimensional Optimization

**When to use**: Parameter fitting, design optimization, machine learning

```python
from agents.optimization_agent import OptimizationAgent
import numpy as np

agent = OptimizationAgent()

# Example: Fit a curve to data
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([1.1, 2.9, 5.2, 7.1, 8.9])

def residual(params):
    a, b = params
    y_pred = a * x_data + b
    return np.sum((y_pred - y_data)**2)

result = agent.process({
    'task': 'minimize',
    'function': residual,
    'x0': [1, 0],
    'method': 'L-BFGS-B'
})

print(f"Best fit: y = {result.data['x'][0]:.2f}x + {result.data['x'][1]:.2f}")
```

### Use Case 3: Large Linear Systems

**When to use**: Finite element analysis, graph algorithms, data analysis

```python
from agents.linear_algebra_agent import LinearAlgebraAgent
import numpy as np

agent = LinearAlgebraAgent()

# Example: Solve a large sparse system
n = 1000
A = np.random.randn(n, n)
A = A @ A.T  # Make symmetric positive definite
b = np.random.randn(n)

result = agent.process({
    'task': 'solve_linear_system',
    'A': A,
    'b': b
})

print(f"Solution computed for {n}x{n} system")
print(f"Residual norm: {np.linalg.norm(A @ result.data['x'] - b):.2e}")
```

### Use Case 4: Orchestrated Workflows

**When to use**: Complex multi-step analyses, pipeline processing

```python
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep
from agents.optimization_agent import OptimizationAgent
from agents.ode_pde_solver_agent import ODEPDESolverAgent

orchestrator = WorkflowOrchestrationAgent()

# Create a workflow: optimize parameters, then simulate
optimizer = OptimizationAgent()
solver = ODEPDESolverAgent()

steps = [
    WorkflowStep(
        step_id='optimize',
        agent=optimizer,
        method='process',
        inputs={
            'task': 'minimize',
            'function': lambda x: (x[0] - 2)**2 + (x[1] - 3)**2,
            'x0': [0, 0]
        }
    ),
    WorkflowStep(
        step_id='simulate',
        agent=solver,
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
print(f"Workflow completed: {result.success}")
```

---

## Learning Path

### Week 1: Foundations

**Day 1-2**: Core Agents
- [ ] Run ODE solver examples
- [ ] Try optimization problems
- [ ] Solve linear systems

**Day 3-4**: Advanced Features
- [ ] Use performance profiler
- [ ] Create basic workflows
- [ ] Explore parallel execution

**Day 5**: Integration
- [ ] Combine multiple agents
- [ ] Profile your workflows
- [ ] Optimize performance

### Week 2: Real Applications

**Day 1-3**: Your Domain
- [ ] Adapt examples to your field
- [ ] Create custom workflows
- [ ] Benchmark performance

**Day 4-5**: Production
- [ ] Set up monitoring
- [ ] Deploy to production
- [ ] Share results with team

### Recommended Examples

Work through these in order:

1. **examples/01_basic_ode.py**: Simple ODE solving
2. **examples/02_optimization.py**: Parameter optimization
3. **examples/03_linear_systems.py**: Large linear systems
4. **examples/04_workflow.py**: Multi-agent workflows
5. **examples/05_profiling.py**: Performance optimization

---

## Getting Help

### Documentation

- **Getting Started**: `docs/GETTING_STARTED.md`
- **API Reference**: `docs/API_REFERENCE.md` (if available)
- **Examples**: `examples/` directory
- **Deployment**: `docs/DEPLOYMENT.md`

### Community

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions, share projects
- **Stack Overflow**: Tag `scientific-computing-agents`

### Support Channels

- **Email**: support@example.com
- **Slack**: #sci-agents-users
- **Office Hours**: Thursdays 2-4 PM UTC

### Common Questions

**Q: Which agent should I use for my problem?**
A: See the decision tree in `docs/AGENT_SELECTION.md`

**Q: How do I improve performance?**
A: Use the PerformanceProfilerAgent to identify bottlenecks

**Q: Can I run workflows in parallel?**
A: Yes! Use `orchestrator.execute_workflow(steps, parallel=True)`

**Q: How do I contribute?**
A: See `CONTRIBUTING.md` for guidelines

---

## Next Steps

### Immediate Actions

1. **Complete Quick Start** ✓
2. **Run Your First Workflow** ✓
3. **Explore Examples**
   - Navigate to `examples/` directory
   - Run `python examples/01_basic_ode.py`
   - Modify examples for your use case

4. **Join the Community**
   - Star the GitHub repo
   - Join Slack/Discord
   - Follow @sci_agents on Twitter

### Short-term Goals (This Week)

- [ ] Solve a real problem in your domain
- [ ] Create a custom workflow
- [ ] Share your results with the community

### Long-term Goals (This Month)

- [ ] Deploy to production
- [ ] Contribute improvements
- [ ] Publish a case study

---

## Success Checklist

After completing this guide, you should be able to:

- [ ] Install and verify the system
- [ ] Run basic agent computations
- [ ] Create multi-agent workflows
- [ ] Profile and optimize performance
- [ ] Find help when needed
- [ ] Contribute to the community

**If you can check all boxes, you're ready to use the system productively!**

---

## Feedback

We want to improve this onboarding experience. Please share:

- What worked well?
- What was confusing?
- What's missing?

**Feedback Form**: https://forms.example.com/onboarding-feedback

---

## Welcome Again! 🚀

You're now part of the Scientific Computing Agents community. We're excited to see what you'll build!

**Happy Computing!**

---

**Document Version**: 1.0
**Maintained By**: Community Team
**Review Frequency**: Quarterly
