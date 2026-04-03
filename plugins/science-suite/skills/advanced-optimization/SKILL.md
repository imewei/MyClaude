---
name: advanced-optimization
description: "Solve complex optimization problems with convex optimization (CVXPY), integer programming, constraint satisfaction, Bayesian optimization, and surrogate-based methods. Use when formulating optimization problems, implementing convex programs, or building surrogate models for expensive objective functions."
---

# Advanced Optimization

## Expert Agent

For JAX-based optimization, numerical solvers, and scientific computing, delegate to:

- **`jax-pro`**: Expert in JAX scientific computing, Optimistix solvers, and Optax schedulers.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`

## Convex Optimization (CVXPY)

### Portfolio Optimization (QP)

```python
import cvxpy as cp
import numpy as np

n = 5
returns = np.array([0.12, 0.10, 0.07, 0.03, 0.15])
cov = np.random.default_rng(42).uniform(0.01, 0.05, (n, n))
cov = cov @ cov.T

w = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(cp.quad_form(w, cov)),
    [cp.sum(w) == 1, w >= 0, returns @ w >= 0.08],
)
problem.solve(solver=cp.OSQP)
```

### SDP Relaxation (Max-Cut)

```python
n = 4
W = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
X = cp.Variable((n, n), symmetric=True)
constraints = [X >> 0] + [X[i, i] == 1 for i in range(n)]
prob = cp.Problem(cp.Maximize(0.25 * cp.trace(W @ (np.ones((n,n)) - X))), constraints)
prob.solve(solver=cp.SCS)
```

## Mixed-Integer Programming

```python
from mip import Model, xsum, BINARY

values = [60, 100, 120, 80, 50]
weights = [10, 20, 30, 15, 10]
capacity = 50

m = Model("knapsack")
m.sense = "MAX"
x = [m.add_var(var_type=BINARY) for _ in range(len(values))]
m.objective = xsum(values[i] * x[i] for i in range(len(values)))
m += xsum(weights[i] * x[i] for i in range(len(values))) <= capacity
m.optimize()
selected = [i for i in range(len(values)) if x[i].x > 0.5]
```

### Constraint Programming (OR-Tools)

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
jobs = [(5, 15), (3, 10), (7, 20)]  # (duration, deadline)
horizon = 30

starts = [model.new_int_var(0, horizon, f"s{i}") for i in range(len(jobs))]
ends = [model.new_int_var(0, horizon, f"e{i}") for i in range(len(jobs))]
intervals = [model.new_interval_var(starts[i], jobs[i][0], ends[i], f"iv{i}") for i in range(len(jobs))]
model.add_no_overlap(intervals)
for i in range(len(jobs)):
    model.add(ends[i] <= jobs[i][1])
makespan = model.new_int_var(0, horizon, "makespan")
model.add_max_equality(makespan, ends)
model.minimize(makespan)
solver = cp_model.CpSolver()
solver.solve(model)
```

## Bayesian Optimization

### BoTorch

```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(5, 2)
train_Y = -((train_X - 0.5) ** 2).sum(dim=-1, keepdim=True)

gp = SingleTaskGP(train_X, train_Y)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

ei = ExpectedImprovement(model=gp, best_f=train_Y.max())
bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate, _ = optimize_acqf(ei, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
```

### Optuna

```python
import optuna

def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return train_and_evaluate(build_model(n_layers, dropout), lr)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study.optimize(objective, n_trials=100)
```

## Multi-Objective Optimization

```python
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)
# Pareto front: study.best_trials
```

## Problem Formulation Guide

| Type | Solver | Complexity |
|------|--------|------------|
| LP | GLPK, HiGHS | Polynomial |
| QP | OSQP, ECOS | Polynomial |
| SOCP/SDP | SCS, MOSEK | Polynomial |
| MIP | GLPK_MI, HiGHS | NP-hard |
| Black-box | BoTorch, Optuna | Budget-dependent |

## Production Checklist

- [ ] Verify convexity before using CVXPY (DCP rules)
- [ ] Check problem status after solve (OPTIMAL, INFEASIBLE, UNBOUNDED)
- [ ] Set solver tolerances explicitly (eps_abs, eps_rel)
- [ ] Warm-start solvers for related sequential problems
- [ ] For MIP: set time limits and optimality gap tolerances
- [ ] For BO: normalize objectives to [0, 1] range
- [ ] Log objective, constraints, solver time, and status
- [ ] Validate solutions against constraints (numerical precision)
- [ ] Use multi-start for non-convex problems
