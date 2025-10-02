# Phase 4 Quick Reference Card

**Last Updated**: 2025-09-30
**Status**: Weeks 1-3 Complete ✅

---

## What's Available Now

### 1. GPU Acceleration (Week 1)

**Use GPU for quantum simulations**:
```python
from gpu_kernels.quantum_evolution import solve_lindblad

result = solve_lindblad(
    rho0, H, L_ops, gammas, t_span,
    backend='gpu'  # or 'cpu' or 'auto'
)
```

**Performance**: 30-50x speedup, n_dim=20 feasible

---

### 2. Magnus Expansion Solver (Week 2)

**Better energy conservation for time-dependent Hamiltonians**:
```python
from solvers.magnus_expansion import MagnusExpansionSolver

solver = MagnusExpansionSolver(order=4)  # or 2, 6
psi_evolution = solver.solve_unitary(psi0, H_protocol, t_span)
```

**Performance**: 10x better energy conservation than RK4

---

### 3. Pontryagin Maximum Principle (Week 3)

**Optimal control for classical and quantum systems**:
```python
from solvers.pontryagin import PontryaginSolver

solver = PontryaginSolver(
    state_dim, control_dim,
    dynamics, running_cost
)

result = solver.solve(
    x0, xf, duration, n_steps,
    method='multiple_shooting'
)
```

**Applications**: LQR, nonlinear control, quantum protocols

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Code Lines** | 16,450+ |
| **Tests** | 60 (100% pass) |
| **Solvers** | 2 (Magnus, PMP) |
| **GPU Speedup** | 30-50x |
| **Energy Conservation** | 10x better |
| **Examples** | 14 demos |

---

## File Locations

### GPU Acceleration
- Code: `gpu_kernels/quantum_evolution.py`
- Tests: `tests/gpu/test_quantum_gpu.py`

### Magnus Solver
- Code: `solvers/magnus_expansion.py`
- Tests: `tests/solvers/test_magnus.py`
- Demo: `examples/magnus_solver_demo.py`

### PMP Solver
- Code: `solvers/pontryagin.py`
- Tests: `tests/solvers/test_pontryagin.py`
- Demo: `examples/pontryagin_demo.py`

### Documentation
- Master Plan: `docs/phases/PHASE4.md`
- Quick Start: `PHASE4_README.md`
- Week 2 Summary: `PHASE4_WEEK2_SUMMARY.md`
- Week 3 Summary: `PHASE4_WEEK3_SUMMARY.md`
- Complete Summary: `PHASE4_WEEKS1-3_COMPLETE.md`
- Progress Tracker: `PHASE4_PROGRESS.md`

---

## Running Examples

```bash
# Magnus solver demo
python3 examples/magnus_solver_demo.py

# PMP solver demo
python3 examples/pontryagin_demo.py

# Run all tests
pytest tests/gpu/ tests/solvers/ -v
```

---

## Agent Integration

### Use GPU backend
```python
agent = NonequilibriumQuantumAgent()
result = agent.execute({
    'method': 'lindblad_master_equation',
    'parameters': {'backend': 'gpu'}
})
```

### Use Magnus solver
```python
result = agent.execute({
    'method': 'lindblad_master_equation',
    'parameters': {
        'solver': 'magnus',
        'magnus_order': 4
    }
})
```

### Combine GPU + Magnus
```python
result = agent.execute({
    'method': 'lindblad_master_equation',
    'parameters': {
        'solver': 'magnus',
        'magnus_order': 4,
        'backend': 'gpu'
    }
})
```

---

## Performance Guide

| Problem Size | Recommended Backend | Recommended Solver |
|-------------|--------------------|--------------------|
| n_dim ≤ 5 | CPU | RK45 or Magnus |
| n_dim = 6-10 | GPU | Magnus |
| n_dim = 11-20 | GPU | Magnus Order 4 |
| Time-dependent H | GPU | **Magnus** |
| Optimal Control | CPU/GPU | **PMP** |

---

## Next Up (Week 4)

- [ ] JAX integration for PMP
- [ ] Collocation methods
- [ ] Test suite improvements
- [ ] ML foundation

---

## Getting Help

1. **Documentation**: See `PHASE4_README.md`
2. **Examples**: Run demo files
3. **Tests**: Check test files for usage patterns
4. **Summaries**: See weekly summary documents

---

**End of Quick Reference**
