---
name: control-theory
description: "Design control systems with python-control and Julia ControlSystems.jl including PID tuning, state-space models, transfer functions, Bode/Nyquist analysis, stability analysis, and optimal control (LQR/MPC). Use when designing controllers, analyzing system stability, or implementing feedback control loops."
---

# Control Theory

Design, analyze, and implement feedback control systems.

## Expert Agent

For physics-based simulation and dynamical systems modeling, delegate to the expert agent:

- **`simulation-expert`**: Physics simulation specialist for dynamical systems, numerical methods, and multi-physics coupling.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
  - *Capabilities*: ODE/PDE solvers, stability analysis, parameter estimation, multiscale modeling.

## Transfer Functions

```python
import control as ct
import numpy as np

# Define transfer function: G(s) = 10 / (s^2 + 3s + 10)
G = ct.tf([10], [1, 3, 10])

# Series and parallel combinations
C = ct.tf([5, 1], [1, 0])  # PI controller: (5s + 1) / s
L = ct.series(C, G)         # Open-loop: C(s) * G(s)
T = ct.feedback(L)          # Closed-loop: L / (1 + L)

# Poles and zeros
poles = ct.poles(T)
zeros = ct.zeros(T)
print(f"Poles: {poles}")
print(f"Zeros: {zeros}")
```

## State-Space Models

```python
# x_dot = Ax + Bu, y = Cx + Du
A = np.array([[0, 1], [-10, -3]])
B = np.array([[0], [10]])
C = np.array([[1, 0]])
D = np.array([[0]])

sys_ss = ct.ss(A, B, C, D)

# Convert between representations
sys_tf = ct.tf(sys_ss)       # State-space -> Transfer function
sys_ss2 = ct.ss(sys_tf)      # Transfer function -> State-space

# Controllability and observability
Wc = ct.ctrb(A, B)
Wo = ct.obsv(A, C)
print(f"Controllable: {np.linalg.matrix_rank(Wc) == A.shape[0]}")
print(f"Observable: {np.linalg.matrix_rank(Wo) == A.shape[0]}")
```

## PID Tuning

```python
def pid_controller(kp: float, ki: float, kd: float, tau_f: float = 0.01) -> ct.TransferFunction:
    """Create PID with derivative filter: Kp + Ki/s + Kd*s/(tau_f*s + 1)."""
    P = ct.tf([kp], [1])
    I = ct.tf([ki], [1, 0])
    D = ct.tf([kd, 0], [tau_f, 1])
    return ct.parallel(P, I, D)

# Ziegler-Nichols tuning from ultimate gain and period
def ziegler_nichols_pid(ku: float, tu: float) -> dict:
    """Compute PID gains using Ziegler-Nichols method."""
    return {
        "kp": 0.6 * ku,
        "ki": 1.2 * ku / tu,
        "kd": 0.075 * ku * tu,
    }
```

## Frequency Response Analysis

```python
# Bode plot data
mag, phase, omega = ct.bode(L, plot=False)
# ct.bode_plot(L)  # Interactive plot

# Stability margins
gm, pm, wgc, wpc = ct.margin(L)
print(f"Gain margin: {20*np.log10(gm):.1f} dB at {wgc:.2f} rad/s")
print(f"Phase margin: {pm:.1f} deg at {wpc:.2f} rad/s")
```

## Root Locus

```python
# Root locus analysis
# ct.root_locus(G)

# Find gain for specific damping ratio
def find_gain_for_damping(G, zeta_target: float) -> float:
    """Find gain K where closed-loop poles have desired damping."""
    gains = np.linspace(0, 100, 1000)
    best_k = 0
    best_err = float("inf")
    for k in gains:
        T = ct.feedback(k * G)
        p = ct.poles(T)
        for pole in p:
            if np.iscomplex(pole):
                zeta = -np.real(pole) / np.abs(pole)
                err = abs(zeta - zeta_target)
                if err < best_err:
                    best_err = err
                    best_k = k
    return best_k
```

## LQR Optimal Control

```python
def design_lqr(A, B, Q, R):
    """Design LQR controller: u = -Kx minimizing J = integral(x'Qx + u'Ru)."""
    K, S, E = ct.lqr(A, B, Q, R)
    return {
        "gain": K,
        "cost_matrix": S,   # Solution to algebraic Riccati equation
        "cl_poles": E,       # Closed-loop eigenvalues
    }

# Example: double integrator
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
Q = np.diag([10, 1])   # Penalize position error more
R = np.array([[1]])     # Control effort penalty
result = design_lqr(A, B, Q, R)
```

## Model Predictive Control (MPC)

```python
import cvxpy as cp

def mpc_step(
    A: np.ndarray, B: np.ndarray,
    x0: np.ndarray, x_ref: np.ndarray,
    N: int = 20, Q: np.ndarray = None, R: np.ndarray = None,
) -> np.ndarray:
    """Solve one MPC step: minimize sum(x'Qx + u'Ru) over horizon N."""
    nx, nu = B.shape
    Q = Q if Q is not None else np.eye(nx)
    R = R if R is not None else 0.1 * np.eye(nu)

    x = cp.Variable((nx, N + 1))
    u = cp.Variable((nu, N))
    cost = 0
    constraints = [x[:, 0] == x0]

    for k in range(N):
        cost += cp.quad_form(x[:, k] - x_ref, Q) + cp.quad_form(u[:, k], R)
        constraints.append(x[:, k + 1] == A @ x[:, k] + B @ u[:, k])
        constraints.append(cp.norm(u[:, k], "inf") <= 1.0)  # Input constraints

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)
    return u[:, 0].value  # Apply first control action
```

## Control Design Checklist

- [ ] Model the plant dynamics (transfer function or state-space)
- [ ] Verify controllability and observability
- [ ] Define performance specs: settling time, overshoot, steady-state error
- [ ] Check open-loop stability (pole locations)
- [ ] Design controller (PID, LQR, or MPC)
- [ ] Verify gain margin > 6 dB and phase margin > 30 deg
- [ ] Simulate step response and disturbance rejection
- [ ] Test robustness to plant uncertainty (+/- 20% parameter variation)
