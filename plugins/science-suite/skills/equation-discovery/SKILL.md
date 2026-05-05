---
name: equation-discovery
description: Data-driven equation discovery with SINDy (Sparse Identification of Nonlinear Dynamics) using DataDrivenDiffEq.jl (Julia) and PySINDy (Python). Covers library construction, sparse regression (STLSQ, SR3), implicit SINDy, weak-form / integral SINDy, physics-constrained SINDy (conservation-law penalties), Bayesian SINDy (posterior over discovered coefficients via sparsifying priors or HMC on the coefficient vector), symbolic regression, and model validation. Use when identifying governing equations from trajectory data, including when uncertainty quantification on the discovered terms is required.
---

# Equation Discovery

Discover governing equations directly from trajectory data using sparse regression (SINDy) and symbolic regression. Supports both Julia (DataDrivenDiffEq.jl) and Python (PySINDy) ecosystems.

---

## Expert Agents

- **`nonlinear-dynamics-expert`**: Domain expertise for dynamical systems, sparsity-promoting regression, and model selection.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
- **`julia-pro`**: Implementation, debugging, and performance tuning for Julia-based SINDy workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

---

## Mode Flag

- `--mode quick`: routing table + agent delegation only
- `--mode standard` (default): overview, thresholding strategies, and routing
- `--mode deep`: full DataDrivenDiffEq.jl API tables and PySINDy code blocks

---

## SINDy Overview

SINDy (Sparse Identification of Nonlinear Dynamics) recovers governing equations from data:

```
dX/dt = Theta(X) * Xi
```

- **X**: State matrix (n_samples x n_states)
- **dX/dt**: Time derivative matrix (measured or numerically estimated)
- **Theta(X)**: Library matrix of candidate nonlinear functions (polynomials, trig, etc.)
- **Xi**: Sparse coefficient matrix — nonzero entries reveal the active terms

The key insight: most dynamical systems have **sparse** representations in a suitable function basis. Sparsity-promoting regression recovers the few active terms.

---

> **--mode deep required** for full library API reference below.

## DataDrivenDiffEq.jl (Julia)

```julia
using DataDrivenDiffEq, ModelingToolkit

# Define symbolic variables
@variables x y

# Build candidate library (polynomial basis up to degree 3)
basis = Basis(polynomial_basis([x, y], 3), [x, y])

# Create problem from data
prob = DataDrivenProblem(X, DX=DX)  # X: state data, DX: derivative data

# Solve with STLSQ sparse regression
result = solve(prob, basis, STLSQ(threshold=0.1))

# Extract discovered equations
system = result.basis        # Symbolic equations
coefficients = result.coeff  # Sparse coefficient matrix
```

---

## Custom Library Functions

Extend the basis beyond polynomials for domain-specific dynamics:

```julia
@variables x y

# Trigonometric terms
trig_terms = [sin(x), cos(x), sin(y), cos(y)]

# Exponential terms
exp_terms = [exp(-x), exp(-y)]

# Cross terms
cross_terms = [x * sin(y), y * cos(x), x * exp(-y)]

# Combined custom basis
custom_basis = Basis(
    vcat(polynomial_basis([x, y], 2), trig_terms, exp_terms, cross_terms),
    [x, y]
)
```

> **Rule:** Start with a polynomial basis. Add domain-specific terms only when polynomial SINDy fails or physics suggests oscillatory/exponential behavior.

---

## Thresholding Strategies

| Algorithm | Best For | Key Parameter | Notes |
|-----------|----------|---------------|-------|
| **STLSQ** | Clean data, default choice | `threshold` (0.01-0.3) | Sequential thresholded least squares; fast, interpretable |
| **SR3** | Noisy data, relaxed sparsity | `threshold`, `relaxation` | Sparse relaxed regularized regression; more robust to noise |
| **ADMM** | Constrained problems | `threshold`, `rho` | Alternating direction method of multipliers; enforces constraints |

```julia
# STLSQ — default for clean data
result_stlsq = solve(prob, basis, STLSQ(threshold=0.1))

# SR3 — noisy data
result_sr3 = solve(prob, basis, SR3(threshold=0.05, relaxation=1.0))

# ADMM — constrained
result_admm = solve(prob, basis, ADMM(threshold=0.1, rho=1.0))
```

---

## Implicit SINDy

For dynamics that cannot be written as explicit `dX/dt = f(X)` (e.g., implicit ODEs, DAEs):

```julia
# Implicit formulation: F(X, dX/dt) = 0
# Augment library with derivative terms
@variables x dx

implicit_basis = Basis(
    vcat(polynomial_basis([x, dx], 3), [sin(x) * dx, x^2 * dx]),
    [x, dx]
)

# Solve with implicit flag
prob_implicit = DataDrivenProblem(X, DX=DX)
result = solve(prob_implicit, implicit_basis, ImplicitOptimizer(STLSQ(threshold=0.1)))
```

---

## DifferentialEquations.jl Integration

Discover-simulate-validate loop:

```julia
using DifferentialEquations, DataDrivenDiffEq, ModelingToolkit

# 1. Discover equations from data
result = solve(prob, basis, STLSQ(threshold=0.1))

# 2. Convert to ODESystem for simulation
@named discovered_sys = ODESystem(result.basis)

# 3. Simulate discovered model
ode_prob = ODEProblem(discovered_sys, u0, tspan)
sol = solve(ode_prob, Tsit5())

# 4. Validate: compare simulation vs held-out data
error = norm(sol(t_test) .- X_test) / norm(X_test)
```

> **Rule:** Always validate on held-out data not used in the SINDy fit. In-sample error is misleading for sparse models.

---

> **--mode deep required** for full PySINDy code blocks below.

## PySINDy (Python)

```python
import pysindy as ps
import numpy as np

# Basic SINDy
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=3),
    feature_names=["x", "y"]
)
model.fit(X, t=t)
model.print()  # Display discovered equations

# Custom library with GeneralizedLibrary
lib = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree=2),
     ps.FourierLibrary(n_frequencies=3),
     ps.CustomLibrary(library_functions=[lambda x: np.exp(-x)],
                       function_names=[lambda x: f"exp(-{x})"])]
)
model = ps.SINDy(feature_library=lib)
model.fit(X, t=t)
```

PySINDy ships `STLSQ`, `SR3`, `SSR`, `FROLS`, `ConstrainedSR3`, `MIOSR` optimizers; feature libraries include `Polynomial`, `Fourier`, `Custom`, `PDE`, `WeakForm`, `Generalized`, `Tensored`; differentiation methods cover finite difference, smoothed FD, spectral, Savitzky-Golay, and Kalman. Supports control inputs (SINDyc), implicit dynamics, trapping theorem, and ensemble/bagging methods for UQ. NumPy/scikit-learn based — **not JAX-native**.

---

## Related Python Packages

| Package | Role | Notes |
|---------|------|-------|
| **PySINDy** | Sparse regression (SINDy, PDE/weak-form, ensemble) | NumPy/sklearn — see above |
| **PyDMD** | DMD family (exact, FbDMD, CDMD, MrDMD, Hankel, EDMD, DMDc, BOPDMD, PiDMD) | Koopman operator approximation; complementary to SINDy |
| **PySR** | Symbolic regression via Julia `SymbolicRegression.jl` backend | sklearn-compat; exports to SymPy/LaTeX/JAX/PyTorch via `model.jax()` |
| **gplearn** | Classical genetic-programming symbolic regression | NumPy; `SymbolicRegressor`/`Classifier`/`Transformer` |

> **No mature JAX-native SINDy library exists.** For a JAX-first workflow, hand-roll STLSQ via `jax.lax.scan` over polynomial libraries — the regression step is trivially vectorizable. PySR's `model.jax()` exporter is the cleanest bridge into a JAX pipeline.

---

## Symbolic Regression

When SINDy's predefined basis is too restrictive, use evolutionary symbolic regression:

```julia
using SymbolicRegression

# Search for symbolic expressions
options = SymbolicRegression.Options(
    binary_operators=[+, -, *, /],
    unary_operators=[sin, cos, exp, sqrt],
    populations=30,
    maxsize=25
)

hall_of_fame = equation_search(X, y;
    options=options,
    niterations=100
)

# Pareto front: complexity vs accuracy
for member in hall_of_fame
    println("Complexity: $(member.complexity), Loss: $(member.loss)")
    println("  Equation: $(member.equation)")
end
```

> **Rule:** Use the Pareto front (complexity vs loss) to select models. Prefer the simplest equation whose loss is within 5% of the best.

---

## Validation Strategies

1. **Cross-prediction on held-out data**: Split trajectories into train/test; fit on train, predict on test
2. **Pareto front sweep**: Vary the sparsity threshold and plot model complexity vs prediction error
3. **Multi-trajectory validation**: Fit on one trajectory, validate on independent initial conditions
4. **Long-time stability**: Simulate discovered equations well beyond the training time horizon

```julia
# Pareto front sweep over thresholds
thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
for thresh in thresholds
    res = solve(prob_train, basis, STLSQ(threshold=thresh))
    err = validate_on_test(res, X_test, t_test)
    n_terms = count(!iszero, res.coeff)
    println("Threshold=$thresh, Terms=$n_terms, Error=$err")
end
```

---

## Handling Noisy Data

| Noise Level | Strategy | Details |
|-------------|----------|---------|
| **Clean** (SNR > 100) | Direct STLSQ | Use raw derivatives; `threshold=0.05-0.1` |
| **Moderate** (SNR 10-100) | Smoothed derivatives + SR3 | Apply Savitzky-Golay or total variation smoothing before SINDy; use SR3 for robustness |
| **High** (SNR < 10) | Integral SINDy or ensemble | Use weak-form / integral formulation to avoid derivative estimation; ensemble averaging over subsampled data |

> **Rule:** Never use finite differences on noisy data. Use smoothed or integral formulations.

---

## Connection to UDE

After training a Universal Differential Equation (UDE), use SINDy on the trained neural network to extract symbolic equations -- the **UDE+SINDy pipeline**:

1. Train UDE with neural network closure (see **sciml-modern-stack** skill)
2. Generate synthetic data from the trained UDE
3. Apply SINDy to the neural network output to recover interpretable equations
4. Validate the symbolic model against the original data

This pipeline combines the flexibility of neural networks with the interpretability of symbolic equations.

---

## Bayesian SINDy — posterior uncertainty on discovered coefficients

Bayesian SINDy with horseshoe priors, ensemble SINDy, and UQ-SINDy are covered in the dedicated **[bayesian-sindy-workflow](../bayesian-sindy-workflow/SKILL.md)** skill. That skill contains a full Lorenz-63 worked example (generate data → build candidate library → fit horseshoe prior with NumPyro + NUTS → diagnose with ArviZ PSIS-LOO → extract inclusion probabilities with credible intervals), a prior-sensitivity sweep, and a Julia Turing sidebar. Use it when you need credible intervals on SINDy coefficients, inclusion probabilities for library terms, or Bayesian model comparison between candidate libraries.

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Library too small | High residual, poor fit | Add domain-specific terms (trig, exp, cross terms) |
| Library too large | Overfitting, spurious terms | Increase sparsity threshold; use information criteria (AIC/BIC) |
| Noisy derivatives | Unstable coefficients across thresholds | Smooth data or use integral SINDy formulation |
| Insufficient data | Underdetermined system | Collect longer trajectories or multiple initial conditions |
| Wrong coordinate system | Complex equations with many terms | Transform to physically meaningful coordinates before SINDy |
| Threshold too aggressive | Missing true dynamics terms | Sweep thresholds and inspect Pareto front for elbow |

## Checklist

- [ ] Verify derivative estimation method matches data noise level (finite difference for clean, smoothed/integral for noisy)
- [ ] Confirm candidate library includes domain-appropriate basis functions (polynomial, trig, exponential)
- [ ] Sweep sparsity thresholds and inspect the Pareto front (complexity vs prediction error) for the elbow
- [ ] Validate discovered equations on held-out trajectories not used during SINDy fitting
- [ ] Check long-time stability by simulating discovered equations well beyond training time horizon
- [ ] Ensure multi-trajectory validation uses independent initial conditions
- [ ] Compare STLSQ, SR3, and ADMM results to assess robustness of discovered terms
- [ ] Verify coordinate system is physically meaningful before applying SINDy
- [ ] Confirm UDE+SINDy pipeline extracts symbolic equations consistent with known physics
