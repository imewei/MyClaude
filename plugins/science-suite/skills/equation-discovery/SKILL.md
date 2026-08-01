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
- `--mode deep`: full DataDrivenDiffEq.jl / PySINDy API — see Reference Files below

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

## Thresholding Strategies

| Algorithm | Best For | Key Parameter | Notes |
|-----------|----------|---------------|-------|
| **STLSQ** | Clean data, default choice | `threshold` (0.01-0.3) | Sequential thresholded least squares; fast, interpretable |
| **SR3** | Noisy data, relaxed sparsity | `threshold`, `relaxation` | Sparse relaxed regularized regression; more robust to noise |
| **ADMM** | Constrained problems | `threshold`, `rho` | Alternating direction method of multipliers; enforces constraints |

Full solver calls for each algorithm are in `references/julia-datadriven-diffeq.md`.

---

## Validation Strategies

1. **Cross-prediction on held-out data**: Split trajectories into train/test; fit on train, predict on test
2. **Pareto front sweep**: Vary the sparsity threshold and plot model complexity vs prediction error
3. **Multi-trajectory validation**: Fit on one trajectory, validate on independent initial conditions
4. **Long-time stability**: Simulate discovered equations well beyond the training time horizon

> **Rule:** Always validate on held-out data not used in the SINDy fit. In-sample error is misleading for sparse models. The threshold-sweep script lives in `references/julia-datadriven-diffeq.md`.

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

> **--mode deep required** for the full API reference below.

## Additional Resources

### Reference Files

- **`references/julia-datadriven-diffeq.md`** - Basis construction, thresholding algorithm calls, implicit SINDy, DifferentialEquations.jl discover-simulate-validate loop, threshold sweep script
- **`references/pysindy-and-symbolic-regression.md`** - Full PySINDy API, related Python packages (PyDMD, PySR, gplearn), Julia `SymbolicRegression.jl` symbolic regression

## Routing Decision Tree

```
Data-driven equation discovery from trajectory data (SINDy / PySINDy)?
  → This skill (science-suite:equation-discovery)

Need posterior uncertainty / inclusion probabilities on discovered terms?
  → science-suite:bayesian-sindy-workflow (../bayesian-sindy-workflow/SKILL.md)

Discovering residual dynamics inside a physics-based ODE?
  → UDE + SINDy pipeline — start with science-suite:bayesian-ude-workflow
```

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
