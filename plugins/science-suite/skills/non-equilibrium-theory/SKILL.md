---
name: non-equilibrium-theory
description: Apply non-equilibrium thermodynamics including fluctuation theorems, entropy production, and linear response theory. Use when modeling irreversible processes, analyzing driven systems, or deriving transport coefficients.
---

# Non-Equilibrium Theory

Theoretical frameworks for systems far from thermal equilibrium.

## Expert Agent

For non-equilibrium thermodynamics, fluctuation theorems, and active matter theory, delegate to the expert agent:

- **`statistical-physicist`**: Unified specialist for Non-Equilibrium Physics.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: Jarzynski equality analysis, entropy production quantification, and linear response theory.

## Core Theorems

| Theorem | Formula | Application |
|---------|---------|-------------|
| Entropy Production | σ = Σ J_i X_i ≥ 0 | Quantify irreversibility |
| Crooks | P_F(W)/P_R(-W) = exp(β(W-ΔF)) | Work distributions |
| Jarzynski | ⟨exp(-βW)⟩ = exp(-βΔF) | Free energy from non-eq |
| FDT | χ(t) = β d/dt⟨A(t)B(0)⟩ | Response from fluctuations |
| Onsager | L_ij = L_ji | Transport symmetry |

## Linear Response Theory

**Response function**:
```
χ(ω) = ∫₀^∞ dt e^(iωt) ⟨A(t)B(0)⟩
```

**Kubo formula (conductivity)**:
```
σ = β ∫₀^∞ ⟨j(t)j(0)⟩ dt
```

## Entropy Production

```
σ = Σ J_i X_i ≥ 0
```

| Symbol | Meaning |
|--------|---------|
| J_i | Thermodynamic fluxes (heat, particle, charge) |
| X_i | Thermodynamic forces (gradients) |
| σ > 0 | Irreversibility, arrow of time |

## Fluctuation Theorems

### Jarzynski Equality

Extract free energy from non-equilibrium measurements:
- Single-molecule pulling
- Molecular motors
- Nanoscale energy conversion

### FDT Violations

| Observation | Interpretation |
|-------------|----------------|
| T_eff > T | Non-equilibrium driving |
| χ ≠ βdC/dt | Active matter, aging |

## Applications

| Application | Principle |
|-------------|-----------|
| Molecular motors | Extract work from ATP hydrolysis |
| Energy harvesting | Brownian ratchets, Landauer bound |
| Active matter | Compute entropy production |
| Self-assembly | Balance driving vs dissipation |
| Dissipative structures | Turing patterns, convection rolls |

## Computing Jarzynski free energies from non-eq work data

The Jarzynski equality `⟨exp(-βW)⟩ = exp(-βΔF)` suffers from catastrophic sample noise — the exponential is dominated by rare low-work trajectories. Use Bennett-Acceptance-Ratio (BAR) or the symmetric Crooks estimator instead of naive exponential averaging.

```python
import numpy as np
from scipy.optimize import brentq

def jarzynski_naive(W_fwd, beta):
    """Biased small-sample estimator. Use only for validation on large N."""
    return -np.log(np.mean(np.exp(-beta * W_fwd))) / beta

def crooks_bar(W_fwd, W_rev, beta):
    """
    Bennett Acceptance Ratio — statistically optimal free-energy estimator
    from forward and reverse work distributions.
        W_fwd: work values from forward protocol (shape Nf)
        W_rev: work values from reverse protocol (shape Nr)
    Solves Crooks' fixed-point equation for ΔF.
    """
    M = np.log(len(W_rev) / len(W_fwd))
    def lhs(dF):
        return (
            np.sum(1 / (1 + np.exp(M + beta * (W_fwd - dF))))
            - np.sum(1 / (1 + np.exp(-M - beta * (W_rev + dF))))
        )
    return brentq(lhs, W_fwd.min() - 10 / beta, W_fwd.max() + 10 / beta)
```

Use BAR in single-molecule pulling experiments (optical trap, AFM) and in alchemical free-energy calculations (`alchemlyb` — Python ecosystem wrapper around BAR / MBAR).

## Entropy production from trajectories

The **stochastic entropy production** along a single trajectory `x(t)` under Langevin dynamics with drift `μF` and noise `ξ` is

```
σ[x] = β ∫ F(x) ∘ dx      (Stratonovich)
```

Discretized on a trajectory (after simulating with `stochastic-dynamics`):

```python
def entropy_production(trajectory, force_fn, beta, dt):
    """
    Stratonovich-integrated trajectory entropy production.
    trajectory: shape (T, d)
    force_fn:   callable returning F(x), shape (d,)
    """
    F_mid = 0.5 * (force_fn(trajectory[:-1]) + force_fn(trajectory[1:]))
    dx = trajectory[1:] - trajectory[:-1]
    return beta * np.sum(F_mid * dx, axis=-1)
```

For active-matter systems, this diagnostic distinguishes *active* from *passive* components by identifying the trajectories where `⟨σ⟩ > 0`. See the `active-matter` skill for the full active-Brownian / run-and-tumble pattern.

## Large-deviation theory & avalanche statistics

For rare events in driven systems — avalanches in sandpile / depinning models, intermittency in glassy dynamics, extreme values in active matter — the relevant framework is large-deviation theory (LDT): the probability of an atypical trajectory scales as `P(x_T → a) ~ exp(-T · I(a))` where `I` is the rate function.

| Method | Role |
|--------|------|
| **Cloning algorithm** | Population Monte Carlo for rare-event rate functions. `ParRep` and `milestoning` variants. Julia reference: `RareEvents.jl`; Python: hand-rolled on `jax.vmap` replicas |
| **Scaled Cumulant Generating Function (SCGF)** | `λ(k) = lim_T (1/T) log⟨exp(k · A_T)⟩` — fit from trajectory ensembles; Legendre-transform to `I(a)` |
| **Giardinà-Kurchan-Lecomte-Tailleur cloning** | Canonical scheme for computing SCGF; biased dynamics + resampling |
| **Tilted generator / Doob transform** | Exact optimal bias for rare events; numerical via principal eigenvector of tilted operator |

For **avalanche size / duration distributions** in self-organized-criticality models (BTW sandpile, Manna, Oslo rice-pile), the observables are the exponents `τ` (avalanche size), `α` (duration), and the avalanche-shape collapse `s(t/T)/T^γ`. Fit via maximum-likelihood power-law estimators — see `ewstools` and `powerlaw` (Alstott et al.) for Python, `HeavyTails.jl` for Julia.

```python
import powerlaw                                   # pip install powerlaw
fit = powerlaw.Fit(avalanche_sizes, discrete=True)
print(fit.alpha, fit.xmin)                        # exponent and cutoff
R, p = fit.distribution_compare("power_law", "lognormal")
```

See `advanced-simulations` for rare-event sampler machinery (WESTPA, OpenPathSampling) and `chaos-attractors` for dynamical-systems indicators (Lyapunov, DFA) that complement avalanche statistics.

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Check thermodynamic consistency | σ ≥ 0, Onsager symmetry |
| Validate FDT | Compare response to correlation |
| Measure effective temperature | T_eff from χ vs C |
| Use Green-Kubo | Transport from equilibrium correlations |
| Prefer BAR over naive Jarzynski | Exponential average has catastrophic variance |
| Fit avalanche exponents with MLE | Least-squares on log-log plots biases the exponent |

## Checklist

- [ ] Entropy production positive
- [ ] Onsager symmetry checked
- [ ] Response functions computed
- [ ] FDT validity assessed
- [ ] Transport coefficients extracted
- [ ] Thermodynamic consistency verified
