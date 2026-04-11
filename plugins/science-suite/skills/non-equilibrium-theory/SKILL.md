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

## Worked example — BAR from Langevin work samples

End-to-end bridge from the `stochastic-dynamics` Langevin ensemble pattern to a BAR free-energy estimate. Jarzynski's identity gives an unbiased ΔF from non-equilibrium work samples but with catastrophic variance — the estimator is dominated by rare low-work trajectories. BAR uses both forward and reverse work distributions for the minimum-variance two-state estimator.

### When to prefer BAR vs Jarzynski vs MBAR

| Situation | Estimator |
|-----------|-----------|
| Forward-only work samples | Naive Jarzynski (validation only) |
| Forward + reverse, two states | **BAR** (optimal two-state) |
| Multiple states (alchemical ladder) | **MBAR** (multistate) |
| Asymmetric, forward-only | Cumulant-expansion Jarzynski |

### Stage 1 — Generate forward and reverse work samples

A switching protocol varies λ from 0→1 (forward) or 1→0 (reverse) at finite rate; along each trajectory accumulate `W = ∫ (∂H/∂λ) dλ`. Use the `stochastic-dynamics` Langevin pattern vectorized with `jax.vmap` for N independent replicas.

```python
import jax, jax.numpy as jnp, jax.random as jr

# H(x; λ) = (1 − λ) U_A(x) + λ U_B(x); two harmonic wells, exact ΔF = 0.
U_A = lambda x: 0.5 * x**2
U_B = lambda x: 0.5 * (x - 2.0)**2
H     = lambda x, lam: (1 - lam) * U_A(x) + lam * U_B(x)
dH_dλ = lambda x: U_B(x) - U_A(x)

def switch_langevin(key, x0, lam0, lam1, n_steps, dt, kT, gamma=1.0):
    """Overdamped Langevin on a linear λ schedule; accumulate work."""
    def body(carry, key_t):
        x, W, i = carry
        lam   = lam0 + (lam1 - lam0) * (i / n_steps)
        dW    = dH_dλ(x) * ((lam1 - lam0) / n_steps)
        force = -jax.grad(lambda x_: H(x_, lam))(x)
        noise = jr.normal(key_t, ())
        x_next = x + (force / gamma) * dt + jnp.sqrt(2 * kT * dt / gamma) * noise
        return (x_next, W + dW, i + 1), None
    (_, W, _), _ = jax.lax.scan(body, (x0, 0.0, 0.0), jr.split(key, n_steps))
    return W

N, kT, dt, n_steps = 1000, 1.0, 0.01, 1000
W_F = jax.vmap(lambda k: switch_langevin(k, 0.0, 0.0, 1.0, n_steps, dt, kT))(jr.split(jr.PRNGKey(0), N))
W_R = jax.vmap(lambda k: switch_langevin(k, 2.0, 1.0, 0.0, n_steps, dt, kT))(jr.split(jr.PRNGKey(1), N))
```

### Stage 2 — Fit BAR with pymbar

`pymbar` (Chodera lab) is canonical. The BAR entry point lives at `pymbar.other_estimators.bar` and returns a dict with keys `Delta_f`, `dDelta_f` in `kT` units.

```python
from pymbar.other_estimators import bar   # [verified 2026-04 — dict return, Delta_f / dDelta_f]

result = bar(W_F / kT, W_R / kT, compute_uncertainty=True,
             method="self-consistent-iteration")
dF     = result["Delta_f"]  * kT          # back to energy units
dF_err = result["dDelta_f"] * kT
print(f"BAR: ΔF = {dF:+.4f} ± {dF_err:.4f}   (expected ~0.0)")
```

`method="self-consistent-iteration"` is Bennett's (1976) iterative solver; switch to `"false-position"` when work distributions are very asymmetric and the Gibbs-density crossing falls near the sample range boundary.

### Stage 3 — Variance comparison with naive Jarzynski

Naive Jarzynski uses only forward samples: `ΔF = −kT log⟨exp(−βW_F)⟩`. Unbiased but its variance scales as `exp(2 σ²_W / (kT)²)` — an order-of-magnitude increase in work spread blows up the variance by orders. Running both estimators on the same forward+reverse subsamples (N=50..1000) shows BAR already near the true ΔF at N=50 while naive Jarzynski remains biased at small N. This is the single strongest empirical argument for using BAR whenever both protocol directions are available.

### Stage 4 — Multi-state MBAR

BAR is optimal for *two* states. For a ladder of intermediate states (alchemical λ-windows, Hamiltonian replica exchange, umbrella sampling with multiple restraints), use MBAR — the multistate generalization estimating all `K(K−1)/2` pairwise ΔFs from data pooled across all K states.

```python
from pymbar import MBAR                   # [verified 2026-04 — compute_free_energy_differences()]
# u_kn: reduced potential of sample n evaluated at state k, shape (K, N_total)
# N_k:  samples per state, shape (K,)
result = MBAR(u_kn, N_k).compute_free_energy_differences()
Delta_f_ij = result["Delta_f"]            # K × K pairwise ΔF in kT
```

For production MD, use `alchemlyb` — the ecosystem wrapper that ingests LAMMPS / GROMACS / NAMD / AMBER / OpenMM alchemical output and calls pymbar under the hood. See `advanced-simulations` for the surrounding production MD stack.

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
| **Tilted generator / Doob transform** | Exact optimal bias for rare events; numerical via principal eigenvector of tilted operator — the machinery is the same as the Fokker-Planck stationary eigenproblem in `stochastic-dynamics` (discretize L†_tilted, solve `eigs` for the smallest eigenvalue). Use `scipy.sparse.linalg.eigs` / `slepc4py` in Python, `KrylovKit.jl` / `ArnoldiMethod.jl` in Julia |

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
