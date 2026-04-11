---
name: point-processes
description: Model self-exciting event data with Hawkes processes, mutually exciting multivariate processes, and Bayesian Hawkes with non-parametric backgrounds. Use when the observations are irregular event timestamps rather than a regularly sampled series — earthquake aftershocks, trade arrivals, neuron spike trains, social-media cascades, infectious-disease events, or any clustered-in-time process. Use proactively when the user mentions Hawkes, self-exciting, point process, temporal point process, NRPP, branching ratio, excitation kernel, or the `tick` library.
---

# Point Processes & Self-Exciting Dynamics

Workhorse tools for event-time data where ARMA/GARCH do not apply — the observations are a sequence of irregular timestamps, not a regular grid.

## Expert Agent

For event-time modeling, self-exciting dynamics, and Bayesian point-process inference, delegate to:

- **`statistical-physicist`**: Stochastic processes, branching-process theory, inference on rate functions.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
- **`ml-expert`** (secondary): ML workflows for Hawkes fitting and survival modeling.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`

---

## When to reach for a point process

| Data shape | Right tool |
|------------|------------|
| Regular time grid (samples per second, hour, day) | ARMA/SARIMA/GARCH — see `time-series-analysis` |
| Irregular event timestamps; clustering / aftershocks suspected | **Hawkes process** (parametric excitation kernel) |
| Event rate depends on an external regressor, but no self-excitation | Inhomogeneous Poisson / Cox process with GP intensity |
| Multiple event types that influence each other | **Multivariate Hawkes** (cross-excitation matrix) |
| Non-parametric background rate μ(t) with parametric excitation | Bayesian Hawkes with **HSGP** background — see below |
| Deterministic continuous dynamics between stochastic jumps | **PDMP** — see `catalyst-reactions` |
| Pure chemical master equation (small populations, discrete states) | Gillespie SSA — see `catalyst-reactions` |

---

## Parametric Hawkes via `tick` (Python reference)

`tick` is the mature reference for parametric Hawkes in Python — fast on Linux, thinner wheels on macOS arm64. It handles univariate and multivariate exponential-kernel Hawkes, sum-of-exponentials, and fully non-parametric EM (`HawkesEM`).

```python
from tick.hawkes import HawkesExpKern, SimuHawkes
import numpy as np

# Simulate a 2-node mutually exciting exponential-kernel Hawkes
baseline  = np.array([0.1, 0.2])
adjacency = np.array([[0.3, 0.0], [0.1, 0.4]])       # branching ratios
decay     = 3.0
sim = SimuHawkes(
    baseline  = baseline,
    decays    = decay * np.ones_like(adjacency),
    adjacency = adjacency,
    end_time  = 1000.0,
    seed      = 0,
)
sim.simulate()
timestamps = sim.timestamps          # list[np.ndarray], one per node

# Fit
learner = HawkesExpKern(decays=decay, solver="agd", verbose=False)
learner.fit(timestamps)
baseline_hat, adjacency_hat = learner.baseline, learner.adjacency
```

The **branching ratio** — the spectral radius of the adjacency matrix times the kernel integral — controls stability. `spectral_radius < 1` is the stationarity condition; above it, the process is explosive.

---

## Bayesian Hawkes with HSGP background (JAX / NumPyro)

When the background rate μ(t) is non-parametric and you want fully-Bayesian uncertainty on both μ and the excitation kernel, combine NumPyro's **Hilbert-space Gaussian process** approximation (`numpyro.contrib.hsgp`) — O(M) cost with M basis functions — with a parametric excitation factor inside a single `@model`.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.approximation import hsgp_squared_exponential

def hawkes_hsgp(event_times, T_end, M=20):
    length = numpyro.sample("length", dist.LogNormal(0.0, 1.0))
    alpha  = numpyro.sample("alpha",  dist.HalfNormal(1.0))       # excitation scale
    beta   = numpyro.sample("beta",   dist.HalfNormal(1.0))       # excitation decay
    # HSGP log-rate background
    log_mu_t = hsgp_squared_exponential(
        x=event_times, alpha=1.0, length=length, ell=T_end, m=M,
    )
    # Self-excitation term: sum over prior events
    dt     = event_times[:, None] - event_times[None, :]
    mask   = dt > 0
    excite = alpha * jnp.sum(jnp.exp(-beta * dt) * mask, axis=1)
    lam_t  = jnp.exp(log_mu_t) + excite
    # Log-likelihood of the inhomogeneous Hawkes process on [0, T_end]
    numpyro.factor(
        "loglik",
        jnp.sum(jnp.log(lam_t)) - jnp.trapezoid(lam_t, event_times),
    )
```

Sample with NUTS from `numpyro-core-mastery`, diagnose with `mcmc-diagnostics`. If the posterior is multimodal (e.g. excitation kernel competes with a flexible background), route to `consensus-mcmc-pigeons` via juliacall.

---

## Julia ecosystem

| Package | Role |
|---------|------|
| **`PointProcesses.jl`** | Unified `AbstractPointProcess` interface — Poisson, Hawkes, marked processes; `simulate`, `logdensity`, `fit_mle`. Best starting point in Julia. |
| **`HawkesProcesses.jl`** | Lightweight Hawkes simulation and MLE fitting; focused on univariate / low-dimensional multivariate cases. |
| **`MultivariateHawkesProcesses.jl`** | Large-dimensional mutually exciting Hawkes; branching-ratio estimation on graphs. |
| **`Turing.jl` + `PointProcesses.jl`** | Fully-Bayesian Hawkes by wrapping the log-density in a `Turing.@model`. Compose with `consensus-mcmc-pigeons` for multimodal posteriors. |

---

## Diagnostics specific to point processes

Standard time-series diagnostics don't apply. The key tool is the **time-rescaling theorem**: a correctly fit point process, when transformed by its cumulative conditional intensity, becomes a unit-rate Poisson process. The transformed inter-event intervals should then be IID `Exponential(1)`.

- **Rescaled-residuals KS test** — transform events via `Λ(t) = ∫ λ(s) ds`; test the resulting inter-arrival times against `Exponential(1)`.
- **QQ plot of rescaled residuals** — visual analog of the KS test.
- **Autocorrelation of residuals** — should be flat if excitation is fully captured.

`tick` exposes `learner.score()` for log-likelihood. For the Bayesian path, use posterior-predictive rescaled residuals and compare to the null via PSIS-LOO in ArviZ.

---

## Composition with neighboring skills

- **Time series analysis** — regularly sampled series, GARCH, state-space. See `time-series-analysis`.
- **NumPyro core mastery** — NUTS sampler, HSGP, `numpyro.factor` patterns. See `numpyro-core-mastery`.
- **Turing model design** — Julia-side Bayesian Hawkes via `@model`. See `turing-model-design`.
- **Consensus MCMC with Pigeons** — multimodal escape hatch for Bayesian Hawkes. See `consensus-mcmc-pigeons`.
- **MCMC diagnostics** — rescaled-residuals workflow post-fit. See `mcmc-diagnostics`.
- **Catalyst reactions** — jump-process side (PDMP, Gillespie SSA, jump-diffusion) for reaction-network systems. See `catalyst-reactions`.
- **Bayesian inference hub** — parent routing. See `bayesian-inference`.

---

## Checklist

- [ ] Confirmed the data is event-time (timestamps), not a regular grid
- [ ] Picked univariate vs multivariate vs marked based on the event types
- [ ] Verified **spectral radius < 1** for stationarity on any fitted Hawkes adjacency
- [ ] Used HSGP background (`numpyro.contrib.hsgp`) when μ(t) is non-parametric
- [ ] Ran the rescaled-residuals KS test on the final fit
- [ ] For Bayesian fits: checked R-hat / ESS and PSIS-LOO via `mcmc-diagnostics`
- [ ] If the posterior was multimodal, escalated to Pigeons via `consensus-mcmc-pigeons`
