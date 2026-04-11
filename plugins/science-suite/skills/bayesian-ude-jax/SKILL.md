---
name: bayesian-ude-jax
description: Build Bayesian Universal Differential Equations end-to-end in JAX with Diffrax + Equinox + NumPyro + Optax. Use when the surrounding training loop is already JAX, when GPU vmap-parallel chain sampling dominates wall-clock, when tight integration with Optax schedulers / HuggingFace models matters more than Julia SciML tooling, or when the team's codebase is Python-first. Covers differentiable ODE solves, Lux-equivalent NN modules, NUTS posterior sampling, warm-start MAP optimization, and the Python analog of Julia's ComponentArrays parameter packing. Use proactively when the user mentions Diffrax, Equinox, neural ODE, Bayesian neural ODE, NumPyro factor, Optax lbfgs, or JAX-first UDE workflow.
---

# Bayesian Universal Differential Equations in JAX

The Python / JAX counterpart to `bayesian-ude-workflow` — same staged pipeline (deterministic warm-start → Bayesian posterior sampling → diagnostics → optional tempering), different ecosystem.

## Expert Agents

For JAX-based Bayesian UDE workflows, delegate to:

- **`jax-pro`** (primary): JAX scientific computing, Diffrax, Equinox, NumPyro integration, Optax schedulers.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
- **`statistical-physicist`** (secondary): Bayesian inference theory, identifiability, sampler geometry, PSIS-LOO.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`

---

## Why a JAX-first path?

The Julia Turing + DiffEq + Lux path (see `bayesian-ude-workflow`) remains the richer ecosystem for stiff SciML with sensitivity-analysis options, symbolic MTK pipelines, and Pigeons non-reversible parallel tempering. **Drop to the JAX stack** when:

- The surrounding training loop is already JAX (RAG + NumPyro, JAX-based physics simulation, HuggingFace integration)
- GPU `vmap`-parallel chain sampling dominates wall-clock and you want one memory budget for both the ODE solve and the NUTS chains
- The team's codebase is Python-first and switching languages per workflow is expensive
- You need tight coupling with Optax learning-rate schedulers, gradient clipping, or pre-trained neural networks from the HuggingFace ecosystem

---

## Stack

| Role | Package | Key API |
|---|---|---|
| Differentiable ODE solve | **`diffrax`** | `diffeqsolve(ODETerm(rhs), ...)`, solvers `Tsit5`, `Dopri5`, `Kvaerno5` (stiff), `ImplicitEuler`; adjoints `BacksolveAdjoint`, `RecursiveCheckpointAdjoint`, `DirectAdjoint` |
| Lux-equivalent NN module | **`equinox`** | `eqx.Module`, `eqx.nn.MLP`, `eqx.partition`, `eqx.combine`, `eqx.filter_jit` |
| Posterior sampling | **`numpyro`** | `NUTS`, `MCMC`, `numpyro.factor` for custom log-densities, `Predictive` for posterior-predictive checks |
| Warm-start MAP optimizer | **`optax`** | `adam`, `adamw`, `lbfgs`, `chain`, `apply_updates`, `cosine_decay_schedule` |
| Parameter packing (ComponentArrays analog) | **`jax.tree_util`** | `tree_flatten`, `tree_unflatten`, `tree_map`; `eqx.partition` for trainable / frozen split |
| Diagnostics | **`arviz`** + **`az.from_numpyro`** | R-hat, ESS, PSIS-LOO, trace plots — see `mcmc-diagnostics` |
| Multimodal escape hatch | **`blackjax`** tempered SMC | closest JAX analog to Pigeons NRPT (see `bayesian-inference` hub ecosystem map) |

---

## Stage 1 — Deterministic warm-start with Optax

Same rationale as the Julia path: initializing NUTS at a MAP estimate dramatically reduces warm-up cost on UDE geometry. Optax gives you the full composable scheduler chain.

```python
import jax, jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import optax

class NeuralCorrection(eqx.Module):
    mlp: eqx.nn.MLP
    alpha: jax.Array
    beta:  jax.Array
    def __call__(self, t, y, args):
        corr = self.mlp(y)
        return jnp.stack([self.alpha * y[0] - corr[0],
                          -self.beta * y[1] + corr[1]])

def loss(model, t_obs, y_obs):
    term = dfx.ODETerm(model)
    sol  = dfx.diffeqsolve(
        term, dfx.Tsit5(),
        t0=t_obs[0], t1=t_obs[-1], dt0=0.01,
        y0=jnp.array([1.0, 1.0]),
        saveat=dfx.SaveAt(ts=t_obs),
        adjoint=dfx.RecursiveCheckpointAdjoint(),   # memory-bounded adjoint
    )
    return jnp.mean((sol.ys - y_obs) ** 2)

model  = NeuralCorrection(
    mlp=eqx.nn.MLP(2, 2, width_size=16, depth=2, key=jax.random.PRNGKey(0)),
    alpha=jnp.array(1.0), beta=jnp.array(1.0),
)
optim  = optax.adam(1e-3)
state  = optim.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def step(model, state, t_obs, y_obs):
    grads = eqx.filter_grad(loss)(model, t_obs, y_obs)
    updates, state = optim.update(grads, state)
    model = eqx.apply_updates(model, updates)
    return model, state

for _ in range(2000):
    model, state = step(model, state, t_obs, y_obs)
```

`eqx.filter` + `eqx.filter_grad` give the clean split between trainable parameters (the arrays inside `NeuralCorrection`) and static structure — the equivalent of `ComponentArrays.getaxes` in Julia.

---

## Stage 2 — Bayesian posterior sampling with NumPyro

Wrap the same `diffeqsolve` call inside a NumPyro `@model` and feed sampled parameters through `eqx.tree_at`:

```python
import numpyro
import numpyro.distributions as dist

def bayesian_ude(y_obs, t_obs, map_model):
    alpha = numpyro.sample("alpha", dist.TruncatedNormal(1.0, 0.5, low=0.0))
    beta  = numpyro.sample("beta",  dist.TruncatedNormal(1.0, 0.5, low=0.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    # Repack the MAP-initialized Equinox module with sampled physical params
    model = eqx.tree_at(lambda m: (m.alpha, m.beta), map_model, (alpha, beta))

    sol = dfx.diffeqsolve(
        dfx.ODETerm(model), dfx.Tsit5(),
        t0=t_obs[0], t1=t_obs[-1], dt0=0.01,
        y0=jnp.array([1.0, 1.0]),
        saveat=dfx.SaveAt(ts=t_obs),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    numpyro.sample("obs", dist.Normal(sol.ys, sigma), obs=y_obs)

mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(bayesian_ude, target_accept_prob=0.85),
    num_warmup=500, num_samples=1500, num_chains=4, chain_method="vectorized",
)
mcmc.run(jax.random.PRNGKey(0), y_obs, t_obs, map_model=model)
```

**`chain_method="vectorized"`** is the key NumPyro / JAX lever: all four chains run under a single `jit`-compiled `vmap`, so the ODE solve happens once per leapfrog step across all chains. This is the JAX equivalent of `MCMCThreads()` in Turing and is typically 2–4× faster per-sample on a GPU.

For NN weights, sample a flat vector with `dist.MultivariateNormal` and repack into the Equinox MLP with `eqx.tree_at` or a hand-rolled unravel (the `jax.tree_util` pattern).

---

## Stage 3 — Sampler selection and diagnostics

| Situation | Sampler | Notes |
|---|---|---|
| Unimodal posterior, NUTS converges from MAP init | `NUTS(target_accept_prob=0.85)` | Fastest; use `chain_method="vectorized"` |
| Suspect multimodality | **BlackJAX tempered SMC** (see `bayesian-inference` ecosystem map) | closest JAX analog to Pigeons; no drop-in NRPT yet |
| High dimension, NUTS too slow | **NumPyro SVI** (`AutoNormal`, `AutoLowRankMultivariateNormal`) | coarse exploration; not a substitute for full MCMC |

**Diagnostics** via `az.from_numpyro(mcmc)`:

```python
import arviz as az

idata = az.from_numpyro(mcmc)
az.summary(idata, hdi_prob=0.95)                # R-hat < 1.01, ESS bulk/tail, MCSE, HDI
az.loo(idata, pointwise=True)                    # PSIS-LOO cross-validation
az.plot_trace(idata, var_names=["alpha", "beta", "sigma"])
```

See `mcmc-diagnostics` for the full diagnostic checklist.

---

## AD / adjoint choice

Unlike Julia's SciMLSensitivity where `GaussAdjoint` is generally preferred, Diffrax's adjoint choice depends on memory vs compute trade-offs:

| Adjoint | Memory | Use for |
|---|---|---|
| `DirectAdjoint()` | High | Small problems; fastest per-step but saves all solver state |
| `RecursiveCheckpointAdjoint()` | Bounded (checkpointing) | **Default for most Bayesian UDEs** — memory-efficient and handles long time spans |
| `BacksolveAdjoint()` | Lowest | Non-stiff only; avoid for stiff solvers and DAEs (same caveats as Julia `BacksolveAdjoint`) |

JAX uses forward-mode AD via Dual numbers on `jax.grad` at low parameter counts — analogous to Julia's ForwardDiff path, but you never specify a `sensealg` for it; JAX auto-dispatches.

---

## Common pitfalls

- **`chain_method="parallel"` needs multiple devices** — without `jax.devices()` returning > 1 it silently runs sequentially. Use `"vectorized"` for single-device multi-chain.
- **`eqx.is_inexact_array` vs `eqx.is_array`** — use the former for optimizer state so integer indices are not treated as trainable.
- **Stiff ODE + `BacksolveAdjoint`** — same instability as Julia; switch to `RecursiveCheckpointAdjoint` on stiff problems.
- **`numpyro.factor` vs `numpyro.sample(..., obs=...)`** — prefer `obs=` for observed data; reserve `factor` for custom likelihoods that don't decompose as a distribution.
- **`dist.MultivariateNormal` on flat NN weights scales poorly** — for wide NNs prefer `dist.Normal(0, prior_std).expand([n])` with a plate, or layer-wise priors.

---

## Composition with SINDy and neighboring skills

- **Bayesian UDE workflow (Julia)** — the parent of this skill; the staged pipeline and symptom-driven sampler selection are shared. See `bayesian-ude-workflow`.
- **NumPyro core mastery** — NUTS tuning, SVI, ArviZ integration, reparameterization patterns. See `numpyro-core-mastery`.
- **JAX Bayesian pro** — JAX-side MCMC internals, NumPyro implementation details. See `jax-bayesian-pro`.
- **JAX diffeq pro** — Diffrax solver catalog, event handling, stiff solver pairings. See `jax-diffeq-pro`.
- **MCMC diagnostics** — R-hat / ESS / PSIS-LOO / ArviZ across Turing / NumPyro / Pigeons chains. See `mcmc-diagnostics`.
- **Consensus MCMC with Pigeons** — Julia-side multimodal escape hatch; BlackJAX tempered SMC is the JAX-side analog. See `consensus-mcmc-pigeons`.
- **Equation discovery** — apply SINDy to the trained neural correction to recover symbolic form. See `equation-discovery`.

---

## Checklist

- [ ] Chose JAX over Julia deliberately (surrounding pipeline is JAX, GPU chain parallelism matters, HuggingFace / Optax integration needed)
- [ ] Stage 1: warm-started with Optax to a MAP estimate before MCMC
- [ ] Used `eqx.filter` + `eqx.filter_grad` for clean trainable / static split
- [ ] Paired the Diffrax adjoint with problem characteristics — `RecursiveCheckpointAdjoint` for most cases, `DirectAdjoint` for small fast problems, avoided `BacksolveAdjoint` on stiff systems
- [ ] Stage 2: used `chain_method="vectorized"` for single-device multi-chain NUTS
- [ ] Initialized NUTS at the MAP estimate via `init_to_value` or `eqx.tree_at` repacking
- [ ] If multimodal: escalated to BlackJAX tempered SMC (or cross-language bridge to Pigeons via juliacall)
- [ ] Ran posterior predictive checks and converted to ArviZ `InferenceData` for R-hat / ESS / PSIS-LOO
- [ ] Documented priors, adjoint choice, and `chain_method` alongside the chain for reproducibility
