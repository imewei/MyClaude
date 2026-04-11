---
name: rare-events-sampling
description: Sample rare events and compute their rates with Forward Flux Sampling (FFS), Transition Path Sampling (TPS/TIS/RETIS), Adaptive Multilevel Splitting (AMS), Weighted Ensemble (WE), milestoning, and large-deviation cloning algorithms. Use when direct simulation never reaches the basin of interest, when you need the rate of a thermally-activated barrier crossing, when computing Jarzynski / Crooks free energies from non-equilibrium pulling, or when extracting avalanche / cascade statistics from driven systems. Use proactively when the user mentions FFS, TIS, RETIS, TPS, OPS, WESTPA, pyretis, weighted ensemble, milestoning, metadynamics, umbrella sampling, cloning, SCGF, large-deviation, rate function, or rare event.
---

# Rare-Events Sampling

Consolidated entry point for the rare-event family — the set of methods that make vanishingly-probable trajectories statistically accessible without waiting for them to happen on their own.

## Expert Agents

For rare-event sampling, path-sampling ensembles, and large-deviation machinery, delegate to:

- **`simulation-expert`**: Production-grade path sampling, WESTPA / OPS / pyretis workflows, HPC scaling.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
- **`statistical-physicist`** (secondary): Large-deviation theory, SCGF / rate functions, Jarzynski / Crooks theory, avalanche statistics.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`

---

## When to reach for rare-event sampling

Symptom-driven decision table:

| Symptom | Right tool family |
|---|---|
| Direct Langevin / MD never crosses the barrier in any feasible wall-clock time | FFS, TPS / TIS / RETIS, WE |
| You need a free-energy difference, not just a rate | BAR / Crooks / Jarzynski (see `non-equilibrium-theory`) driven by **metadynamics** or **umbrella sampling** |
| Reaction pathway / transition-state structure is unknown | **TPS** (shooting moves) or **string method** |
| The committor function matters | **milestoning** on a committor-aligned partition |
| You want the SCGF `λ(k)` / rate function `I(a)` for a large-deviation observable | **cloning** / **Giardinà-Kurchan-Lecomte-Tailleur** |
| Event data is a bursty sequence (avalanches, aftershocks) rather than a single barrier crossing | power-law exponent fits + point-process rate function (see `point-processes` and `non-equilibrium-theory`) |

---

## Method selection

| Method | Core idea | Best for |
|---|---|---|
| **Forward Flux Sampling (FFS)** | Sequential interfaces `λ_0 < λ_1 < … < λ_n`; cross one at a time | Cheap in driven non-equilibrium; no committor needed upfront |
| **Transition Path Sampling (TPS)** | Metropolis MC in trajectory space (shooting moves); target is the reactive-path ensemble | Pathway / transition-state discovery when the endpoints are known |
| **Transition Interface Sampling (TIS/RETIS)** | Interface-wise path ensembles + replica exchange | Rate + path ensemble + committor in one shot |
| **Adaptive Multilevel Splitting (AMS)** | Genetic resampling at adaptive level sets | Simple, unbiased, easy to automate |
| **Weighted Ensemble (WE)** | Parallel walkers with replication + pruning under a fixed bin structure | Large-scale MD; embarrassingly parallel; what WESTPA provides |
| **Metadynamics / Umbrella Sampling** | Bias-potential methods that flatten the free-energy landscape in CV space | Free-energy surfaces when good CVs exist |
| **Milestoning** | Short trajectories between Voronoi cells in committor space | Rate via first-hitting matrix; no bias applied |
| **Cloning / GKLT** | Population Monte Carlo with replica branching | SCGF of large-deviation observables; avalanche exponents |

---

## Python ecosystem

| Role | Package | Key API |
|---|---|---|
| Path sampling (TPS / TIS / RETIS) | **`openpathsampling` (OPS)** | `Engine`, `Volume`, `CollectiveVariable`, `MoveScheme`, `PathHistogram`, `Storage` (HDF5) |
| Weighted ensemble | **`westpa`** | `w_init`, `w_run`, `w_assign`, `w_pdist`; HDF5 iteration files; supports OpenMM, GROMACS, LAMMPS engines |
| FFS / TIS / RETIS (lightweight) | **`pyretis`** | `RETIS`, `FFS`, `Path`, ensemble-crossing analysis, plug-in engines |
| Generic walker framework | **`wepy`** | `RandomWalkProcess`, resamplers, walker checkpointing, custom boundary conditions |
| Bias-potential MD (metadynamics, umbrella) | **`PLUMED`** (C++ library with Python bindings + plugins for LAMMPS / GROMACS / OpenMM / NAMD) | `plumed.Plumed().cmd(...)`, `hills` files, COLVAR analysis |
| Bias-potential analysis | **`pymbar`** / **`alchemlyb`** | `MBAR`, `BAR`, `TI`, subsampling, PMF extraction |
| Large-deviation cloning | hand-rolled on `jax.vmap` replicas | population + branching + SCGF estimation |
| Avalanche exponent fits | **`powerlaw`** (Alstott et al.) | MLE power-law, KS statistic, distribution comparison |

---

## Julia ecosystem

Julia has no direct equivalent to OPS / WESTPA — instead, rare-event workflows **compose from SciML primitives**:

| Role | Package | Pattern |
|---|---|---|
| Event-driven SDE / jump simulation | `StochasticDiffEq`, `JumpProcesses` | solve with `CallbackSet` on interface-crossing events |
| Interface-crossing callbacks | `DiffEqCallbacks` | `ContinuousCallback` on `λ_k(u) - λ_k^*` sign changes |
| Walker population for AMS / WE | hand-rolled ensemble + resampling | `EnsembleProblem` + `EnsembleThreads` + custom resampler |
| Tipping / escape indicators | **`CriticalTransitions.jl`** | escape rates, tipping proximity, mean-first-passage times |
| Agent-based cloning populations | **`Agents.jl`** | particle branching for SCGF / population MC |
| Power-law avalanche fits | **`HeavyTails.jl`** | MLE power-law exponents and cutoffs |

---

## Minimal pattern — OPS two-state TPS

```python
import openpathsampling as paths
from openpathsampling import engines

# Define the state volumes A and B via a collective variable
cv  = paths.MDTrajFunctionCV("q", md_q_function, topology)
A   = paths.CVDefinedVolume(cv, -1e6, 0.0)    # reactant basin
B   = paths.CVDefinedVolume(cv,  1.0, 1e6)    # product basin

# Wrap an engine (OpenMM / LAMMPS / GROMACS) via OPS's engine adapter
engine = engines.OpenMMEngine(topology, system, integrator)
engine.initialized = True

# Build the TPS network + move scheme
network = paths.TPSNetwork(initial_states=A, final_states=B)
scheme  = paths.OneWayShootingMoveScheme(network=network, engine=engine)

# Run Metropolis in trajectory space
storage = paths.Storage("tps.nc", mode="w")
sampler = paths.PathSampling(storage=storage, move_scheme=scheme, sample_set=...)
sampler.run(5000)                              # 5000 Metropolis steps in path space
```

## Minimal pattern — JAX cloning for SCGF

```python
import jax, jax.numpy as jnp, jax.random as jr

def simulate_one(step_fn, x0, n_steps, key):
    keys = jr.split(key, n_steps)
    return jax.lax.scan(step_fn, x0, keys)

def cloning_step(replicas, weights, s, observable):
    # Weight replicas by exp(s * A_T) for the biased dynamics target
    log_w = s * jax.vmap(observable)(replicas)
    w = jnp.exp(log_w - log_w.max())
    w /= w.sum()
    # Resample (multinomial cloning)
    n = replicas.shape[0]
    idx = jr.choice(jr.PRNGKey(0), n, shape=(n,), p=w, replace=True)
    return replicas[idx], jnp.log(jnp.mean(jnp.exp(log_w)))   # branching factor contributes to λ(s)
```

Accumulate the branching-factor logs to estimate `λ(s) = (1/T) log⟨exp(s · A_T)⟩`, then Legendre-transform to recover the rate function `I(a)`.

---

## Diagnostics

| Check | What to measure |
|---|---|
| **Efficiency ratio** | `τ_direct / τ_rare-event` — should be orders of magnitude in your favour |
| **Variance reduction** | Observed estimator variance vs naive MC at the same wall-clock cost |
| **Rate self-consistency** | Forward and reverse FFS / TIS rates should satisfy detailed balance at equilibrium |
| **Committor test** | Short trajectories launched from the transition state should commit to product ~50% of the time |
| **Replica mixing** (RETIS / parallel tempering) | Replicas should round-trip through the full interface ladder |
| **Avalanche exponent stability** | MLE exponents stable under sub-sampling and choice of cutoff (`xmin`) |

---

## Composition with neighboring skills

- **Non-equilibrium theory** — Crooks / Jarzynski / BAR free-energy estimation driven by rare-event samplers; SCGF / rate-function theory. See `non-equilibrium-theory`.
- **Stochastic dynamics** — the underlying Langevin / SDE / master-equation dynamics that rare-event samplers wrap. See `stochastic-dynamics`.
- **Advanced simulations** — production MD engines (OpenMM, LAMMPS, GROMACS) that OPS / WESTPA / pyretis drive. See `advanced-simulations`.
- **Chaos and attractors** — escape from metastable basins and tipping-point indicators complement rare-event sampling theoretically. See `chaos-attractors`.
- **Point processes** — for bursty event-time data (avalanches, aftershocks) model the rate as a Hawkes / Cox process. See `point-processes`.
- **MCMC diagnostics** — when the rare-event estimator is itself Bayesian (e.g., Bayesian inference of rate constants), validate with R-hat / ESS. See `mcmc-diagnostics`.

---

## Checklist

- [ ] Identified the symptom (no barrier crossing / need free energy / need rate / need avalanche exponents) before picking a method
- [ ] Chose a tool that matches the engine: **WESTPA** for production MD, **OPS** for pathway discovery, **pyretis** for standalone FFS/TIS, **cloning** for SCGF
- [ ] For AMS / WE / FFS: verified that the level sets / interfaces are well-separated and coverage-safe
- [ ] For TPS: checked the reactive-path ensemble is sampling both forward and reverse directions
- [ ] Ran a committor test from the estimated transition state
- [ ] Compared the rare-event rate against a short direct-simulation check at elevated temperature or lowered barrier
- [ ] For avalanche / large-deviation observables: fit exponents with MLE, not log-log least squares
- [ ] Documented the collective variable / order parameter choice — rare-event rates are CV-dependent
