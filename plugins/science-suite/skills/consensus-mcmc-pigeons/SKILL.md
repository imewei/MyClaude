---
name: consensus-mcmc-pigeons
description: Sample multimodal posteriors with Pigeons.jl non-reversible parallel tempering. Use when NUTS chains fail to mix across modes, R-hat stays elevated, or the posterior is multimodal by construction (mixtures, weakly identified UDEs, label-switching). Also use when scaling MCMC across MPI ranks or threads. Use proactively when the user mentions consensus Monte Carlo, parallel tempering, NRPT, multimodal posterior, or Pigeons.
---

# Consensus MCMC with Pigeons.jl

Non-Reversible Parallel Tempering (NRPT) for posteriors that defeat single-chain samplers.

> **Two different things share the "Consensus MC" name — read this first.**
>
> | Flavor | Algorithm | When it helps | Where it lives |
> |--------|-----------|---------------|-----------------|
> | **Pigeons NRPT** (this skill) | Run `n_chains` tempered replicas of the *same* full-data posterior on one machine (or across MPI ranks); replicas swap non-reversibly along the temperature ladder. Cold chain samples the posterior. | Multimodal posteriors, weakly identified UDEs, label-switching, thin-neck posteriors. Data is not partitioned. | This skill. |
> | **Scott-2016 Consensus Monte Carlo** (divide-and-conquer) | Shard the data into `K` pieces, run NUTS on each shard with a rescaled prior `p(θ)^{1/K}`, then combine the `K` sub-posteriors by weighted-average (or density-product) to approximate the full-data posterior. | Posterior is *unimodal* but the full dataset is too large to fit on one node; embarrassingly parallel is acceptable. | Not covered in depth here — the NumPyro / Turing `@model` is identical to the single-machine version; only the orchestration differs. Use `numpyro-core-mastery` + a thin sharding layer. |
>
> These solve different problems. If your issue is multimodality or a bad posterior geometry, you want Pigeons. If your issue is data volume and the posterior is well-behaved, you want Scott-2016 divide-and-conquer. The rest of this skill is about Pigeons NRPT.

## Expert Agent

For multimodal Bayesian inference and parallel tempering workflows, delegate to:

- **`statistical-physicist`**: Bayesian inference, MCMC theory, replica-exchange methods.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
- **`julia-pro`** (secondary): Julia integration patterns with Turing and SciML.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

---

## When to use Pigeons (symptom-driven)

Reach for Pigeons when one or more of these holds:

| Symptom | Why NUTS fails | Why Pigeons helps |
|---------|----------------|-------------------|
| R-hat > 1.01 with high ESS in each chain individually | Chains visit disjoint modes; per-chain mixing is fine but inter-chain disagreement is large | Tempered chains traverse a flattened reference; cold chain inherits global mixing |
| Posterior is a mixture of well-separated components | NUTS gradient never crosses the trough between modes | Reference distribution lifts the trough; replicas swap across it |
| UDE / weakly identified parameters | Likelihood ridge with multiple basins (sign degeneracies, neuron permutations) | Tempering smooths the ridge; consensus across replicas resolves identifiability |
| Label-switching in mixture / cluster models | NUTS gets stuck in one labeling | Tempered swaps re-explore the labeling combinatorics |

**Don't use Pigeons** for low-dimensional unimodal targets — NUTS already converges and Pigeons pays a per-iteration overhead.

---

## Basic pattern (Turing target)

Pigeons frames itself broadly as a tool for **sampling from intractable distributions** — multimodal posteriors are the most common motivation, but the same machinery handles weakly identified models, posteriors with thin necks, and Lebesgue-integration problems generally. Wrap any Turing model as a `TuringLogPotential` and pass it to `pigeons`:

```julia
using Pigeons, Turing

@model function my_model(data)
    μ ~ MixtureModel([Normal(-3, 1), Normal(3, 1)])
    σ ~ truncated(Normal(0, 1), 0, Inf)
    data .~ Normal(μ, σ)
end

target = TuringLogPotential(my_model(observed))
pt = pigeons(
    target    = target,
    n_chains  = 10,                # cold + 9 tempered replicas
    n_rounds  = 10,                # doubling rounds; total iters ≈ 2^n_rounds
    seed      = 1,
)

samples = Chains(pt)              # MCMCChains.Chains for downstream tooling
```

The **cold chain** (chain index 1) targets the actual posterior. Other chains target tempered intermediates between a **reference distribution** (default: the prior) and the posterior.

### Variational reference (`GaussianReference`)

When the prior is a poor reference (very wide, very different shape from the posterior), tempered chains spend most of their time in low-density regions and round-trip rates collapse. Pigeons can **learn** a Gaussian variational reference instead and replace the prior endpoint of the temperature ladder:

```julia
pt = pigeons(
    target      = TuringLogPotential(my_model(observed)),
    n_chains    = 10,
    n_rounds    = 10,
    variational = GaussianReference(),     # learn a Gaussian reference between rounds
    seed        = 1,
)
```

This often improves the global communication barrier `Λ` by an order of magnitude on UDE-style posteriors and other ill-conditioned problems where the prior is wildly broader than the posterior. Combine with `n_chains` tuning until `Λ / n_chains` is small.

### Advanced configuration (`Inputs`)

Every keyword to `pigeons(...)` is also a field of the `Pigeons.Inputs` type, which lets you build a configuration object up incrementally and re-use it across runs (helpful for benchmarks or HPC jobs that rerun the same target with different seeds).

---

## Composition with arbitrary log-densities

Pigeons targets are not Turing-specific. Anything implementing the `LogDensityProblems` interface works:

```julia
struct MyTarget end
LogDensityProblems.logdensity(::MyTarget, x) = -0.5 * sum(abs2, x .- [3, -3])
LogDensityProblems.dimension(::MyTarget) = 2

pt = pigeons(target = MyTarget(), n_chains = 8)
```

This is how Bayesian UDE workflows feed a custom likelihood (DiffEq solve + observation model) to Pigeons. See `bayesian-ude-workflow` for the full pattern.

---

## Why non-reversible PT?

Standard parallel tempering proposes swaps between adjacent temperature rungs symmetrically — replicas perform a random walk on the temperature ladder. **Non-reversible** PT biases swap proposals so replicas drift monotonically up or down the ladder, achieving an order-of-magnitude better round-trip rate. The cold chain therefore samples a fresh global state every few rounds instead of decorrelating only locally.

The **global communication barrier** (`Λ`) that Pigeons reports estimates how much information must flow between rungs. Add chains until `Λ / n_chains < 1` for healthy mixing.

---

## Scaling

| Backend | When to use |
|---------|-------------|
| Single-process multithreaded | Default. Chains run on Julia threads; one machine. |
| MPI launcher (`mpi_run`) | Multi-node clusters; scales to **thousands of MPI-communicating ranks**. Pigeons handles the rank↔chain mapping. |
| Distributed.jl | Heterogeneous node mix or non-MPI clusters. |

Use the multithreaded backend until you exceed a single node, then switch to MPI — the target wrapper code does not change.

---

## Diagnostics

Pigeons reports its own diagnostics each round:

- **Global communication barrier (`Λ`)** — primary tuning signal. Increase `n_chains` until `Λ / n_chains` is small.
- **Round-trip rate** — how often a replica completes a full ladder traversal. Higher is better.
- **Log normalization constant estimate** — bonus output, useful for model comparison.

For posterior diagnostics on the cold chain, convert to `MCMCChains.Chains` and use the standard tooling. See `mcmc-diagnostics` for R-hat, ESS, BFMI, PSIS-LOO, and ArviZ workflows.

---

## Composition with neighboring skills

- **Turing model design** — author the `@model`, then wrap with `TuringLogPotential`. See `turing-model-design`.
- **Bayesian UDE workflow** — Pigeons is the multimodal escape hatch when NUTS fails on UDE posteriors. See `bayesian-ude-workflow`.
- **MCMC diagnostics** — post-processing of cold chain. See `mcmc-diagnostics`.
- **Variational inference patterns** — VI is the alternative when even tempering is too expensive. See `variational-inference-patterns`.

---

## Checklist

- [ ] Confirmed the posterior is multimodal (or NUTS R-hat persistently > 1.01) before reaching for Pigeons
- [ ] Started with the multithreaded backend; only switched to MPI after exceeding one node
- [ ] Picked `n_chains` so that `Λ / n_chains < 1` after a tuning round
- [ ] Used the **cold chain** (index 1) for posterior summaries — never a tempered intermediate
- [ ] Converted output to `MCMCChains.Chains` and ran R-hat / ESS via `mcmc-diagnostics`
- [ ] Verified round-trip rate is non-trivial (replicas actually traverse the ladder)
- [ ] Documented the choice of reference distribution (default is the prior for `TuringLogPotential`)
