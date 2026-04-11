---
name: statistical-physics
description: Comprehensive statistical physics suite covering equilibrium and non-equilibrium statistical mechanics, active matter, stochastic dynamics, and correlation analysis. Master the bridge between microscopic laws and macroscopic behavior. Use when computing partition functions, analyzing phase transitions, deriving fluctuation-dissipation relations, or modeling active matter systems.
---

# Statistical Physics

Master the theoretical and computational tools of statistical mechanics.

## Expert Agent

For complex statistical mechanics problems, active matter simulations, and theoretical derivations, delegate to the expert agent:

- **`statistical-physicist`**: Unified specialist for Statistical Physics and Soft Matter.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: Non-equilibrium thermodynamics, active matter simulations, correlation function analysis, and stochastic calculus.

## Core Skills

### [Stochastic Dynamics](../stochastic-dynamics/SKILL.md)
Langevin equations, Fokker-Planck modeling, and noise-driven systems.

### [Non-Equilibrium Theory](../non-equilibrium-theory/SKILL.md)
Fluctuation theorems, entropy production, and driven systems.

### [Active Matter](../active-matter/SKILL.md)
Self-propelled particles, MIPS, and collective behavior.

### [Correlation Analysis](../data-analysis/SKILL.md)
FFT-based analysis, structure factors, and dynamic heterogeneity.

### [Correlation Math Foundations](../correlation-math-foundations/SKILL.md)
Mathematical basis for two-point and higher-order correlation functions.

### [Correlation Physical Systems](../correlation-physical-systems/SKILL.md)
S(q), g(r), and χ₄(t) across different physical states.

### [Correlation Computational Methods](../correlation-computational-methods/SKILL.md)
O(N log N) algorithms and multi-tau implementation.

### [Correlation Experimental Data](../correlation-experimental-data/SKILL.md)
Extracting physical parameters from DLS, SAXS, and microscopy.

## 1. Equilibrium Statistical Mechanics

- **Ensembles**: Microcanonical (NVE), Canonical (NVT), Grand Canonical (μVT).
- **Phase Transitions**: Critical phenomena, scaling laws, and order parameters.
- **Fluctuations**: Fluctuation-Dissipation Theorem (FDT) and linear response.

### Monte Carlo samplers — ecosystem selection

| Update rule | Use for | Python production | Julia idiom |
|-------------|---------|--------------------|-------------|
| Metropolis-Hastings | Generic equilibrium sampling, Ising / XY / Heisenberg / lattice gauge, classical fluids | HOOMD-blue, LAMMPS (`fix sgcmc`), ESPResSo | Hand-rolled with `Random` + `StaticArrays.jl`; ~20 lines |
| Heat-bath / Glauber | Ising-like lattice models | LAMMPS, custom | Hand-rolled (deterministic transition probabilities, no accept/reject) |
| Cluster (Wolff / Swendsen-Wang) | Critical-slowing-down systems near Tc — breaks critical-slowing bottleneck | HOOMD-blue cluster moves, custom | Hand-rolled union-find cluster build |
| Wang-Landau | Density of states, flat histograms across energy | LAMMPS (WL via REXMC), custom | Hand-rolled; one counter + one log-g array |
| Replica exchange (parallel tempering) | Multimodal or rough landscapes | LAMMPS `temper`, HOOMD-blue + MPI | `Distributed.jl` or `MPI.jl` + manual exchange step |
| Event-driven / kinetic MC | Dynamics on a lattice with time-scale separation | `kmcpy`, `SPPARKS` | Hand-rolled; priority queue via `DataStructures.jl` |

**Why Python dominates production MC**: LAMMPS and HOOMD-blue ship optimized C++/CUDA implementations of every canonical update rule plus replica-exchange orchestration at MPI scale. Julia has no equivalent production package — for ensemble sampling on >10⁶ spins or >10⁴ atoms, the honest path is to drive Python from Julia via `PyCall.jl` / `PythonCall.jl`, or stage the MC run as a separate HOOMD-blue / LAMMPS job.

**Why Julia is idiomatic for research prototyping**: the Metropolis inner loop is ~20 lines of Julia using stdlib + `StaticArrays.jl`, runs at C-equivalent speed thanks to the type system, and integrates naturally with `OnlineStats.jl` (verified active) for streaming observable accumulation and `Distributed.jl` for embarrassingly-parallel replica ensembles. The canonical pattern:

```julia
using StaticArrays, Random, OnlineStats, Distributed

# Energy on a square-lattice Ising model
@inline function ΔE(spins, i, j, L, J=1.0)
    s = spins[i, j]
    nn = spins[mod1(i+1, L), j] + spins[mod1(i-1, L), j] +
         spins[i, mod1(j+1, L)] + spins[i, mod1(j-1, L)]
    return 2 * J * s * nn
end

function metropolis_sweep!(spins, β, rng)
    L = size(spins, 1)
    for _ in 1:L^2
        i, j = rand(rng, 1:L), rand(rng, 1:L)
        δE  = ΔE(spins, i, j, L)
        if δE ≤ 0 || rand(rng) < exp(-β * δE)
            spins[i, j] = -spins[i, j]
        end
    end
end

function run_chain(L, β, n_sweeps; thermalize=1000, seed=0)
    rng   = Xoshiro(seed)
    spins = rand(rng, (-1, 1), L, L)
    for _ in 1:thermalize
        metropolis_sweep!(spins, β, rng)
    end
    mag = Series(Mean(), Variance())        # OnlineStats accumulators
    for _ in 1:n_sweeps
        metropolis_sweep!(spins, β, rng)
        fit!(mag, sum(spins) / L^2)
    end
    return value(mag)                        # (mean, variance) of magnetization
end
```

Parallel tempering across β-replicas: spin up workers with `Distributed.addprocs(N)`, run independent chains with `pmap`, and exchange configurations at Metropolis-weighted swap rates. For exact error bars on correlated-sample means, combine `OnlineStats.jl` with log-binning analysis — the `BinningAnalysis.jl` package (Carsten Bauer's group) implements this; verify its current status on GitHub before depending on it for production work.

**When to reach outside Julia**: if the system is large enough that a single Julia thread can't keep up with the per-sweep cost, *or* if you need an MC update rule (e.g. geometric cluster moves, directed-loop, worm algorithm) that has an existing LAMMPS / HOOMD-blue implementation, delegate to the Python production path via `PythonCall.jl`. See `advanced-simulations` for the HOOMD-blue / LAMMPS / ESPResSo side of the ecosystem.

### Error analysis for correlated samples

MC samples are autocorrelated; naïve `std / sqrt(N)` underestimates the error. Tools:

- **Log-binning** — `BinningAnalysis.jl` (Julia, well-known, not Context7-indexed) or hand-rolled recursive binning; the plateau in the binned variance gives the integrated autocorrelation time.
- **Jackknife** — robust for derived quantities (ratios, susceptibilities).
- **Bootstrap** — `scipy.stats.bootstrap` (Python), `Bootstrap.jl` (Julia).
- **Auto-correlation time** — `emcee.autocorr` (Python), or Madras-Sokal adaptive estimator hand-rolled in Julia.

## 2. Non-Equilibrium & Active Matter

- **Stochastic Dynamics**: Langevin equations, Fokker-Planck, and Master equations.
- **Active Matter**: Self-propelled particles (ABPs), motility-induced phase separation (MIPS).
- **Fluctuation Theorems**: Jarzynski equality, Crooks fluctuation theorem, and entropy production.

## 3. Correlation Analysis

- **Mathematical Foundations**: Wiener-Khinchin theorem, Fourier transforms, and Green's functions.
- **Computational Methods**: FFT-based autocorrelation, multi-tau algorithms, and block averaging.
- **Physical Systems**: Structure factors S(q), radial distribution functions g(r), and dynamic heterogeneity χ₄(t).

## 4. Experimental Connection

- **Scattering**: SAXS/SANS/DLS interpretation.
- **Rheology**: Viscoelasticity and transport coefficients.
- **Microscopy**: Particle tracking and trajectory analysis.

## Checklist

- [ ] Verify ensemble selection matches experimental constraints (NVE, NVT, NPT, or grand canonical)
- [ ] Confirm partition function calculation accounts for all relevant microstates
- [ ] Check that order parameter correctly identifies the phase transition (e.g., magnetization, density)
- [ ] Validate fluctuation-dissipation theorem connects measured susceptibility to equilibrium fluctuations
- [ ] Ensure critical exponents satisfy known scaling relations (e.g., Rushbrooke, Widom)
- [ ] Verify finite-size scaling extrapolation to thermodynamic limit L -> infinity
- [ ] Confirm entropy production calculation is non-negative for irreversible processes
- [ ] Check that Langevin/Fokker-Planck parameters match the physical noise source
- [ ] Validate correlation function analysis against known sum rules and asymptotic behaviors
