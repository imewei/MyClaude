---
name: catalyst-reactions
maturity: "5-Expert"
specialization: Reaction Networks
description: Model chemical reaction networks with Catalyst.jl for deterministic and stochastic simulations. Use when modeling biochemical pathways, chemical kinetics, gene regulatory networks, enzyme kinetics, or metabolic networks. Also use when defining reaction stoichiometry with @reaction_network, converting between ODE/SDE/Jump systems, running Gillespie stochastic simulations, or building systems biology models in Julia, even if the user doesn't mention Catalyst.jl by name.
---

# Catalyst.jl Reaction Networks

## Expert Agent

For reaction network modeling with Catalyst.jl, delegate to:

- **`julia-pro`**: Julia SciML ecosystem, differential equations, and symbolic modeling.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

Chemical and biochemical reaction network modeling.

---

## Reaction Network Pattern

```julia
using Catalyst, DifferentialEquations

rn = @reaction_network begin
    k1, A + B --> C
    k2, C --> A + B
end k1 k2

odesys = convert(ODESystem, rn)
prob = ODEProblem(odesys, [A => 10, B => 10, C => 0],
                  (0.0, 10.0), [k1 => 0.1, k2 => 0.05])
sol = solve(prob, Tsit5())
```

---

## Simulation Types

| Type | Convert To | Use Case |
|------|------------|----------|
| Deterministic | ODESystem | Large populations |
| Stochastic | JumpSystem | Small populations |
| SDE | SDESystem | Chemical noise (Gaussian approximation) |
| Jump-diffusion | `JumpProblem(sde_prob, ...)` | Continuous noise + rare discrete jumps |
| PDMP | `PDMPProblem(F!, R!, DX, ...)` | Deterministic flow between stochastic events — see Jump processes section |

---

## Applications

- Systems biology
- Metabolic networks
- Gene regulation
- Enzyme kinetics
- Chemical engineering

---

## Python Counterpart (SBML ecosystem)

JAX / NumPy has **no direct equivalent** to Catalyst.jl's symbolic reaction-network compilation. The closest Python stack is SBML-based:

| Role | Python package | Notes |
|------|----------------|-------|
| Antimony DSL → SBML simulation | **tellurium** | `te.loada()` → libRoadRunner LLVM-JIT ODE solver; round-trips Antimony ↔ SBML ↔ CellML; SED-ML support |
| Low-level SBML I/O | **python-libsbml** | `SBMLReader`/`SBMLWriter`; programmatic network construction; L3 package support (comp, fbc, qual) |
| Gillespie SSA | **gillespy2** | Exact SSA (Direct), tau-leaping, hybrid ODE+jump; compiled C++ solvers. Check current release cadence before relying for production use |

```python
import tellurium as te

model = te.loada("""
    A + B -> C; k1*A*B
    C -> A + B; k2*C
    k1 = 0.1; k2 = 0.05
    A = 10; B = 10; C = 0
""")
result = model.simulate(0, 10, 100)
# model.exportToSBML('network.xml')  # round-trip to SBML
```

> **Bridge pattern**: for symbolic features unique to Catalyst.jl (DSL-driven index reduction, symbolic Jacobians, unified ODE/SDE/Jump conversion), call Julia from Python via `juliacall`. Tellurium covers the simulation path but not the symbolic modeling depth of Catalyst + ModelingToolkit.

---

## Jump processes beyond pure SSA

Pure Gillespie/SSA is only one point in the jump-process design space. When the dynamics mix discrete jumps with continuous evolution — continuous populations with rare switching events, piecewise-deterministic trajectories, stiff reaction networks with slow/fast timescale separation — reach for the richer toolkit:

| Regime | Julia package | Pattern |
|--------|---------------|---------|
| **Exact SSA, tau-leaping, RSSA, coupled constant-rate** | `JumpProcesses.jl` (part of `DifferentialEquations.jl`) | `JumpProblem(prob, Direct(), jumps...)`; composes with any `ODEProblem` / `SDEProblem` |
| **Piecewise-deterministic Markov processes (PDMP)** — continuous ODE flow punctuated by rate-dependent jumps | `PiecewiseDeterministicMarkovProcesses.jl` | `PDMPProblem(F!, R!, DX, xc0, xd0, ...)` — specify the continuous flow `F!`, rate function `R!`, and discrete state update `DX` |
| **Jump-diffusion** — SDE with additional discrete jump events | `DifferentialEquations.jl` `JumpProblem(sde_prob, Direct(), jumps...)` | Wrap an `SDEProblem` instead of `ODEProblem` — the Catalyst compile path can emit either |
| **Adaptive switching SSA ↔ tau-leap ↔ ODE** | `RSSA` / `DirectCR` / `ExtendedJumpArray` | Use when stoichiometries span many orders of magnitude; `SSAStepper` for pure jump, `Tsit5` for the continuous portion |

```julia
using Catalyst, JumpProcesses, DifferentialEquations

rn = @reaction_network begin
    k_on,  S + E --> SE        # fast
    k_off, SE --> S + E        # fast
    k_cat, SE --> P + E        # slow
end k_on k_off k_cat

u0 = [:S => 100, :E => 10, :SE => 0, :P => 0]
tspan = (0.0, 50.0)
p = [:k_on => 0.01, :k_off => 0.1, :k_cat => 0.05]

dprob = DiscreteProblem(rn, u0, tspan, p)
jprob = JumpProblem(rn, dprob, Direct())         # exact SSA
sol   = solve(jprob, SSAStepper())
```

For PDMP, define the continuous flow and rate separately:

```julia
using PiecewiseDeterministicMarkovProcesses
# Continuous flow between jumps
F!(ẋ, xc, xd, p, t) = (ẋ[1] = -0.1 * xc[1]; nothing)
# Rate of the single jump channel (depends on continuous state)
R!(rate, xc, xd, p, t, sum_rate) = (rate[1] = 0.5 * xc[1]; sum_rate ? 0.5 * xc[1] : 0.0)
# Discrete state update when a jump fires
DX!(xc, xd, p, t, ind_reaction) = (xd[1] += 1; nothing)

problem = PDMPProblem(F!, R!, DX!, 1, [1.0], [0], (0.0, 10.0))
sol = solve(problem, CHV(Tsit5()))
```

> **When to use PDMP vs SDE vs pure SSA**: pure SSA when every reaction fires discretely and the population is small; SDE (`SDESystem`) when chemical noise is approximately Gaussian and the population is large; **PDMP** when between jumps the dynamics are *deterministic* (e.g. receptor occupancy driven by an ODE, with stochastic binding/unbinding events) — PDMP is much cheaper than SSA and avoids the Gaussian approximation of chemical Langevin.

### Python counterparts

| Role | Python package | Notes |
|------|----------------|-------|
| Gillespie + tau-leaping + hybrid | `gillespy2` | See above — closest Python counterpart to JumpProcesses.jl's exact SSA / tau-leap branch |
| General jump-diffusion simulation | `sdeint` | NumPy implementations of Euler-Maruyama, Milstein, Strong-Taylor; no symbolic compilation |
| PDMP | *no maintained reference* | Fall back to `juliacall` + `PiecewiseDeterministicMarkovProcesses.jl` — the Python ecosystem has no direct equivalent |

---

## Checklist

- [ ] Reactions defined with @reaction_network
- [ ] Rates and species specified
- [ ] Converted to appropriate system type
- [ ] Initial conditions set

---

**Version**: 1.0.5
