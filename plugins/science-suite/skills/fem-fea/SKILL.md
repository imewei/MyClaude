---
name: fem-fea
description: Finite Element Modeling/Analysis — weak-form formulation, mesh strategy, element selection, and convergence verification. Use when setting up a FEM simulation, choosing element types, diagnosing mesh-convergence or locking issues, or picking between FEniCS, scikit-fem, Gridap.jl, or Ferrite.jl.
---

# FEM/FEA

Weak-form PDE discretization for continuum mechanics problems.

## Expert Agent

- **`continuum-mechanics-engineer`** — FEM formulation, mesh strategy, convergence verification.

## Weak-Form Workflow

1. Derive the strong-form governing PDE (equilibrium/conservation law) with boundary conditions.
2. Multiply by a test function from an appropriate function space; integrate by parts to shift a derivative onto the test function (this is what makes the weak form only need one order less continuity than the strong form).
3. Choose trial/test function spaces: Lagrange P1/P2 for scalar/vector displacement fields; mixed spaces (e.g. Taylor-Hood) for problems with a constraint (incompressibility).
4. Assemble the global stiffness matrix and load vector; solve (direct or iterative, depending on system size).

## Toolchain

| Ecosystem | Library | Notes |
|-----------|---------|-------|
| Python | FEniCS/FEniCSx | UFL symbolic weak-form specification, mature ecosystem |
| Python | scikit-fem | Lightweight, good for teaching/prototyping |
| Julia | Gridap.jl | Similar UFL-like symbolic weak forms |
| Julia | Ferrite.jl | Lower-level, more manual control over assembly |

## Convergence & Correctness Checklist

- [ ] Mesh convergence study: halve characteristic element size, confirm error decreases at the expected order for the chosen element (e.g. O(h²) for linear elements in energy norm)
- [ ] Boundary conditions correctly classified and imposed (Dirichlet — essential, imposed on the function space; Neumann — natural, appears in the weak form's boundary integral)
- [ ] Locking checked for nearly-incompressible materials (pure displacement formulation with low-order elements locks — use a mixed formulation or reduced integration)
- [ ] Nonlinear problems: Newton-Raphson residual norm decreasing each iteration, tangent stiffness matrix correctly linearized

## Common Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Volumetric locking | Overly stiff response for incompressible/near-incompressible materials with low-order elements | Mixed formulation (u-p) or reduced integration |
| Shear locking | Overly stiff bending response in low-order elements | Use higher-order elements or reduced integration |
| Non-converging Newton iteration | Residual not decreasing | Check tangent stiffness derivation, add line search or load stepping |
| Mesh-dependent results | Solution changes qualitatively with refinement | Under-resolved mesh — refine and re-check convergence order |

## Delegation

For the constitutive law plugged into the weak form (what stress-strain relation to use), see `science-suite:constitutive-equations`. For neural-network PDE solvers as an alternative to classical FEM, delegate to `pinn-engineer`. For mesh-connectivity graph representations, see `science-suite:graph-theory`.
