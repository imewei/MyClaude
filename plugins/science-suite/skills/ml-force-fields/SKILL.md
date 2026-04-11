---
name: ml-force-fields
description: Design, train, and deploy ML interatomic potentials (MLIPs) including equivariant GNNs (NequIP, MACE, Allegro, SchNet, PaiNN, TensorNet), Julia ACE potentials (ACEpotentials.jl, PotentialLearning.jl), and foundation/universal potentials (MACE-MP, fairchem UMA). Use when training neural network potentials on DFT data, fine-tuning a foundation MLIP, running active learning loops, deploying potentials in LAMMPS (pair_style mliap / pace / mace), OpenMM, HOOMD-blue, or Molly.jl, quantifying ensemble uncertainty, or choosing between equivariant GNN and linear ACE architectures. Use proactively when the user mentions MLIP, MLFF, neural network potential, machine learning force field, foundation potential, or wants DFT-quality forces at classical MD speed.
---

# ML Force Fields

Learned energy/force surrogates that replace classical force fields (Lennard-Jones, EAM, AMBER, CHARMM, GAFF, ReaxFF) with ~meV/atom accuracy at 100-1000x DFT speed, while respecting translation/rotation/permutation invariance and (for equivariant models) rotational equivariance of forces and stresses.

## Expert Agent

- **`ml-expert`** — architecture design and training of neural network potentials. Location: `plugins/science-suite/agents/ml-expert.md`.
- **`simulation-expert`** — deployment of trained potentials in production MD and validation of physical properties. Location: `plugins/science-suite/agents/simulation-expert.md`.
- **`julia-pro`** — Julia ACE stack (ACEpotentials.jl, PotentialLearning.jl, Molly.jl). Location: `plugins/science-suite/agents/julia-pro.md`.

## Sibling skills

- `md-simulation-setup` — classical engines (LAMMPS, GROMACS, HOOMD, OpenMM) where trained MLIPs are deployed.
- `advanced-simulations` — non-equilibrium, multiscale, rare-event sampling that consume MLIPs.
- `trajectory-analysis` — post-simulation structure/dynamics analysis.
- `jax-physics-applications` — JAX-MD-based differentiable MD with in-graph learned potentials.
- `julia-ml-and-dl` — Lux.jl / Flux.jl neural architectures used by Julia-side MLIP training.

## Landscape (2025)

### Python / PyTorch (+ JAX) — equivariant GNN family

| Framework | Architecture class | Body order | Equivariance | Locality | Entry point |
|---|---|---|---|---|---|
| NequIP | E(3)-equivariant MPNN (e3nn) | 2-body msgs, stacked | E(3) | Global (configurable cutoff) | `nequip-train config.yaml` |
| Allegro | NequIP descendant | 2-body | E(3) | Strictly local (no MP) | Model class inside NequIP framework |
| MACE | Higher-body-order equivariant MPNN | ν=2-4 per layer | E(3) | Local, few layers | `mace_run_train ...` |
| MACE-JAX | JAX port of MACE | same | E(3) | same | `mace_jax` package |
| SchNet / SchNetPack | Continuous-filter CNN | 2-body | Invariant | Local | `spktrain experiment=...` |
| PaiNN | Polarizable + invariant | 2-body vector | Equivariant vectors | Local | SchNetPack model flag |
| TensorNet / TensorNet2 | Cartesian tensor MPNN | 2-3 | O(3) | Local | `torchmd-train` |
| fairchem (UMA, eSEN, EquiformerV2) | Foundation potentials (Meta FAIR) | varies | E(3)/SO(3) | Global | `FAIRChemCalculator` + UMA weights |

### Julia — ACE family

| Package | Role |
|---|---|
| `ACEpotentials.jl` (v0.10.x, Julia 1.12) | Primary entry point for fitting ACE potentials; internally migrated to `EquivariantTensors.jl`; AtomsBase-integrated. |
| `ACE1.jl` | Legacy backend (ACEpotentials v0.6.x). Maintenance-only. Use v0.10 for new work. |
| `PotentialLearning.jl` (CESMIX-MIT) | Data-centric fitting interface: DPP/kDPP active subsampling, PCA-ACE, linear basis potentials (`LBasisPotential`), metrics (MAE/MSE/RSQ). |
| `Molly.jl` | Native Julia MD engine. Supports any potential implementing the `AtomsCalculators.jl` interface, CUDA/KernelAbstractions GPU, Float32/Float64, and differentiable MD as a first-class feature. |
| `LAMMPS.jl` | Julia wrapper for LAMMPS (verify current status before depending on it). |

Foundation-potential note: MACE-MP-0 checkpoints ("small"/"medium"/"large") and fairchem UMA models (`uma-s-1p1`, `uma-m-1p1`) cover large chunks of the periodic table and can often be used zero-shot or with light fine-tuning before training from scratch.

## Training workflow

1. **Data generation** — AIMD or single-point DFT (VASP, Quantum ESPRESSO, CP2K) at 1-10k configs minimum; for complex chemistry start from a foundation model instead.
2. **Active learning** — committee disagreement or MC-dropout selects high-uncertainty configs for new DFT labels.
3. **Training** — energy + force (+ stress + virial) loss; force-weighted objective; EMA; mixed precision.
4. **Validation** — held-out test set, physical observables (RDF, phonons, elastic constants, melting point), and out-of-distribution probes.
5. **Deployment** — export to LAMMPS / OpenMM / HOOMD / Molly.jl via a pair style or calculator.

## Concrete examples (verified against upstream docs)

### MACE — fine-tune a foundation model (Python)

```bash
mace_run_train \
  --name="MACE_finetune" \
  --foundation_model="medium" \
  --train_file="train.xyz" \
  --valid_fraction=0.05 \
  --test_file="test.xyz" \
  --energy_weight=1.0 \
  --forces_weight=10.0 \
  --E0s="average" \
  --lr=0.01 \
  --scaling="rms_forces_scaling" \
  --batch_size=2 \
  --max_num_epochs=10 \
  --ema --ema_decay=0.99 --amsgrad \
  --default_dtype="float32" --device=cuda --seed=42
```

Python API entry point: `mace.tools.build_default_arg_parser` + `mace.tools.run`.

### NequIP — compile and deploy to LAMMPS ML-IAP

```bash
# (1) Train with nequip-train config.yaml   (see upstream docs for full config schema)
# (2) Compile for deployment
nequip-compile model.nequip.zip compiled.nequip.pt2 \
    --mode aotinductor --device cuda --target ase --tf32

# (3) Prepare LAMMPS ML-IAP interface file
nequip-prepare-lmp-mliap compiled.nequip.pt2 output.nequip.lmp.pt
```

```lammps
units         metal
atom_style    atomic
atom_modify   map yes
newton        on
read_data     system.data
pair_style    mliap unified output.nequip.lmp.pt 0
pair_coeff    * * H O
```

### fairchem UMA — zero-shot ASE calculator

```python
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
add_adsorbate(slab, molecule("CO"), 2.0, "bridge")
slab.calc = calc
LBFGS(slab).run(fmax=0.05, steps=100)
```

### ACEpotentials.jl — install and fit (Julia 1.12, v0.10.x)

```julia
# Pkg setup (shell)
# julia --project=.
# pkg> add ACEpotentials

using ACEpotentials
# ACEpotentials.copy_tutorial()   # drops Jupyter notebook tutorials into cwd
# Then follow the upstream tutorials in examples/ for `acemodel` construction,
# `acefit!` fitting, and export. The v0.10 API surface is stable relative to
# v0.9 — see https://acesuit.github.io/ACEpotentials.jl/dev for current calls.
```

[unverified API shape: exact symbol names for `acemodel` / `acefit!` / `export2lammps` in v0.10; consult `ACEpotentials.jl` docs before committing downstream code.]

### Molly.jl — native Julia MD (Lennard-Jones shown; any AtomsCalculators-compatible ACE potential plugs in the same way)

```julia
using Molly

n_atoms  = 100
boundary = CubicBoundary(2.0u"nm")
temp     = 298.0u"K"
atoms    = [Atom(mass=10.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:n_atoms]
coords   = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
velocities = [random_velocity(10.0u"g/mol", temp) for _ in 1:n_atoms]

sys = System(
    atoms=atoms, coords=coords, velocities=velocities, boundary=boundary,
    pairwise_inters=(LennardJones(),),
    loggers=(temp=TemperatureLogger(100),),
)
simulate!(sys, VelocityVerlet(dt=0.002u"ps",
    coupling=AndersenThermostat(temp, 1.0u"ps")), 10_000)
```

Molly runs on CUDA (and other KernelAbstractions backends), supports Float32/Float64, and treats differentiable MD as a first-class feature — suitable for gradient-based potential fitting and rare-event reweighting.

## Deployment matrix

| Trained in | Deploys to | Channel |
|---|---|---|
| MACE (PyTorch) | LAMMPS | `pair_style mace` (via ML-IAP or `mace_lammps` interface) |
| NequIP / Allegro | LAMMPS | `nequip-prepare-lmp-mliap` + `pair_style mliap unified` |
| NequIP / MACE / SchNet | OpenMM | openmm-ml / OpenMM-TorchMD bridge |
| NequIP / MACE | ASE | `nequip-compile --target ase` / MACE ASE calculator |
| fairchem UMA | ASE / downstream MD | `FAIRChemCalculator(task_name=...)` |
| ACEpotentials.jl | LAMMPS | `pair_style pace` (ACE LAMMPS pair style) |
| ACEpotentials.jl | Molly.jl | `AtomsCalculators.jl` interface |
| Any MLIP | HOOMD-blue | HOOMD-ML plug-ins / custom force compute |

## Datasets and benchmarks

| Dataset | Scope |
|---|---|
| ANI-1x, ANI-2x | Organic molecules (CHNO, then +FSCl) |
| SPICE / SPICE2 | Drug-like and biomolecular small systems |
| QM7-X, QM9 | Small molecule quantum properties |
| MD17 / rMD17 | Small-molecule dynamics (revised for tighter convergence) |
| OC20 / OC22 (via fairchem) | Catalysis, adsorbates on surfaces |
| MPtrj | Materials Project trajectories — basis for MACE-MP foundation |
| Alexandria, MatPES | Broader materials coverage |

## Universal tooling

- **ASE** (`ase`) — Python universal atomistic I/O, calculator abstraction, geometry ops, MD drivers. Every Python MLIP exposes an ASE calculator.
- **AtomsBase.jl / AtomsCalculators.jl** — Julia-side equivalent; common interface for Molly.jl, ACEpotentials, DFTK.jl.

## Decision table

| Situation | Recommendation |
|---|---|
| New chemistry, small-medium dataset, need SOTA accuracy | Start from MACE-MP foundation, fine-tune |
| Periodic materials, catalysis | fairchem UMA or EquiformerV2 zero-shot, then fine-tune |
| Strictly local / huge systems / GPU scaling | Allegro (via NequIP framework) |
| Julia-native workflow, interpretable linear basis, small species count | ACEpotentials.jl (v0.10) + Molly.jl |
| Data-starved but DFT-budget-limited | PotentialLearning.jl kDPP/DPP subsampling → ACE fit |
| Differentiable MD (end-to-end gradient through trajectory) | JAX-MD + MACE-JAX (Python) or Molly.jl (Julia) |
| Molecular QM properties (dipoles, polarizability) | SchNetPack with PaiNN |
| Production LAMMPS with minimal fuss | MACE → `pair_style mace`, NequIP → `pair_style mliap`, ACE → `pair_style pace` |

## Cross-language handoff pattern

Python has the richest training ecosystem; Julia has the cleanest differentiable-MD stack and the ACE basis. Honest state of the ecosystem:

1. **Train in Python** (MACE / NequIP / fairchem) or **Julia** (ACEpotentials) depending on architecture choice.
2. **Freeze** to a portable format (TorchScript, AOTInductor `.pt2`, ACE `.json` / LAMMPS `pace` file).
3. **Deploy** in LAMMPS / OpenMM / HOOMD (production C++/CUDA performance) or Molly.jl (native Julia, differentiable).
4. **Validate** structural and dynamical observables in the target engine, not just the training loss.

## Uncertainty quantification

- **Committee / deep ensemble** — train N seeds; σ(force) drives active learning queries.
- **MC dropout** — cheaper proxy, works with any NN backbone.
- **Bayesian linear last layer** — natural for ACE linear models; `PotentialLearning.jl` computes covariance estimates directly.
- **Validation beyond training distribution** — probe elevated T, strained cells, defect configurations.

## Parallelization

| Stage | Strategy |
|---|---|
| Training | PyTorch DDP / NequIP `SimpleDDPStrategy` / multi-GPU Lightning |
| Data generation | MPI-parallel VASP / Quantum ESPRESSO / CP2K |
| Inference (production MD) | LAMMPS MPI + GPU domain decomposition; OpenMM CUDA; Molly.jl KernelAbstractions |
| Active learning loop | Parallel candidate scoring + parallel DFT labeling |

## Checklist

- [ ] Decided: foundation-model fine-tune vs train-from-scratch before committing DFT budget
- [ ] Training set spans bulk, surface, defect, and high-T configurations
- [ ] Force error < 50 meV/Å on held-out test set
- [ ] Ensemble or MC-dropout uncertainty quantified for OOD probes
- [ ] Physical observables (RDF, phonons, elastic constants) match reference DFT/experiment
- [ ] Potential exported to the target production engine (LAMMPS / OpenMM / Molly.jl)
- [ ] Deployment smoke-tested in the target engine at production system size
- [ ] Checkpoint, config, training data, and random seeds archived for reproducibility
