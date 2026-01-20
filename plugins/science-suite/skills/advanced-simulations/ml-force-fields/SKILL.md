---
name: ml-force-fields
version: "2.1.1"
description: Develop ML force fields (NequIP, MACE, DeepMD) achieving quantum accuracy with 1000x speedup. Use when training neural network potentials or deploying ML force fields in MD.
---

# ML Force Fields

Neural network potentials achieving ~1 meV/atom accuracy.

## Expert Agent

For training, validating, and deploying machine learning force fields, delegate to the expert agent:

- **`ml-expert`** or **`simulation-expert`**:
  - **`ml-expert`**: For architecture design and training of neural network potentials (NequIP, MACE).
    - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - **`simulation-expert`**: For deploying trained potentials in MD simulations and validating physical properties.
    - *Location*: `plugins/science-suite/agents/simulation-expert.md`

## Modern MLFFs

| Framework | Accuracy | Notes |
|-----------|----------|-------|
| NequIP/Allegro | ~0.5 meV/atom | E(3)-equivariant |
| MACE | < 1 meV/atom | State-of-the-art |
| DeepMD | ~1 meV/atom | Production-ready |
| SchNet/PaiNN | ~1 meV/atom | Efficient molecules |

## Training Workflow

1. **Data Generation**: AIMD (1000-10000 configs)
2. **Active Learning**: On-the-fly DFT labeling
3. **Training**: Force-centric loss, ensemble
4. **Validation**: Test transferability
5. **Deployment**: LAMMPS/GROMACS integration

## LAMMPS Deployment

```lammps
pair_style deepmd frozen_model.pb
pair_coeff * *
```

## Uncertainty Quantification

- **Ensemble disagreement**: Committee of models
- **Active learning**: Query high-uncertainty configs
- **Validation**: Test beyond training distribution

## Parallelization

| Strategy | Implementation |
|----------|----------------|
| **Training** | DistributedDataParallel (DDP) for multi-GPU training |
| **Data Gen** | MPI for parallel DFT calculations (VASP/QE) |
| **Inference** | MPI+GPU domain decomposition in LAMMPS |
| **Active Learning** | Parallel candidate selection and labeling |

## Checklist

- [ ] Training data diverse (bulk, surface, defects)
- [ ] Force errors < 50 meV/Å
- [ ] Transferability validated
- [ ] Uncertainty quantified
- [ ] Deployed in production MD
