---
name: ml-force-fields
description: Develop ML force fields (NequIP, MACE, DeepMD) with near-DFT accuracy and 1000-10000x speedups. Use when training neural network potentials from AIMD data, implementing active learning, performing uncertainty quantification, or deploying MLFFs in LAMMPS/GROMACS for production simulations.
---

# ML Force Fields Development

Train, validate, and deploy machine learning force fields achieving quantum accuracy (~1 meV/atom) with 1000-10000x speedups over DFT.

## Modern MLFFs

**NequIP/Allegro**: E(3)-equivariant, ~0.5 meV/atom
**MACE**: State-of-the-art, < 1 meV/atom
**DeepMD**: Production-ready, mature
**SchNet/PaiNN**: Efficient for molecules

## Training Workflow

1. **Data Generation**: AIMD (1000-10000 configs)
2. **Active Learning**: On-the-fly DFT labeling
3. **Training**: Force-centric loss, ensemble models
4. **Validation**: Test transferability (surfaces, defects)
5. **Deployment**: Integrate with LAMMPS/GROMACS

## Deployment

**LAMMPS:**
```lammps
pair_style deepmd frozen_model.pb
pair_coeff * *
```

**Uncertainty quantification**: Ensemble disagreement

References for architecture details, hyperparameter tuning, and active learning strategies.
