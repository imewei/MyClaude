---
name: ml-force-fields
description: Develop and deploy machine learning force fields (NequIP, MACE, DeepMD, SchNet, PaiNN) achieving near-DFT quantum accuracy (~1 meV/atom) with 1000-10000x computational speedups for molecular dynamics simulations. Use this skill when training neural network potentials from ab initio molecular dynamics (AIMD) or DFT reference data. Use when implementing active learning workflows for on-the-fly data generation and model refinement. Use when performing uncertainty quantification through ensemble models or model committee methods. Use when deploying trained ML force fields in LAMMPS (pair_style deepmd), GROMACS, or other MD engines for production simulations. Use when working with Python training scripts for NequIP, Allegro, or MACE frameworks. Use when designing training datasets with diverse atomic configurations (bulk, surfaces, defects, interfaces). Use when validating ML force field transferability beyond training data. Use when selecting appropriate neural network architectures (message passing, equivariant networks, graph neural networks) for molecular systems. Use when optimizing force-centric loss functions and hyperparameters for accurate force predictions. Use when replacing expensive DFT/AIMD calculations with ML-accelerated simulations while maintaining quantum accuracy.
---

# ML Force Fields Development

## When to use this skill

- When training neural network potentials (NequIP, MACE, DeepMD, SchNet, PaiNN) from DFT or AIMD reference data
- When writing Python training scripts for ML force field frameworks
- When implementing active learning workflows to generate training data on-the-fly
- When performing uncertainty quantification using ensemble models or model committees
- When deploying trained ML force fields in LAMMPS using `pair_style deepmd` or similar interfaces
- When integrating ML potentials into GROMACS or other MD simulation engines
- When designing diverse training datasets spanning bulk structures, surfaces, defects, and interfaces
- When validating ML force field accuracy against quantum mechanical calculations
- When testing transferability of ML models to configurations beyond the training set
- When optimizing neural network architectures (message passing depth, cutoff radius, hidden dimensions)
- When tuning force-centric loss functions for improved force predictions
- When replacing expensive DFT/AIMD simulations with ML-accelerated MD while maintaining quantum accuracy
- When working with systems requiring reactive chemistry or bond breaking/formation at quantum accuracy
- When analyzing model performance metrics (energy MAE, force RMSE, uncertainty estimates)

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
