---
name: ml-force-fields
version: "1.0.7"
maturity: "5-Expert"
specialization: ML Potentials
description: Develop ML force fields (NequIP, MACE, DeepMD) achieving quantum accuracy with 1000x speedup. Use when training neural network potentials or deploying ML force fields in MD.
---

# ML Force Fields

Neural network potentials achieving ~1 meV/atom accuracy.

---

## Modern MLFFs

| Framework | Accuracy | Notes |
|-----------|----------|-------|
| NequIP/Allegro | ~0.5 meV/atom | E(3)-equivariant |
| MACE | < 1 meV/atom | State-of-the-art |
| DeepMD | ~1 meV/atom | Production-ready |
| SchNet/PaiNN | ~1 meV/atom | Efficient molecules |

---

## Training Workflow

1. **Data Generation**: AIMD (1000-10000 configs)
2. **Active Learning**: On-the-fly DFT labeling
3. **Training**: Force-centric loss, ensemble
4. **Validation**: Test transferability
5. **Deployment**: LAMMPS/GROMACS integration

---

## LAMMPS Deployment

```lammps
pair_style deepmd frozen_model.pb
pair_coeff * *
```

---

## Uncertainty Quantification

- **Ensemble disagreement**: Committee of models
- **Active learning**: Query high-uncertainty configs
- **Validation**: Test beyond training distribution

---

## Checklist

- [ ] Training data diverse (bulk, surface, defects)
- [ ] Force errors < 50 meV/Å
- [ ] Transferability validated
- [ ] Uncertainty quantified
- [ ] Deployed in production MD

---

**Version**: 1.0.5
