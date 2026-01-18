---
name: deep-learning-experimentation
version: "1.0.7"
maturity: "5-Expert"
specialization: ML Experiment Design
description: Design systematic deep learning experiments with hyperparameter optimization, ablation studies, and reproducible workflows. Use when tuning hyperparameters, conducting ablations, setting up experiment tracking (W&B, TensorBoard, MLflow), or managing reproducibility.
---

# Deep Learning Experimentation

Systematic experiment design with hyperparameter optimization and reproducibility.

---

## Hyperparameter Priority

| Priority | Parameters | Search Strategy |
|----------|------------|-----------------|
| Critical | Learning rate, batch size, architecture | Tune first |
| Important | Optimizer, weight decay, dropout | Tune second |
| Fine-tuning | LR schedule, initialization, augmentation | Tune last |

---

## Search Strategies

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Grid | Complete coverage | Exponential | Few params |
| Random | More efficient | May miss optimal | Many params |
| Bayesian | Sample-efficient | Complex setup | Limited budget |

```python
# LR Range Test (Leslie Smith)
# Start: 1e-7, increase exponentially, stop when loss explodes
# Optimal: steepest descent point, use 1/10 to 1/3 of max stable LR

# Random search example
learning_rate = loguniform(1e-5, 1e-2)
dropout = uniform(0.1, 0.5)
```

---

## Ablation Studies

```
Full model:                    95.0% accuracy
- Without skip connections:    92.1% (-2.9%)
- Without batch norm:          93.5% (-1.5%)
- Without data augmentation:   90.2% (-4.8%)  ← Most critical
- Without dropout:             94.7% (-0.3%)  ← Least important
```

**Process**: Full model → remove one component → measure drop → identify critical vs unnecessary

---

## Experiment Tracking

```python
# Weights & Biases
import wandb
wandb.init(project="my-project", config=config)
wandb.log({"loss": loss, "accuracy": acc})

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)

# MLflow
import mlflow
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("accuracy", acc)
```

---

## Reproducibility

```python
# allow-torch
import torch, numpy as np, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Track**: Code (Git), data versions (DVC), checkpoints, dependencies (requirements.txt)

---

## Configuration Management

```yaml
# config.yaml
model:
  architecture: "resnet50"
  num_classes: 10

training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 100
  optimizer: "adam"
```

---

## Directory Structure

```
experiments/
├── exp001_baseline/
│   ├── config.yaml
│   ├── logs/
│   ├── checkpoints/
│   └── results.json
├── exp002_higher_lr/
└── exp003_deeper_network/
```

---

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Controlled comparisons | Change one variable at a time |
| Multiple seeds | Run 3-5 seeds, report mean ± std |
| Statistical significance | t-tests or bootstrap confidence intervals |
| Warm starting | Initialize from previous checkpoint |
| Early stopping | Stop if val loss doesn't improve |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| No seed management | Always set random seeds |
| Single run claims | Report over multiple seeds |
| No baseline comparison | Always compare to baselines |
| Test set peeking | Evaluate test set only once |

---

## Checklist

**Before Training:**
- [ ] Set random seed
- [ ] Log all hyperparameters
- [ ] Sanity check: overfit on small batch

**During Training:**
- [ ] Monitor train and val losses
- [ ] Save checkpoints regularly
- [ ] Log to experiment tracker

**After Training:**
- [ ] Evaluate test set (only once)
- [ ] Document what worked
- [ ] Compare with baselines

**Reporting:**
- [ ] Mean ± std over seeds
- [ ] Learning curves
- [ ] Ablation studies
- [ ] All hyperparameters specified

---

**Version**: 1.0.5
