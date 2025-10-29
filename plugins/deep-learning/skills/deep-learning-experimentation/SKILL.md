---
name: deep-learning-experimentation
description: Systematic experiment design, hyperparameter tuning, and ablation studies for neural networks. Covers learning rate finding, grid/random/bayesian search, experiment tracking, and reproducible research. Use when optimizing models or conducting rigorous experiments.
---

# Deep Learning Experimentation

Systematic frameworks for experiment design, hyperparameter optimization, and reproducible deep learning research.

## When to Use

- Hyperparameter tuning (learning rate, batch size, architecture choices)
- Ablation studies (understanding component contributions)
- Comparing multiple approaches systematically
- Optimizing model performance
- Conducting reproducible research
- Experiment tracking and management

## Hyperparameter Optimization

### 1. Learning Rate Finding

**LR Range Test (Leslie Smith):**
```python
# Start with very small LR, exponentially increase
# Plot loss vs LR
# Optimal LR: steepest descent before loss explodes
```

**Process:**
1. Start LR: 1e-7
2. Increase exponentially each batch
3. Stop when loss explodes
4. Select LR at steepest descent point
5. Use 1/10th to 1/3rd of max stable LR

### 2. Search Strategies

**Grid Search:**
```python
# Exhaustive search over discrete values
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]
# Try all combinations (3 × 3 = 9 experiments)
```
- Pros: Complete coverage
- Cons: Exponential in number of hyperparameters

**Random Search:**
```python
# Sample from distributions
learning_rate = loguniform(1e-5, 1e-2)
dropout = uniform(0.1, 0.5)
# Try N random combinations
```
- Pros: More efficient than grid
- Cons: May miss optimal region

**Bayesian Optimization:**
```python
# Use Optuna, Ray Tune, or Weights & Biases Sweeps
# Model p(performance|hyperparameters)
# Select next point to try based on expected improvement
```
- Pros: Most sample-efficient
- Cons: More complex setup

### 3. Important Hyperparameters (Priority Order)

**Critical (tune first):**
1. Learning rate
2. Batch size
3. Architecture (width, depth)

**Important:**
4. Optimizer (Adam vs SGD)
5. Weight decay / L2 regularization
6. Dropout rate

**Fine-tuning:**
7. Learning rate schedule
8. Initialization scheme
9. Augmentation strength
10. Loss function weights

## Experiment Design

### Ablation Studies

**Purpose:** Understand what components contribute to performance

**Process:**
1. Start with full model (all components)
2. Remove one component at a time
3. Measure performance drop
4. Identify critical vs unnecessary components

**Example:**
```
Full model:                    95.0% accuracy
- Without skip connections:    92.1% (-2.9%)
- Without batch norm:          93.5% (-1.5%)
- Without data augmentation:   90.2% (-4.8%)
- Without dropout:             94.7% (-0.3%)
```

**Insights:** Data augmentation most critical, dropout least important

### Controlled Comparisons

**Change one variable at a time:**
```
Baseline: LR=0.001, batch=32, dropout=0.5
Experiment 1: LR=0.01 (everything else same)
Experiment 2: batch=64 (everything else same)
```

**Use multiple seeds:**
- Run each experiment with 3-5 different random seeds
- Report mean ± std
- Ensures results are robust, not lucky

**Statistical Significance:**
- Use t-tests or bootstrap confidence intervals
- Don't claim improvement without statistical evidence

## Experiment Tracking

### What to Track

**Hyperparameters:**
- All settings (LR, batch size, architecture details)
- Optimizer configuration
- Data preprocessing

**Metrics:**
- Training loss, validation loss
- Task-specific metrics (accuracy, F1, etc.)
- Gradient norms, learning rates
- Computational cost (time, memory)

**Artifacts:**
- Model checkpoints
- Configuration files
- Training logs
- Visualizations

### Tools

**Weights & Biases:**
```python
import wandb
wandb.init(project="my-project", config=config)
wandb.log({"loss": loss, "accuracy": acc})
```

**TensorBoard:**
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)
```

**MLflow:**
```python
import mlflow
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("accuracy", acc)
```

## Reproducibility

### Random Seed Management

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Version Control

**Track everything:**
- Code (Git)
- Data versions
- Model checkpoints
- Dependencies (requirements.txt, environment.yml)

**DVC (Data Version Control):**
```bash
dvc add data/train.csv
dvc add models/checkpoint.pt
git add data/train.csv.dvc models/checkpoint.pt.dvc
```

### Configuration Management

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

data:
  dataset: "cifar10"
  augmentation: true
```

## Advanced Experiment Strategies

### Multi-Objective Optimization

**Trade-offs:**
- Accuracy vs Latency
- Performance vs Model Size
- Training Time vs Final Accuracy

**Pareto Frontier:**
- Plot trade-off curves
- Identify non-dominated solutions

### Transfer Learning Experiments

**Strategy:**
1. Pretrain on large dataset (ImageNet, BERT corpus)
2. Fine-tune on target task
3. Compare with training from scratch

**Hyperparameters:**
- Lower learning rate for fine-tuning (1/10th of pretraining)
- Freeze early layers, train later layers
- Gradual unfreezing

### Curriculum Learning

**Idea:** Train on easy examples first, gradually increase difficulty

**Implementation:**
1. Define difficulty metric (e.g., loss on pretrained model)
2. Sort training data by difficulty
3. Start with easiest 20%, gradually add harder examples

## Quick Reference: Experiment Checklist

**Before Training:**
- [ ] Set random seed for reproducibility
- [ ] Log all hyperparameters
- [ ] Validate data loader (check samples)
- [ ] Sanity check: Overfit on small batch

**During Training:**
- [ ] Monitor train and val losses
- [ ] Track gradient norms
- [ ] Save checkpoints regularly
- [ ] Log to experiment tracker

**After Training:**
- [ ] Evaluate on test set (only once!)
- [ ] Document what worked and what didn't
- [ ] Save final model and config
- [ ] Compare with baselines

**Reporting Results:**
- [ ] Report mean ± std over multiple seeds
- [ ] Show learning curves
- [ ] Include ablation studies
- [ ] Specify all hyperparameters
- [ ] Make code available

## Best Practices

### Experiment Management

1. **Naming Convention:**
   ```
   exp_name = f"{model}_{dataset}_{timestamp}_{unique_id}"
   # Example: "resnet50_cifar10_20250127_a3f2"
   ```

2. **Directory Structure:**
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

3. **Documentation:**
   - README.md for each experiment
   - Motivation for trying this configuration
   - Results and insights

### Computational Efficiency

**Warm Starting:**
- Initialize from previous checkpoint
- Saves training time

**Early Stopping:**
- Stop if validation loss doesn't improve
- Saves compute, prevents overfitting

**Progressive Resizing:**
- Start with small images, increase size
- Faster initial training

**Mixed Precision:**
```python
# Automatic Mixed Precision (AMP)
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

*Systematic frameworks for deep learning experimentation, hyperparameter optimization, and reproducible research with proper experiment tracking.*
