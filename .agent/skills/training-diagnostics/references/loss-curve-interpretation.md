# allow-torch
# Loss Curve Interpretation Guide

Comprehensive guide to understanding training and validation loss curves, identifying pathologies, and taking corrective action.

---

## Table of Contents

1. [Healthy Training Curves](#healthy-training-curves)
2. [Overfitting Patterns](#overfitting-patterns)
3. [Underfitting Patterns](#underfitting-patterns)
4. [Loss Spikes and Instability](#loss-spikes-and-instability)
5. [Plateau Patterns](#plateau-patterns)
6. [Learning Rate Issues](#learning-rate-issues)
7. [Double Descent Phenomenon](#double-descent-phenomenon)

---

## Healthy Training Curves

### Ideal Pattern

```
Loss
  │
  │  ╲
  │   ╲___training loss
  │    ╲
  │     ╲─────validation loss
  │      ╲___
  │         ╲___
  └─────────────────── Epoch
```

**Characteristics:**
1. Both losses decrease monotonically
2. Validation loss follows training loss closely
3. Small gap between train and val loss
4. Smooth curves without oscillations
5. Convergence to stable value

**What it means:**
- Model is learning generalizable patterns
- Optimization is stable
- Hyperparameters are well-tuned
- Continue training until convergence

---

## Overfitting Patterns

### Classic Overfitting

```
Loss
  │
  │  ╲
  │   ╲___training loss
  │      ╲___
  │         ╲___
  │
  │    ╱──validation loss
  │   ╱
  │  ╱
  └─────────────────── Epoch
```

**Characteristics:**
1. Training loss continues to decrease
2. Validation loss increases after initial decrease
3. Growing gap between curves
4. Typically starts after 50-70% of training

**Root Causes:**
- Model too complex for dataset size
- Insufficient regularization
- Too many training epochs
- Data leakage or overfitting to specific samples

**Solutions:**

1. **Early Stopping**
   ```python
   class EarlyStopping:
       def __init__(self, patience=5, min_delta=0):
           self.patience = patience
           self.min_delta = min_delta
           self.counter = 0
           self.best_loss = None

       def __call__(self, val_loss):
           if self.best_loss is None:
               self.best_loss = val_loss
           elif val_loss > self.best_loss - self.min_delta:
               self.counter += 1
               if self.counter >= self.patience:
                   return True  # Stop training
           else:
               self.best_loss = val_loss
               self.counter = 0
           return False
   ```

2. **Regularization**
   ```python
   # L2 regularization
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

   # Dropout
   model = nn.Sequential(
       nn.Linear(784, 256),
       nn.ReLU(),
       nn.Dropout(0.5),  # Drop 50% during training
       nn.Linear(256, 10)
   )
   ```

3. **Data Augmentation**
   ```python
   from torchvision import transforms

   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ColorJitter(brightness=0.2),
       transforms.ToTensor()
   ])
   ```

4. **Model Simplification**
   - Reduce number of layers
   - Reduce layer width
   - Use smaller kernel sizes (CNNs)

### Severe Overfitting

```
Loss
  │
  │  ╲
  │   ╲___training loss → 0
  │
  │
  │ ↗↗↗ validation loss → ∞
  │↗
  │
  └─────────────────── Epoch
```

**Characteristics:**
- Training loss approaches zero
- Validation loss increases dramatically
- Model memorizes training data

**Immediate Actions:**
1. Stop training immediately
2. Load earlier checkpoint (before overfitting began)
3. Add aggressive regularization (dropout 0.5-0.7)
4. Reduce model capacity by 50%
5. Increase training data (augmentation or collection)

---

## Underfitting Patterns

### High Bias (Both Losses High)

```
Loss
  │
  │ ─────training loss
  │
  │ ─────validation loss
  │
  │ (Both high and flat)
  │
  └─────────────────── Epoch
```

**Characteristics:**
1. Both losses remain high
2. Minimal improvement over epochs
3. Curves are flat or very slow to decrease
4. Small gap between train and val

**Root Causes:**
- Model too simple (insufficient capacity)
- Learning rate too low
- Poor feature engineering
- Inappropriate architecture for task

**Solutions:**

1. **Increase Model Capacity**
   ```python
   # Too simple
   model = nn.Sequential(
       nn.Linear(784, 10)  # Single layer
   )

   # Better
   model = nn.Sequential(
       nn.Linear(784, 512),
       nn.ReLU(),
       nn.Linear(512, 256),
       nn.ReLU(),
       nn.Linear(256, 10)
   )
   ```

2. **Increase Learning Rate**
   ```python
   # Try 10x higher
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
   ```

3. **More Complex Architecture**
   - Add more layers
   - Add skip connections
   - Use attention mechanisms
   - Try different architecture family

4. **Feature Engineering**
   - Add polynomial features
   - Domain-specific transformations
   - Better preprocessing

### Slow Convergence

```
Loss
  │
  │  ╲
  │   ╲
  │    ╲__
  │       ╲__
  │          ╲__
  │             ╲__ (Very slow decrease)
  └─────────────────── Epoch
```

**Characteristics:**
- Loss decreases but very slowly
- Would converge eventually but takes too long
- Both curves parallel

**Solutions:**
1. Increase learning rate by 5-10x
2. Use learning rate finder (Leslie Smith method)
3. Add batch normalization
4. Better initialization (He/Xavier)
5. Use adaptive optimizers (Adam instead of SGD)

---

## Loss Spikes and Instability

### Occasional Spikes

```
Loss
  │     ↑
  │  ╲  │  ╱
  │   ╲_│_╱___
  │      spike
  │
  └─────────────────── Epoch
```

**Characteristics:**
- Sudden increase in loss
- Usually recovers within few epochs
- May occur once or periodically

**Causes:**
1. **Bad Batch**: Outliers or corrupted data
2. **Learning Rate Too High**: Overshoots on steep loss landscape
3. **Gradient Explosion**: Temporary numerical instability
4. **Learning Rate Schedule**: Sudden LR increase

**Solutions:**
```python
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Reduce learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# 3. Data cleaning
# Remove outliers, check for corrupted samples

# 4. Gentle LR schedules
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)
```

### Persistent Oscillation

```
Loss
  │  ╱╲╱╲╱╲
  │ ╱  ╲  ╲╱
  │╱    ╲
  │      ╲
  └─────────────────── Epoch
```

**Characteristics:**
- Loss oscillates up and down
- No clear downward trend
- Training unstable

**Causes:**
1. Learning rate too high (most common)
2. Batch size too small
3. Poor architecture choice
4. Adversarial examples in data

**Solutions:**
```python
# 1. Reduce learning rate dramatically (10x)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. Increase batch size
train_loader = DataLoader(dataset, batch_size=128)  # Was 32

# 3. Add batch normalization
model.add_module('bn', nn.BatchNorm1d(features))

# 4. Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Loss Becomes NaN

```
Loss
  │
  │  ╲
  │   ╲
  │    ╲
  │     ↓ NaN
  └─────────────────── Epoch
```

**Critical Issue!**

**Causes:**
1. Numerical overflow (exp, log operations)
2. Division by zero
3. Learning rate too high
4. Gradient explosion

**Immediate Actions:**
```python
# 1. Check for NaN in data
assert not torch.isnan(inputs).any(), "NaN in inputs"
assert not torch.isnan(targets).any(), "NaN in targets"

# 2. Add numerical stability
# Bad:
loss = -torch.log(probs)

# Good:
loss = -torch.log(probs + 1e-8)  # Numerical stability

# 3. Use stable loss functions
criterion = nn.CrossEntropyLoss()  # Includes LogSoftmax with stability

# 4. Gradient clipping (mandatory)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. Reduce learning rate by 100x
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

---

## Plateau Patterns

### Early Plateau

```
Loss
  │  ╲
  │   ╲____
  │        ────── (Stuck)
  │
  └─────────────────── Epoch
```

**Characteristics:**
- Loss decreases initially then plateaus
- Occurs early in training (< 20% complete)
- Both train and val stuck

**Causes:**
1. Learning rate too low
2. Poor initialization
3. Model stuck in local minimum
4. Saturated activations (sigmoid/tanh)

**Solutions:**
```python
# 1. Increase learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 2. Learning rate warmup
def warmup_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

# 3. Restart with different initialization
torch.manual_seed(new_seed)
model = Model()  # Re-initialize

# 4. Switch activation functions
# Replace sigmoid/tanh with ReLU
```

### Mid-Training Plateau

```
Loss
  │  ╲
  │   ╲
  │    ╲_______
  │         ────── (Plateau at 50-70%)
  │
  └─────────────────── Epoch
```

**Characteristics:**
- Good initial progress
- Plateau midway through training
- Validation loss also plateaus

**Causes:**
1. Learning rate needs adjustment
2. Model capacity reached
3. Need for regularization changes

**Solutions:**
```python
# 1. Learning rate decay
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# 2. Reduce regularization temporarily
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-6  # Reduced from 1e-5
)

# 3. Cyclical learning rates
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-4, max_lr=1e-2,
    step_size_up=2000, mode='triangular'
)
```

---

## Learning Rate Issues

### Too High

```
Loss
  │ ╱╲╱╲╱╲
  │╱  ╲  ╲╱
  │    ╲
  └─────────────────── Epoch
```

**Symptoms:**
- Oscillating loss
- Overshooting minima
- Slow or no convergence

**Fix:** Reduce by 10x

### Too Low

```
Loss
  │
  │ ╲
  │  ╲___
  │      ───── (Very slow)
  └─────────────────── Epoch
```

**Symptoms:**
- Extremely slow progress
- Linear decrease (no exponential decay phase)

**Fix:** Increase by 10x

### Just Right (Goldilocks)

```
Loss
  │
  │ ╲
  │  ╲
  │   ╲___
  │      ╲___
  └─────────────────── Epoch
```

**Characteristics:**
- Smooth exponential decay
- Transitions to linear phase
- Reaches convergence in reasonable time

---

## Double Descent Phenomenon

### Modern Deep Learning Curve

```
Test Error
  │   ╲
  │    ╲    ╱
  │     ╲  ╱
  │      ╲╱  ← Classical regime
  │       ╲
  │        ╲___ ← Modern regime
  │            ╲___
  └─────────────────── Model Complexity
          ↑
    Interpolation threshold
```

**Explanation:**
1. **Under-parameterized**: Classic bias-variance tradeoff
2. **Interpolation threshold**: Peak test error (can fit training data exactly)
3. **Over-parameterized**: Test error decreases again! (Double descent)

**Practical Implications:**
- Very large models can generalize despite perfect training fit
- "Overfitting" may be temporary - keep training
- Modern networks operate in over-parameterized regime

**Reference:** "Deep Double Descent" - Nakkiran et al., 2019

---

## Quick Diagnosis Flowchart

```
Start: Look at loss curves
│
├─ Both losses high and flat?
│  └─ UNDERFITTING: Increase capacity or LR
│
├─ Training ↓, Validation ↑?
│  └─ OVERFITTING: Add regularization or early stop
│
├─ Both decrease but slowly?
│  └─ LOW LR: Increase learning rate
│
├─ Oscillating?
│  └─ HIGH LR: Decrease learning rate, add clipping
│
├─ NaN?
│  └─ NUMERICAL INSTABILITY: Clip gradients, add stability
│
├─ Plateau?
│  └─ Adjust LR schedule or reduce regularization
│
└─ Smooth decrease?
   └─ HEALTHY: Continue training
```

---

## Recommended Monitoring Setup

```python
import wandb  # or tensorboard

# Initialize tracking
wandb.init(project="my-project")

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # Log everything
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'grad_norm': compute_grad_norm(model),
    })

    # Check for issues
    if math.isnan(train_loss):
        print("⚠️ NaN loss detected! Stopping.")
        break

    if val_loss > 1.1 * min_val_loss:
        print("⚠️ Validation loss increasing - possible overfitting")
```

---

## References

1. "Deep Double Descent" - Nakkiran et al., 2019
2. "Cyclical Learning Rates for Training Neural Networks" - Smith, 2017
3. "A disciplined approach to neural network hyper-parameters" - Smith, 2018
