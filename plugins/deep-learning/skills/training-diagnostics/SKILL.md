---
name: training-diagnostics
version: "1.0.6"
maturity: "5-Expert"
specialization: Neural Network Training Debugging
description: Diagnose and resolve neural network training failures through systematic analysis of gradient pathologies, loss curves, and convergence issues. Use when encountering vanishing/exploding gradients, dead ReLU neurons, loss anomalies (NaN, spikes, plateaus), overfitting/underfitting patterns, or when debugging training scripts requiring systematic troubleshooting.
---

# Training Diagnostics

Systematic frameworks for diagnosing and resolving neural network training issues.

---

## Symptom → Solution Quick Reference

| Symptom | Likely Cause | First Fix |
|---------|--------------|-----------|
| Loss = NaN | Exploding gradients | Gradient clipping |
| Very slow loss decrease | Vanishing gradients | ReLU + residuals |
| Low train, high val loss | Overfitting | Dropout, weight decay |
| High train + val loss | Underfitting | More capacity |
| Loss spike then recover | High LR | Reduce LR, clip grads |
| Loss plateau | Saddle point | Cyclical LR |
| >50% zero activations | Dead ReLU | Leaky ReLU |
| Activations at ±1 | Saturation | ReLU, batch norm |

---

## Gradient Pathologies

### Vanishing Gradients

**Symptoms**: Gradients < 1e-7 in early layers, slow training, shallow network behavior

**Causes**: Deep nets with sigmoid/tanh, poor initialization, no skip connections

**Solutions**:
```python
# 1. ReLU instead of sigmoid/tanh
activation = nn.ReLU()

# 2. Residual connections
def forward(self, x):
    return x + self.block(x)

# 3. He initialization
nn.init.kaiming_normal_(layer.weight, mode='fan_in')

# 4. Batch normalization
self.bn = nn.BatchNorm1d(features)
```

### Exploding Gradients

**Symptoms**: Gradients > 100, NaN loss, unstable training

**Solutions**:
```python
# Gradient clipping (essential)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = torch.optim.Adam(params, lr=1e-4)  # Start smaller

# Spectral normalization
layer = torch.nn.utils.spectral_norm(nn.Linear(in_f, out_f))
```

### Dead ReLU

**Symptoms**: >50% zero activations, learning stops

**Solutions**:
```python
# Leaky ReLU
activation = nn.LeakyReLU(0.01)

# PReLU (learnable slope)
activation = nn.PReLU()

# Slightly positive bias
layer.bias.data.fill_(0.01)
```

---

## Loss Curve Patterns

### Overfitting

**Pattern**: Train loss ↓, val loss ↑, increasing gap

**Solutions**:
```python
# Dropout
self.dropout = nn.Dropout(p=0.5)

# Weight decay
optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-4)

# Early stopping
if val_loss > best_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
```

### Underfitting

**Pattern**: Both train and val loss remain high

**Solutions**:
- Increase model capacity (more layers/width)
- Train longer
- Reduce regularization
- Increase learning rate

### Loss Spikes

**Pattern**: Sudden large increase then recovery

**Solutions**:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

# Learning rate warmup
warmup_steps = 1000
lr = base_lr * min(step / warmup_steps, 1.0)

# LR decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### Plateaus

**Pattern**: Loss stops decreasing, gradients near zero

**Solutions**:
```python
# Cyclical learning rate
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-4, max_lr=1e-3)

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10)

# Momentum to escape saddle points
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
```

---

## Gradient Diagnostics

```python
def diagnose_gradients(model, data_loader, criterion):
    """Compute gradient statistics per layer."""
    model.train()
    inputs, targets = next(iter(data_loader))
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    model.zero_grad()
    loss.backward()

    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            stats[name] = {
                'norm': g.norm().item(),
                'mean': g.mean().item(),
                'max': g.max().item(),
                'zero_pct': (g == 0).float().mean().item()
            }
    return stats

def check_issues(stats):
    """Identify gradient pathologies."""
    norms = [s['norm'] for s in stats.values()]

    if max(norms) / (min(norms) + 1e-10) > 1000:
        print("⚠️ VANISHING: gradient ratio > 1000")

    if any(n > 100 for n in norms):
        print("⚠️ EXPLODING: gradient norm > 100")

    for name, s in stats.items():
        if s['zero_pct'] > 0.5:
            print(f"⚠️ DEAD NEURONS: {name} has {s['zero_pct']:.0%} zeros")
```

---

## Learning Rate Selection

```python
# Learning Rate Range Test
def lr_range_test(model, train_loader, start_lr=1e-7, end_lr=10, steps=100):
    """Find optimal LR by plotting loss vs LR."""
    lr_mult = (end_lr / start_lr) ** (1 / steps)
    lr = start_lr
    losses = []

    for batch in train_loader:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss = train_step(batch)
        losses.append((lr, loss))
        lr *= lr_mult

        if loss > losses[0][1] * 4:  # Loss exploded
            break

    # Optimal LR: steepest descent region
    return losses
```

**Interpretation**:
- Too low: Slow convergence
- Too high: Divergence, oscillations
- Sweet spot: Steepest descent before explosion

---

## Batch Size Effects

| Batch Size | Pros | Cons |
|------------|------|------|
| Small | Better generalization, exploration | Slow, noisy |
| Large | Faster, stable | Sharp minima, worse generalization |

**Linear Scaling Rule**: If batch × k, then LR × k

```python
# Warmup essential for large batches
def lr_with_warmup(step, warmup_steps, base_lr):
    return base_lr * min(step / warmup_steps, 1.0)
```

---

## Pre-Training Sanity Checks

```python
# 1. Overfit on single batch (should reach ~0 loss)
single_batch = next(iter(train_loader))
for _ in range(1000):
    loss = train_step(single_batch)
assert loss < 0.01, "Cannot overfit single batch"

# 2. Check gradient flow
stats = diagnose_gradients(model, train_loader, criterion)
check_issues(stats)

# 3. Verify data correctness
for x, y in train_loader:
    assert not torch.isnan(x).any(), "NaN in inputs"
    assert not torch.isnan(y).any(), "NaN in labels"
    break
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Monitor everything | Loss, gradients, activations, LR |
| Use TensorBoard/W&B | Visual diagnostics |
| Set random seeds | Reproducibility |
| Save checkpoints | Resume from last good state |
| Change one thing at a time | Isolate cause |
| Test on small data first | Fast iteration |

---

## Prevention Checklist

- [ ] Use ReLU/LeakyReLU (not sigmoid/tanh in hidden layers)
- [ ] Apply He/Xavier initialization
- [ ] Add batch normalization
- [ ] Include residual connections (deep nets)
- [ ] Enable gradient clipping
- [ ] Use learning rate warmup
- [ ] Monitor train AND validation loss
- [ ] Implement early stopping
- [ ] Save regular checkpoints

---

**Version**: 1.0.5
