---
name: training-diagnostics
version: "1.0.7"
description: Diagnose and resolve neural network training failures through systematic analysis of gradient pathologies, loss curves, and convergence issues. Use when encountering vanishing/exploding gradients, dead ReLU neurons, loss anomalies (NaN, spikes, plateaus), overfitting/underfitting patterns, or when debugging training scripts requiring systematic troubleshooting.
---

# Training Diagnostics

## Quick Reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss = NaN | Exploding gradients | Gradient clipping |
| Slow decrease | Vanishing gradients | ReLU + residuals |
| Low train, high val | Overfitting | Dropout, weight decay |
| High train + val | Underfitting | More capacity |
| Loss spike | High LR | Reduce LR, clip grads |
| Plateau | Saddle point | Cyclical LR |
| >50% zero activations | Dead ReLU | Leaky ReLU |

## Gradient Pathologies

### Vanishing (<1e-7 in early layers)

```python
# ReLU + residuals + He init + batch norm
activation = nn.ReLU()
def forward(self, x): return x + self.block(x)
nn.init.kaiming_normal_(layer.weight, mode='fan_in')
self.bn = nn.BatchNorm1d(features)
```

### Exploding (>100, NaN loss)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer = torch.optim.Adam(params, lr=1e-4)  # Smaller LR
layer = torch.nn.utils.spectral_norm(nn.Linear(in_f, out_f))
```

### Dead ReLU (>50% zeros)

```python
activation = nn.LeakyReLU(0.01)  # or nn.PReLU()
layer.bias.data.fill_(0.01)
```

## Loss Curves

### Overfitting (train↓, val↑)

```python
self.dropout = nn.Dropout(p=0.5)
optimizer = torch.optim.Adam(params, weight_decay=1e-4)
if val_loss > best_loss:
    patience_counter += 1
    if patience_counter >= patience: break
```

### Underfitting (both high)
- Increase capacity
- Train longer
- Reduce regularization

### Spikes

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
lr = base_lr * min(step / warmup_steps, 1.0)  # Warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
```

### Plateaus

```python
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-4, 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
```

## Diagnostics

```python
def diagnose_gradients(model, data_loader, criterion):
    inputs, targets = next(iter(data_loader))
    loss = criterion(model(inputs), targets)
    model.zero_grad(); loss.backward()

    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            stats[name] = {'norm': g.norm().item(), 'zero_pct': (g==0).float().mean().item()}

    norms = [s['norm'] for s in stats.values()]
    if max(norms)/(min(norms)+1e-10) > 1000: print("⚠️ VANISHING")
    if any(n > 100 for n in norms): print("⚠️ EXPLODING")
    for name, s in stats.items():
        if s['zero_pct'] > 0.5: print(f"⚠️ DEAD: {name}")
```

## LR Selection

```python
def lr_range_test(model, train_loader, start_lr=1e-7, end_lr=10):
    lr_mult = (end_lr / start_lr) ** (1 / steps)
    lr, losses = start_lr, []
    for batch in train_loader:
        optimizer.param_groups[0]['lr'] = lr
        losses.append((lr, train_step(batch)))
        lr *= lr_mult
        if losses[-1][1] > losses[0][1] * 4: break
    return losses  # Optimal: steepest descent before explosion
```

## Batch Size Effects

- **Small**: Better generalization, noisy
- **Large**: Faster, sharp minima
- **Linear Scaling**: If batch×k, then LR×k (with warmup)

## Pre-Training Checks

```python
# 1. Overfit single batch
single_batch = next(iter(train_loader))
for _ in range(1000):
    loss = train_step(single_batch)
assert loss < 0.01

# 2. Check gradient flow
diagnose_gradients(model, train_loader, criterion)

# 3. Verify data
for x, y in train_loader:
    assert not torch.isnan(x).any() and not torch.isnan(y).any()
    break
```

**Outcome**: ReLU/LeakyReLU, He init, batch norm, residuals, gradient clipping, warmup, monitoring
