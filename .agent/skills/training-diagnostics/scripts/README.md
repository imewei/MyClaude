# allow-torch
# Training Diagnostics Scripts

Practical Python tools for diagnosing neural network training issues.

## Available Scripts

### 1. diagnose_gradients.py

Analyzes gradient flow through neural network layers to detect vanishing or exploding gradient problems.

**Features:**
- Computes comprehensive gradient statistics (mean, std, norm, zeros, NaN/Inf)
- Detects vanishing gradients (norm < 1e-7)
- Detects exploding gradients (norm > 100)
- Identifies dead neurons (high percentage of zero gradients)
- Provides actionable recommendations

**Usage:**

```python
from diagnose_gradients import (
    compute_gradient_stats,
    diagnose_gradient_pathology,
    print_gradient_report
)

# Compute gradient statistics
grad_stats = compute_gradient_stats(
    model=model,
    data_loader=train_loader,
    criterion=nn.CrossEntropyLoss()
)

# Diagnose issues
diagnosis = diagnose_gradient_pathology(grad_stats)

# Print report
print_gradient_report(grad_stats, diagnosis, verbose=True)
```

**Example Output:**
```
================================================================================
GRADIENT DIAGNOSTIC REPORT
================================================================================

🚨 CRITICAL ISSUES:
  ❄️  VANISHING GRADIENTS: Min norm = 3.45e-09 (threshold: 1e-7)
     Affected layers: layer1.weight, layer2.weight

⚠️  WARNINGS:
  💀 layer3.weight: 45.2% zero gradients (dead neurons)

================================================================================
RECOMMENDATIONS:
================================================================================

🔧 IMMEDIATE ACTIONS:
  1. Replace sigmoid/tanh with ReLU activations
  2. Add residual/skip connections
  3. Use proper initialization (He for ReLU, Xavier for tanh)
  4. Consider batch normalization
```

---

### 2. analyze_activations.py

Analyzes activation distributions to detect dead neurons, saturation, and other activation pathologies.

**Features:**
- Registers hooks to capture intermediate activations
- Computes activation statistics (mean, std, min, max, percentiles)
- Detects dead ReLU neurons (>50% zeros)
- Detects saturated sigmoid/tanh activations (>80% in extreme ranges)
- Identifies very small or large activations

**Usage:**

```python
from analyze_activations import (
    register_activation_hooks,
    analyze_activation_statistics,
    diagnose_activation_pathologies,
    print_activation_report
)

# Register hooks
hooks = register_activation_hooks(model)

# Forward pass
model.eval()
with torch.no_grad():
    output = model(sample_input)

# Analyze activations
activation_stats = {}
for name, hook in hooks.items():
    if hook.activations is not None:
        activation_stats[name] = analyze_activation_statistics(
            hook.activations,
            name,
            activation_type='relu'  # or 'sigmoid', 'tanh'
        )

# Diagnose
diagnosis = diagnose_activation_pathologies(activation_stats, model)

# Print report
print_activation_report(activation_stats, diagnosis, verbose=True)
```

**Example Output:**
```
================================================================================
ACTIVATION DIAGNOSTIC REPORT
================================================================================

🚨 CRITICAL ISSUES:
  💀 conv2.relu: 67.3% dead neurons (ReLU)
     → Learning rate may be too high or initialization poor

⚠️  WARNINGS:
  📉 fc1.sigmoid: 85.2% saturated activations
     → Consider switching to ReLU or adjusting input scale

================================================================================
RECOMMENDATIONS:
================================================================================

🔧 IMMEDIATE ACTIONS:
  Dead ReLU Neurons Detected:
    1. Reduce learning rate (try 10x smaller)
    2. Use He initialization: nn.init.kaiming_normal_(weights)
    3. Consider Leaky ReLU instead: nn.LeakyReLU(0.01)
    4. Add batch normalization before activation
```

---

### 3. compare_training_runs.py

Compare multiple training runs to identify differences in hyperparameters, metrics, and outcomes.

**Features:**
- Loads training logs from multiple experiments
- Compares hyperparameters to identify what changed
- Compares metric trajectories (loss, accuracy, etc.)
- Identifies best performing runs
- Provides insights on convergence speed and stability

**Usage:**

```python
from compare_training_runs import (
    load_training_log,
    extract_config,
    extract_metrics,
    compare_configs,
    compare_metrics,
    print_comparison_report
)

# Load runs
run_dirs = [Path('exp1'), Path('exp2'), Path('exp3')]

configs = {}
metrics = {}

for run_dir in run_dirs:
    data = load_training_log(run_dir)
    if data:
        configs[run_dir.name] = extract_config(data)
        metrics[run_dir.name] = extract_metrics(data)

# Print comparison
print_comparison_report(configs, metrics, primary_metric='val_loss')
```

**Example Output:**
```
================================================================================
TRAINING RUN COMPARISON REPORT
================================================================================

Comparing 3 runs: exp1, exp2, exp3

--------------------------------------------------------------------------------
CONFIGURATION DIFFERENCES:
--------------------------------------------------------------------------------

Parameter                      exp1            exp2            exp3
--------------------------------------------------------------------------------
learning_rate                  0.001           0.01            0.0001
batch_size                     32              64              32
dropout                        0.5             0.3             0.5

📋 Common parameters: 15 (use --verbose to see all)

--------------------------------------------------------------------------------
METRIC COMPARISON: val_loss
--------------------------------------------------------------------------------

Run                  Final        Best     Mean±Std    Converged
--------------------------------------------------------------------------------
exp1                0.2341      0.2156   0.3245±0.08     Epoch 42
exp2                0.3521      0.3012   0.4123±0.12     Epoch 28
exp3                0.1987      0.1834   0.2876±0.06     Epoch 55 🏆

================================================================================
SUMMARY & RECOMMENDATIONS:
================================================================================

🏆 BEST RUN: exp3
   val_loss: 0.1834 (converged at epoch 55)

🔑 KEY HYPERPARAMETERS FOR BEST RUN:
   learning_rate: 0.0001
   batch_size: 32
   dropout: 0.5

💡 INSIGHTS:
   ⚠️  exp2: High variance in val_loss
      → Training may be unstable, consider reducing learning rate
```

---

## Expected Data Formats

### For gradient analysis:

Your model should be a standard PyTorch `nn.Module`:
```python
model = YourModel()
model.load_state_dict(torch.load('checkpoint.pt'))
```

### For activation analysis:

Same as gradient analysis - any PyTorch model works.

### For training run comparison:

Each experiment directory should contain:
```
experiment1/
├── config.json          # Hyperparameters
└── metrics.json         # Training history
```

**config.json example:**
```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "num_epochs": 100,
  "optimizer": "adam",
  "architecture": "resnet50"
}
```

**metrics.json example:**
```json
{
  "train_loss": [0.85, 0.73, 0.65, ...],
  "val_loss": [0.92, 0.81, 0.74, ...],
  "train_acc": [0.65, 0.71, 0.76, ...],
  "val_acc": [0.62, 0.68, 0.73, ...]
}
```

---

## Integration Example

Full training loop with diagnostics:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diagnose_gradients import compute_gradient_stats, diagnose_gradient_pathology
from analyze_activations import register_activation_hooks, analyze_activation_statistics

# Setup
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Register activation hooks
activation_hooks = register_activation_hooks(model)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Periodic diagnostics (every 10 epochs)
    if epoch % 10 == 0:
        print(f"\n=== Diagnostics at Epoch {epoch} ===")

        # 1. Check gradients
        grad_stats = compute_gradient_stats(
            model, train_loader, criterion
        )
        grad_diagnosis = diagnose_gradient_pathology(grad_stats)

        if grad_diagnosis['errors']:
            print("\n⚠️ Gradient issues detected:")
            for error in grad_diagnosis['errors']:
                print(f"  {error}")

        # 2. Check activations
        model.eval()
        with torch.no_grad():
            sample_input, _ = next(iter(val_loader))
            _ = model(sample_input)

        activation_stats = {}
        for name, hook in activation_hooks.items():
            if hook.activations is not None:
                activation_stats[name] = analyze_activation_statistics(
                    hook.activations, name, 'relu'
                )

        act_diagnosis = diagnose_activation_pathologies(activation_stats, model)

        if act_diagnosis['errors']:
            print("\n⚠️ Activation issues detected:")
            for error in act_diagnosis['errors']:
                print(f"  {error}")
```

---

## Requirements

```bash
pip install torch numpy
```

Optional for visualizations:
```bash
pip install matplotlib seaborn wandb tensorboard
```

---

## Tips

1. **Run diagnostics periodically** (every 5-10 epochs), not every epoch
2. **Save diagnostic outputs** to compare across training runs
3. **Use verbose mode** for detailed analysis when issues detected
4. **Automate responses** - automatically adjust LR or add clipping when issues detected
5. **Compare baselines** - run diagnostics on working models to understand healthy ranges

---

## Contributing

These scripts are templates. Contributions welcome:
- Additional diagnostic metrics
- Support for more frameworks (TensorFlow, JAX)
- Visualization tools (loss landscape, gradient flow diagrams)
- Automated remediation actions
