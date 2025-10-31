---
name: training-diagnostics
description: Diagnose and resolve neural network training failures through systematic analysis of gradient pathologies, loss curves, convergence issues, and performance bottlenecks. Use this skill when encountering vanishing gradients (gradients <1e-7, slow convergence), exploding gradients (NaN losses, gradient norms >100), or dead ReLU neurons (>50% zero activations). Apply when debugging loss curve anomalies (training divergence, validation spikes, plateaus, overfitting gaps). Use when models fail to converge, train slowly, or exhibit unstable behavior with oscillating losses. Apply when analyzing learning rate sensitivity, optimizer pathologies, or batch size effects on training dynamics. Use when diagnosing overfitting (train-val gap), underfitting (high losses), or double descent phenomena. Apply when investigating NaN/Inf values, gradient clipping needs, or numerical stability issues. Use when working with training logs, TensorBoard visualizations, gradient norm plots, or loss curve analysis. Apply when debugging training scripts (train.py), optimization loops, or convergence monitoring code requiring systematic troubleshooting.
---

# Training Diagnostics

This skill provides systematic frameworks for diagnosing and resolving neural network training issues. It covers gradient pathologies, loss curve interpretation, convergence analysis, and practical debugging workflows.

## When to use this skill

- When training fails to converge or loss remains high after many epochs (underfitting, capacity issues)
- When encountering vanishing gradients (gradients <1e-7 in early layers, very slow learning, shallow network behavior)
- When experiencing exploding gradients (NaN/Inf losses, gradient norms >100, parameter updates too large)
- When observing dead ReLU neurons (>40% zero activations, gradients permanently zero, learning stops)
- When seeing saturated sigmoid/tanh activations (activations near ±1, near-zero gradients, training plateaus)
- When debugging loss curve anomalies (sudden spikes, unexpected plateaus, training divergence)
- When analyzing overfitting patterns (low train loss, high validation loss, increasing train-val gap over time)
- When diagnosing underfitting issues (both train and validation losses remain high, insufficient model capacity)
- When investigating double descent phenomena (performance worsens then improves with more capacity/training)
- When encountering training instability (oscillating losses, sudden loss spikes, unstable convergence)
- When models exhibit NaN/Inf values in losses, gradients, or parameters during training
- When analyzing learning rate sensitivity (loss explodes with high LR, stagnates with low LR)
- When comparing optimizers (SGD vs Adam behavior differences, convergence speed variations)
- When investigating batch size effects on training stability and generalization
- When debugging gradient flow through deep networks (checking layer-wise gradient statistics)
- When setting up gradient clipping thresholds to prevent explosions
- When analyzing activation distributions to detect saturation or dead neuron issues
- When implementing proper weight initialization strategies (He, Xavier, orthogonal)
- When configuring normalization layers (BatchNorm, LayerNorm, GroupNorm) to stabilize training
- When working with training diagnostic scripts, gradient analysis tools, or TensorBoard logging
- When interpreting loss curve plots, learning rate schedules, or validation metrics over time
- When systematically debugging unknown training failures through hypothesis testing

## Diagnostic Framework

### Systematic Diagnosis Process

To diagnose training issues systematically:

1. **Characterize Symptoms**
   - Observe loss curves (training and validation)
   - Check gradient statistics (norms, distribution)
   - Monitor parameter updates (magnitude, direction)
   - Track metrics over time (accuracy, other task-specific metrics)

2. **Hypothesize Root Causes**
   - Match symptoms to known pathologies
   - Consider multiple potential causes
   - Rank by likelihood based on symptoms

3. **Test Hypotheses**
   - Design experiments to isolate causes
   - Use diagnostic tools and visualizations
   - Verify with controlled changes

4. **Apply Solutions**
   - Implement theoretically-grounded fixes
   - Start with least disruptive changes
   - Validate improvements

5. **Document and Iterate**
   - Record what worked and why
   - Build knowledge base of solutions
   - Refine diagnosis skills

## Common Training Pathologies

### Gradient Pathologies

#### Vanishing Gradients

**Symptoms:**
- Gradients become very small (< 1e-7) in early layers
- Training loss decreases very slowly or not at all
- Early layers don't update while later layers do
- Network behaves like shallow network

**Root Causes:**
- Deep networks with sigmoid/tanh activations (derivatives < 1)
- Poor weight initialization (weights too small)
- No skip connections to propagate gradients
- Exponential decay of gradients through depth

**Diagnosis:**
```python
# Use scripts/diagnose_gradients.py
python scripts/diagnose_gradients.py --checkpoint model.pt --log-dir logs/
```

Check gradient norms per layer - vanishing shows exponential decay.

**Solutions:**

1. **ReLU Activation Functions:**
   - Replace sigmoid/tanh with ReLU family
   - Gradient = 1 for positive activations (no decay)
   - Variants: Leaky ReLU, PReLU, ELU

2. **Residual Connections:**
   ```python
   def residual_block(x):
       residual = x
       x = layer1(x)
       x = activation(x)
       x = layer2(x)
       return x + residual  # Skip connection
   ```

3. **Proper Initialization:**
   - Xavier/Glorot: `std = sqrt(2/(fan_in + fan_out))`
   - He initialization: `std = sqrt(2/fan_in)` (for ReLU)

4. **Batch Normalization:**
   - Normalize activations across batch
   - Reduces internal covariate shift
   - Allows higher learning rates

5. **Gradient Clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

**Verification:**
- Gradient norms should be similar across layers (within 1-2 orders of magnitude)
- Training loss should decrease steadily

#### Exploding Gradients

**Symptoms:**
- Gradients become very large (> 100 or NaN)
- Loss increases or becomes NaN
- Parameter updates are huge
- Training is unstable

**Root Causes:**
- Large learning rate relative to gradient magnitude
- Weights initialized too large
- Recurrent connections without proper handling
- Large batch size without learning rate scaling
- Eigenvalues of weight matrices > 1

**Diagnosis:**
```python
# scripts/diagnose_gradients.py checks for:
# - Gradient norms > threshold
# - NaN/Inf in gradients
# - Sudden spikes in gradient magnitude
```

**Solutions:**

1. **Gradient Clipping (Essential):**
   ```python
   # By norm (most common)
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

   # By value
   torch.nn.utils.clip_grad_value_(parameters, clip_value=0.5)
   ```

2. **Reduce Learning Rate:**
   - Start with smaller LR (e.g., 1e-4 instead of 1e-3)
   - Use learning rate warmup for first few epochs

3. **Proper Initialization:**
   - Use He/Xavier initialization
   - Check initial weight magnitudes (should be ~0.01 to 0.1)

4. **Normalize Inputs:**
   ```python
   # Normalize to zero mean, unit variance
   x = (x - x.mean()) / x.std()
   ```

5. **Spectral Normalization (for stability):**
   ```python
   # Constrain weight matrices to have spectral norm ≤ 1
   layer = torch.nn.utils.spectral_norm(torch.nn.Linear(in_features, out_features))
   ```

6. **For RNNs/LSTMs:**
   - Use gradient clipping (essential)
   - Consider GRU instead of vanilla RNN
   - Use layer normalization

**Verification:**
- Gradient norms should be < 10
- No NaN/Inf values
- Stable loss decrease

#### Dead ReLUs

**Symptoms:**
- Large fraction of neurons output zero (>50%)
- Learning stagnates after initial progress
- Validation performance plateaus while training continues
- Gradient flow stops in affected neurons

**Root Causes:**
- Large negative bias shifts neurons into non-active region
- Very large learning rate causes weights to update past zero-gradient region
- Poor initialization biases neurons negative
- Once dead, ReLU gradient = 0 (irreversible without special techniques)

**Diagnosis:**
```python
# scripts/analyze_activations.py
# Checks percentage of zero activations per layer
python scripts/analyze_activations.py --model model.pt --data val_loader
```

Look for layers with >40% zero activations.

**Solutions:**

1. **Leaky ReLU:**
   ```python
   activation = torch.nn.LeakyReLU(negative_slope=0.01)
   ```
   Allows small gradient for negative inputs.

2. **Parametric ReLU (PReLU):**
   ```python
   activation = torch.nn.PReLU()
   ```
   Learns negative slope parameter.

3. **Reduce Learning Rate:**
   - Prevents large updates that kill neurons
   - Use learning rate warmup

4. **Better Initialization:**
   - He initialization with proper scaling
   - Check bias initialization (should be small, often 0)

5. **Lower Negative Bias:**
   - If biases are very negative, adjust initialization
   ```python
   layer.bias.data.fill_(0.01)  # Slightly positive bias
   ```

**Prevention:**
- Monitor dead neuron percentage during training
- Use Leaky/PReLU instead of ReLU
- Careful learning rate tuning

#### Saturation (Sigmoid/Tanh)

**Symptoms:**
- Gradients near zero despite non-zero loss
- Training plateaus early
- Activations clustered near -1 or +1 (tanh) or 0/1 (sigmoid)

**Root Causes:**
- Inputs to sigmoid/tanh in flat regions (|x| > 3)
- Large weight magnitudes push activations to extremes
- Derivative ≈ 0 in saturated regions

**Diagnosis:**
```python
# Check activation distributions
# If most activations near extremes, saturation likely
```

**Solutions:**

1. **Switch to ReLU Family:**
   - No saturation for positive inputs
   - Gradient = 1 (ReLU) or constant (Leaky/PReLU)

2. **Batch Normalization:**
   - Keeps activations in reasonable range
   - Prevents saturation

3. **Reduce Weight Magnitude:**
   - Smaller weights → smaller pre-activations
   - Use weight decay/L2 regularization

4. **Better Initialization:**
   - Xavier initialization designed to prevent saturation

5. **For Output Layer:**
   - Keep sigmoid/tanh for output if needed (e.g., binary classification)
   - But use ReLU for hidden layers

### Loss Curve Pathologies

#### Underfitting

**Symptoms:**
- High training loss (not decreasing to acceptable level)
- High validation loss (similar to training)
- Gap between training and validation loss is small
- Model performance poor on both train and val

**Diagnosis:**
- Check if model capacity is sufficient
- Verify data is learnable (not too noisy)
- Ensure sufficient training time

**Root Causes:**
- Model too simple (insufficient capacity)
- Training stopped too early
- Learning rate too low (slow convergence)
- Strong regularization limiting capacity

**Solutions:**

1. **Increase Model Capacity:**
   ```python
   # Add more layers
   # Increase width (more neurons per layer)
   # Use more expressive architecture
   ```

2. **Train Longer:**
   - Increase number of epochs
   - Check if loss still decreasing

3. **Increase Learning Rate:**
   - If loss decreasing too slowly, try higher LR
   - Use learning rate finder

4. **Reduce Regularization:**
   - Lower weight decay
   - Reduce dropout probability
   - Fewer data augmentation transforms

5. **Verify Data Quality:**
   - Check labels are correct
   - Ensure input preprocessing is appropriate
   - Visualize some training examples

**Verification:**
- Training loss should reach low value (< 0.1 for many tasks)
- If loss plateaus at high value, needs more capacity or training time

#### Overfitting

**Symptoms:**
- Low training loss, high validation loss
- Gap between train and val loss increases over time
- Validation performance degrades while training improves
- Model memorizes training data

**Diagnosis:**
```python
# Plot train vs val loss
# Look for divergence point
# Check if train loss continues decreasing while val increases
```

**Root Causes:**
- Model too complex for data
- Insufficient training data
- No/insufficient regularization
- Training too long

**Solutions:**

1. **Early Stopping:**
   ```python
   # Monitor validation loss
   # Stop when val loss stops improving
   # Save best checkpoint based on validation performance
   ```

2. **Regularization:**

   **Dropout:**
   ```python
   dropout = torch.nn.Dropout(p=0.5)  # Drop 50% of neurons during training
   ```

   **Weight Decay (L2):**
   ```python
   optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-4)
   ```

   **Data Augmentation:**
   ```python
   # For images: rotation, flipping, color jitter
   # For text: synonym replacement, back-translation
   # For tabular: noise injection
   ```

3. **Get More Data:**
   - Collect more training examples
   - Use data augmentation to artificially increase data
   - Try transfer learning if data limited

4. **Reduce Model Capacity:**
   - Fewer layers
   - Fewer neurons per layer
   - Simpler architecture

5. **Ensemble Methods:**
   - Train multiple models
   - Average predictions (reduces overfitting)

6. **Cross-Validation:**
   - K-fold cross-validation for robust validation
   - Helps detect overfitting

**Verification:**
- Train-val gap should be reasonable (< 2x)
- Validation performance should be primary metric

#### Double Descent

**Symptoms:**
- Performance improves, then worsens, then improves again
- Occurs when increasing model size or training time
- Peak at "interpolation threshold" (model exactly fits training data)

**Recognition:**
- Plot test error vs model size (parameters)
- Or test error vs training epochs
- Look for characteristic double descent curve

**Interpretation:**
- Classical regime: Bias-variance tradeoff (left of peak)
- Interpolation threshold: Model fits all training data exactly (peak)
- Modern regime: Overparameterized, benefits from implicit regularization (right of peak)

**Action:**
- **Don't stop at the peak!** Continue training or use larger models
- Modern deep learning operates in overparameterized regime
- More parameters can help even when overfitting seems to occur

#### Loss Spikes

**Symptoms:**
- Sudden large increase in loss during training
- Loss jumps up then may or may not recover
- Can happen once or repeatedly

**Root Causes:**
- Learning rate too high (escaped local minimum)
- Batch size too small (noisy gradients)
- Hit high-curvature region of loss landscape
- Bad batch (outliers or corrupted data)
- Gradient explosion event

**Diagnosis:**
```python
# Check correlation with:
# - Learning rate schedule changes
# - Data batches (are some batches problematic?)
# - Gradient norms (spike in gradients?)
```

**Solutions:**

1. **Learning Rate Decay:**
   ```python
   # Reduce LR over time
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
   ```

2. **Gradient Clipping:**
   - Prevents single bad batch from derailing training
   ```python
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
   ```

3. **Increase Batch Size:**
   - Reduces gradient noise
   - More stable updates

4. **Learning Rate Warmup:**
   ```python
   # Start with small LR, gradually increase
   # Prevents early instability
   ```

5. **Checkpoint and Resume:**
   - Save checkpoints frequently
   - If spike occurs, resume from last good checkpoint with lower LR

6. **Investigate Bad Batches:**
   ```python
   # If spikes correlate with specific batches:
   # - Check for outliers in data
   # - Look for corrupted examples
   # - Consider data cleaning
   ```

**Verification:**
- Spikes should not occur with proper LR and gradient clipping
- If recurring, investigate data quality

#### Plateaus

**Symptoms:**
- Loss stops decreasing for extended period
- Gradient norms near zero
- No progress for many epochs

**Root Causes:**
- Local minimum (rare in high dimensions)
- Saddle point (more common)
- Learning rate too small
- Batch size too large (smooths loss landscape)

**Diagnosis:**
```python
# Check gradient norms - if very small, at critical point
# Check Hessian eigenvalues (if feasible) - saddle point has negative eigenvalues
```

**Solutions:**

1. **Learning Rate Scheduling:**
   ```python
   # Cyclical learning rates
   scheduler = torch.optim.lr_scheduler.CyclicalLR(optimizer, base_lr=1e-4, max_lr=1e-3)

   # Reduce on plateau
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
   ```

2. **Momentum:**
   - Helps escape saddle points
   ```python
   optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
   ```

3. **Change Optimizer:**
   - Try different optimizer (Adam → SGD or vice versa)
   - Different optimizers have different implicit biases

4. **Learning Rate Warmup:**
   - For transformers and large models
   - Helps escape early plateaus

5. **Adjust Batch Size:**
   - Smaller batches add noise (exploration)
   - Can help escape flat regions

**Verification:**
- Loss should start decreasing again after intervention
- If persistent plateau, may be at minimum (stop training)

### Convergence Analysis

#### Learning Rate Sensitivity

**Systematic LR Testing:**

```python
# Learning Rate Range Test
# scripts/lr_range_test.py

# Start with very small LR, exponentially increase
# Plot loss vs LR
# Optimal LR: steepest descent region, before loss explodes
```

**Interpretation:**
- **Too low**: Slow convergence, may not reach minimum
- **Too high**: Divergence, oscillations, instability
- **Sweet spot**: Fast convergence, stable training

**Recommendations:**
- Start with LR from range test
- Use learning rate schedules (decay over time)
- For large models: Warmup important

#### Optimizer Pathologies

**SGD Issues:**
- Gets stuck in sharp minima
- Sensitive to learning rate
- Slow convergence on plateaus

**Adam Issues:**
- Sometimes worse generalization than SGD
- Can overshoot minima with high learning rate
- Weight decay not equivalent to L2 regularization (use AdamW)

**Solutions:**
- Try multiple optimizers
- Use AdamW instead of Adam for better weight decay
- Consider SGD with momentum for final fine-tuning
- Use adaptive learning rates (Adam) for quick prototyping

#### Batch Size Effects

**Small Batches:**
- Pros: Noisy gradients provide exploration, often better generalization
- Cons: Slow per-epoch training, unstable updates

**Large Batches:**
- Pros: Faster per-epoch, stable updates
- Cons: Sharp minima, worse generalization, requires LR scaling

**Linear Scaling Rule:**
```
If batch_size increases by factor k,
increase learning_rate by factor k
```

**Warmup for Large Batches:**
- Essential when using large batch sizes
- Gradually increase LR over first few epochs

## Diagnostic Tools and Scripts

### Gradient Analysis Script

**scripts/diagnose_gradients.py:**
```python
#!/usr/bin/env python3
"""
Diagnose gradient flow issues in neural network training.

Usage:
    python diagnose_gradients.py --checkpoint model.pt --data val_loader.pkl

Checks:
- Gradient norms per layer
- Vanishing gradients (exponential decay)
- Exploding gradients (very large norms or NaN)
- Dead neurons (zero gradients)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def diagnose_gradients(model, data_loader, criterion):
    """Compute and analyze gradient statistics."""
    model.train()

    # Forward pass on batch
    inputs, targets = next(iter(data_loader))
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Collect gradient statistics
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'norm': param.grad.norm().item(),
                'max': param.grad.max().item(),
                'min': param.grad.min().item(),
                'num_zeros': (param.grad == 0).sum().item(),
                'total_params': param.grad.numel()
            }

    return grad_stats

def plot_gradient_flow(grad_stats):
    """Visualize gradient flow across layers."""
    layers = list(grad_stats.keys())
    norms = [grad_stats[layer]['norm'] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.semilogy(range(len(norms)), norms, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Flow Across Layers')
    plt.xticks(range(len(layers)), [l.split('.')[0] for l in layers], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
    print("Saved gradient flow plot to gradient_flow.png")

def identify_issues(grad_stats):
    """Identify gradient pathologies."""
    issues = []

    # Check for vanishing gradients
    norms = [grad_stats[layer]['norm'] for layer in grad_stats]
    if max(norms) / min(norms) > 1000:
        issues.append("VANISHING GRADIENTS: Ratio of max/min gradient norms > 1000")

    # Check for exploding gradients
    if any(norm > 100 for norm in norms):
        issues.append("EXPLODING GRADIENTS: Some gradient norms > 100")

    # Check for dead neurons (ReLU)
    for layer, stats in grad_stats.items():
        zero_fraction = stats['num_zeros'] / stats['total_params']
        if zero_fraction > 0.5:
            issues.append(f"DEAD NEURONS: {layer} has {zero_fraction:.1%} zero gradients")

    # Check for NaN/Inf
    if any(np.isnan(stats['mean']) or np.isinf(stats['mean']) for stats in grad_stats.values()):
        issues.append("NaN/Inf GRADIENTS: Found NaN or Inf values")

    return issues

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', required=True, help='Path to validation data loader')
    args = parser.parse_args()

    # Load model and data
    model = torch.load(args.checkpoint)
    data_loader = torch.load(args.data)
    criterion = torch.nn.CrossEntropyLoss()

    # Diagnose
    grad_stats = diagnose_gradients(model, data_loader, criterion)
    plot_gradient_flow(grad_stats)

    # Identify issues
    issues = identify_issues(grad_stats)
    if issues:
        print("\nPOTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo major gradient issues detected.")
```

### Activation Analysis Script

**scripts/analyze_activations.py:**

Analyzes activation distributions to detect dead neurons, saturation, and other issues.

### Loss Curve Comparison

**scripts/compare_training_runs.py:**

Compares multiple training runs to identify best hyperparameters and diagnose issues.

## Workflow: Systematic Training Debugging

To debug a training issue systematically:

1. **Gather Evidence:**
   ```bash
   # Collect training logs
   # Run gradient diagnostics
   python scripts/diagnose_gradients.py --checkpoint checkpoint.pt --data data.pkl

   # Analyze activations
   python scripts/analyze_activations.py --checkpoint checkpoint.pt --data data.pkl

   # Plot loss curves
   python scripts/plot_losses.py --log-dir logs/
   ```

2. **Characterize Symptoms:**
   - Review plots and statistics
   - Identify patterns in loss curves
   - Check gradient and activation statistics
   - List all observed symptoms

3. **Form Hypotheses:**
   - Match symptoms to known pathologies
   - Consider multiple potential causes
   - Rank by likelihood

4. **Design Experiments:**
   - Test most likely hypotheses first
   - Change one thing at a time
   - Use controlled comparisons

5. **Implement Solutions:**
   - Apply theoretically-grounded fixes
   - Start with least disruptive changes
   - Monitor improvements carefully

6. **Validate:**
   - Verify fix resolves symptoms
   - Ensure no new issues introduced
   - Compare to baseline

7. **Document:**
   - Record what worked and why
   - Note failed approaches
   - Build debugging knowledge base

## Quick Reference: Symptom → Solution

| Symptom | Likely Cause | First Solution to Try |
|---------|-------------|---------------------|
| Loss = NaN | Exploding gradients | Gradient clipping |
| Loss decreasing very slowly | Vanishing gradients or low LR | Check gradients, try ReLU, increase LR |
| Train low, Val high | Overfitting | Add regularization (dropout, weight decay) |
| Both train and val high | Underfitting | Increase model capacity or train longer |
| Loss spike then recover | High LR or bad batch | Reduce LR, add gradient clipping |
| Loss plateau | Saddle point or low LR | Try cyclical LR or increase LR |
| >50% zero activations | Dead ReLUs | Switch to Leaky/PReLU, lower LR |
| Activations near ±1 | Saturation (tanh/sigmoid) | Switch to ReLU, add batch norm |

## Best Practices

### Prevention > Cure

1. **Start with known-good hyperparameters:**
   - Use defaults from papers/repos for similar tasks
   - Proven architectures (ResNet, Transformer)

2. **Monitor from start:**
   - Log everything: losses, gradients, activations
   - Use TensorBoard or Weights & Biases
   - Set up alerts for NaN/Inf

3. **Sanity checks before full training:**
   - Overfit on small batch (should reach ~0 loss)
   - Check gradient flow on first batch
   - Verify data loader correctness

4. **Use validation set properly:**
   - Monitor both train and val
   - Early stopping based on val performance
   - Keep test set completely separate

### Debugging Hygiene

1. **Change one thing at a time**
2. **Keep good records** of all experiments
3. **Use version control** for code and configs
4. **Set random seeds** for reproducibility
5. **Save checkpoints frequently**

### When to Stop Debugging

Stop debugging and consider alternatives when:
- Multiple interventions don't help
- Problem may be fundamental (data quality, task difficulty)
- Cost of debugging > cost of different approach

Consider:
- Different architecture
- Different problem formulation
- More/better data
- Consulting experts or literature

---

## Practical Tools & Resources

### Diagnostic Scripts

This skill includes ready-to-use Python scripts in `scripts/`:

1. **diagnose_gradients.py** - Gradient flow analysis
   - Detects vanishing/exploding gradients
   - Identifies dead neurons
   - Computes layer-wise gradient statistics
   - Usage: `python scripts/diagnose_gradients.py --checkpoint model.pt --data sample.pt`

2. **analyze_activations.py** - Activation distribution analysis
   - Detects dead ReLU neurons
   - Identifies saturated sigmoid/tanh activations
   - Monitors activation health
   - Usage: `python scripts/analyze_activations.py --model model.pt --data sample.pt`

3. **compare_training_runs.py** - Multi-experiment comparison
   - Compares hyperparameters across runs
   - Identifies best performing configurations
   - Analyzes convergence patterns
   - Usage: `python scripts/compare_training_runs.py --runs exp1/ exp2/ exp3/`

**Quick Integration:**
```python
from scripts.diagnose_gradients import compute_gradient_stats, diagnose_gradient_pathology

# During training loop (every 10 epochs)
grad_stats = compute_gradient_stats(model, train_loader, criterion)
diagnosis = diagnose_gradient_pathology(grad_stats)

if diagnosis['errors']:
    print("⚠️ Training issues detected - see recommendations")
```

See `scripts/README.md` for detailed documentation and integration examples.

### Reference Materials

Comprehensive guides in `references/`:

1. **gradient-pathologies-reference.md**
   - Complete guide to vanishing/exploding gradients
   - Dead neuron analysis
   - Mathematical foundations
   - Prevention strategies with code

2. **loss-curve-interpretation.md**
   - Overfitting and underfitting patterns
   - Loss spikes and instability
   - Learning rate diagnostics
   - Double descent phenomenon
   - Quick diagnosis flowcharts

---

*This skill provides systematic frameworks for diagnosing and resolving neural network training issues, from gradient pathologies to convergence problems, with practical tools and workflows.*
