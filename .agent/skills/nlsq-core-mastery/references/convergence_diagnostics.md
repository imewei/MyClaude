# NLSQ Convergence Diagnostics Reference

## Quick Convergence Checklist

When optimization fails or produces poor results, work through this checklist:

### 1. Initial Cost Reduction
```
Good: >50% reduction from initial cost
Warning: 10-50% reduction
Problem: <10% reduction
```

**If poor reduction:**
- Improve initial guess using domain knowledge
- Check that model matches data
- Try different loss function

### 2. Gradient Norm
```
Excellent: <1e-6
Good: 1e-6 to 1e-4
Warning: 1e-4 to 1e-2
Problem: >1e-2
```

**If large gradient:**
- Increase max_nfev
- Check for numerical issues
- Verify convergence criteria not too tight

### 3. Jacobian Condition Number
```
Excellent: <1e8
Good: 1e8 to 1e10
Warning: 1e10 to 1e12
Critical: >1e12
```

**If ill-conditioned:**
- Scale parameters to similar magnitudes
- Check for redundant parameters
- Consider parameter transformation
- Add regularization if needed

### 4. Residual Analysis
```
Good: Mean ≈ 0, symmetric distribution
Warning: Systematic bias (|mean| > 0.1 * std)
Problem: Non-random patterns
```

**If systematic bias:**
- Model may be misspecified
- Consider additional terms
- Check for missing physics

### 5. Parameters at Bounds
```
Good: No active constraints
Warning: 1-2 parameters at bounds
Problem: Multiple parameters at bounds
```

**If at bounds:**
- Relax bounds if physically reasonable
- Check if bound is artificial
- Consider reparameterization

## Common Failure Modes

### Divergence (Final Cost > Initial Cost)
**Causes:**
- Poor initial guess
- Model doesn't match data
- Numerical instability
- Too large step size

**Solutions:**
- Improve p0 using domain knowledge
- Switch to LM method (smaller steps)
- Add parameter bounds
- Check model implementation

### Stagnation (Cost stops decreasing)
**Symptoms:**
- Cost change < 1e-8 for many iterations
- Gradient not small
- Parameters still changing

**Causes:**
- Flat cost function region
- Ill-conditioned Jacobian
- Numerical precision limits

**Solutions:**
- Increase max_nfev
- Tighten tolerances (ftol, xtol, gtol)
- Scale parameters
- Try different algorithm

### Oscillation (Cost oscillates without converging)
**Symptoms:**
- Cost increases and decreases
- Parameters jump around
- Many iterations without convergence

**Causes:**
- Step size too large
- Poorly scaled parameters
- Conflicting constraints

**Solutions:**
- Use TRF method (adaptive step size)
- Normalize parameters
- Check bound conflicts
- Increase regularization

### Slow Convergence
**Symptoms:**
- Iterations >> 100
- Cost decreasing slowly
- Small but non-zero gradient

**Causes:**
- Poor conditioning
- Tight tolerances
- Complex model

**Solutions:**
- Loosen tolerances slightly
- Improve parameter scaling
- Use analytical Jacobian
- Simplify model if possible

## Tolerance Tuning Guide

### Default Tolerances (balanced)
```python
ftol=1e-8   # Function tolerance
xtol=1e-8   # Parameter tolerance
gtol=1e-8   # Gradient tolerance
```

### Strict Tolerances (high accuracy)
```python
ftol=1e-12  # Very tight
xtol=1e-12
gtol=1e-10
max_nfev=1000  # Allow more iterations
```
**Use when:** High precision required, computational cost not critical

### Loose Tolerances (faster)
```python
ftol=1e-6   # Relaxed
xtol=1e-6
gtol=1e-6
max_nfev=50  # Fewer iterations
```
**Use when:** Quick approximation acceptable, computational speed critical

### Problem-Specific Tuning

**For noisy data:**
```python
ftol=1e-6   # Looser (noise limits precision)
xtol=1e-8   # Standard
gtol=1e-6   # Looser
loss='huber'  # Robust loss
```

**For ill-conditioned problems:**
```python
ftol=1e-10  # Tighter (more careful)
xtol=1e-10
gtol=1e-10
x_scale='jac'  # Automatic scaling
```

**For many parameters (>20):**
```python
ftol=1e-9   # Slightly tighter
xtol=1e-9
gtol=1e-9
max_nfev=200 * n_params  # More iterations per parameter
```

## Performance Troubleshooting

### Memory Issues

**Symptoms:**
- Out of memory errors
- GPU memory errors
- Slow performance

**Solutions:**
1. Estimate memory: `n_points * n_params * 4 bytes`
2. If > GPU memory, use StreamingOptimizer
3. Reduce chunk size
4. Use float32 instead of float64

**Example:**
```python
# Check if streaming needed
memory_gb = n_points * n_params * 4 / 1e9
gpu_memory = 8  # GB

if memory_gb > gpu_memory * 0.8:
    # Use streaming
    chunk_size = int(n_points * gpu_memory * 0.8 / memory_gb)
    optimizer = StreamingOptimizer(model, p0, chunk_size)
else:
    # Regular optimization
    optimizer = CurveFit(model, x, y, p0)
```

### Slow Jacobian Computation

**Symptoms:**
- Each iteration takes >1s
- GPU underutilized
- Most time in Jacobian

**Solutions:**
1. Provide analytical Jacobian if possible
2. Use jacrev for many parameters (>20)
3. Simplify model if possible
4. Check for unnecessary complexity

### JIT Compilation Overhead

**Symptoms:**
- First call very slow (>10s)
- Second call much faster
- Large compilation time

**Solutions:**
1. Accept one-time cost for production
2. Cache compiled functions
3. Use simpler model if compilation critical
4. Consider ahead-of-time compilation

## Real-Time Monitoring

### Log Convergence Progress
```python
def monitor_optimization(result):
    """Print convergence metrics."""
    print(f"Iteration: {result.nfev}")
    print(f"Cost: {result.cost:.6e}")
    print(f"Gradient: {jnp.linalg.norm(result.grad):.6e}")
    print(f"Params: {result.x}")

# Call periodically during optimization
```

### Detect Stagnation
```python
class StagnationDetector:
    def __init__(self, window=10, threshold=1e-8):
        self.window = window
        self.threshold = threshold
        self.costs = []

    def check(self, cost):
        self.costs.append(cost)
        if len(self.costs) > self.window:
            recent_change = abs(self.costs[-1] - self.costs[-self.window])
            relative_change = recent_change / self.costs[-self.window]
            if relative_change < self.threshold:
                return True  # Stagnation detected
        return False
```

### Plot Convergence History
```python
import matplotlib.pyplot as plt

def plot_convergence(iterations, costs, grads):
    """Visualize convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(iterations, costs)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_yscale('log')
    ax1.set_title('Cost Function')
    ax1.grid(True)

    ax2.plot(iterations, grads)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('||Gradient||')
    ax2.set_yscale('log')
    ax2.set_title('Gradient Norm')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
```
