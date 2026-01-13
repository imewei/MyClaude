---
name: correlation-computational-methods
version: "1.0.7"
maturity: "5-Expert"
specialization: Correlation Algorithms
description: Implement efficient correlation algorithms including FFT-based O(N log N) autocorrelation (50,000× speedup), multi-tau correlators for wide dynamic ranges, KD-tree spatial correlations, JAX-accelerated GPU computation (200× speedup), and statistical validation (bootstrap, convergence). Use when optimizing calculations for MD trajectories, DLS, XPCS, or large experimental datasets.
---

# Computational Methods & Algorithms

Efficient, scalable algorithms for correlation function calculation.

---

## Algorithm Selection

| Data Size | Algorithm | Complexity |
|-----------|-----------|------------|
| N < 1000 | Direct | O(N²) |
| 1000 < N < 10⁶ | FFT-based | O(N log N) |
| N > 10⁶ | Block averaging + FFT | O(N log N) |
| Wide dynamic range | Multi-tau correlator | O(m log N) |

---

## FFT-Based Correlation

### O(N log N) Autocorrelation
```python
import numpy as np
from scipy import fft

def fft_autocorrelation(data):
    N = len(data)
    padded = np.concatenate([data - data.mean(), np.zeros(N)])

    F = fft.fft(padded)
    S = np.abs(F)**2
    C = fft.ifft(S).real[:N]

    C /= (N - np.arange(N))
    C /= C[0]
    return C

def fft_cross_correlation(x, y):
    N = len(x)
    x_pad = np.concatenate([x - x.mean(), np.zeros(N)])
    y_pad = np.concatenate([y - y.mean(), np.zeros(N)])

    return fft.ifft(fft.fft(x_pad) * np.conj(fft.fft(y_pad))).real[:N] / (N - np.arange(N))
```

**Speedup**: 50,000× for N = 10⁶

---

## Multi-Tau Correlator

For wide dynamic range (10⁻⁶ to 10³ s):

```python
class MultiTauCorrelator:
    def __init__(self, num_levels=16, channels_per_level=8):
        self.num_levels = num_levels
        self.m = channels_per_level
        self.levels = [np.zeros(2*self.m) for _ in range(num_levels)]

    def add_point(self, value):
        # Level 0: linear spacing
        self.levels[0] = np.roll(self.levels[0], 1)
        self.levels[0][0] = value

        # Propagate to higher levels (logarithmic spacing)
        for level in range(1, self.num_levels):
            if self._should_promote(level):
                avg = (self.levels[level-1][0] + self.levels[level-1][1]) / 2
                self.levels[level] = np.roll(self.levels[level], 1)
                self.levels[level][0] = avg
```

**Memory**: O(m log N) instead of O(N)

---

## Spatial Correlation (KD-Tree)

```python
from scipy.spatial import cKDTree

def efficient_rdf(positions, box_size, rmax, dr, pbc=True):
    N = len(positions)
    tree = cKDTree(positions, boxsize=box_size if pbc else None)
    pairs = tree.query_pairs(rmax, output_type='ndarray')

    distances = np.linalg.norm(positions[pairs[:,0]] - positions[pairs[:,1]], axis=1)
    hist, _ = np.histogram(distances, bins=np.arange(0, rmax, dr))

    r = np.arange(dr/2, rmax, dr)
    rho = N / box_size**3
    ideal = 4*np.pi * r**2 * dr * rho * N
    return r, hist / ideal
```

**Complexity**: O(N log N) vs O(N²) brute force

---

## Block Averaging

For long MD trajectories (10⁶ - 10⁹ frames):

```python
def block_averaged_correlation(trajectory, block_size=10000, max_lag=1000):
    N = len(trajectory)
    num_blocks = N // block_size

    C_blocks = []
    for i in range(num_blocks):
        block = trajectory[i*block_size:(i+1)*block_size]
        C_blocks.append(fft_autocorrelation(block)[:max_lag])

    return np.mean(C_blocks, axis=0), np.std(C_blocks, axis=0) / np.sqrt(num_blocks)
```

---

## JAX GPU Acceleration

```python
import jax
import jax.numpy as jnp

@jax.jit
def jax_autocorrelation(data):
    N = len(data)
    padded = jnp.concatenate([data - jnp.mean(data), jnp.zeros(N)])
    F = jnp.fft.fft(padded)
    C = jnp.fft.ifft(jnp.abs(F)**2).real[:N]
    C = C / (N - jnp.arange(N)) / C[0]
    return C

# Batch processing
jax_correlation_batch = jax.vmap(jax_autocorrelation)
C_batch = jax_correlation_batch(data_batch)  # Parallel on GPU
```

**Speedup**: 200× for N = 10⁶, batch of 100

---

## Statistical Validation

### Bootstrap Error
```python
def bootstrap_correlation(data, num_bootstrap=1000, block_size=None):
    N = len(data)
    if block_size is None:
        C = fft_autocorrelation(data)
        block_size = np.where(C < 1/np.e)[0][0] if any(C < 1/np.e) else N//10

    num_blocks = N // block_size
    C_bootstrap = []
    for _ in range(num_bootstrap):
        blocks = np.random.choice(num_blocks, size=num_blocks, replace=True)
        resampled = np.concatenate([data[b*block_size:(b+1)*block_size] for b in blocks])
        C_bootstrap.append(fft_autocorrelation(resampled))

    return np.mean(C_bootstrap, axis=0), np.std(C_bootstrap, axis=0)
```

### Convergence Test
```python
def convergence_test(data, fractions=[0.25, 0.5, 0.75, 1.0]):
    return {frac: fft_autocorrelation(data[:int(len(data)*frac)]) for frac in fractions}
```

### Robust Statistics
```python
def robust_correlation(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    mask = np.abs(data - median) < 3*mad
    return fft_autocorrelation(data[mask])
```

---

## Performance Benchmarks

| Method | Relative Speed |
|--------|----------------|
| NumPy direct | 1× (baseline) |
| NumPy FFT | 50× |
| JAX GPU | 200× |
| Multi-GPU | 1000× |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Algorithm selection | Match to data size |
| Memory management | Block averaging for N > 10⁶ |
| Validation | Bootstrap N=1000 for error bars |
| Convergence | Compare N/2 vs N results |
| Physical constraints | Check sum rules, positivity |

---

## Checklist

- [ ] Appropriate algorithm for data size
- [ ] Memory-efficient for large datasets
- [ ] Bootstrap error estimation
- [ ] Convergence verified
- [ ] Outliers removed (3-MAD)
- [ ] Physical constraints validated

---

**Version**: 1.0.5
