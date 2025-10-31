---
name: correlation-computational-methods
description: Implement efficient algorithms for correlation analysis including FFT-based O(N log N) autocorrelation and cross-correlation computation (50,000× speedup for N=10⁶), multi-tau correlators for wide dynamic ranges (10⁻⁶ to 10³ seconds with logarithmic time spacing), spatial correlations with KD-tree O(N log N) pair distribution functions, multi-scale analysis spanning femtoseconds to hours and nanometers to micrometers, statistical validation (bootstrap resampling with block bootstrap for correlated data, convergence analysis, bias correction for finite-size effects, outlier detection with robust statistics), and JAX-accelerated GPU computation for large datasets with JIT compilation and batch processing (200× CPU speedup for N=10⁶, 1000× multi-GPU). Use when optimizing correlation calculations for molecular dynamics trajectories (10⁶-10⁹ frames), dynamic light scattering with wide dynamic ranges, XPCS data analysis, glassy dynamics with multiple relaxation timescales, or processing large experimental datasets requiring memory-efficient algorithms.
---

# Computational Methods & Algorithms

## When to use this skill

- Optimizing autocorrelation calculations from direct O(N²) to FFT-based O(N log N) for time series with N > 1000 data points (*.py, *.jl implementations)
- Computing cross-correlation functions efficiently using convolution theorem and FFT for multi-variate time series analysis or signal processing (*.py NumPy/SciPy, *.jl Julia FFT)
- Implementing multi-tau correlators for DLS, XPCS, or FCS data spanning wide dynamic ranges (10 ns to 10 s, 1 ms to 1000 s) with logarithmic time spacing and minimal memory (O(m log N) storage)
- Calculating radial distribution functions g(r) efficiently using KD-tree spatial indexing (O(N log N)) for molecular dynamics simulations with periodic boundary conditions (*.py SciPy cKDTree)
- Performing block averaging for memory-efficient correlation analysis of long MD trajectories (10⁶-10⁹ frames) without storing full correlation arrays
- Implementing multi-scale temporal analysis with adaptive time stepping for systems spanning femtoseconds (atomic vibrations) to hours (aging, creep) using hierarchical binning
- Computing hierarchical spatial correlations across length scales from angstroms (atomic) to millimeters (macroscopic) using coarse-graining and Fourier transforms
- Combining multiscale data from different sampling rates (high-frequency 1 MHz and low-frequency 1 Hz) with proper stitching at overlap regions
- Performing bootstrap resampling with block bootstrap for uncertainty estimation in correlated data (N=1000 bootstrap samples recommended) with automatic correlation length detection
- Conducting convergence tests by comparing correlations from data subsets (25%, 50%, 75%, 100%) to ensure sufficient trajectory length and statistical reliability
- Estimating required trajectory length for target error using N_required = (1/target_error)² × correlation_time formula
- Correcting finite-size effects in correlations using periodic image summations for systems where correlation length ξ approaches system size L
- Detecting and removing outliers using robust statistics (median absolute deviation with 3-MAD threshold or 10% trimmed mean) before correlation analysis
- Accelerating correlation calculations with JAX JIT-compiled GPU functions for repeated calls on batches of time series (200× speedup vs NumPy for N=10⁶)
- Processing large datasets (> RAM) using chunked GPU correlation with automatic memory management to avoid out-of-memory errors
- Implementing vectorized batch correlation processing for multiple time series in parallel on GPU (batch of 100 gives 1000× speedup with multi-GPU)
- Analyzing experimental correlation data from DLS, SAXS/SANS, XPCS, or FCS with proper statistical validation and uncertainty quantification
- Optimizing algorithm selection based on data size: direct method (N < 1000), FFT (1000 < N < 10⁶), block averaging (N > 10⁶), or multi-tau for wide dynamic ranges
- Implementing streaming algorithms for real-time correlation analysis of experimental data with continuous data acquisition
- Performing performance benchmarking comparing CPU (NumPy), CPU with FFT, GPU (JAX), and multi-GPU implementations for computational method selection
- Validating correlation results against physical constraints including sum rules (number conservation, compressibility), non-negativity C(0) ≥ |C(t)|, and causality χ(t<0) = 0

Implement efficient, scalable algorithms for correlation function calculation across multiple timescales and spatial scales with rigorous statistical validation.

## Efficient Algorithms

### FFT-Based Correlation Calculation

**Direct Correlation (O(N²)):**
C(τ) = (1/N) ∑ᵢ₌₁^(N-τ) x(i) x(i+τ)

**FFT Method (O(N log N)):**
Based on convolution theorem: correlation ↔ product in Fourier space

```python
import numpy as np
from scipy import fft

def fft_autocorrelation(data):
    """
    Compute autocorrelation using FFT (O(N log N))
    
    Faster than direct method for N > ~100
    """
    N = len(data)
    # Zero-pad to prevent circular correlation
    padded = np.concatenate([data - data.mean(), np.zeros(N)])
    
    # FFT, power spectrum, inverse FFT
    F = fft.fft(padded)
    S = np.abs(F)**2
    C = fft.ifft(S).real[:N]
    
    # Normalize by decreasing number of points
    C /= (N - np.arange(N))
    C /= C[0]  # Normalize to C(0) = 1
    
    return C

def fft_cross_correlation(x, y):
    """
    Cross-correlation C_xy(τ) = ⟨x(t)y(t+τ)⟩
    """
    N = len(x)
    x_pad = np.concatenate([x - x.mean(), np.zeros(N)])
    y_pad = np.concatenate([y - y.mean(), np.zeros(N)])
    
    Fx = fft.fft(x_pad)
    Fy = fft.fft(y_pad)
    C_xy = fft.ifft(Fx * np.conj(Fy)).real[:N]
    
    C_xy /= (N - np.arange(N))
    return C_xy
```

**Performance Comparison:**
- Direct: O(N²) → 1M points: ~10⁶ seconds
- FFT: O(N log N) → 1M points: ~20 seconds
- Speedup: ~50,000× for large N

### Multi-Tau Correlators

**Problem**: Wide dynamic range (10⁻⁶ to 10³ seconds)
- Uniform sampling: Prohibitive memory (10⁹ time points)
- Solution: Logarithmic time spacing

**Algorithm:**
- Level 0: Linear spacing, Δt = dt
- Level 1: Spacing Δt = 2dt, average pairs from level 0
- Level n: Spacing Δt = 2ⁿdt

```python
class MultiTauCorrelator:
    """
    Multi-tau correlator for wide dynamic range
    
    Memory: O(m log(N)) where m = channels per level
    Typical: m=8, achieves 10⁶ dynamic range with ~500 points
    """
    def __init__(self, num_levels=16, channels_per_level=8):
        self.num_levels = num_levels
        self.m = channels_per_level
        
        # Storage for each level
        self.levels = [np.zeros(2*self.m) for _ in range(num_levels)]
        self.correlation = []
        self.counts = []
        
    def add_point(self, value):
        """Add single data point and update correlations"""
        # Level 0: Store directly
        self.levels[0] = np.roll(self.levels[0], 1)
        self.levels[0][0] = value
        
        # Compute correlations at level 0
        for k in range(self.m):
            C = self.levels[0][0] * self.levels[0][k]
            self._update_correlation(0, k, C)
        
        # Propagate to higher levels (coarse-graining)
        for level in range(1, self.num_levels):
            if self._should_promote(level):
                # Average two points from previous level
                avg = (self.levels[level-1][0] + self.levels[level-1][1]) / 2
                self.levels[level] = np.roll(self.levels[level], 1)
                self.levels[level][0] = avg
                
                # Compute correlations at this level
                for k in range(self.m):
                    C = self.levels[level][0] * self.levels[level][k]
                    self._update_correlation(level, k, C)
    
    def _should_promote(self, level):
        """Check if enough points accumulated for promotion"""
        # Promote every 2^level points
        return (self.num_points % (2**level)) == 0
    
    def get_correlation(self):
        """Return correlation function with logarithmic time spacing"""
        times = []
        correlations = []
        
        for level in range(self.num_levels):
            dt = 2**level
            for k in range(self.m):
                times.append(k * dt)
                correlations.append(self.correlation[level][k] / self.counts[level][k])
        
        return np.array(times), np.array(correlations)
```

**Applications:**
- DLS: 10 ns to 10 s
- XPCS: 1 ms to 1000 s
- Glassy dynamics: Wide spectrum of relaxation times

### Spatial Correlation with KD-Trees

**Problem**: Pair distribution g(r) for N particles → N² pairs
**Solution**: KD-tree spatial indexing → O(N log N)

```python
from scipy.spatial import cKDTree

def efficient_rdf(positions, box_size, rmax, dr, pbc=True):
    """
    Efficient radial distribution function using KD-tree
    
    Complexity: O(N log N) vs O(N²) brute force
    """
    N = len(positions)
    
    if pbc:
        # Periodic boundary conditions
        tree = cKDTree(positions, boxsize=box_size)
    else:
        tree = cKDTree(positions)
    
    # Find all pairs within rmax
    pairs = tree.query_pairs(rmax, output_type='ndarray')
    
    # Compute distances
    if pbc:
        distances = np.linalg.norm(
            minimum_image(positions[pairs[:,0]] - positions[pairs[:,1]], box_size),
            axis=1
        )
    else:
        distances = np.linalg.norm(
            positions[pairs[:,0]] - positions[pairs[:,1]], 
            axis=1
        )
    
    # Histogram
    r_bins = np.arange(0, rmax, dr)
    hist, _ = np.histogram(distances, bins=r_bins)
    
    # Normalize by ideal gas
    r_centers = r_bins[:-1] + dr/2
    rho = N / box_size**3
    ideal = 4*np.pi * r_centers**2 * dr * rho * N
    
    g_r = hist / ideal
    
    return r_centers, g_r

def minimum_image(dr, box_size):
    """Apply minimum image convention for PBC"""
    return dr - box_size * np.round(dr / box_size)
```

### Block Averaging for Long Trajectories

**Problem**: Long MD trajectories (10⁶ - 10⁹ frames)
**Memory**: Cannot store full correlation

**Solution**: Block averaging
- Divide trajectory into blocks
- Compute correlation per block
- Average over blocks

```python
def block_averaged_correlation(trajectory, block_size=10000, max_lag=1000):
    """
    Memory-efficient correlation for long trajectories
    """
    N = len(trajectory)
    num_blocks = N // block_size
    
    C_blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = trajectory[start:end]
        
        # Compute correlation for this block
        C = fft_autocorrelation(block)[:max_lag]
        C_blocks.append(C)
    
    # Average over blocks
    C_avg = np.mean(C_blocks, axis=0)
    C_std = np.std(C_blocks, axis=0) / np.sqrt(num_blocks)
    
    return C_avg, C_std
```

## Multi-Scale Analysis

### Temporal Multi-Scale

**Timescale Hierarchy:**
- Atomic vibrations: femtoseconds (10⁻¹⁵ s)
- Molecular rotations: picoseconds (10⁻¹² s)
- Diffusion: nanoseconds to microseconds (10⁻⁹ - 10⁻⁶ s)
- Collective relaxation: milliseconds to seconds (10⁻³ - 10⁰ s)
- Aging, creep: hours (10³ - 10⁴ s)

**Adaptive Time Stepping:**
```python
def adaptive_correlation_sampling(data, times, target_points=1000):
    """
    Adaptive time sampling for multi-scale correlations
    
    Dense sampling at short times, sparse at long times
    """
    # Logarithmic time grid
    log_times = np.logspace(
        np.log10(times[1]), 
        np.log10(times[-1]), 
        target_points
    )
    
    # Interpolate correlation to log grid
    C_full = fft_autocorrelation(data)
    C_sampled = np.interp(log_times, times[:len(C_full)], C_full)
    
    return log_times, C_sampled
```

### Spatial Multi-Scale

**Length Scale Hierarchy:**
- Atomic: angstroms (10⁻¹⁰ m)
- Molecular: nanometers (10⁻⁹ m)
- Mesoscale: micrometers (10⁻⁶ m)
- Macroscopic: millimeters (10⁻³ m)

**Hierarchical Binning:**
```python
def hierarchical_spatial_correlation(positions, levels=4):
    """
    Multi-scale spatial correlation
    
    Coarse-grain to larger length scales
    """
    correlations = {}
    
    for level in range(levels):
        grid_size = 2**level
        
        # Bin particles to grid
        grid = bin_to_grid(positions, grid_size)
        
        # Compute correlation at this scale
        C_k = np.abs(np.fft.fftn(grid))**2
        
        # Radial average
        k, C = radial_average_3d(C_k)
        correlations[level] = (k, C)
    
    return correlations
```

### Handling Heterogeneous Data

**Different Sampling Rates:**
- High-frequency instrument: 1 MHz
- Low-frequency readout: 1 Hz
- Combine datasets with different Δt

```python
def combine_multiscale_data(data_fast, dt_fast, data_slow, dt_slow):
    """
    Combine correlation from different sampling rates
    """
    # Fast timescale correlation
    C_fast = fft_autocorrelation(data_fast)
    t_fast = np.arange(len(C_fast)) * dt_fast
    
    # Slow timescale correlation
    C_slow = fft_autocorrelation(data_slow)
    t_slow = np.arange(len(C_slow)) * dt_slow
    
    # Stitch together at overlap
    overlap_idx = np.where(t_fast > t_slow[0])[0][0]
    
    t_combined = np.concatenate([t_fast[:overlap_idx], t_slow])
    C_combined = np.concatenate([C_fast[:overlap_idx], C_slow])
    
    return t_combined, C_combined
```

## Statistical Validation

### Bootstrap Resampling

**Purpose**: Estimate uncertainty in correlation functions

```python
def bootstrap_correlation(data, num_bootstrap=1000, block_size=None):
    """
    Bootstrap error estimation for correlations
    
    For correlated data, use block bootstrap
    """
    N = len(data)
    
    if block_size is None:
        # Estimate correlation length
        C = fft_autocorrelation(data)
        block_size = np.where(C < 1/np.e)[0][0] if len(np.where(C < 1/np.e)[0]) > 0 else N//10
    
    num_blocks = N // block_size
    
    C_bootstrap = []
    for _ in range(num_bootstrap):
        # Resample blocks
        blocks = np.random.choice(num_blocks, size=num_blocks, replace=True)
        resampled = np.concatenate([data[b*block_size:(b+1)*block_size] for b in blocks])
        
        # Compute correlation
        C = fft_autocorrelation(resampled)
        C_bootstrap.append(C)
    
    # Statistics
    C_mean = np.mean(C_bootstrap, axis=0)
    C_std = np.std(C_bootstrap, axis=0)
    C_95 = np.percentile(C_bootstrap, [2.5, 97.5], axis=0)
    
    return C_mean, C_std, C_95
```

### Convergence Analysis

**Check**: Does correlation converge with more data?

```python
def convergence_test(data, fractions=[0.25, 0.5, 0.75, 1.0]):
    """
    Test convergence of correlation with data length
    """
    N = len(data)
    
    correlations = {}
    for frac in fractions:
        n = int(N * frac)
        C = fft_autocorrelation(data[:n])
        correlations[frac] = C
    
    # Plot or compare
    return correlations

def estimate_required_length(C, target_error=0.01):
    """
    Estimate trajectory length needed for target error
    
    Error ~ 1/√(N/τ) where τ = correlation time
    """
    # Find correlation time
    tau = np.trapz(C)
    
    # Required independent samples
    N_indep = (1 / target_error)**2
    
    # Total length
    N_required = N_indep * tau
    
    return int(N_required)
```

### Bias Correction

**Finite-Size Effects:**
```python
def finite_size_correction(C, L, xi):
    """
    Correct correlation for finite system size
    
    True correlation: C_∞(r)
    Finite system: C_L(r) affected by periodic images
    """
    # Exponential decay assumption
    r = np.arange(len(C))
    
    # Sum over periodic images
    correction = 0
    for n in [-1, 0, 1]:
        for m in [-1, 0, 1]:
            for p in [-1, 0, 1]:
                if n == m == p == 0:
                    continue
                r_image = np.sqrt((r + n*L)**2 + (m*L)**2 + (p*L)**2)
                correction += np.exp(-r_image / xi)
    
    C_corrected = C - correction
    return C_corrected
```

### Outlier Detection

**Robust Statistics:**
```python
def robust_correlation(data, method='median'):
    """
    Robust correlation using median or trimmed mean
    
    Less sensitive to outliers than standard correlation
    """
    if method == 'median':
        # Median absolute deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Remove outliers (> 3 MAD)
        mask = np.abs(data - median) < 3*mad
        data_clean = data[mask]
        
    elif method == 'trimmed':
        # Trim 10% from each tail
        data_sorted = np.sort(data)
        n_trim = int(0.1 * len(data))
        data_clean = data_sorted[n_trim:-n_trim]
    
    C = fft_autocorrelation(data_clean)
    return C
```

## GPU Acceleration with JAX

### JAX-Optimized Correlation

```python
import jax
import jax.numpy as jnp

@jax.jit
def jax_autocorrelation(data):
    """
    JAX JIT-compiled autocorrelation
    
    Faster than NumPy for repeated calls
    """
    N = len(data)
    data_centered = data - jnp.mean(data)
    
    # Zero-pad
    padded = jnp.concatenate([data_centered, jnp.zeros(N)])
    
    # FFT
    F = jnp.fft.fft(padded)
    S = jnp.abs(F)**2
    C = jnp.fft.ifft(S).real[:N]
    
    # Normalize
    norm = N - jnp.arange(N)
    C = C / norm / C[0]
    
    return C

# Vectorize over batch dimension
jax_correlation_batch = jax.vmap(jax_autocorrelation)

# Example: Process multiple time series in parallel
data_batch = jnp.array([data1, data2, data3, ...])  # Shape: (batch, time)
C_batch = jax_correlation_batch(data_batch)  # Parallel on GPU
```

### Large Dataset Handling

**Chunked Processing:**
```python
def gpu_correlation_large(data, chunk_size=10**6):
    """
    Process large dataset in chunks on GPU
    
    Avoids OOM errors for massive datasets
    """
    N = len(data)
    num_chunks = (N + chunk_size - 1) // chunk_size
    
    C_chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, N)
        
        # Move chunk to GPU
        chunk = jnp.array(data[start:end])
        
        # Compute correlation
        C = jax_autocorrelation(chunk)
        
        # Move back to CPU
        C_chunks.append(np.array(C))
    
    # Weighted average
    weights = [len(c) for c in C_chunks]
    C_avg = np.average(C_chunks, axis=0, weights=weights)
    
    return C_avg
```

### Performance Benchmarks

**Speedup Examples:**
- CPU (NumPy): 1.0× baseline
- CPU (NumPy + FFT): 50× for N=10⁶
- GPU (JAX): 200× for N=10⁶, batch of 100
- Multi-GPU: 1000× for very large batches

## Best Practices

### Algorithm Selection
- **N < 1000**: Direct method acceptable
- **1000 < N < 10⁶**: FFT-based
- **N > 10⁶**: Block averaging + FFT
- **Wide dynamic range**: Multi-tau correlator

### Memory Management
- Streaming algorithms for real-time analysis
- Chunked processing for datasets > RAM
- Block averaging for long trajectories

### Validation
- Bootstrap (N=1000) for error bars
- Convergence test: Compare N/2 vs N
- Cross-validation: Train/test splits
- Physical constraints: Sum rules, positivity

### Performance Optimization
- JIT compilation (JAX, Numba)
- Vectorization over batch dimension
- GPU for large N or batch processing
- Parallel processing for independent samples

References for advanced methods: adaptive mesh refinement for spatial correlations, compressed sensing for sparse sampling, quantum algorithms for correlation functions.
