# JAX Installation Guide for Phase 4

**Purpose**: Install JAX for GPU acceleration and automatic differentiation in Phase 4

**Status**: Required for testing JAX PMP solver and GPU-accelerated features

---

## Quick Installation

### Option 1: CPU-Only (Simplest)

```bash
pip install jax jaxlib diffrax
```

This installs:
- `jax`: Core JAX library
- `jaxlib`: JAX backend
- `diffrax`: Modern ODE solver for JAX

**Use Case**: Development, testing, small problems

### Option 2: GPU Support (CUDA 12)

```bash
pip install jax[cuda12_pip] jaxlib diffrax
```

**Requirements**: NVIDIA GPU with CUDA 12 installed

**Use Case**: Production, large problems (n_dim > 10)

### Option 3: From Requirements File

```bash
pip install -r requirements-gpu.txt
```

Contents of `requirements-gpu.txt`:
```
jax[cuda12_pip]>=0.4.20
jaxlib>=0.4.20
diffrax>=0.4.1
```

---

## Installation Verification

### Test 1: Check JAX is Installed

```python
python3 -c "import jax; print(f'JAX version: {jax.__version__}')"
```

Expected output:
```
JAX version: 0.4.20 (or higher)
```

### Test 2: Check Available Devices

```python
python3 -c "import jax; print(f'Devices: {jax.devices()}')"
```

Expected output (CPU):
```
Devices: [CpuDevice(id=0)]
```

Expected output (GPU):
```
Devices: [cuda(id=0)]
```

### Test 3: Run Simple JAX Computation

```python
python3 -c "import jax.numpy as jnp; x = jnp.array([1, 2, 3]); print(f'Result: {jnp.sum(x)}')"
```

Expected output:
```
Result: 6
```

---

## Troubleshooting

### Issue 1: `externally-managed-environment` Error

**Error**:
```
error: externally-managed-environment
× This environment is externally managed
```

**Solutions**:

1. **Use Virtual Environment** (Recommended):
   ```bash
   python3 -m venv phase4-env
   source phase4-env/bin/activate  # On macOS/Linux
   pip install jax jaxlib diffrax
   ```

2. **User Installation**:
   ```bash
   pip install --user jax jaxlib diffrax
   ```

3. **Break System Packages** (Not Recommended):
   ```bash
   pip install --break-system-packages jax jaxlib diffrax
   ```

### Issue 2: JAX Not Found After Installation

**Problem**: JAX installed but import fails

**Solution**: Check Python path
```bash
python3 -c "import sys; print('\n'.join(sys.path))"
```

Ensure installation directory is in path.

### Issue 3: CUDA Version Mismatch

**Error**: `RuntimeError: CUDA version mismatch`

**Solution**: Match JAX CUDA version to system CUDA

```bash
# Check system CUDA version
nvcc --version

# Install matching JAX
# CUDA 11:
pip install jax[cuda11_pip]

# CUDA 12:
pip install jax[cuda12_pip]
```

### Issue 4: GPU Not Detected

**Problem**: JAX shows only CPU devices despite having GPU

**Diagnostic**:
```python
import jax
print(jax.devices())  # Should show cuda(id=0)
```

**Solutions**:

1. **Install GPU Version**:
   ```bash
   pip uninstall jax jaxlib
   pip install jax[cuda12_pip] jaxlib
   ```

2. **Check CUDA Installation**:
   ```bash
   nvidia-smi  # Should show GPU info
   ```

3. **Set CUDA Path**:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

---

## Platform-Specific Instructions

### macOS (Apple Silicon M1/M2/M3)

**Note**: No GPU support on Apple Silicon for JAX

```bash
# Install CPU-only version
pip install jax jaxlib diffrax
```

Performance is still good due to optimized CPU kernels.

### Linux (NVIDIA GPU)

```bash
# Install with CUDA 12 support
pip install jax[cuda12_pip] jaxlib diffrax

# Verify GPU
python3 -c "import jax; print(jax.devices())"
```

### Windows

```bash
# CPU-only (simplest)
pip install jax jaxlib diffrax

# For GPU, follow official JAX docs
# GPU support on Windows is experimental
```

---

## Running Phase 4 Tests After Installation

### Test JAX PMP Suite

```bash
# Navigate to project root
cd /path/to/nonequilibrium-physics-agents

# Run JAX PMP tests
python3 tests/solvers/test_pontryagin_jax.py

# Or with pytest
python3 -m pytest tests/solvers/test_pontryagin_jax.py -v
```

### Test GPU Kernels

```bash
# Run GPU quantum evolution tests
python3 -m pytest tests/gpu/test_quantum_gpu.py -v
```

### Run All Phase 4 Tests

```bash
# Run complete Phase 4 test suite
python3 run_phase4_tests.py
```

---

## Performance Expectations

### CPU Performance

| Task | Expected Time |
|------|---------------|
| Small LQR (n=2) | < 1 sec |
| Large LQR (n=10) | < 5 sec |
| Quantum control (n=2) | 1-2 sec |
| JAX PMP test suite | 30-60 sec |

### GPU Performance (If Available)

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| n_dim=10, 100 steps | 32 sec | 1 sec | 31x |
| n_dim=20, 50 steps | 180 sec | 6 sec | 30x |
| Batch 100, n_dim=4 | 80 sec | 0.8 sec | 100x |

---

## Minimal Test Script

Save as `test_jax_install.py`:

```python
"""Quick JAX installation test."""
import sys

def test_jax_installation():
    """Test JAX is properly installed."""
    try:
        import jax
        import jax.numpy as jnp
        import diffrax

        print(f"✅ JAX version: {jax.__version__}")
        print(f"✅ Devices: {jax.devices()}")

        # Simple computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print(f"✅ Simple computation: sum([1,2,3]) = {y}")

        # Test JIT
        @jax.jit
        def square(x):
            return x ** 2

        result = square(jnp.array(2.0))
        print(f"✅ JIT compilation: 2^2 = {result}")

        # Test grad
        def f(x):
            return x ** 2

        grad_f = jax.grad(f)
        gradient = grad_f(3.0)
        print(f"✅ Autodiff: d/dx(x^2) at x=3 = {gradient}")

        print("\n🎉 JAX installation successful!")
        print("✅ Ready to run Phase 4 JAX PMP tests")
        return True

    except ImportError as e:
        print(f"❌ JAX not installed: {e}")
        print("Install with: pip install jax jaxlib diffrax")
        return False

if __name__ == "__main__":
    success = test_jax_installation()
    sys.exit(0 if success else 1)
```

Run:
```bash
python3 test_jax_install.py
```

Expected output:
```
✅ JAX version: 0.4.20
✅ Devices: [CpuDevice(id=0)]
✅ Simple computation: sum([1,2,3]) = 6.0
✅ JIT compilation: 2^2 = 4.0
✅ Autodiff: d/dx(x^2) at x=3 = 6.0

🎉 JAX installation successful!
✅ Ready to run Phase 4 JAX PMP tests
```

---

## Advanced Configuration

### JIT Compilation Settings

```python
import jax

# Disable JIT (for debugging)
jax.config.update("jax_disable_jit", True)

# Enable 64-bit precision (default is 32-bit)
jax.config.update("jax_enable_x64", True)
```

### Memory Management

```python
# Limit GPU memory allocation
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use 50% of GPU
```

### Backend Selection

```python
# Force CPU backend
jax.config.update("jax_platform_name", "cpu")

# Use GPU if available, else CPU
# (This is the default behavior)
```

---

## Phase 4 Integration

Once JAX is installed, Phase 4 features automatically use GPU acceleration:

### GPU-Accelerated Quantum Evolution

```python
from gpu_kernels.quantum_evolution import solve_lindblad

# Automatic GPU/CPU selection
result = solve_lindblad(rho0, H, L_ops, gammas, t_span, backend='auto')
```

### JAX-Accelerated Optimal Control

```python
from solvers.pontryagin_jax import PontryaginSolverJAX

solver = PontryaginSolverJAX(state_dim, control_dim, dynamics, cost)
result = solver.solve(x0, xf, duration, n_steps, backend='cpu')
```

### Magnus Solver with GPU

```python
from solvers.magnus_expansion import MagnusExpansionSolver

solver = MagnusExpansionSolver(order=4)
psi = solver.solve_unitary(psi0, H_protocol, t_span)
```

---

## Resources

### Official Documentation

- JAX: https://jax.readthedocs.io
- JAX Installation: https://github.com/google/jax#installation
- Diffrax: https://docs.kidger.site/diffrax/

### Phase 4 Documentation

- `PHASE4_COMPLETE_README.md` - Complete usage guide
- `PHASE4_QUICK_REFERENCE.md` - API cheat sheet
- `NEXT_STEPS.md` - Development roadmap

---

## Frequently Asked Questions

**Q: Do I need a GPU to use Phase 4?**
A: No. Phase 4 works on CPU with automatic fallback. GPU provides 30-50x speedup for large problems.

**Q: Which JAX version should I use?**
A: JAX >= 0.4.20 recommended. Check `requirements-gpu.txt` for exact versions.

**Q: Can I use JAX on Apple Silicon (M1/M2/M3)?**
A: Yes, CPU-only. Install with `pip install jax jaxlib diffrax`. No GPU support yet.

**Q: Will Phase 4 work without JAX?**
A: Partially. GPU features require JAX, but CPU features (scipy-based) work without it.

**Q: How do I verify GPU is being used?**
A: Check `jax.devices()` shows `cuda(id=0)`, and verify speedup > 10x for n_dim=10 problems.

---

**Status**: Ready for JAX installation
**Next Step**: Install JAX and run `python3 test_jax_install.py`
**After Installation**: Run Phase 4 JAX PMP tests

---

*For full Phase 4 documentation, see `PHASE4_COMPLETE_README.md`*
