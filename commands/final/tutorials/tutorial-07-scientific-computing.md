# Tutorial 07: Scientific Computing

**Duration**: 90 minutes | **Level**: Advanced

---

## Learning Objectives

- Master Python/Julia/JAX workflows
- Optimize GPU code
- Adopt legacy Fortran/C code
- Ensure numerical accuracy
- Create reproducible research

---

## Part 1: Python/Julia/JAX Support (20 minutes)

### Python NumPy Optimization
```bash
# Analyze scientific Python code
/optimize --language=python --scientific simulation.py

# AI automatically:
# - Vectorizes loops
# - Uses NumPy broadcasting
# - Optimizes memory layout
# - Enables parallel processing
```

### Julia Performance
```bash
# Optimize Julia code
/optimize --language=julia --scientific solver.jl

# Optimizations:
# - Type stability improvements
# - Multiple dispatch optimization
# - SIMD vectorization
# - Memory allocation reduction
```

### JAX for GPU
```bash
# Optimize JAX GPU code
/optimize --language=jax --gpu model.py

# Transformations:
# - Efficient use of vmap/pmap
# - XLA compilation optimization
# - Device memory management
# - Gradient computation efficiency
```

---

## Part 2: GPU Debugging and Optimization (25 minutes)

### Debug GPU Memory Issues
```bash
# Profile GPU usage
/debug --gpu --profile --monitor simulation.py

# Output:
# 🎮 GPU Analysis
# ══════════════════════════════════════════════════
# Device: NVIDIA A100 (40GB)
# Memory Usage: 38.5GB / 40GB (96% - WARNING!)
#
# Memory Leaks Detected:
#   - Line 45: Tensor not freed (2.3GB leak)
#   - Line 67: Gradient accumulation (5.1GB leak)
#
# Optimization Opportunities:
#   - Use gradient checkpointing (save 15GB)
#   - Batch size reduction (save 8GB)
#   - Mixed precision (save 12GB)
```

### Optimize GPU Code
```bash
# Apply GPU optimizations
/optimize --implement --gpu --category=memory simulation.py

# Optimizations applied:
# ✅ Gradient checkpointing enabled
# ✅ Mixed precision (FP16) enabled
# ✅ Memory-efficient attention
# ✅ Batch size auto-tuning

# Result: 38.5GB → 18.2GB (53% reduction)
```

---

## Part 3: Legacy Code Adoption (25 minutes)

### Adopt Fortran to Python
```bash
# Analyze legacy Fortran code
/adopt-code legacy_solver.f90 --target=python --parallel=jax

# AI translation:
# - Fortran DO loops → NumPy/JAX operations
# - COMMON blocks → Python classes
# - Subroutines → Functions
# - F77/F90 → Modern Python

# Generated: solver.py (with JAX acceleration)
```

### Adopt C to Julia
```bash
# Migrate C code to Julia
/adopt-code physics_sim.c --target=julia --optimize

# Translation with optimizations:
# - C pointers → Julia arrays
# - malloc/free → Automatic memory management
# - Loops → Julia @simd/@threads
# - MPI → Julia parallel primitives
```

### Validation
```bash
# Verify numerical equivalence
/run-all-tests --scientific --reproducible \
  --compare-with=legacy_output.dat

# Output:
# Numerical Validation:
# ✅ Results match within tolerance (1e-12)
# ✅ Conservation laws preserved
# ✅ Physical constraints satisfied
# ✅ Statistical properties identical
```

---

## Part 4: Reproducible Research (20 minutes)

### Setup Reproducibility
```bash
# Create reproducible environment
/generate-tests --type=scientific --reproducible research_code/

# Generates:
# - requirements.txt (pinned versions)
# - environment.yml (Conda)
# - Dockerfile (containerization)
# - seed_manager.py (RNG control)
```

### Document Computational Methods
```bash
# Generate methods documentation
/update-docs --type=research --format=latex research_code/

# Creates:
# - methods.tex (LaTeX methods section)
# - computational_details.tex
# - algorithm_descriptions.tex
# - performance_benchmarks.tex
```

---

## Part 5: HPC and Parallel Computing (10 minutes)

### MPI Parallelization
```bash
# Optimize for MPI
/optimize --parallel=mpi --distributed simulation.py

# Adds:
# - MPI communication optimization
# - Domain decomposition
# - Load balancing
# - Collective operations
```

### Multi-GPU Scaling
```bash
# Scale to multiple GPUs
/optimize --gpu --distributed --agents=scientific model.py

# Optimizations:
# - Data parallelism across GPUs
# - Model parallelism for large models
# - Pipeline parallelism
# - Gradient synchronization
```

---

## Practice Projects

**Project 1**: Modernize Legacy Fortran Physics Code
- Input: 5000-line F77 simulation
- Output: Modern Python/JAX with 10x speedup
- Time: 30 minutes

**Project 2**: Optimize Deep Learning Training
- Input: Slow PyTorch training (2 days/epoch)
- Output: JAX training (4 hours/epoch)
- Time: 45 minutes

**Project 3**: Create Reproducible Research Package
- Package complete research codebase
- Generate publication-ready documentation
- Setup CI/CD for validation
- Time: 30 minutes

---

## Summary

✅ Multi-language scientific computing
✅ GPU optimization mastery
✅ Legacy code adoption
✅ Reproducible research setup
✅ HPC parallelization

**Next**: [Tutorial 08: Enterprise →](tutorial-08-enterprise.md)