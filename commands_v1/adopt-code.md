---
title: "Adopt Code"
description: "Analyze, integrate, and optimize scientific computing codebases for modern frameworks"
category: scientific-computing
subcategory: legacy-modernization
complexity: expert
argument-hint: "[--analyze] [--integrate] [--optimize] [--language=fortran|c|cpp|python|julia|mixed] [--target=python|jax|julia] [--parallel=mpi|openmp|cuda|jax] [--agents=scientific|quality|orchestrator|all] [codebase-path]"
allowed-tools: "*"
model: inherit
tags: scientific-computing, legacy-code, optimization, integration, numerical-accuracy
dependencies: []
related: [optimize, jax-essentials, julia-jit-like, debug, check-code-quality, refactor-clean, multi-agent-optimize, generate-tests, run-all-tests]
workflows: [legacy-modernization, scientific-migration, performance-optimization]
version: "2.0"
last-updated: "2025-09-28"
---

# Scientific Computing Code Adoption Tool

Transform legacy scientific computing codebases into modern, optimized frameworks while preserving numerical accuracy and computational efficiency.

## Overview

The adopt-code tool bridges the gap between legacy scientific computing code and modern high-performance frameworks. It provides comprehensive analysis, seamless integration, and performance optimization for scientific codebases across multiple languages and target platforms.

## Quick Start

### Essential Workflows
```bash
# 1. Analyze legacy codebase structure and dependencies
/adopt-code ./legacy_solver --analyze

# 2. Integrate Fortran code with Python ecosystem
/adopt-code ./fortran_code --integrate --language=fortran --target=python

# 3. Optimize existing code for GPU acceleration
/adopt-code ./molecular_dynamics --optimize --parallel=cuda

# 4. Complete modernization pipeline
/adopt-code ./scientific_code --analyze --integrate --optimize --target=jax
```

### Domain-Specific Examples
```bash
# Climate modeling: Legacy Fortran → Python with MPI
/adopt-code ./climate_model --integrate --language=fortran --target=python --parallel=mpi

# High-performance computing: CUDA optimization
/adopt-code ./hpc_solver --optimize --parallel=cuda

# Machine learning: Mixed languages → JAX ecosystem
/adopt-code ./mixed_codebase --language=mixed --integrate --target=jax

# Quantum chemistry: C++ → Julia with performance focus
/adopt-code ./quantum_solver --integrate --language=cpp --target=julia --optimize
```

### Argument Reference

| Argument | Description | Options |
|----------|-------------|---------|
| `codebase-path` | Path to source codebase (moved to end) | Directory or file path |
| `--analyze` | Perform comprehensive code analysis | Flag |
| `--integrate` | Enable cross-language integration | Flag |
| `--optimize` | Apply performance optimizations | Flag |
| `--language` | Source language specification | `fortran`, `c`, `cpp`, `python`, `julia`, `mixed` |
| `--target` | Target framework/language | `python`, `jax`, `julia` |
| `--parallel` | Parallelization strategy | `mpi`, `openmp`, `cuda`, `jax` |

## Core Functionality

Transform scientific computing codebases while maintaining numerical accuracy and computational efficiency.

### Primary Functions
- **Code Analysis**: Understand scientific algorithms, data structures, and dependencies
- **Integration**: Bridge legacy code with modern frameworks (Python, JAX, Julia)
- **Performance Optimization**: Improve speed and efficiency while preserving precision
- **Cross-Language Support**: Handle Fortran, C, C++, Python, Julia codebases

### Scientific Domain Support
- **Numerical Methods**: Linear algebra, differential equations, optimization, FFTs, Monte Carlo
- **Modern Libraries**: JAX/Flax, Julia SciML, NumPy, SciPy, PyTorch
- **Legacy Libraries**: BLAS/LAPACK, PETSc, FFTW, MPI, OpenMP, CUDA
- **Applications**: Physics simulations, climate modeling, computational chemistry, bioinformatics

## Analysis Framework

### Step 1: Initial Assessment
**Codebase Structure Analysis**
- Identify core algorithms and performance hotspots
- Map data flow patterns and dependencies
- Catalog external libraries and requirements
- Assess parallelization strategies
- Document mathematical foundations

**Integration Feasibility**
```python
class CodebaseAnalyzer:
    def __init__(self, codebase_path: str):
        self.path = codebase_path
        self.languages = self._detect_languages()
        self.dependencies = self._catalog_dependencies()
        self.kernels = self._identify_computational_kernels()

    def analyze_patterns(self):
        return {
            'numerical_methods': self._analyze_algorithms(),
            'data_structures': self._analyze_memory_patterns(),
            'parallelization': self._analyze_parallel_constructs(),
            'optimization_potential': self._assess_bottlenecks()
        }
```

### Step 2: Integration Planning
**Compatibility Assessment**
- Language interoperability analysis
- Numerical precision requirements
- Memory model alignment
- Performance bottleneck identification

**Risk Analysis**
- Numerical instability risks
- Race conditions and synchronization issues
- Platform dependencies
- Testing coverage gaps

### Step 3: Implementation Strategy
**Modular Integration Pattern**
```python
class ScientificAdapter:
    """Wrapper for integrating legacy scientific code."""

    def __init__(self, config: dict):
        self.validate_dependencies()
        self.initialize_backends()
        self.setup_error_handling()

    def validate_inputs(self, data: np.ndarray) -> dict:
        """Input validation with scientific constraints."""
        return {
            'dimension_check': self._check_dimensions(data),
            'conservation_laws': self._verify_conservation(data),
            'numerical_stability': self._check_stability(data)
        }

    def execute_computation(self, inputs: dict) -> dict:
        """Execute with safety monitoring."""
        prepared_data = self._marshal_data(inputs)

        with self._performance_monitor() as monitor:
            results = self._call_kernel(prepared_data)

        self._verify_stability(results)
        return {
            'results': self._unmarshal_results(results),
            'metrics': monitor.get_metrics()
        }
```

## Performance Optimization

### Memory Optimization
```fortran
! Cache-aware algorithm optimization
module optimization
  use iso_fortran_env
  implicit none

  integer, parameter :: BLOCK_SIZE_X = 64
  integer, parameter :: BLOCK_SIZE_Y = 32

contains

subroutine compute_stencil_optimized(data, result, nx, ny, nz)
  implicit none
  integer, intent(in) :: nx, ny, nz
  real(real64), intent(in) :: data(nx, ny, nz)
  real(real64), intent(out) :: result(nx, ny, nz)

  integer :: i, j, k, ib, jb

  !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(i,j,k,ib,jb)
  do jb = 1, ny, BLOCK_SIZE_Y
    do ib = 1, nx, BLOCK_SIZE_X
      do k = 1, nz
        do j = jb, min(jb+BLOCK_SIZE_Y-1, ny)
          !$OMP SIMD
          do i = ib, min(ib+BLOCK_SIZE_X-1, nx)
            result(i,j,k) = stencil_kernel(data, i, j, k, nx, ny, nz)
          end do
        end do
      end do
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine
end module
```

### Optimization Strategies
- **Vectorization**: SIMD operations and compiler intrinsics
- **Cache Optimization**: Blocking techniques for cache efficiency
- **Linear Algebra**: Optimized BLAS/LAPACK integration
- **Algorithmic Improvements**: Better complexity algorithms
- **GPU Acceleration**: CUDA/OpenCL/JAX kernel porting

### Parallel Optimization
- **Load Balancing**: Optimal workload distribution
- **Communication**: MPI overhead reduction
- **Overlap**: Computation/communication latency hiding
- **Memory**: NUMA architecture optimization

## Quality Assurance

### Testing Framework
```python
class ScientificTestSuite:
    """Testing framework for scientific code validation."""

    def test_numerical_accuracy(self, original_func, optimized_func, test_cases):
        """Verify numerical accuracy preservation."""
        for case in test_cases:
            tolerance = self._compute_tolerance(case)
            original_result = original_func(case)
            optimized_result = optimized_func(case)

            relative_error = np.abs((original_result - optimized_result) / original_result)
            assert np.all(relative_error < tolerance), f"Accuracy violation: {relative_error.max()}"

    def test_conservation_laws(self, func, initial_state, steps=1000):
        """Verify conservation law preservation."""
        state = initial_state.copy()
        initial_quantities = self._compute_conserved_quantities(state)

        for step in range(steps):
            state = func(state)
            if step % 100 == 0:
                current_quantities = self._compute_conserved_quantities(state)
                drift = np.abs((current_quantities - initial_quantities) / initial_quantities)
                assert np.all(drift < 1e-10), f"Conservation violation at step {step}"

    def test_convergence_rates(self, solver, analytical_solution, resolutions):
        """Verify convergence rate expectations."""
        errors = []
        for resolution in resolutions:
            numerical = solver(resolution)
            analytical = analytical_solution(resolution)
            error = np.linalg.norm(numerical - analytical)
            errors.append(error)

        # Check convergence rate
        rates = np.log(errors[:-1] / errors[1:]) / np.log(2)
        assert np.mean(rates) >= self.expected_rate - 0.1, f"Poor convergence: {np.mean(rates)}"
```

### Validation Checklist
- [ ] **Numerical Accuracy**: Results within acceptable tolerance
- [ ] **Performance**: Meets speed/memory targets
- [ ] **Scalability**: Proper parallel scaling
- [ ] **Reproducibility**: Deterministic results when required
- [ ] **Error Handling**: Robust error detection and recovery
- [ ] **Documentation**: Complete API and usage documentation

## Integration Examples

### Legacy Fortran Climate Model
```python
class AtmosphericSolverAdapter:
    """Python interface for legacy Fortran climate code."""

    def __init__(self, grid_config: dict):
        self.nx, self.ny, self.nz = grid_config['dimensions']
        self.dt = grid_config['timestep']

        # Compile Fortran module
        self._compile_fortran_module()

        # Import and initialize
        import atmospheric_solver_f90
        self.f90_solver = atmospheric_solver_f90
        self.f90_solver.init_solver(self.nx, self.ny, self.nz)

        # Allocate arrays
        self._allocate_arrays()

    def advance_timestep(self, boundary_conditions: dict) -> dict:
        """Execute one timestep with error handling."""
        self._validate_boundary_conditions(boundary_conditions)

        try:
            self.f90_solver.advance_timestep(
                self.u, self.v, self.w, self.pressure,
                self.dt, boundary_conditions
            )
        except Exception as e:
            raise RuntimeError(f"Timestep failed: {str(e)}")

        # Verify stability
        if not self._check_stability():
            raise ValueError("Numerical instability detected")

        return self._package_results()
```

### GPU Molecular Dynamics
```cuda
// CUDA kernel for molecular dynamics force calculation
__global__ void compute_forces_optimized(
    double3* __restrict__ positions,
    double3* __restrict__ forces,
    int* __restrict__ neighbor_list,
    int n_atoms,
    double cutoff_sq) {

    int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_i >= n_atoms) return;

    double3 pos_i = positions[atom_i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    // Process neighbors
    for (int j = 0; j < MAX_NEIGHBORS; j++) {
        int atom_j = neighbor_list[atom_i * MAX_NEIGHBORS + j];
        if (atom_j == -1) break;

        double3 pos_j = positions[atom_j];
        double dx = pos_i.x - pos_j.x;
        double dy = pos_i.y - pos_j.y;
        double dz = pos_i.z - pos_j.z;
        double r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < cutoff_sq && r2 > 0.01) {
            double r2_inv = 1.0 / r2;
            double r6_inv = r2_inv * r2_inv * r2_inv;
            double force_mag = 24.0 * r2_inv * (2.0 * r6_inv * r6_inv - r6_inv);

            fx += force_mag * dx;
            fy += force_mag * dy;
            fz += force_mag * dz;
        }
    }

    forces[atom_i] = make_double3(fx, fy, fz);
}
```

## Usage Examples

### Basic Operations
```bash
# Analyze codebase structure
/adopt-code ./legacy_solver --analyze

# Integrate with Python
/adopt-code ./fortran_code --integrate --language=fortran --target=python

# Optimize for performance
/adopt-code ./solver --optimize --parallel=cuda

# Complete workflow
/adopt-code ./scientific_code --analyze --integrate --optimize --target=jax
```

### Advanced Integration
```bash
# Climate modeling pipeline
/adopt-code ./climate_model --integrate --language=fortran --target=python --parallel=mpi

# Mixed language optimization
/adopt-code ./mixed_codebase --language=mixed --integrate --target=jax

# High-performance computing
/adopt-code ./hpc_solver --optimize --parallel=cuda
```

## Best Practices

### Integration Guidelines
1. **Scientific Accuracy**: Never sacrifice correctness for performance
2. **Reproducibility**: Ensure deterministic results when required
3. **Documentation**: Document physical/mathematical assumptions
4. **Version Control**: Track modifications systematically
5. **Benchmarking**: Profile before and after changes

### Common Issues to Avoid
- **Precision Loss**: Monitor floating-point precision degradation
- **Invariant Violations**: Preserve domain-specific constraints
- **Boundary Conditions**: Verify correct implementation
- **Error Propagation**: Handle numerical errors appropriately
- **Platform Dependencies**: Ensure cross-platform compatibility

### Performance Targets
- **Molecular Dynamics**: 2-5x speedup, 30-50% memory reduction
- **Climate Modeling**: 3-8x compute speedup, 10-20x I/O improvement
- **Quantum Chemistry**: 5-15x integral computation, 50-80% faster convergence

## Common Workflows

### Legacy Fortran to Modern Python
```bash
# 1. Analyze legacy Fortran codebase
/adopt-code ./legacy_fortran --analyze --language=fortran

# 2. Integrate with Python ecosystem
/adopt-code ./legacy_fortran --integrate --language=fortran --target=python

# 3. Optimize performance
/optimize ./python_integration --language=python --implement

# 4. Validate numerical accuracy
/generate-tests ./python_integration --type=scientific --coverage=95
/run-all-tests --scientific --reproducible
```

### Scientific Computing Migration to JAX
```bash
# 1. Analyze mixed-language scientific code
/adopt-code ./mixed_science_code --analyze --language=mixed

# 2. Migrate to JAX ecosystem
/adopt-code ./mixed_science_code --integrate --target=jax --parallel=cuda

# 3. JAX-specific optimization
/jax-essentials --operation=jit --static-args
/jax-performance --technique=caching --gpu-accel

# 4. Test scientific accuracy
/run-all-tests --scientific --gpu --reproducible
```

### High-Performance Computing Modernization
```bash
# 1. Analyze C++ HPC code
/adopt-code ./hpc_solver --analyze --language=cpp

# 2. Optimize for modern hardware
/adopt-code ./hpc_solver --optimize --parallel=cuda

# 3. Performance validation
/debug --gpu --profile --monitor
/optimize --category=memory --implement

# 4. Quality assurance
/check-code-quality --analysis=gpu --auto-fix
/double-check "HPC modernization results" --deep-analysis
```

## Related Commands

**Prerequisites**: Commands to run before code adoption
- `/check-code-quality --analysis=scientific` - Assess legacy code quality
- `/debug --auto-fix` - Fix runtime issues in legacy code
- `/explain-code` - Understand legacy codebase structure
- Version control - Backup original codebase before migration

**Core Integration**: JAX and Julia ecosystem commands
- `/jax-essentials` - JAX operations and transformations after migration
- `/jax-performance --gpu-accel` - JAX-specific GPU performance optimization
- `/julia-jit-like --type-stability` - Julia performance and type optimization
- `/julia-prob-model` - Julia probabilistic modeling integration

**Optimization**: Performance improvement commands
- `/optimize --implement` - Apply performance optimizations after adoption
- `/multi-agent-optimize --focus=performance` - Complex optimization analysis
- `/python-debug-prof --suggest-opts` - Python performance profiling
- `/refactor-clean --patterns=modern` - Modernization and cleanup

**Validation**: Testing and quality assurance
- `/generate-tests --type=scientific` - Scientific computing test generation
- `/run-all-tests --scientific --reproducible` - Comprehensive scientific validation
- `/double-check --deep-analysis` - Verify adoption results systematically
- `/reflection --type=scientific` - Analyze adoption process effectiveness

## Integration Patterns

### Scientific Computing Migration Pipeline
```bash
# Complete scientific codebase modernization
/adopt-code legacy/ --analyze --language=fortran
/adopt-code legacy/ --integrate --target=jax --parallel=cuda
/jax-performance --gpu-accel --optimization
/run-all-tests --scientific --gpu --reproducible
```

### Performance-First Adoption
```bash
# Legacy code with performance focus
/adopt-code old_solver/ --analyze --optimize
/optimize modern_solver/ --language=julia --implement
/julia-jit-like modern_solver/ --type-stability --precompile
```

### Quality-Assured Migration
```bash
# Migration with comprehensive quality checks
/check-code-quality legacy/ --analysis=scientific
/adopt-code legacy/ --integrate --target=python
/generate-tests modern/ --type=scientific --coverage=95
/double-check "migration results" --deep-analysis --auto-complete
```