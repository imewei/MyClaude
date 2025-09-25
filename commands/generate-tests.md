---
description: Generate comprehensive test suites and test cases for Python and Julia scientific computing projects
category: code-analysis-testing
argument-hint: [target-file-or-module] [--type=all|unit|integration|performance|cases] [--framework=auto|pytest|julia]
allowed-tools: Read, Write, Edit, Grep, Glob, TodoWrite, Bash
---

# Advanced Test Suite & Test Case Generator

Generate production-ready, executable test suites and comprehensive test cases for Python and Julia scientific computing projects with emphasis on numerical accuracy, performance validation, statistical rigor, and research reproducibility.

## Usage

```bash
# Generate complete test suite for file/module
/generate-tests homodyne/analysis/core.py

# Generate specific test types
/generate-tests src/optimization.py --type=performance
/generate-tests homodyne/mcmc.py --type=cases          # Focus on test case generation

# Generate Julia package test suite
/generate-tests MyPackage.jl --framework=julia

# Generate tests with interactive planning and TodoWrite tracking
/generate-tests homodyne/analysis/ --interactive

# Generate tests matching existing project patterns
/generate-tests --auto-detect

# Generate comprehensive test architecture
/generate-tests --type=all --coverage=95 --benchmark

# Generate test cases for specific functions
/generate-tests homodyne/optimization/mcmc.py::MCMCSampler --type=cases
```

## Comprehensive Test Generation Workflow

### 1. **Intelligent Project Analysis & Context Detection**

#### **Multi-Language Framework Detection**
- **Python Projects**: Automatically identify pytest, unittest, nose2, or custom frameworks
- **Julia Projects**: Detect Test.jl, PkgBenchmark.jl, Documenter.jl testing patterns
- **Mixed Environments**: Handle polyglot scientific computing projects
- **Configuration Parsing**: Analyze pytest.ini, pyproject.toml, Project.toml, tox.ini

#### **Scientific Domain Recognition & Context**
- **Numerical Computing**: NumPy arrays, SciPy functions, mathematical algorithms
- **Statistical Analysis**: MCMC sampling, hypothesis testing, parameter estimation
- **Data Science**: Pandas dataframes, scikit-learn models, visualization
- **High-Performance Computing**: JAX, CuPy, Dask, parallel computing patterns
- **Research Workflows**: Jupyter notebooks, experimental data processing

#### **Target Code Deep Analysis**
- **Function Signature Analysis**: Parameters, types, return values, exceptions, docstrings
- **Algorithm Classification**: Identify numerical methods, optimization algorithms, statistical procedures
- **Computational Characteristics**: Complexity analysis, performance bottlenecks, memory usage
- **Scientific Constraints**: Physical parameters, units, mathematical bounds
- **Dependency Mapping**: External libraries, hardware requirements, API integrations

### 2. **Advanced Test Case Strategy Design**

#### **Scientific Test Case Categories**

##### **Numerical Accuracy & Stability Tests**
```python
# Floating-point precision validation
def test_algorithm_numerical_stability():
    """Test numerical stability across different input ranges."""
    test_cases = [
        (np.array([1e-12, 1e-6, 1e-3]), "very small values"),
        (np.array([1e3, 1e6, 1e9]), "very large values"),
        (np.ones(100) * np.finfo(float).eps, "machine epsilon"),
    ]
    
    for data, description in test_cases:
        result = numerical_algorithm(data)
        assert np.all(np.isfinite(result)), f"Non-finite result for {description}"
        assert not np.any(np.isnan(result)), f"NaN result for {description}"

# Algorithm convergence validation
@pytest.mark.parametrize("tolerance", [1e-6, 1e-9, 1e-12])
def test_iterative_algorithm_convergence(tolerance):
    """Test iterative algorithm convergence at different tolerances."""
    result = iterative_solver(test_data, tolerance=tolerance)
    assert result.converged, f"Failed to converge at tolerance {tolerance}"
    assert result.error < tolerance, f"Error {result.error} exceeds tolerance {tolerance}"
```

##### **Statistical Rigor & Validation Tests**
```python
# Random seed reproducibility
def test_stochastic_reproducibility():
    """Ensure stochastic computations are reproducible."""
    np.random.seed(42)
    result1 = monte_carlo_simulation(n_samples=1000)
    
    np.random.seed(42)  
    result2 = monte_carlo_simulation(n_samples=1000)
    
    np.testing.assert_array_equal(result1, result2)

# Statistical significance validation
def test_hypothesis_test_statistical_power():
    """Test statistical hypothesis testing power and validity."""
    # Generate data with known effect size
    effect_size = 0.5
    data1, data2 = generate_test_data_with_effect(effect_size, n=100)
    
    p_value, test_statistic = statistical_test(data1, data2)
    
    assert p_value < 0.05, "Failed to detect significant effect"
    assert abs(test_statistic) > 1.96, "Test statistic below critical value"

# MCMC convergence diagnostics
@pytest.mark.mcmc
def test_mcmc_convergence_diagnostics():
    """Test MCMC chain convergence using multiple diagnostics."""
    chains = run_mcmc_chains(n_chains=4, n_samples=1000, target_accept=0.8)
    
    # R-hat convergence diagnostic
    r_hat = compute_r_hat(chains)
    assert np.all(r_hat < 1.1), f"Poor convergence: R-hat = {r_hat}"
    
    # Effective sample size
    ess = compute_effective_sample_size(chains)
    assert np.all(ess > 100), f"Insufficient effective samples: {ess}"
```

##### **Performance & Scalability Test Cases**
```python
# Algorithmic complexity validation
@pytest.mark.performance
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_algorithm_complexity(size, benchmark):
    """Validate algorithm scales as expected."""
    data = generate_test_data(size=size)
    
    result = benchmark(target_algorithm, data)
    
    # Verify linear scaling (adjust for actual complexity)
    expected_time = BASELINE_TIME * (size / BASELINE_SIZE)
    assert result.stats.median < expected_time * 2.0

# Memory efficiency validation
def test_memory_usage_scaling():
    """Test memory usage scales appropriately with input size."""
    sizes = [100, 1000, 10000]
    memory_usage = []
    
    for size in sizes:
        tracemalloc.start()
        data = generate_test_data(size=size)
        result = memory_intensive_algorithm(data)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage.append(peak)
    
    # Memory should scale roughly linearly
    scaling_factors = [m2/m1 for m1, m2 in zip(memory_usage[:-1], memory_usage[1:])]
    assert all(2 < factor < 20 for factor in scaling_factors)
```

### 3. **Python Scientific Computing Test Generation**

#### **NumPy/SciPy Integration Tests**
```python
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import kstest, normaltest

class TestScientificComputing:
    """Test scientific computing functions with NumPy/SciPy integration."""
    
    @pytest.fixture(scope="class")
    def test_datasets(self):
        """Generate comprehensive test datasets."""
        np.random.seed(42)
        return {
            "gaussian": np.random.normal(0, 1, 1000),
            "exponential": np.random.exponential(2.0, 1000),
            "uniform": np.random.uniform(-1, 1, 1000),
            "sparse": sp.sparse.random(100, 100, density=0.1),
            "structured": np.rec.fromarrays([
                np.arange(100),
                np.random.normal(0, 1, 100)
            ], names=['index', 'value'])
        }
    
    def test_array_operations_preserve_shape(self, test_datasets):
        """Test that array operations preserve expected shapes."""
        for name, data in test_datasets.items():
            if isinstance(data, np.ndarray):
                result = scientific_function(data)
                assert result.shape == data.shape, f"Shape mismatch for {name}"
    
    def test_statistical_properties(self, test_datasets):
        """Validate statistical properties are preserved."""
        gaussian_data = test_datasets["gaussian"]
        result = normalize_data(gaussian_data)
        
        # Test normality is preserved
        stat, p_value = normaltest(result)
        assert p_value > 0.05, "Normality not preserved after transformation"
        
        # Test mean and variance
        assert abs(np.mean(result)) < 0.1, "Mean significantly different from 0"
        assert abs(np.std(result) - 1.0) < 0.1, "Standard deviation not normalized"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
    def test_dtype_preservation(self, dtype):
        """Test functions handle different data types correctly."""
        data = np.array([1.0, 2.0, 3.0], dtype=dtype)
        result = scientific_function(data)
        
        # Check dtype is preserved or appropriately promoted
        if np.issubdtype(dtype, np.floating):
            assert np.issubdtype(result.dtype, np.floating)
        elif np.issubdtype(dtype, np.complexfloating):
            assert np.issubdtype(result.dtype, np.complexfloating)
```

#### **Performance Benchmarking & Regression Tests**
```python
class TestPerformanceBaselines:
    """Performance regression testing with baseline management."""
    
    @pytest.fixture(scope="class")
    def performance_baseline(self):
        """Load or create performance baselines."""
        baseline_file = Path(__file__).parent / "performance_baselines.json"
        
        if baseline_file.exists():
            with open(baseline_file) as f:
                return json.load(f)
        else:
            return {}
    
    def test_algorithm_performance_regression(self, benchmark, performance_baseline):
        """Test for performance regression against baseline."""
        large_dataset = generate_performance_dataset(size=10000)
        
        result = benchmark(target_algorithm, large_dataset)
        
        baseline_key = "target_algorithm_large"
        if baseline_key in performance_baseline:
            baseline_time = performance_baseline[baseline_key]
            regression_threshold = baseline_time * 1.2  # 20% tolerance
            
            assert result.stats.median < regression_threshold, \
                f"Performance regression: {result.stats.median:.6f}s > {regression_threshold:.6f}s"
        
        # Update baseline if this is better performance
        if baseline_key not in performance_baseline or result.stats.median < performance_baseline[baseline_key]:
            performance_baseline[baseline_key] = result.stats.median
            
    @pytest.mark.memory
    def test_memory_usage_baseline(self):
        """Test memory usage against established baselines."""
        import tracemalloc
        
        tracemalloc.start()
        
        large_data = generate_memory_test_data(size=50000)
        result = memory_intensive_function(large_data)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be within reasonable bounds
        assert peak < 500e6, f"Memory usage too high: {peak/1e6:.1f} MB > 500 MB"
        assert current < 100e6, f"Memory not released: {current/1e6:.1f} MB > 100 MB"
```

### 4. **Julia Scientific Computing Test Generation**

#### **Type Stability & Performance Tests**
```julia
using Test, BenchmarkTools, JET
import MyPackage: scientific_function, optimization_algorithm

@testset "Type Stability Analysis" begin
    @testset "Type Inference" begin
        # Test type inference for different input types
        @test @inferred scientific_function(1.0, 2.0) isa Float64
        @test @inferred scientific_function(1.0f0, 2.0f0) isa Float32
        @test @inferred scientific_function([1.0, 2.0], [3.0, 4.0]) isa Vector{Float64}
    end
    
    @testset "Type Warnings" begin
        # Ensure no type instabilities
        @test_nowarn @code_warntype scientific_function(1.0, 2.0)
        @test_nowarn @code_warntype optimization_algorithm(rand(100))
    end
    
    @testset "Method Dispatch" begin
        # Test multiple dispatch works correctly
        @test scientific_function(1) !== scientific_function(1.0)
        @test scientific_function(Vector{Int}) !== scientific_function(Vector{Float64})
    end
end

@testset "Performance Benchmarking" begin
    @testset "Allocation Analysis" begin
        # Test memory allocations
        small_data = rand(100)
        medium_data = rand(1000)
        large_data = rand(10000)
        
        @test @allocated scientific_function(small_data) < 1000
        @test @allocated scientific_function(medium_data) < 10000
        @test @allocated scientific_function(large_data) < 100000
    end
    
    @testset "Timing Benchmarks" begin
        data_sizes = [100, 1000, 10000]
        
        for n in data_sizes
            test_data = rand(n)
            
            b = @benchmark scientific_function($test_data)
            median_time = median(b.times)
            
            # Ensure reasonable scaling (adjust for actual complexity)
            expected_time = 1e6 * n  # nanoseconds
            @test median_time < expected_time
        end
    end
end

@testset "Numerical Accuracy" begin
    @testset "Analytical Solutions" begin
        # Test against known analytical solutions
        x = collect(range(-5, 5, length=100))
        analytical = exp.(-x.^2)
        numerical = gaussian_function(x, σ=1.0)
        
        @test isapprox(numerical, analytical, rtol=1e-10)
    end
    
    @testset "Conservation Laws" begin
        # Test physical conservation laws
        initial_energy = compute_energy(initial_state)
        final_state = simulate_system(initial_state, time=10.0)
        final_energy = compute_energy(final_state)
        
        # Energy should be conserved (within numerical precision)
        @test isapprox(initial_energy, final_energy, rtol=1e-8)
    end
end
```

### 5. **Advanced Test Data Generation & Management**

#### **Scientific Test Data Factories**
```python
class ScientificTestDataFactory:
    """Factory for generating scientifically-relevant test data."""
    
    @staticmethod
    def correlation_function_data(tau_range=(1e-6, 1e-1), n_points=100, 
                                 noise_level=0.01, model="exponential"):
        """Generate synthetic correlation function data."""
        t = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_points)
        
        if model == "exponential":
            correlation = np.exp(-t / 1e-3)
        elif model == "stretched_exponential":
            correlation = np.exp(-(t / 1e-3)**0.8)
        elif model == "kohlrausch":
            correlation = np.exp(-(t / 1e-3)**0.5)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=correlation.shape)
            correlation += noise
            
        return {"time": t, "correlation": correlation, "model": model}
    
    @staticmethod  
    def mcmc_test_chains(n_chains=4, n_samples=1000, target_distribution="normal"):
        """Generate MCMC test chains with known statistical properties."""
        chains = []
        
        for i in range(n_chains):
            if target_distribution == "normal":
                chain = np.random.normal(0, 1, n_samples)
            elif target_distribution == "gamma":
                chain = np.random.gamma(2, 2, n_samples)
            elif target_distribution == "beta":
                chain = np.random.beta(2, 5, n_samples)
                
            chains.append(chain)
            
        return np.array(chains)
    
    @staticmethod
    def optimization_test_problems():
        """Generate standard optimization test problems."""
        problems = {}
        
        # Rosenbrock function
        problems["rosenbrock"] = {
            "function": lambda x: sum(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2),
            "gradient": lambda x: np.array([
                -400*x[0]*(x[1] - x[0]**2) - 2*(1-x[0]),
                200*(x[1] - x[0]**2)
            ]),
            "minimum": np.array([1.0, 1.0]),
            "minimum_value": 0.0
        }
        
        # Quadratic bowl
        problems["quadratic"] = {
            "function": lambda x: 0.5 * np.sum(x**2),
            "gradient": lambda x: x,
            "minimum": np.zeros(2),
            "minimum_value": 0.0
        }
        
        return problems

@pytest.fixture(scope="session")
def scientific_test_datasets():
    """Session-scoped fixture providing comprehensive test datasets."""
    factory = ScientificTestDataFactory()
    
    return {
        "correlation_data": {
            "exponential": factory.correlation_function_data(model="exponential"),
            "stretched": factory.correlation_function_data(model="stretched_exponential"), 
            "noisy": factory.correlation_function_data(noise_level=0.1)
        },
        "mcmc_data": {
            "normal_chains": factory.mcmc_test_chains(target_distribution="normal"),
            "gamma_chains": factory.mcmc_test_chains(target_distribution="gamma"),
            "converged_chains": factory.mcmc_test_chains(n_samples=10000)
        },
        "optimization_data": factory.optimization_test_problems()
    }
```

### 6. **Property-Based & Hypothesis Testing Integration**

#### **Advanced Property-Based Tests**
```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

class TestMathematicalProperties:
    """Property-based testing for mathematical functions."""
    
    @given(arrays(np.float64, (10,), elements=st.floats(0.1, 10.0)))
    def test_monotonic_function_property(self, x):
        """Test that monotonic functions preserve ordering."""
        assume(len(x) > 1)
        
        sorted_x = np.sort(x)
        result = monotonic_function(sorted_x)
        
        # Result should also be monotonic
        assert np.all(result[1:] >= result[:-1]), "Function not monotonic"
    
    @given(st.floats(0.1, 10.0), st.floats(0.1, 10.0))
    def test_function_homogeneity(self, x, scale):
        """Test function scaling properties."""
        f_x = scientific_function(x)
        f_scaled = scientific_function(scale * x)
        
        # Test homogeneity property: f(ax) = a^k * f(x)
        expected_scaling = scale**HOMOGENEITY_DEGREE
        assert abs(f_scaled - expected_scaling * f_x) / f_x < 1e-10
    
    @given(arrays(np.float64, (20,), elements=st.floats(-10, 10)))
    def test_statistical_properties(self, data):
        """Test statistical function properties."""
        assume(len(data) > 5)
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        normalized = normalize_function(data)
        
        # Normalized data should have mean ≈ 0, std ≈ 1
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10

    @settings(max_examples=50, deadline=None)
    @given(st.integers(10, 1000))
    def test_algorithm_invariants(self, n):
        """Test algorithm invariants across different input sizes."""
        data = np.random.randn(n)
        
        result = algorithm_with_invariants(data)
        
        # Test invariant properties
        assert len(result) == n, "Output size should match input"
        assert np.all(np.isfinite(result)), "All outputs should be finite"
        assert np.sum(result) >= 0, "Sum should be non-negative"
```

### 7. **Integration & System Test Generation**

#### **End-to-End Scientific Workflows**
```python
class TestScientificWorkflows:
    """Test complete scientific analysis pipelines."""
    
    @pytest.fixture
    def experimental_data_mock(self):
        """Mock experimental data with realistic characteristics."""
        return {
            "time_series": generate_realistic_time_series(length=1000),
            "correlation_matrix": generate_correlation_matrix(size=50),
            "metadata": {
                "experiment_date": "2024-01-15",
                "temperature": 298.15,
                "pressure": 1.013e5,
                "sample_id": "TEST_001"
            }
        }
    
    def test_complete_analysis_pipeline(self, experimental_data_mock):
        """Test complete data analysis pipeline."""
        # Load and validate data
        data = load_experimental_data(experimental_data_mock)
        assert validate_data_integrity(data), "Data integrity check failed"
        
        # Preprocessing
        preprocessed = preprocess_data(data, 
                                     remove_outliers=True,
                                     apply_smoothing=True)
        assert preprocessed.shape[0] > 0, "No data after preprocessing"
        
        # Analysis
        analysis_results = run_analysis(preprocessed,
                                      method="robust_estimation",
                                      convergence_threshold=1e-6)
        assert analysis_results["converged"], "Analysis did not converge"
        
        # Validation
        validation_results = validate_results(analysis_results,
                                            physics_constraints=True)
        assert validation_results["physically_valid"], "Results violate physics"
        
        # Output generation
        output = generate_output(analysis_results, 
                               format="publication_ready")
        assert "parameters" in output, "Missing parameter estimates"
        assert "uncertainties" in output, "Missing uncertainty estimates"
        assert "figures" in output, "Missing visualization outputs"

    @pytest.mark.integration
    def test_multi_method_comparison(self):
        """Test comparison of multiple analysis methods."""
        synthetic_data = generate_synthetic_data_with_known_truth()
        
        methods = ["classical_optimization", "robust_optimization", "bayesian_mcmc"]
        results = {}
        
        for method in methods:
            results[method] = run_analysis(synthetic_data, method=method)
            assert results[method]["success"], f"{method} analysis failed"
        
        # Compare results consistency
        parameter_estimates = [r["parameters"] for r in results.values()]
        
        # All methods should give similar results for synthetic data
        pairwise_differences = []
        for i, est1 in enumerate(parameter_estimates):
            for est2 in parameter_estimates[i+1:]:
                diff = np.abs(est1 - est2) / est1
                pairwise_differences.append(diff)
        
        max_relative_difference = np.max(pairwise_differences)
        assert max_relative_difference < 0.1, "Methods give inconsistent results"
```

### 8. **Quality Assurance & Test Validation**

#### **Test Suite Completeness Validation**
```python
def test_comprehensive_coverage():
    """Validate comprehensive test coverage of scientific functions."""
    
    # Get all public scientific functions
    import inspect
    from homodyne import analysis, optimization
    
    modules = [analysis, optimization]
    scientific_functions = []
    
    for module in modules:
        for name, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) and 
                not name.startswith('_') and 
                has_scientific_computation(obj)):
                scientific_functions.append(f"{module.__name__}.{name}")
    
    # Check test coverage
    test_files = glob.glob("tests/**/test_*.py", recursive=True)
    all_test_functions = []
    
    for test_file in test_files:
        with open(test_file) as f:
            content = f.read()
            test_funcs = re.findall(r'def (test_\w+)', content)
            all_test_functions.extend(test_funcs)
    
    # Validate coverage
    missing_tests = []
    for func_name in scientific_functions:
        simple_name = func_name.split('.')[-1]
        has_test = any(simple_name.lower() in test_name.lower() 
                      for test_name in all_test_functions)
        
        if not has_test:
            missing_tests.append(func_name)
    
    assert len(missing_tests) == 0, f"Missing tests for: {missing_tests}"

def test_numerical_stability_coverage():
    """Ensure all numerical functions have stability tests."""
    numerical_functions = identify_numerical_functions()
    
    for func_name in numerical_functions:
        stability_tests = [test for test in get_all_test_names() 
                          if "stability" in test and func_name in test]
        assert len(stability_tests) > 0, f"Missing stability test for {func_name}"

def test_performance_regression_coverage():
    """Ensure performance-critical functions have regression tests."""
    performance_critical_functions = identify_performance_critical_functions()
    
    for func_name in performance_critical_functions:
        performance_tests = [test for test in get_all_test_names()
                           if "performance" in test and func_name in test]
        assert len(performance_tests) > 0, f"Missing performance test for {func_name}"
```

## Advanced Features & Integration

### **AI-Powered Test Generation**
- **Pattern Recognition**: ML-based identification of test patterns
- **Smart Test Case Generation**: AI-powered edge case discovery
- **Optimization Suggestions**: Intelligent test improvement recommendations
- **Documentation Integration**: Auto-generated test documentation

### **Continuous Testing & Monitoring**
- **Performance Baselines**: Automated baseline management and regression detection
- **Statistical Validation**: Continuous validation of statistical test properties
- **Quality Metrics**: Real-time test quality and coverage monitoring
- **Team Collaboration**: Shared test quality dashboards and metrics

### **Research Workflow Integration**
- **Publication Standards**: Tests that meet research publication requirements
- **Reproducibility Validation**: Ensure research reproducibility through testing
- **Collaboration Support**: Multi-institution research code validation
- **Data Provenance**: Test data lineage and source validation

## Command Options

- `--type=<type>`: Test types (all, unit, integration, performance, cases, regression)
- `--framework=<framework>`: Testing framework (auto, pytest, julia, unittest)
- `--coverage=<threshold>`: Target coverage percentage (default: 85%)
- `--interactive`: Interactive test generation with TodoWrite planning
- `--cases`: Focus on comprehensive test case generation
- `--performance`: Include performance benchmarking and regression tests
- `--statistical`: Emphasize statistical validation and hypothesis testing
- `--numerical`: Focus on numerical accuracy and stability testing
- `--property-based`: Include property-based testing with Hypothesis/QuickCheck
- `--mock-external`: Auto-generate mocks for external dependencies
- `--baseline-update`: Update performance and quality baselines
- `--research`: Generate research-quality validation tests
- `--publication`: Generate publication-ready test validation

## Integration Capabilities

### **Scientific Computing Ecosystem**
- **Python**: NumPy, SciPy, PyMC, JAX, pytest, Hypothesis integration
- **Julia**: Test.jl, BenchmarkTools, JET.jl, Aqua.jl, QuickCheck.jl
- **Performance**: Memory profiling, allocation analysis, regression detection
- **Documentation**: Sphinx, Documenter.jl, notebook validation

### **Development Workflow**
- **CI/CD**: GitHub Actions, GitLab CI, automated test execution
- **Quality Gates**: Pre-commit hooks, code review integration
- **Team Collaboration**: Shared baselines, quality metrics, knowledge transfer
- **Research Standards**: Publication-ready validation and documentation

Target: Generate comprehensive test suites and test cases for $ARGUMENTS following scientific computing best practices, numerical rigor, and research reproducibility standards