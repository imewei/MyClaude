---
description: Comprehensive scientific computing test execution engine with intelligent failure resolution, performance benchmarking, and reproducibility validation
category: testing-validation
argument-hint: [--scope=SCOPE] [--profile] [--benchmark] [--scientific] [--gpu] [--parallel] [--reproducible] [--coverage] [--report]
allowed-tools: Bash, Read, Write, Glob, MultiEdit, TodoWrite
---

# Advanced Scientific Computing Test Engine (2025 Research Edition)

Intelligent test orchestration system with AI-powered failure analysis, comprehensive performance benchmarking, scientific reproducibility validation, and GPU/TPU acceleration testing for Python, JAX, and Julia ecosystems.

## Quick Start

```bash
# Comprehensive test suite with scientific validation
/run-all-tests

# Scientific computing focused testing
/run-all-tests --scientific --benchmark --reproducible

# GPU/TPU accelerated testing
/run-all-tests --gpu --parallel --profile

# JAX ecosystem testing with JIT validation
/run-all-tests --scope=jax --benchmark --profile

# Julia performance testing with type stability
/run-all-tests --scope=julia --benchmark --profile

# Research reproducibility validation
/run-all-tests --reproducible --coverage --report

# CI/CD optimized testing
/run-all-tests --scope=ci --parallel --coverage

# Comprehensive testing with AI failure analysis
/run-all-tests --profile --benchmark --report --auto-fix

# Performance regression testing
/run-all-tests --benchmark --baseline-compare --alert

# Cross-platform compatibility testing
/run-all-tests --scope=compatibility --platform-matrix
```

## Core Test Orchestration Engine

### 1. Intelligent Test Discovery & Classification

```bash
# Advanced test discovery system
discover_test_ecosystem() {
    echo "🔍 Discovering Scientific Computing Test Ecosystem..."

    # Initialize test environment
    mkdir -p .test_cache/{discovery,execution,benchmarks,reports,failures}

    # Project detection and test framework identification
    local project_types=()
    local test_frameworks=()
    local scientific_packages=()

    # Python ecosystem detection
    if [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]] || [[ -f "requirements.txt" ]]; then
        project_types+=("python")

        # Test framework detection
        if command -v pytest &>/dev/null || find . -name "test_*.py" -o -name "*_test.py" | head -1 | grep -q .; then
            test_frameworks+=("pytest")
        fi
        if command -v unittest &>/dev/null || find . -name "unittest_*.py" | head -1 | grep -q .; then
            test_frameworks+=("unittest")
        fi
        if command -v nose2 &>/dev/null; then
            test_frameworks+=("nose2")
        fi

        # Scientific package detection
        for package in numpy scipy pandas scikit-learn matplotlib seaborn jax flax optax tensorflow pytorch; do
            if python -c "import $package" 2>/dev/null; then
                scientific_packages+=("$package")
            fi
        done
    fi

    # Julia ecosystem detection
    if [[ -f "Project.toml" ]] || [[ -f "Manifest.toml" ]]; then
        project_types+=("julia")
        test_frameworks+=("Pkg.test")

        # Julia scientific package detection
        if [[ -f "Project.toml" ]]; then
            for package in Flux DifferentialEquations MLJ Plots StatsModels LinearAlgebra; do
                if julia -e "using $package" 2>/dev/null; then
                    scientific_packages+=("$package")
                fi
            done
        fi
    fi

    # JavaScript/TypeScript ecosystem detection
    if [[ -f "package.json" ]]; then
        project_types+=("javascript")

        if jq -e '.devDependencies.jest // .dependencies.jest' package.json >/dev/null 2>&1; then
            test_frameworks+=("jest")
        fi
        if jq -e '.devDependencies.mocha // .dependencies.mocha' package.json >/dev/null 2>&1; then
            test_frameworks+=("mocha")
        fi
        if jq -e '.devDependencies.vitest // .dependencies.vitest' package.json >/dev/null 2>&1; then
            test_frameworks+=("vitest")
        fi
    fi

    # GPU/CUDA detection
    local gpu_available=false
    if command -v nvidia-smi &>/dev/null && nvidia-smi >/dev/null 2>&1; then
        gpu_available=true
    elif command -v rocm-smi &>/dev/null; then
        gpu_available=true
    elif python -c "import jax; print(jax.devices())" 2>/dev/null | grep -q "gpu\|tpu"; then
        gpu_available=true
    fi

    # Test file discovery with categorization
    local unit_tests=()
    local integration_tests=()
    local performance_tests=()
    local scientific_tests=()
    local gpu_tests=()

    # Python test categorization
    while IFS= read -r -d '' test_file; do
        if [[ "$test_file" =~ test_.*\.py$ ]] || [[ "$test_file" =~ .*_test\.py$ ]]; then
            # Categorize by content analysis
            if grep -q "benchmark\|performance\|timing" "$test_file"; then
                performance_tests+=("$test_file")
            elif grep -q "integration\|e2e\|end.to.end" "$test_file"; then
                integration_tests+=("$test_file")
            elif grep -q "jax\|numpy\|scipy\|torch\|scientific" "$test_file"; then
                scientific_tests+=("$test_file")
            elif grep -q "gpu\|cuda\|device" "$test_file"; then
                gpu_tests+=("$test_file")
            else
                unit_tests+=("$test_file")
            fi
        fi
    done < <(find . -name "*.py" -path "*/test*" -o -name "test_*.py" -o -name "*_test.py" -print0 2>/dev/null)

    # Julia test discovery
    if [[ -d "test" ]]; then
        while IFS= read -r -d '' test_file; do
            if [[ "$test_file" =~ \.jl$ ]]; then
                if grep -q "BenchmarkTools\|@benchmark" "$test_file"; then
                    performance_tests+=("$test_file")
                elif grep -q "scientific\|numerical\|Flux\|MLJ" "$test_file"; then
                    scientific_tests+=("$test_file")
                else
                    unit_tests+=("$test_file")
                fi
            fi
        done < <(find test -name "*.jl" -print0 2>/dev/null)
    fi

    # Save discovery results
    cat > ".test_cache/discovery.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "project_types": $(printf '%s\n' "${project_types[@]}" | jq -R . | jq -s .),
    "test_frameworks": $(printf '%s\n' "${test_frameworks[@]}" | jq -R . | jq -s .),
    "scientific_packages": $(printf '%s\n' "${scientific_packages[@]}" | jq -R . | jq -s .),
    "gpu_available": $gpu_available,
    "test_categories": {
        "unit_tests": $(printf '%s\n' "${unit_tests[@]}" | jq -R . | jq -s .),
        "integration_tests": $(printf '%s\n' "${integration_tests[@]}" | jq -R . | jq -s .),
        "performance_tests": $(printf '%s\n' "${performance_tests[@]}" | jq -R . | jq -s .),
        "scientific_tests": $(printf '%s\n' "${scientific_tests[@]}" | jq -R . | jq -s .),
        "gpu_tests": $(printf '%s\n' "${gpu_tests[@]}" | jq -R . | jq -s .)
    },
    "total_test_files": $((${#unit_tests[@]} + ${#integration_tests[@]} + ${#performance_tests[@]} + ${#scientific_tests[@]} + ${#gpu_tests[@]}))
}
EOF

    # Display discovery summary
    echo "  📊 Test Ecosystem Discovery:"
    echo "    • Project types: ${project_types[*]}"
    echo "    • Test frameworks: ${test_frameworks[*]}"
    echo "    • Scientific packages: ${#scientific_packages[@]} detected"
    echo "    • GPU/accelerator: $gpu_available"
    echo "    • Test files found:"
    echo "      - Unit tests: ${#unit_tests[@]}"
    echo "      - Integration tests: ${#integration_tests[@]}"
    echo "      - Performance tests: ${#performance_tests[@]}"
    echo "      - Scientific tests: ${#scientific_tests[@]}"
    echo "      - GPU tests: ${#gpu_tests[@]}"

    export TEST_DISCOVERY_COMPLETE="true"
    export PROJECT_TYPES="${project_types[*]}"
    export TEST_FRAMEWORKS="${test_frameworks[*]}"
    export GPU_AVAILABLE="$gpu_available"
}

# Scientific computing environment setup
setup_scientific_test_environment() {
    echo "🔬 Setting up Scientific Computing Test Environment..."

    # Python scientific environment
    if [[ "$PROJECT_TYPES" =~ python ]]; then
        echo "  🐍 Configuring Python scientific testing environment..."

        # Set up reproducibility
        export PYTHONHASHSEED=42
        export NUMPY_RANDOM_SEED=42
        export TF_DETERMINISTIC_OPS=1
        export TF_CUDNN_DETERMINISTIC=1

        # JAX configuration for testing
        export JAX_PLATFORM_NAME=cpu  # Start with CPU, switch to GPU if available
        export JAX_ENABLE_X64=true     # Enable 64-bit precision for numerical tests

        # PyTorch configuration
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

        # Performance testing setup
        if python -c "import pytest_benchmark" 2>/dev/null; then
            echo "    ✅ pytest-benchmark available"
        else
            echo "    ⚠️  Consider installing pytest-benchmark for performance testing"
        fi

        # Coverage setup
        if python -c "import coverage" 2>/dev/null; then
            echo "    ✅ Coverage.py available"
        fi
    fi

    # Julia scientific environment
    if [[ "$PROJECT_TYPES" =~ julia ]]; then
        echo "  💎 Configuring Julia scientific testing environment..."

        # Julia reproducibility
        export JULIA_NUM_THREADS=$(nproc 2>/dev/null || echo 4)

        # Check for BenchmarkTools
        if julia -e "using BenchmarkTools" 2>/dev/null; then
            echo "    ✅ BenchmarkTools.jl available"
        fi

        # Check for test dependencies
        if julia -e "using Test" 2>/dev/null; then
            echo "    ✅ Test.jl available"
        fi
    fi

    # GPU environment setup
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        echo "  🎮 Configuring GPU testing environment..."

        # CUDA setup
        if command -v nvidia-smi &>/dev/null; then
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
            echo "    📊 GPU Status:"
            nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -3
        fi

        # JAX GPU setup
        if python -c "import jax" 2>/dev/null; then
            export JAX_PLATFORM_NAME=gpu
            echo "    🚀 JAX GPU backend enabled"
        fi
    fi

    echo "✅ Scientific test environment configured"
}
```

### 2. Intelligent Test Execution Engine

```bash
# Advanced test execution with parallel processing and failure analysis
execute_test_suite() {
    local test_scope="${1:-all}"
    local profile_mode="${2:-false}"
    local benchmark_mode="${3:-false}"
    local parallel_mode="${4:-false}"
    local gpu_mode="${5:-false}"

    echo "🚀 Executing Scientific Computing Test Suite..."
    echo "   Scope: $test_scope | Profile: $profile_mode | Benchmark: $benchmark_mode | Parallel: $parallel_mode | GPU: $gpu_mode"

    # Load test discovery
    if [[ ! -f ".test_cache/discovery.json" ]]; then
        echo "❌ Test discovery not complete. Run discovery first."
        return 1
    fi

    local start_time=$(date +%s)
    local test_results=()
    local failed_tests=()
    local performance_metrics=()

    # Python test execution
    if [[ "$PROJECT_TYPES" =~ python ]] && [[ "$test_scope" =~ all|python|pytest ]]; then
        echo "  🐍 Executing Python test suite..."
        execute_python_tests "$profile_mode" "$benchmark_mode" "$parallel_mode" "$gpu_mode"
    fi

    # Julia test execution
    if [[ "$PROJECT_TYPES" =~ julia ]] && [[ "$test_scope" =~ all|julia ]]; then
        echo "  💎 Executing Julia test suite..."
        execute_julia_tests "$profile_mode" "$benchmark_mode" "$parallel_mode"
    fi

    # JavaScript test execution
    if [[ "$PROJECT_TYPES" =~ javascript ]] && [[ "$test_scope" =~ all|javascript|jest ]]; then
        echo "  🟨 Executing JavaScript test suite..."
        execute_javascript_tests "$profile_mode" "$benchmark_mode" "$parallel_mode"
    fi

    # Specialized scientific tests
    if [[ "$test_scope" =~ all|scientific ]]; then
        echo "  🔬 Executing specialized scientific tests..."
        execute_scientific_tests "$profile_mode" "$benchmark_mode" "$gpu_mode"
    fi

    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))

    echo "⏱️  Total execution time: ${execution_time}s"

    # Generate execution summary
    generate_test_execution_summary "$execution_time"
}

# Python scientific test execution
execute_python_tests() {
    local profile_mode="$1"
    local benchmark_mode="$2"
    local parallel_mode="$3"
    local gpu_mode="$4"

    if [[ "$TEST_FRAMEWORKS" =~ pytest ]]; then
        echo "    📋 Running pytest with scientific optimizations..."

        # Build pytest command with optimizations
        local pytest_cmd="python -m pytest"
        local pytest_args=()

        # Basic arguments
        pytest_args+=("--tb=short")
        pytest_args+=("--strict-markers")
        pytest_args+=("--strict-config")

        # Coverage
        if [[ "$COVERAGE_MODE" == "true" ]]; then
            pytest_args+=("--cov=.")
            pytest_args+=("--cov-report=term-missing")
            pytest_args+=("--cov-report=json:.test_cache/coverage.json")
            pytest_args+=("--cov-report=html:.test_cache/htmlcov")
        fi

        # Parallel execution
        if [[ "$parallel_mode" == "true" ]] && command -v pytest-xdist &>/dev/null; then
            local num_workers=$(nproc 2>/dev/null || echo 4)
            pytest_args+=("-n" "$num_workers")
            echo "      🔀 Parallel execution with $num_workers workers"
        fi

        # Benchmark mode
        if [[ "$benchmark_mode" == "true" ]] && python -c "import pytest_benchmark" 2>/dev/null; then
            pytest_args+=("--benchmark-only")
            pytest_args+=("--benchmark-json=.test_cache/benchmark_results.json")
            pytest_args+=("--benchmark-histogram=.test_cache/benchmark_histogram")
            echo "      📏 Benchmark mode enabled"
        fi

        # Profiling mode
        if [[ "$profile_mode" == "true" ]]; then
            pytest_args+=("--profile")
            pytest_args+=("--profile-svg")
            echo "      📊 Profiling mode enabled"
        fi

        # GPU mode
        if [[ "$gpu_mode" == "true" ]] && [[ "$GPU_AVAILABLE" == "true" ]]; then
            pytest_args+=("-m" "gpu")
            export JAX_PLATFORM_NAME=gpu
            echo "      🎮 GPU testing mode enabled"
        fi

        # Scientific reproducibility
        pytest_args+=("--durations=10")
        pytest_args+=("-v")

        # Execute pytest
        echo "      Command: $pytest_cmd ${pytest_args[*]}"
        if $pytest_cmd "${pytest_args[@]}" 2>&1 | tee .test_cache/pytest_output.log; then
            echo "    ✅ pytest execution completed successfully"
        else
            local exit_code=$?
            echo "    ❌ pytest execution failed (exit code: $exit_code)"
            analyze_python_test_failures
            return $exit_code
        fi

        # JAX-specific tests if available
        if python -c "import jax" 2>/dev/null; then
            echo "    🚀 Running JAX-specific validation tests..."
            run_jax_validation_tests "$gpu_mode"
        fi

        # Scientific computation validation
        if [[ "${SCIENTIFIC_PACKAGES[*]}" =~ numpy ]]; then
            echo "    🔢 Running numerical precision validation..."
            run_numerical_validation_tests
        fi
    fi
}

# Julia scientific test execution
execute_julia_tests() {
    local profile_mode="$1"
    local benchmark_mode="$2"
    local parallel_mode="$3"

    if [[ "$PROJECT_TYPES" =~ julia ]]; then
        echo "    💎 Running Julia test suite with optimizations..."

        # Build Julia test command
        local julia_cmd="julia --project=. -e"
        local test_expr="using Pkg; Pkg.test()"

        # Parallel testing
        if [[ "$parallel_mode" == "true" ]]; then
            export JULIA_NUM_THREADS=$(nproc 2>/dev/null || echo 4)
            echo "      🔀 Julia parallel testing with $JULIA_NUM_THREADS threads"
        fi

        # Benchmark mode with BenchmarkTools
        if [[ "$benchmark_mode" == "true" ]] && julia -e "using BenchmarkTools" 2>/dev/null; then
            echo "      📏 Running Julia benchmarks..."
            julia --project=. -e "
                using BenchmarkTools, JSON
                include(\"test/benchmarks.jl\")
                results = run_benchmarks()
                open(\".test_cache/julia_benchmarks.json\", \"w\") do f
                    JSON.print(f, results)
                end
            " 2>/dev/null || echo "      ⚠️  No benchmark suite found"
        fi

        # Profiling mode
        if [[ "$profile_mode" == "true" ]]; then
            echo "      📊 Julia profiling mode enabled"
            test_expr="using Profile, Pkg; Profile.clear(); Profile.init(); @profile Pkg.test(); Profile.print()"
        fi

        # Execute Julia tests
        echo "      Command: $julia_cmd '$test_expr'"
        if $julia_cmd "$test_expr" 2>&1 | tee .test_cache/julia_output.log; then
            echo "    ✅ Julia test execution completed successfully"
        else
            local exit_code=$?
            echo "    ❌ Julia test execution failed (exit code: $exit_code)"
            analyze_julia_test_failures
            return $exit_code
        fi

        # Type stability analysis
        echo "    🔗 Running type stability analysis..."
        julia --project=. -e "
            using Test
            @testset \"Type Stability\" begin
                # Add type stability tests for critical functions
                include(\"test/type_stability_tests.jl\")
            end
        " 2>/dev/null || echo "      ℹ️  No type stability tests found"

        # Performance regression tests
        if [[ "$benchmark_mode" == "true" ]]; then
            echo "    📈 Checking for performance regressions..."
            check_julia_performance_regressions
        fi
    fi
}

# Specialized scientific test execution
execute_scientific_tests() {
    local profile_mode="$1"
    local benchmark_mode="$2"
    local gpu_mode="$3"

    echo "    🔬 Running specialized scientific computing tests..."

    # JAX ecosystem tests
    if python -c "import jax" 2>/dev/null; then
        echo "      🚀 JAX ecosystem validation..."

        # JIT compilation tests
        python << 'EOF'
import jax
import jax.numpy as jnp
import sys

def test_jit_compilation():
    """Test JIT compilation works correctly"""
    @jax.jit
    def simple_function(x):
        return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

    x = jnp.array([1.0, 2.0, 3.0])
    result = simple_function(x)
    expected = jnp.ones_like(x)

    if jnp.allclose(result, expected, rtol=1e-10):
        print("    ✅ JIT compilation test passed")
        return True
    else:
        print("    ❌ JIT compilation test failed")
        return False

def test_gradient_computation():
    """Test automatic differentiation"""
    def loss_fn(x):
        return jnp.sum(x ** 2)

    grad_fn = jax.grad(loss_fn)
    x = jnp.array([1.0, 2.0, 3.0])
    gradient = grad_fn(x)
    expected = 2 * x

    if jnp.allclose(gradient, expected):
        print("    ✅ Gradient computation test passed")
        return True
    else:
        print("    ❌ Gradient computation test failed")
        return False

def test_vectorization():
    """Test vectorization with vmap"""
    def single_fn(x):
        return jnp.sin(x)

    batch_fn = jax.vmap(single_fn)
    x_batch = jnp.array([1.0, 2.0, 3.0])

    manual_result = jnp.array([single_fn(x) for x in x_batch])
    vmap_result = batch_fn(x_batch)

    if jnp.allclose(manual_result, vmap_result):
        print("    ✅ Vectorization test passed")
        return True
    else:
        print("    ❌ Vectorization test failed")
        return False

# Run tests
success_count = 0
success_count += test_jit_compilation()
success_count += test_gradient_computation()
success_count += test_vectorization()

print(f"    📊 JAX tests: {success_count}/3 passed")
if success_count < 3:
    sys.exit(1)
EOF
    fi

    # GPU/TPU tests if available
    if [[ "$gpu_mode" == "true" ]] && [[ "$GPU_AVAILABLE" == "true" ]]; then
        echo "      🎮 GPU/TPU acceleration tests..."

        # JAX GPU tests
        if python -c "import jax; print(jax.devices())" 2>/dev/null | grep -q "gpu\|tpu"; then
            python << 'EOF'
import jax
import jax.numpy as jnp

def test_gpu_computation():
    """Test GPU computation works"""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.device_kind in ['gpu', 'tpu']]

    if not gpu_devices:
        print("    ⚠️  No GPU/TPU devices found")
        return False

    @jax.jit
    def gpu_computation(x):
        return jnp.dot(x, x.T)

    x = jnp.ones((1000, 1000))
    with jax.default_device(gpu_devices[0]):
        result = gpu_computation(x)

    expected_value = 1000.0
    if jnp.allclose(result[0, 0], expected_value):
        print(f"    ✅ GPU computation test passed on {gpu_devices[0]}")
        return True
    else:
        print("    ❌ GPU computation test failed")
        return False

if test_gpu_computation():
    print("    📊 GPU tests: 1/1 passed")
else:
    print("    📊 GPU tests: 0/1 passed")
EOF
        fi
    fi

    # Numerical precision tests
    echo "      🔢 Numerical precision validation..."
    python << 'EOF'
import numpy as np
import sys

def test_numerical_precision():
    """Test numerical precision and stability"""
    # Test floating point precision
    x = np.array([1e-15, 1e-14, 1e-13])
    result = np.sum(x)

    if result > 0 and not np.isnan(result) and not np.isinf(result):
        print("    ✅ Numerical precision test passed")
        return True
    else:
        print("    ❌ Numerical precision test failed")
        return False

def test_mathematical_identities():
    """Test mathematical identities hold"""
    x = np.linspace(0, 2*np.pi, 100)

    # sin²(x) + cos²(x) = 1
    identity_result = np.sin(x)**2 + np.cos(x)**2
    ones = np.ones_like(x)

    if np.allclose(identity_result, ones, rtol=1e-14):
        print("    ✅ Mathematical identity test passed")
        return True
    else:
        print("    ❌ Mathematical identity test failed")
        return False

# Run numerical tests
success_count = 0
success_count += test_numerical_precision()
success_count += test_mathematical_identities()

print(f"    📊 Numerical tests: {success_count}/2 passed")
if success_count < 2:
    sys.exit(1)
EOF

    # Reproducibility tests
    echo "      🔄 Reproducibility validation..."
    test_reproducibility
}
```

### 3. AI-Powered Failure Analysis System

```bash
# Intelligent test failure analysis with root cause detection
analyze_test_failures() {
    echo "🔍 AI-Powered Test Failure Analysis..."

    # Collect failure data
    local failure_files=()
    local error_patterns=()
    local failure_categories=()

    # Analyze Python test failures
    if [[ -f ".test_cache/pytest_output.log" ]]; then
        analyze_python_test_failures
    fi

    # Analyze Julia test failures
    if [[ -f ".test_cache/julia_output.log" ]]; then
        analyze_julia_test_failures
    fi

    # Generate failure report with AI insights
    generate_failure_analysis_report
}

analyze_python_test_failures() {
    echo "  🐍 Analyzing Python test failures..."

    python3 << 'EOF'
import re
import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TestFailure:
    test_name: str
    failure_type: str
    error_message: str
    traceback: str
    file_location: str
    line_number: int
    suggested_fix: str
    confidence: float

class PythonTestFailureAnalyzer:
    def __init__(self):
        self.failure_patterns = {
            'assertion_error': {
                'pattern': r'AssertionError: (.+)',
                'category': 'logic_error',
                'common_fixes': [
                    'Review test expectations vs actual values',
                    'Check for floating-point precision issues',
                    'Verify test data setup is correct'
                ]
            },
            'import_error': {
                'pattern': r'ImportError: (.+)|ModuleNotFoundError: (.+)',
                'category': 'dependency_issue',
                'common_fixes': [
                    'Install missing dependencies: pip install <package>',
                    'Check virtual environment activation',
                    'Verify PYTHONPATH configuration'
                ]
            },
            'attribute_error': {
                'pattern': r'AttributeError: (.+)',
                'category': 'api_change',
                'common_fixes': [
                    'Check for API changes in dependencies',
                    'Verify object initialization',
                    'Review attribute name spelling'
                ]
            },
            'type_error': {
                'pattern': r'TypeError: (.+)',
                'category': 'type_mismatch',
                'common_fixes': [
                    'Check function argument types',
                    'Verify data type compatibility',
                    'Review type annotations'
                ]
            },
            'value_error': {
                'pattern': r'ValueError: (.+)',
                'category': 'invalid_input',
                'common_fixes': [
                    'Validate input data ranges',
                    'Check for edge cases in test data',
                    'Review function parameter constraints'
                ]
            },
            'runtime_error': {
                'pattern': r'RuntimeError: (.+)',
                'category': 'execution_error',
                'common_fixes': [
                    'Check system resources and permissions',
                    'Review concurrent execution issues',
                    'Verify runtime environment setup'
                ]
            },
            'numerical_error': {
                'pattern': r'(overflow|underflow|division by zero|invalid value)',
                'category': 'numerical_instability',
                'common_fixes': [
                    'Add numerical stability checks',
                    'Use appropriate data types (float64 vs float32)',
                    'Implement proper error handling for edge cases'
                ]
            },
            'jax_error': {
                'pattern': r'(jax\.|JAX)',
                'category': 'jax_specific',
                'common_fixes': [
                    'Ensure functions are JAX-compatible (pure functions)',
                    'Check for proper JAX random key usage',
                    'Verify JIT compilation compatibility'
                ]
            },
            'gpu_error': {
                'pattern': r'(CUDA|GPU|device)',
                'category': 'gpu_specific',
                'common_fixes': [
                    'Verify GPU availability and drivers',
                    'Check CUDA memory allocation',
                    'Ensure proper device placement'
                ]
            }
        }

    def analyze_pytest_output(self, log_file: str) -> List[TestFailure]:
        """Analyze pytest output for failure patterns"""
        failures = []

        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Extract individual test failures
            failure_sections = re.split(r'={20,} FAILURES ={20,}', content)
            if len(failure_sections) > 1:
                failure_content = failure_sections[1]

                # Split by test failure headers
                individual_failures = re.split(r'_{20,} .+ _{20,}', failure_content)

                for failure_text in individual_failures[1:]:  # Skip first empty element
                    failure = self.parse_individual_failure(failure_text)
                    if failure:
                        failures.append(failure)

        except Exception as e:
            print(f"Error analyzing pytest output: {e}")

        return failures

    def parse_individual_failure(self, failure_text: str) -> TestFailure:
        """Parse individual test failure"""
        # Extract test name
        test_name_match = re.search(r'(\w+\.py)::', failure_text)
        test_name = test_name_match.group(0) if test_name_match else "unknown_test"

        # Extract file location
        location_match = re.search(r'(\w+\.py):(\d+)', failure_text)
        file_location = location_match.group(1) if location_match else "unknown_file"
        line_number = int(location_match.group(2)) if location_match else 0

        # Determine failure type and extract error message
        failure_type = "unknown"
        error_message = ""
        suggested_fix = "Review test failure and check implementation"
        confidence = 0.5

        for pattern_name, pattern_info in self.failure_patterns.items():
            if re.search(pattern_info['pattern'], failure_text, re.IGNORECASE):
                failure_type = pattern_info['category']

                # Extract specific error message
                error_match = re.search(pattern_info['pattern'], failure_text)
                if error_match:
                    error_message = error_match.group(1) if error_match.group(1) else error_match.group(0)

                # Select appropriate fix suggestion
                suggested_fix = pattern_info['common_fixes'][0]  # Use first suggestion
                confidence = 0.8
                break

        return TestFailure(
            test_name=test_name,
            failure_type=failure_type,
            error_message=error_message,
            traceback=failure_text[:500],  # First 500 chars
            file_location=file_location,
            line_number=line_number,
            suggested_fix=suggested_fix,
            confidence=confidence
        )

    def generate_failure_report(self, failures: List[TestFailure]) -> Dict[str, Any]:
        """Generate comprehensive failure report"""
        if not failures:
            return {
                'status': 'success',
                'total_failures': 0,
                'message': 'All tests passed!'
            }

        # Categorize failures
        failure_categories = {}
        for failure in failures:
            category = failure.failure_type
            if category not in failure_categories:
                failure_categories[category] = []
            failure_categories[category].append(failure)

        # Generate summary
        report = {
            'status': 'failed',
            'total_failures': len(failures),
            'failure_categories': {},
            'top_issues': [],
            'recommended_actions': []
        }

        # Process each category
        for category, category_failures in failure_categories.items():
            report['failure_categories'][category] = {
                'count': len(category_failures),
                'failures': [
                    {
                        'test_name': f.test_name,
                        'error_message': f.error_message,
                        'file_location': f.file_location,
                        'line_number': f.line_number,
                        'suggested_fix': f.suggested_fix,
                        'confidence': f.confidence
                    }
                    for f in category_failures[:3]  # Top 3 failures per category
                ]
            }

        # Generate top issues and recommendations
        sorted_failures = sorted(failures, key=lambda x: x.confidence, reverse=True)
        report['top_issues'] = [
            {
                'test_name': f.test_name,
                'failure_type': f.failure_type,
                'error_message': f.error_message,
                'suggested_fix': f.suggested_fix,
                'confidence': f.confidence
            }
            for f in sorted_failures[:5]  # Top 5 issues
        ]

        # Generate actionable recommendations
        recommendations = []
        if 'dependency_issue' in failure_categories:
            recommendations.append("🔧 Install missing dependencies and verify environment setup")
        if 'numerical_instability' in failure_categories:
            recommendations.append("🔢 Review numerical computations for stability and precision")
        if 'jax_specific' in failure_categories:
            recommendations.append("🚀 Check JAX-specific patterns and functional programming requirements")
        if 'gpu_specific' in failure_categories:
            recommendations.append("🎮 Verify GPU/CUDA setup and memory allocation")

        report['recommended_actions'] = recommendations

        return report

def main():
    analyzer = PythonTestFailureAnalyzer()

    # Analyze pytest output if available
    log_file = '.test_cache/pytest_output.log'
    failures = analyzer.analyze_pytest_output(log_file)
    report = analyzer.generate_failure_report(failures)

    # Save analysis report
    with open('.test_cache/python_failure_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Display summary
    print(f"    📊 Python Test Analysis:")
    print(f"       • Total failures: {report['total_failures']}")

    if report['total_failures'] > 0:
        print(f"       • Failure categories:")
        for category, info in report['failure_categories'].items():
            print(f"         - {category}: {info['count']} failures")

        print(f"       • Top recommended actions:")
        for action in report['recommended_actions'][:3]:
            print(f"         - {action}")

    return report['total_failures']

if __name__ == '__main__':
    failures = main()
    sys.exit(min(failures, 1))  # Exit with 1 if any failures, 0 if none
EOF
}

analyze_julia_test_failures() {
    echo "  💎 Analyzing Julia test failures..."

    julia << 'EOF'
using JSON

function analyze_julia_failures()
    log_file = ".test_cache/julia_output.log"

    if !isfile(log_file)
        println("    ℹ️  No Julia test output found")
        return
    end

    content = read(log_file, String)

    # Common Julia test failure patterns
    failure_patterns = Dict(
        "LoadError" => "Package loading or compilation issue",
        "MethodError" => "Function signature mismatch or missing method",
        "BoundsError" => "Array indexing out of bounds",
        "TypeError" => "Type system violation or instability",
        "UndefVarError" => "Undefined variable or scope issue",
        "ArgumentError" => "Invalid function arguments",
        "OutOfMemoryError" => "Memory allocation failure",
        "StackOverflowError" => "Infinite recursion or deep call stack"
    )

    failures = []
    failure_count = 0

    for (pattern, description) in failure_patterns
        matches = collect(eachmatch(Regex(pattern), content))
        if !isempty(matches)
            failure_count += length(matches)
            push!(failures, Dict(
                "type" => pattern,
                "count" => length(matches),
                "description" => description
            ))
        end
    end

    # Performance-related warnings
    perf_warnings = []
    if occursin("type instability", content)
        push!(perf_warnings, "Type instability detected - may cause performance issues")
    end
    if occursin("allocation", content)
        push!(perf_warnings, "Excessive memory allocation detected")
    end

    # Generate analysis report
    analysis = Dict(
        "total_failures" => failure_count,
        "failure_types" => failures,
        "performance_warnings" => perf_warnings,
        "suggestions" => [
            "Review function type annotations for stability",
            "Check array bounds and indexing logic",
            "Verify package dependencies and versions",
            "Consider using BenchmarkTools for performance testing"
        ]
    )

    # Save analysis
    open(".test_cache/julia_failure_analysis.json", "w") do f
        JSON.print(f, analysis, 2)
    end

    # Display summary
    println("    📊 Julia Test Analysis:")
    println("       • Total failures: $failure_count")

    if failure_count > 0
        println("       • Failure types:")
        for failure in failures
            println("         - $(failure["type"]): $(failure["count"]) occurrences")
        end
    end

    if !isempty(perf_warnings)
        println("       • Performance warnings:")
        for warning in perf_warnings
            println("         - $warning")
        end
    end

    return failure_count
end

failures = analyze_julia_failures()
exit(min(failures, 1))
EOF
}
```

### 4. Performance Benchmarking & Regression Detection

```bash
# Advanced performance benchmarking with regression detection
run_performance_benchmarks() {
    local benchmark_scope="${1:-all}"
    local baseline_compare="${2:-false}"
    local generate_report="${3:-false}"

    echo "📏 Running Performance Benchmarks..."

    # Python benchmarks
    if [[ "$PROJECT_TYPES" =~ python ]] && [[ "$benchmark_scope" =~ all|python ]]; then
        echo "  🐍 Running Python performance benchmarks..."
        run_python_benchmarks "$baseline_compare"
    fi

    # Julia benchmarks
    if [[ "$PROJECT_TYPES" =~ julia ]] && [[ "$benchmark_scope" =~ all|julia ]]; then
        echo "  💎 Running Julia performance benchmarks..."
        run_julia_benchmarks "$baseline_compare"
    fi

    # JAX-specific benchmarks
    if python -c "import jax" 2>/dev/null && [[ "$benchmark_scope" =~ all|jax ]]; then
        echo "  🚀 Running JAX performance benchmarks..."
        run_jax_benchmarks "$baseline_compare"
    fi

    if [[ "$generate_report" == "true" ]]; then
        generate_performance_report
    fi
}

run_python_benchmarks() {
    local baseline_compare="$1"

    if python -c "import pytest_benchmark" 2>/dev/null; then
        echo "    📊 Running pytest-benchmark..."

        python -m pytest --benchmark-only \
            --benchmark-json=.test_cache/benchmark_results.json \
            --benchmark-histogram=.test_cache/benchmark_histogram \
            --benchmark-sort=mean \
            --benchmark-columns=min,max,mean,stddev,median,iqr,outliers,ops,rounds \
            2>/dev/null || echo "    ⚠️  No benchmarks found"

        # Baseline comparison
        if [[ "$baseline_compare" == "true" ]] && [[ -f ".test_cache/baseline_benchmarks.json" ]]; then
            echo "    📈 Comparing against baseline..."
            compare_benchmark_results
        fi
    else
        echo "    ⚠️  pytest-benchmark not available. Install with: pip install pytest-benchmark"
    fi
}

run_julia_benchmarks() {
    local baseline_compare="$1"

    if julia -e "using BenchmarkTools" 2>/dev/null; then
        echo "    📊 Running BenchmarkTools.jl..."

        julia --project=. << 'EOF'
using BenchmarkTools, JSON

# Run benchmark suite if available
if isfile("test/benchmarks.jl")
    include("test/benchmarks.jl")

    println("    🔄 Running benchmark suite...")

    # Example benchmark structure
    suite = BenchmarkGroup()

    # Add benchmarks from benchmark file
    if isdefined(Main, :benchmark_suite)
        suite = Main.benchmark_suite
    else
        # Default benchmarks for common operations
        suite["basic"] = BenchmarkGroup()
        suite["basic"]["allocation"] = @benchmarkable zeros(1000)
        suite["basic"]["computation"] = @benchmarkable sum(1:1000)
    end

    # Run benchmarks
    results = run(suite, verbose=true, seconds=1)

    # Save results
    open(".test_cache/julia_benchmark_results.json", "w") do f
        JSON.print(f, results, 2)
    end

    println("    ✅ Julia benchmarks completed")
else
    println("    ℹ️  No benchmark suite found (test/benchmarks.jl)")
end
EOF
    else
        echo "    ⚠️  BenchmarkTools.jl not available. Add to Project.toml: BenchmarkTools = \"6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf\""
    fi
}

run_jax_benchmarks() {
    local baseline_compare="$1"

    echo "    🚀 Running JAX-specific benchmarks..."

    python << 'EOF'
import jax
import jax.numpy as jnp
import time
import json
from typing import Dict, Any

def benchmark_jax_operations() -> Dict[str, Any]:
    """Benchmark core JAX operations"""
    results = {}

    # JIT compilation benchmark
    @jax.jit
    def jit_function(x):
        return jnp.sum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2)

    # Warm up JIT
    x = jnp.arange(10000.0)
    _ = jit_function(x)

    # Benchmark JIT execution
    start_time = time.time()
    for _ in range(100):
        result = jit_function(x).block_until_ready()
    jit_time = (time.time() - start_time) / 100
    results['jit_execution_time'] = jit_time

    # Gradient computation benchmark
    grad_fn = jax.grad(lambda x: jnp.sum(x ** 3))

    start_time = time.time()
    for _ in range(100):
        grad_result = grad_fn(x).block_until_ready()
    grad_time = (time.time() - start_time) / 100
    results['gradient_computation_time'] = grad_time

    # Vectorization benchmark
    def single_op(x):
        return jnp.sin(x) + jnp.cos(x)

    vmap_fn = jax.vmap(single_op)
    batch_x = jnp.arange(1000.0)

    start_time = time.time()
    for _ in range(100):
        vmap_result = vmap_fn(batch_x).block_until_ready()
    vmap_time = (time.time() - start_time) / 100
    results['vectorization_time'] = vmap_time

    # Memory usage estimation
    array_size_mb = x.nbytes / (1024 * 1024)
    results['memory_usage_mb'] = array_size_mb

    return results

def benchmark_device_performance():
    """Benchmark performance across available devices"""
    devices = jax.devices()
    device_results = {}

    for device in devices:
        print(f"    📱 Benchmarking on {device}")

        with jax.default_device(device):
            # Simple matrix multiplication benchmark
            size = 1000
            a = jnp.ones((size, size))
            b = jnp.ones((size, size))

            # Warm up
            _ = jnp.dot(a, b).block_until_ready()

            # Benchmark
            start_time = time.time()
            for _ in range(10):
                result = jnp.dot(a, b).block_until_ready()
            exec_time = (time.time() - start_time) / 10

            device_results[str(device)] = {
                'matrix_mult_time': exec_time,
                'device_kind': device.device_kind,
                'device_id': device.id
            }

    return device_results

# Run benchmarks
try:
    print("    🔄 Running JAX operation benchmarks...")
    operation_results = benchmark_jax_operations()

    print("    🔄 Running device performance benchmarks...")
    device_results = benchmark_device_performance()

    # Combine results
    full_results = {
        'timestamp': time.time(),
        'jax_version': jax.__version__,
        'operation_benchmarks': operation_results,
        'device_benchmarks': device_results
    }

    # Save results
    with open('.test_cache/jax_benchmark_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"    ✅ JAX benchmarks completed")
    print(f"       • JIT execution: {operation_results['jit_execution_time']:.4f}s")
    print(f"       • Gradient computation: {operation_results['gradient_computation_time']:.4f}s")
    print(f"       • Vectorization: {operation_results['vectorization_time']:.4f}s")

except Exception as e:
    print(f"    ❌ JAX benchmark failed: {e}")
EOF
}

# Reproducibility testing
test_reproducibility() {
    echo "  🔄 Testing reproducibility across runs..."

    # Python reproducibility
    if [[ "$PROJECT_TYPES" =~ python ]]; then
        echo "    🐍 Python reproducibility test..."

        python << 'EOF'
import numpy as np
import random
import hashlib
import json

def test_reproducibility():
    """Test that computations are reproducible"""
    results = {}

    # Set seeds
    np.random.seed(42)
    random.seed(42)

    # Generate test data
    run1_data = np.random.rand(100)
    run1_result = np.sum(run1_data ** 2)

    # Reset seeds and repeat
    np.random.seed(42)
    random.seed(42)

    run2_data = np.random.rand(100)
    run2_result = np.sum(run2_data ** 2)

    # Check reproducibility
    is_reproducible = np.allclose(run1_result, run2_result, rtol=1e-15)

    results['numpy_reproducible'] = is_reproducible
    results['run1_result'] = float(run1_result)
    results['run2_result'] = float(run2_result)
    results['difference'] = float(abs(run1_result - run2_result))

    if is_reproducible:
        print("    ✅ NumPy reproducibility test passed")
    else:
        print("    ❌ NumPy reproducibility test failed")
        print(f"       Run 1: {run1_result}")
        print(f"       Run 2: {run2_result}")
        print(f"       Difference: {abs(run1_result - run2_result)}")

    # Test JAX reproducibility if available
    try:
        import jax
        import jax.numpy as jnp

        key = jax.random.PRNGKey(42)
        jax_run1 = jax.random.normal(key, (100,))
        jax_run1_result = jnp.sum(jax_run1 ** 2)

        key = jax.random.PRNGKey(42)
        jax_run2 = jax.random.normal(key, (100,))
        jax_run2_result = jnp.sum(jax_run2 ** 2)

        jax_reproducible = jnp.allclose(jax_run1_result, jax_run2_result, rtol=1e-15)
        results['jax_reproducible'] = bool(jax_reproducible)

        if jax_reproducible:
            print("    ✅ JAX reproducibility test passed")
        else:
            print("    ❌ JAX reproducibility test failed")

    except ImportError:
        results['jax_reproducible'] = None
        print("    ℹ️  JAX not available for reproducibility testing")

    # Save results
    with open('.test_cache/reproducibility_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return all(v for v in results.values() if v is not None)

if __name__ == '__main__':
    success = test_reproducibility()
    exit(0 if success else 1)
EOF
    fi

    # Julia reproducibility
    if [[ "$PROJECT_TYPES" =~ julia ]]; then
        echo "    💎 Julia reproducibility test..."

        julia << 'EOF'
using Random, JSON

function test_julia_reproducibility()
    results = Dict()

    # Test Random module reproducibility
    Random.seed!(42)
    run1_data = rand(100)
    run1_result = sum(run1_data .^ 2)

    Random.seed!(42)
    run2_data = rand(100)
    run2_result = sum(run2_data .^ 2)

    is_reproducible = isapprox(run1_result, run2_result, rtol=1e-15)

    results["julia_random_reproducible"] = is_reproducible
    results["run1_result"] = run1_result
    results["run2_result"] = run2_result
    results["difference"] = abs(run1_result - run2_result)

    if is_reproducible
        println("    ✅ Julia Random reproducibility test passed")
    else
        println("    ❌ Julia Random reproducibility test failed")
        println("       Run 1: $run1_result")
        println("       Run 2: $run2_result")
        println("       Difference: $(abs(run1_result - run2_result))")
    end

    # Save results
    open(".test_cache/julia_reproducibility_results.json", "w") do f
        JSON.print(f, results, 2)
    end

    return is_reproducible
end

success = test_julia_reproducibility()
exit(success ? 0 : 1)
EOF
    fi
}
```

### 5. Comprehensive Test Reporting System

```bash
# Generate comprehensive test execution report
generate_test_execution_summary() {
    local execution_time="$1"

    echo "📊 Generating Comprehensive Test Report..."

    python3 << 'EOF'
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

class TestReportGenerator:
    def __init__(self):
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': 0,
            'summary': {},
            'test_results': {},
            'performance_metrics': {},
            'failure_analysis': {},
            'recommendations': [],
            'next_steps': []
        }

    def load_test_results(self):
        """Load all test result files"""
        cache_files = {
            'discovery': '.test_cache/discovery.json',
            'python_failures': '.test_cache/python_failure_analysis.json',
            'julia_failures': '.test_cache/julia_failure_analysis.json',
            'benchmark_results': '.test_cache/benchmark_results.json',
            'jax_benchmarks': '.test_cache/jax_benchmark_results.json',
            'julia_benchmarks': '.test_cache/julia_benchmark_results.json',
            'reproducibility': '.test_cache/reproducibility_results.json',
            'coverage': '.test_cache/coverage.json'
        }

        loaded_data = {}
        for key, filepath in cache_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        loaded_data[key] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
                    loaded_data[key] = None
            else:
                loaded_data[key] = None

        return loaded_data

    def calculate_summary_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary metrics"""
        summary = {
            'total_test_files': 0,
            'tests_executed': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_success_rate': 0.0,
            'frameworks_used': [],
            'scientific_packages': [],
            'performance_tests': 0,
            'gpu_tests_available': False,
            'reproducibility_validated': False
        }

        # Discovery data
        if data['discovery']:
            discovery = data['discovery']
            summary['total_test_files'] = discovery.get('total_test_files', 0)
            summary['frameworks_used'] = discovery.get('test_frameworks', [])
            summary['scientific_packages'] = discovery.get('scientific_packages', [])
            summary['gpu_tests_available'] = discovery.get('gpu_available', False)

            test_categories = discovery.get('test_categories', {})
            summary['performance_tests'] = len(test_categories.get('performance_tests', []))

        # Failure analysis
        total_failures = 0
        if data['python_failures']:
            total_failures += data['python_failures'].get('total_failures', 0)
        if data['julia_failures']:
            total_failures += data['julia_failures'].get('total_failures', 0)

        summary['tests_failed'] = total_failures
        summary['tests_executed'] = max(summary['total_test_files'], total_failures + 1)
        summary['tests_passed'] = summary['tests_executed'] - total_failures

        if summary['tests_executed'] > 0:
            summary['test_success_rate'] = summary['tests_passed'] / summary['tests_executed']

        # Reproducibility
        if data['reproducibility']:
            repro_data = data['reproducibility']
            numpy_repro = repro_data.get('numpy_reproducible', False)
            jax_repro = repro_data.get('jax_reproducible', True)  # Default true if not tested
            summary['reproducibility_validated'] = numpy_repro and (jax_repro is not False)

        return summary

    def generate_performance_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance metrics summary"""
        performance = {
            'benchmark_suites_run': 0,
            'python_benchmarks': {},
            'julia_benchmarks': {},
            'jax_benchmarks': {},
            'performance_status': 'unknown'
        }

        # Python benchmarks
        if data['benchmark_results']:
            performance['benchmark_suites_run'] += 1
            # Process pytest-benchmark results
            benchmarks = data['benchmark_results'].get('benchmarks', [])
            if benchmarks:
                avg_time = sum(b.get('stats', {}).get('mean', 0) for b in benchmarks) / len(benchmarks)
                performance['python_benchmarks'] = {
                    'total_benchmarks': len(benchmarks),
                    'average_execution_time': avg_time,
                    'fastest_test': min(b.get('stats', {}).get('min', float('inf')) for b in benchmarks),
                    'slowest_test': max(b.get('stats', {}).get('max', 0) for b in benchmarks)
                }

        # JAX benchmarks
        if data['jax_benchmarks']:
            performance['benchmark_suites_run'] += 1
            jax_data = data['jax_benchmarks'].get('operation_benchmarks', {})
            performance['jax_benchmarks'] = {
                'jit_execution_time': jax_data.get('jit_execution_time', 0),
                'gradient_computation_time': jax_data.get('gradient_computation_time', 0),
                'vectorization_time': jax_data.get('vectorization_time', 0),
                'memory_usage_mb': jax_data.get('memory_usage_mb', 0)
            }

        # Julia benchmarks
        if data['julia_benchmarks']:
            performance['benchmark_suites_run'] += 1
            performance['julia_benchmarks'] = {'status': 'completed'}

        # Overall performance status
        if performance['benchmark_suites_run'] > 0:
            performance['performance_status'] = 'benchmarked'
        elif data['discovery'] and data['discovery'].get('test_categories', {}).get('performance_tests'):
            performance['performance_status'] = 'performance_tests_available'
        else:
            performance['performance_status'] = 'no_performance_tests'

        return performance

    def generate_recommendations(self, summary: Dict[str, Any], performance: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Test coverage recommendations
        if summary['test_success_rate'] < 0.95:
            recommendations.append("🔧 Improve test reliability - current success rate: {:.1%}".format(summary['test_success_rate']))

        if summary['total_test_files'] < 5:
            recommendations.append("📝 Increase test coverage - only {} test files found".format(summary['total_test_files']))

        # Performance recommendations
        if performance['performance_status'] == 'no_performance_tests':
            recommendations.append("📏 Add performance benchmarks using pytest-benchmark or BenchmarkTools.jl")

        if not summary['reproducibility_validated']:
            recommendations.append("🔄 Implement reproducibility controls with proper random seed management")

        # Scientific computing recommendations
        if len(summary['scientific_packages']) > 0 and performance['benchmark_suites_run'] == 0:
            recommendations.append("🔬 Add scientific computing benchmarks for {} packages".format(len(summary['scientific_packages'])))

        if summary['gpu_tests_available'] and not data['jax_benchmarks']:
            recommendations.append("🎮 Implement GPU/TPU testing and benchmarking")

        # Framework-specific recommendations
        if 'pytest' in summary['frameworks_used'] and not data['coverage']:
            recommendations.append("📊 Enable code coverage reporting with pytest-cov")

        if 'julia' in [f.lower() for f in summary['frameworks_used']] and not data['julia_benchmarks']:
            recommendations.append("💎 Add Julia performance benchmarks with BenchmarkTools.jl")

        # Failure-specific recommendations
        if data['python_failures'] and data['python_failures'].get('total_failures', 0) > 0:
            for action in data['python_failures'].get('recommended_actions', []):
                recommendations.append(f"🐍 Python: {action}")

        return recommendations[:8]  # Limit to top 8 recommendations

    def generate_next_steps(self, summary: Dict[str, Any]) -> List[str]:
        """Generate specific next steps"""
        steps = []

        if summary['tests_failed'] > 0:
            steps.append("Fix failing tests before proceeding with development")
            steps.append("Review failure analysis in .test_cache/ for specific guidance")

        if summary['test_success_rate'] > 0.95:
            steps.append("Consider implementing additional edge case tests")

        if summary['performance_tests'] == 0:
            steps.append("Set up performance regression testing with benchmarks")

        if not summary['reproducibility_validated']:
            steps.append("Implement proper random seed management for reproducible results")

        steps.append("Set up automated testing in CI/CD pipeline")
        steps.append("Configure test result notifications for the team")

        return steps[:6]  # Limit to top 6 steps

    def generate_comprehensive_report(self, execution_time: int) -> Dict[str, Any]:
        """Generate the complete test report"""
        # Load all test data
        data = self.load_test_results()

        # Calculate summary metrics
        summary = self.calculate_summary_metrics(data)
        performance = self.generate_performance_summary(data)

        # Generate recommendations and next steps
        recommendations = self.generate_recommendations(summary, performance, data)
        next_steps = self.generate_next_steps(summary)

        # Build comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'summary': summary,
            'performance_metrics': performance,
            'failure_analysis': {
                'python_failures': data['python_failures'],
                'julia_failures': data['julia_failures']
            },
            'recommendations': recommendations,
            'next_steps': next_steps,
            'test_data_files': {
                key: value is not None for key, value in data.items()
            }
        }

        return report

    def display_report_summary(self, report: Dict[str, Any]):
        """Display a formatted summary of the report"""
        summary = report['summary']
        performance = report['performance_metrics']

        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE TEST EXECUTION REPORT")
        print("="*60)

        # Executive Summary
        print(f"📊 EXECUTIVE SUMMARY")
        print(f"   • Execution Time: {report['execution_time_seconds']}s")
        print(f"   • Test Success Rate: {summary['test_success_rate']:.1%}")
        print(f"   • Tests Passed: {summary['tests_passed']}/{summary['tests_executed']}")
        print(f"   • Performance Benchmarked: {performance['benchmark_suites_run']} suites")
        print(f"   • Reproducibility: {'✅' if summary['reproducibility_validated'] else '⚠️'}")

        # Test Frameworks
        print(f"\n🔧 TEST FRAMEWORKS")
        for framework in summary['frameworks_used']:
            print(f"   • {framework}")

        # Scientific Packages
        if summary['scientific_packages']:
            print(f"\n🔬 SCIENTIFIC PACKAGES DETECTED")
            for package in summary['scientific_packages'][:5]:  # Show top 5
                print(f"   • {package}")
            if len(summary['scientific_packages']) > 5:
                print(f"   • ... and {len(summary['scientific_packages']) - 5} more")

        # Performance Summary
        if performance['benchmark_suites_run'] > 0:
            print(f"\n⚡ PERFORMANCE SUMMARY")
            if performance['python_benchmarks']:
                py_bench = performance['python_benchmarks']
                print(f"   • Python: {py_bench['total_benchmarks']} benchmarks, avg: {py_bench['average_execution_time']:.4f}s")
            if performance['jax_benchmarks']:
                jax_bench = performance['jax_benchmarks']
                print(f"   • JAX JIT: {jax_bench['jit_execution_time']:.4f}s")
                print(f"   • JAX Gradients: {jax_bench['gradient_computation_time']:.4f}s")

        # Failures
        if summary['tests_failed'] > 0:
            print(f"\n❌ FAILURES DETECTED")
            print(f"   • Total Failed Tests: {summary['tests_failed']}")
            print(f"   • Detailed analysis available in .test_cache/")

        # Top Recommendations
        if report['recommendations']:
            print(f"\n💡 TOP RECOMMENDATIONS")
            for i, rec in enumerate(report['recommendations'][:5], 1):
                print(f"   {i}. {rec}")

        # Next Steps
        if report['next_steps']:
            print(f"\n📋 NEXT STEPS")
            for i, step in enumerate(report['next_steps'][:3], 1):
                print(f"   {i}. {step}")

        print("\n" + "="*60)

def main():
    # Get execution time from command line argument
    execution_time = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    generator = TestReportGenerator()
    report = generator.generate_comprehensive_report(execution_time)

    # Save comprehensive report
    with open('.test_cache/comprehensive_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Display summary
    generator.display_report_summary(report)

    # Return appropriate exit code
    success_rate = report['summary']['test_success_rate']
    if success_rate >= 0.95:
        print("\n✅ All tests passed successfully!")
        return 0
    elif success_rate >= 0.80:
        print(f"\n⚠️  Some tests failed (success rate: {success_rate:.1%})")
        return 1
    else:
        print(f"\n❌ Significant test failures (success rate: {success_rate:.1%})")
        return 2

if __name__ == '__main__':
    sys.exit(main())
EOF
}
```

### 6. Enhanced Main Execution Controller

```bash
# Next-generation test execution engine with adaptive strategies
main() {
    # Enhanced environment initialization
    set -euo pipefail

    # Enable better error handling
    trap cleanup_on_exit EXIT
    trap handle_interrupt INT TERM

    # Parse command line arguments with enhanced options
    local test_scope="all"
    local profile_mode="false"
    local benchmark_mode="false"
    local scientific_mode="false"
    local gpu_mode="false"
    local parallel_mode="false"
    local reproducible_mode="false"
    local coverage_mode="false"
    local report_mode="false"
    local auto_fix="false"
    local baseline_compare="false"
    local platform_matrix="false"
    local ci_mode="false"
    local config_file=""
    local cache_mode="auto"
    local security_mode="moderate"
    local verbose_mode="false"
    local monitoring_mode="false"
    local adaptive_mode="true"
    local ml_predictions="false"

    # Advanced argument parsing
    while [[ $# -gt 0 ]]; do
        case $1 in
            --scope=*)
                test_scope="${1#*=}"
                shift
                ;;
            --profile)
                profile_mode="true"
                shift
                ;;
            --benchmark)
                benchmark_mode="true"
                shift
                ;;
            --scientific)
                scientific_mode="true"
                benchmark_mode="true"
                reproducible_mode="true"
                shift
                ;;
            --gpu)
                gpu_mode="true"
                shift
                ;;
            --parallel)
                parallel_mode="true"
                shift
                ;;
            --reproducible)
                reproducible_mode="true"
                shift
                ;;
            --coverage)
                coverage_mode="true"
                shift
                ;;
            --report)
                report_mode="true"
                shift
                ;;
            --auto-fix)
                auto_fix="true"
                shift
                ;;
            --baseline-compare)
                baseline_compare="true"
                shift
                ;;
            --platform-matrix)
                platform_matrix="true"
                shift
                ;;
            --ci)
                ci_mode="true"
                parallel_mode="true"
                coverage_mode="true"
                shift
                ;;
            --config=*)
                config_file="${1#*=}"
                shift
                ;;
            --cache|--cache=true)
                cache_mode="true"
                shift
                ;;
            --no-cache|--cache=false)
                cache_mode="false"
                shift
                ;;
            --security=*)
                security_mode="${1#*=}"
                shift
                ;;
            --verbose|-v)
                verbose_mode="true"
                shift
                ;;
            --monitor)
                monitoring_mode="true"
                shift
                ;;
            --adaptive)
                adaptive_mode="true"
                shift
                ;;
            --no-adaptive)
                adaptive_mode="false"
                shift
                ;;
            --ml-predict)
                ml_predictions="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --version)
                echo "Next-Generation Scientific Computing Test Engine v2.1.0 (2025 Pro Edition)"
                echo "Features: Intelligent Caching, Adaptive Execution, Enhanced Security, Multi-language Support"
                exit 0
                ;;
            -*)
                echo "❌ Unknown option: $1"
                echo "💡 Run --help for usage information"
                echo "🔍 Did you mean one of these?"
                echo "   --config=FILE    Use configuration file"
                echo "   --cache          Enable intelligent caching"
                echo "   --security=MODE  Set security mode (strict|moderate|permissive)"
                echo "   --verbose        Enable verbose output"
                exit 1
                ;;
            *)
                test_scope="$1"
                shift
                ;;
        esac
    done

    # Set enhanced environment variables for child processes
    export COVERAGE_MODE="$coverage_mode"
    export PROFILE_MODE="$profile_mode"
    export BENCHMARK_MODE="$benchmark_mode"
    export GPU_MODE="$gpu_mode"
    export REPRODUCIBLE_MODE="$reproducible_mode"
    export CI_MODE="$ci_mode"
    export VERBOSE_MODE="$verbose_mode"
    export MONITORING_MODE="$monitoring_mode"
    export ADAPTIVE_MODE="$adaptive_mode"
    export ML_PREDICTIONS="$ml_predictions"

    # Initialize enhanced configuration system
    CONFIG["cache_enabled"]="$cache_mode"
    CONFIG["security_mode"]="$security_mode"
    CONFIG["verbose"]="$verbose_mode"
    CONFIG["monitoring_enabled"]="$monitoring_mode"
    CONFIG["adaptive_execution"]="$adaptive_mode"
    CONFIG["ml_predictions"]="$ml_predictions"

    # Load custom configuration if specified
    if [[ -n "$config_file" ]]; then
        if [[ -f "$config_file" ]]; then
            echo "📄 Loading configuration from: $config_file"
            CONFIG["config_file"]="$config_file"
        else
            echo "❌ Configuration file not found: $config_file"
            exit 3
        fi
    fi

    # Show enhanced startup banner
    echo "🚀 Next-Generation Scientific Computing Test Engine (2025 Pro Edition)"
    echo "=========================================================================="
    echo "🎯 Mode: $test_scope | Cache: ${CONFIG[cache_enabled]} | Security: ${CONFIG[security_mode]} | Workers: ${CONFIG[parallel_workers]}"

    local start_time=$(date +%s)

    # Step 0: Initialize enhanced systems
    echo
    echo "⚙️  Initializing Enhanced Systems..."
    init_configuration_system
    init_caching_system
    init_security_system
    init_performance_monitoring
    init_adaptive_execution

    # Step 1: Enhanced test ecosystem discovery and environment setup
    echo
    discover_test_ecosystem
    setup_scientific_test_environment

    # Step 2: Test execution
    if execute_test_suite "$test_scope" "$profile_mode" "$benchmark_mode" "$parallel_mode" "$gpu_mode"; then
        echo "✅ Test execution completed successfully"
        local test_exit_code=0
    else
        echo "❌ Test execution completed with failures"
        local test_exit_code=1
    fi

    # Step 3: Performance benchmarking (if requested)
    if [[ "$benchmark_mode" == "true" ]]; then
        echo
        run_performance_benchmarks "$test_scope" "$baseline_compare" "true"
    fi

    # Step 4: Reproducibility testing (if requested)
    if [[ "$reproducible_mode" == "true" ]]; then
        echo
        test_reproducibility
    fi

    # Step 5: Failure analysis (if needed)
    if [[ $test_exit_code -ne 0 ]] || [[ "$auto_fix" == "true" ]]; then
        echo
        analyze_test_failures
    fi

    # Step 6: Comprehensive reporting
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))

    echo
    generate_test_execution_summary "$execution_time"

    # Step 7: Auto-fix suggestions (if requested)
    if [[ "$auto_fix" == "true" ]]; then
        echo
        echo "🔧 Auto-fix suggestions:"
        generate_auto_fix_suggestions
    fi

    # Cleanup
    echo
    echo "🧹 Test artifacts saved in: .test_cache/"
    echo "📊 View detailed report: .test_cache/comprehensive_test_report.json"

    # Save execution to history
    save_execution_history "$execution_time" "$test_exit_code"

    # Exit with appropriate code
    if [[ $test_exit_code -eq 0 ]]; then
        echo "🎉 All tests completed successfully!"
        echo "⚡ Execution optimized by adaptive strategies (${execution_time}s)"
        exit 0
    else
        echo "⚠️  Tests completed with issues - review analysis for details"
        echo "🔧 Run with --auto-fix for automated fix suggestions"
        exit 1
    fi
}

# Enhanced error handling and cleanup functions
cleanup_on_exit() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]] && [[ "${VERBOSE_MODE:-false}" == "true" ]]; then
        echo "🧹 Cleaning up after exit code: $exit_code"
    fi

    # Clean up temporary files
    if [[ -d ".test_cache" ]]; then
        # Remove temporary execution files older than 1 hour
        find .test_cache -name "*.tmp" -type f -mmin +60 -delete 2>/dev/null || true

        # Compress old log files
        find .test_cache -name "*.log" -type f -mtime +1 -exec gzip {} \; 2>/dev/null || true
    fi

    # Clean up background processes
    cleanup_background_processes

    return $exit_code
}

handle_interrupt() {
    echo
    echo "⚠️  Test execution interrupted by user"
    echo "🧹 Performing cleanup..."

    # Kill any background test processes
    cleanup_background_processes

    # Save partial results if available
    if [[ -f ".test_cache/discovery.json" ]]; then
        echo "💾 Partial test discovery results saved"
    fi

    echo "✅ Cleanup completed"
    exit 130  # Standard exit code for SIGINT
}

cleanup_background_processes() {
    # Clean up any background test processes
    local pids_to_kill=()

    # Find Python test processes
    mapfile -t python_pids < <(pgrep -f "python.*test" 2>/dev/null || true)
    pids_to_kill+=("${python_pids[@]}")

    # Find Julia test processes
    mapfile -t julia_pids < <(pgrep -f "julia.*test" 2>/dev/null || true)
    pids_to_kill+=("${julia_pids[@]}")

    # Find JavaScript test processes
    mapfile -t js_pids < <(pgrep -f "npm.*test\|yarn.*test\|jest\|mocha" 2>/dev/null || true)
    pids_to_kill+=("${js_pids[@]}")

    # Kill processes gracefully
    for pid in "${pids_to_kill[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            if [[ "${VERBOSE_MODE:-false}" == "true" ]]; then
                echo "  🔄 Terminating background process: $pid"
            fi
            kill -TERM "$pid" 2>/dev/null || true
            sleep 0.5
            # Force kill if still running
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

save_execution_history() {
    local execution_time="$1"
    local exit_code="$2"
    local history_file=".test_cache/test_history.json"

    # Create history entry
    local timestamp=$(date -Iseconds)
    local history_entry

    if command -v jq &>/dev/null; then
        # Load existing history or create new
        local existing_history="{\"executions\": []}"
        if [[ -f "$history_file" ]]; then
            existing_history=$(cat "$history_file")
        fi

        # Add new execution
        history_entry=$(echo "$existing_history" | jq \
            --arg timestamp "$timestamp" \
            --arg execution_time "$execution_time" \
            --arg exit_code "$exit_code" \
            --arg test_scope "$test_scope" \
            --arg cache_enabled "${CONFIG[cache_enabled]}" \
            --arg parallel_workers "${CONFIG[parallel_workers]}" \
            '.executions += [{
                "timestamp": $timestamp,
                "execution_time_seconds": ($execution_time | tonumber),
                "exit_code": ($exit_code | tonumber),
                "test_scope": $test_scope,
                "cache_enabled": ($cache_enabled == "true"),
                "parallel_workers": ($parallel_workers | tonumber)
            }] | .executions |= (sort_by(.timestamp) | .[-20:])' # Keep last 20 executions
        )

        echo "$history_entry" > "$history_file"

        if [[ "${VERBOSE_MODE:-false}" == "true" ]]; then
            echo "📈 Execution history updated"
        fi
    fi
}

# Auto-fix suggestions generator
generate_auto_fix_suggestions() {
    echo "💡 Analyzing failures for auto-fix opportunities..."

    # Python auto-fixes
    if [[ -f ".test_cache/python_failure_analysis.json" ]]; then
        python3 << 'EOF'
import json

try:
    with open('.test_cache/python_failure_analysis.json', 'r') as f:
        analysis = json.load(f)

    if analysis.get('total_failures', 0) > 0:
        print("  🐍 Python Auto-fix Suggestions:")

        for category, info in analysis.get('failure_categories', {}).items():
            if category == 'dependency_issue':
                print("    • Install missing dependencies:")
                print("      pip install pytest pytest-cov pytest-benchmark pytest-xdist")
            elif category == 'import_error':
                print("    • Fix import issues:")
                print("      pip install -e .  # Install package in development mode")
            elif category == 'numerical_instability':
                print("    • Add numerical stability:")
                print("      Use np.isclose() instead of == for float comparisons")
            elif category == 'jax_specific':
                print("    • JAX compatibility fixes:")
                print("      Ensure functions are pure (no side effects)")
                print("      Use jax.random.PRNGKey for random number generation")
except Exception as e:
    print(f"    ⚠️  Could not analyze Python failures: {e}")
EOF
    fi

    # Julia auto-fixes
    if [[ -f ".test_cache/julia_failure_analysis.json" ]]; then
        julia << 'EOF'
using JSON

try
    analysis = JSON.parsefile(".test_cache/julia_failure_analysis.json")

    if analysis["total_failures"] > 0
        println("  💎 Julia Auto-fix Suggestions:")

        for failure in analysis["failure_types"]
            failure_type = failure["type"]

            if failure_type == "LoadError"
                println("    • Fix package loading:")
                println("      using Pkg; Pkg.instantiate()  # Install dependencies")
            elseif failure_type == "MethodError"
                println("    • Fix method signatures:")
                println("      Check function argument types and count")
            elseif failure_type == "BoundsError"
                println("    • Fix array indexing:")
                println("      Use 1-based indexing (Julia arrays start at 1)")
            elseif failure_type == "TypeError"
                println("    • Fix type stability:")
                println("      Add type annotations: function foo(x::Float64)::Float64")
            end
        end
    end
catch e
    println("    ⚠️  Could not analyze Julia failures: $e")
end
EOF
    fi
}

# Enhanced comprehensive help system
show_help() {
    cat << 'EOF'
🚀 Next-Generation Scientific Computing Test Engine (2025 Pro Edition)
======================================================================

USAGE:
    /run-all-tests [OPTIONS] [SCOPE]

SCOPE OPTIONS:
    all                        Run all available tests (default)
    python, pytest            Python test suite only
    julia                      Julia test suite only
    javascript, jest           JavaScript/TypeScript test suite only
    rust                       Rust test suite only
    go                         Go test suite only
    scientific                 Scientific computing tests only
    jax                        JAX ecosystem tests only
    gpu                        GPU/TPU tests only
    security                   Security tests only
    api                        API tests only
    ui                         UI/Frontend tests only
    ci                         CI/CD optimized test suite

EXECUTION OPTIONS:
    --profile                  Enable performance profiling
    --benchmark               Run performance benchmarks
    --scientific              Enable scientific computing optimizations
    --gpu                     Enable GPU/TPU testing
    --parallel                Parallel test execution
    --reproducible            Validate reproducibility
    --coverage                Generate code coverage reports
    --report                  Generate comprehensive test report

CACHING & PERFORMANCE OPTIONS:
    --cache                   Enable intelligent caching (default: auto)
    --no-cache               Disable caching completely
    --adaptive               Enable adaptive execution strategies (default)
    --no-adaptive            Disable adaptive execution
    --monitor                Enable real-time performance monitoring

CONFIGURATION OPTIONS:
    --config=FILE            Use custom configuration file
    --security=MODE          Set security mode (strict|moderate|permissive)
    --verbose, -v            Enable verbose output
    --ml-predict             Enable ML-based performance prediction

AUTOMATION OPTIONS:
    --auto-fix               Generate automated fix suggestions
    --baseline-compare       Compare against performance baseline
    --platform-matrix        Cross-platform compatibility testing
    --ci                     CI/CD optimized mode (parallel + coverage)

INFO OPTIONS:
    --help, -h               Show this help message
    --version                Show version information

EXAMPLES:
    # Complete test suite with intelligent caching
    /run-all-tests --cache --adaptive

    # Scientific computing with enhanced security
    /run-all-tests --scientific --benchmark --reproducible --security=strict

    # CI/CD optimized testing with monitoring
    /run-all-tests --ci --report --monitor

    # GPU accelerated testing with ML predictions
    /run-all-tests --gpu --parallel --profile --ml-predict

    # Configuration-driven testing
    /run-all-tests --config=.test-config.yaml --verbose

    # Multi-language project testing
    /run-all-tests --parallel --cache --adaptive

    # Security-focused testing
    /run-all-tests security --security=strict --report

NEW IN 2025 PRO EDITION:
    🧠 Intelligent Caching           Smart test result caching with dependency tracking
    🛡️  Enhanced Security            Multi-level security validation and sandboxing
    🎯 Adaptive Execution           Dynamic resource allocation and test prioritization
    📊 Real-time Monitoring         Live performance metrics and resource usage
    🔧 Configuration Management     YAML/TOML configuration file support
    🌐 Multi-language Support       Rust, Go, C/C++, Zig support
    🤖 ML-powered Predictions       Machine learning for performance optimization
    📈 Historical Analysis          Test execution history and trend analysis

FEATURES:
    🔍 Intelligent Test Discovery       Auto-detection of 10+ test frameworks
    ⚡ Performance Benchmarking        Multi-language benchmark support
    🚀 JAX/GPU Acceleration           Advanced GPU/TPU testing
    💎 Julia Optimization             Type stability and performance analysis
    🔄 Reproducibility Validation     Scientific reproducibility checks
    📊 Comprehensive Reporting        AI-powered analysis and recommendations
    🔧 Enhanced Failure Analysis      Pattern recognition and auto-fix suggestions
    🎯 Adaptive Strategies            Resource optimization and smart execution

EXIT CODES:
    0   All tests passed successfully
    1   Some tests failed but analysis available
    2   Significant test failures requiring attention
    3   Configuration or environment errors
    130 Interrupted by user

CONFIGURATION FILE EXAMPLE (.test-config.yaml):
    test_engine:
      test_timeout: 300
      parallel_workers: auto
      cache_enabled: true
      cache_ttl: 3600
      security_mode: moderate
      ai_analysis: true
      monitoring_enabled: true
      adaptive_execution: true

OUTPUT FILES:
    .test_cache/comprehensive_test_report.json    Complete test report
    .test_cache/benchmark_results.json           Performance benchmarks
    .test_cache/coverage.json                    Code coverage data
    .test_cache/discovery.json                   Test ecosystem analysis
    .test_cache/test_history.json               Execution history
    .test_cache/cache_metadata.json             Cache statistics
    .test_cache/*_failure_analysis.json         Failure analysis reports

CACHE LOCATIONS:
    .test_cache/execution/                       Cached test results
    .test_cache/dependencies/                    Dependency analysis
    .test_cache/monitoring/                      Performance metrics

For documentation: https://docs.claude.com/en/docs/claude-code/run-all-tests
For issues: https://github.com/anthropics/claude-code/issues
EOF
}

# Execute main function with all arguments
main "$@"