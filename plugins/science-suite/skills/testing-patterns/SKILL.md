---
name: testing-patterns
maturity: "5-Expert"
specialization: Julia Testing
description: Master Test.jl, Aqua.jl quality checks, and JET.jl static analysis for Julia testing. Use when writing unit tests, organizing test suites, or validating package quality.
---

# Julia Testing Patterns

## Expert Agent

For Julia testing with Test.jl, Aqua.jl, and JET.jl, delegate to:

- **`julia-pro`**: Julia package quality, testing, and static analysis.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

Test.jl, Aqua.jl, and JET.jl for comprehensive testing.

---

## Test Organization

```julia
using Test, MyPackage

@testset "MyPackage.jl" begin
    @testset "Feature 1" begin
        @test compute(5) == 25
        @test_throws ArgumentError compute(-1)
    end

    @testset "Feature 2" begin
        result = process_data([1,2,3])
        @test length(result) == 3
        @test all(x -> x > 0, result)
    end
end
```

---

## Quality Checks

```julia
using Aqua, JET

# Aqua.jl: Package quality
Aqua.test_all(MyPackage)

# JET.jl: Static analysis
@test_opt my_function(args)
@test_call my_function(args)
```

---

## Advanced Test Patterns

```julia
using Test

# Approximate floating-point comparison
@test result ≈ expected atol=1e-10

# Test that output matches a pattern
@test occursin("converged", log_output)

# Test with temporary files
mktempdir() do dir
    path = joinpath(dir, "output.csv")
    export_data(data, path)
    @test isfile(path)
    @test filesize(path) > 0
end

# Parameterized tests
@testset "solver=$solver" for solver in [Tsit5(), Vern7(), Rodas5()]
    sol = solve(prob, solver)
    @test sol.retcode == ReturnCode.Success
end
```

## Aqua.jl Quality Checks

```julia
using Aqua

# Run all quality checks
Aqua.test_all(MyPackage)

# Individual checks for targeted debugging
Aqua.test_ambiguities(MyPackage)       # No method ambiguities
Aqua.test_unbound_args(MyPackage)      # No unbound type parameters
Aqua.test_undefined_exports(MyPackage) # All exports are defined
Aqua.test_stale_deps(MyPackage)        # No unused dependencies
Aqua.test_deps_compat(MyPackage)       # All deps have compat entries
Aqua.test_piracies(MyPackage)          # No type piracy
```

## JET.jl Static Analysis

```julia
using JET

# Type stability analysis
@test_opt target_modules=(MyPackage,) MyPackage.compute(data)

# Call analysis (detect runtime errors statically)
@test_call target_modules=(MyPackage,) MyPackage.process(input)

# Analyze entire module
report = report_package(MyPackage)
@test length(JET.get_reports(report)) == 0
```

## CI Integration (GitHub Actions)

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        julia-version: ['1.10', '1.11']
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```

## Test Organization

| Pattern | When to Use |
|---------|-------------|
| `@testset` nesting | Group related assertions by feature |
| `@testset for` | Parameterized tests across inputs/solvers |
| Separate test files | Large packages with distinct modules |
| `test/runtests.jl` includes | `include("test_module_a.jl")` for modularity |

## Checklist

- [ ] Unit tests in test/runtests.jl with `@testset` blocks
- [ ] Floating-point comparisons use `≈` with explicit tolerance
- [ ] Parameterized tests cover multiple solvers/inputs
- [ ] Aqua.jl quality checks pass (ambiguities, piracy, stale deps)
- [ ] JET.jl static analysis reports zero type instabilities
- [ ] CI matrix tests against supported Julia versions
- [ ] Edge cases and error paths tested with `@test_throws`

---

**Version**: 1.0.5
