---
name: testing-patterns
description: Master Test.jl best practices, test organization, BenchmarkTools.jl, Aqua.jl quality checks, and JET.jl static analysis for comprehensive Julia testing. Use when writing or editing test files (test/runtests.jl, test/*.jl), creating @testset blocks for organized test suites, writing unit tests with @test and @test_throws, setting up BenchmarkTools.jl for performance benchmarks, implementing Aqua.jl quality checks (12 automated checks for package quality), adding JET.jl static analysis with @test_opt and @test_call, organizing tests by feature or module, ensuring test coverage, debugging test failures, or validating code quality. Essential for all Julia packages and ensuring robust, well-tested code.
---

# Testing Patterns

Master testing Julia packages with Test.jl, Aqua.jl, and JET.jl.

## When to use this skill

- Writing test files in test/ directory (test/runtests.jl)
- Creating organized test suites with @testset blocks
- Writing unit tests with @test, @test_throws, and @test_logs
- Setting up BenchmarkTools.jl for performance regression testing
- Implementing Aqua.jl quality checks (unbound_args, undefined_exports, etc.)
- Adding JET.jl static analysis for type inference validation
- Organizing tests by feature, module, or component
- Debugging test failures and assertion errors
- Ensuring comprehensive test coverage
- Validating package quality before release
- Testing edge cases and error conditions
- Writing property-based tests

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

## Quality Checks
```julia
using Aqua, JET

# Aqua.jl: Package quality
Aqua.test_all(MyPackage)

# JET.jl: Static analysis
@test_opt my_function(args)
@test_call my_function(args)
```

## Resources
- **Test.jl**: https://docs.julialang.org/en/v1/stdlib/Test/
- **Aqua.jl**: https://github.com/JuliaTesting/Aqua.jl
