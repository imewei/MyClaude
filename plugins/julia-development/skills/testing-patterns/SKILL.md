---
name: testing-patterns
description: Test.jl best practices, test organization, BenchmarkTools.jl, Aqua.jl quality checks, and JET.jl static analysis. Use for writing comprehensive test suites and ensuring code quality.
---

# Testing Patterns

Master testing Julia packages with Test.jl, Aqua.jl, and JET.jl.

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
