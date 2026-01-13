---
name: testing-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Testing
description: Master Test.jl, Aqua.jl quality checks, and JET.jl static analysis for Julia testing. Use when writing unit tests, organizing test suites, or validating package quality.
---

# Julia Testing Patterns

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

## Checklist

- [ ] Unit tests in test/runtests.jl
- [ ] @testset blocks for organization
- [ ] Aqua.jl quality checks pass
- [ ] JET.jl type inference validated
- [ ] Edge cases and errors tested

---

**Version**: 1.0.5
