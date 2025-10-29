# /julia-optimize - Performance Profiling and Optimization

**Priority**: 2
**Agent**: julia-pro
**Description**: Profile Julia code and provide optimization recommendations. Analyzes type stability, memory allocations, identifies bottlenecks, and suggests parallelization strategies.

## Usage
```
/julia-optimize "<file_path>"
```

## Analysis Steps
1. Type stability analysis with @code_warntype
2. Memory allocation profiling
3. Execution profiling with @profview
4. Identify bottlenecks and hot paths

## Output Format
- Ranked list of optimization opportunities
- Type instabilities with suggested fixes
- Allocation hotspots with reduction strategies
- Parallelization opportunities (threads, distributed, GPU)
- Algorithm improvement suggestions
- Before/after performance estimates

## Example
```
/julia-optimize "src/slow_function.jl"
```

Provides actionable optimization recommendations.
