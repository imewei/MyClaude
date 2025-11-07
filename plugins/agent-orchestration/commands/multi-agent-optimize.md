---
description: Coordinate specialized agents for comprehensive code optimization across scientific computing, web, and ML domains
allowed-tools: Read, Grep, Glob, Bash(find:*), Bash(git:*), Bash(python:*), Bash(julia:*), Bash(npm:*), Bash(cargo:*)
argument-hint: <target-path> [--mode=scan|analyze|apply] [--agents=AGENTS] [--focus=AREA] [--parallel]
color: magenta
required-plugins:
  - agent-orchestration (local)
  - optional: jax-implementation, hpc-computing, deep-learning
agents:
  primary:
    - multi-agent-orchestrator
    - systems-architect
  conditional:
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|numerical|simulation" OR argument "--focus=scientific"
      fallback: "Skip scientific optimizations (install hpc-computing plugin for NumPy/SciPy analysis)"
    - agent: jax-pro
      trigger: pattern "jax|flax|@jit|@vmap|@pmap|grad\\(|optax"
      fallback: "Skip JAX-specific optimizations (install jax-implementation plugin for GPU acceleration)"
    - agent: neural-architecture-engineer
      trigger: pattern "torch|pytorch|tensorflow|keras|neural.*network"
      fallback: "Skip ML model optimizations (install deep-learning plugin)"
    - agent: correlation-function-expert
      trigger: pattern "correlation|fft|spectral.*analysis|statistical.*physics"
      fallback: "Skip correlation analysis optimizations"
    - agent: simulation-expert
      trigger: pattern "lammps|gromacs|molecular.*dynamics|md.*simulation|ase"
      fallback: "Skip molecular dynamics optimizations"
    - agent: code-quality
      trigger: argument "--focus=quality" OR pattern "test|quality|lint"
      fallback: "Skip code quality analysis"
    - agent: research-intelligence
      trigger: argument "--focus=research" OR pattern "research|publication"
      fallback: "Skip research methodology analysis"
  orchestrated: true
  execution: parallel
execution-modes:
  scan: Quick bottleneck identification (2-5 min)
  analyze: Deep analysis with recommendations (10-30 min)
  apply: Apply optimizations automatically (with confirmation)
output-format: json-report + markdown-summary + code-patches + benchmarks
---

# Multi-Agent Code Optimization

## Quick Start

### Quick Scan (Recommended First Step)
```bash
/multi-agent-optimize src/ --mode=scan
```
**Output**: Priority list of quick wins and bottlenecks (2-5 minutes)

### Deep Analysis
```bash
/multi-agent-optimize src/simulation/ --mode=analyze --focus=scientific --parallel
```
**Output**: Comprehensive report with code patches (10-30 minutes)

### Apply Optimizations
```bash
/multi-agent-optimize src/ --mode=apply --quick-wins
```
**Output**: Applied patches with validation results

## Execution Flow

### Mode: scan (Quick Bottleneck Detection)

**When user invokes**: `/multi-agent-optimize <target-path> --mode=scan`

**Execute these steps**:

1. **Parse target path** from arguments:
   ```
   Target: $ARGUMENTS (first argument, required)
   If no path provided: prompt user "Which directory/file would you like to optimize?"
   ```

2. **Verify target exists**:
   ```bash
   # Check if path is valid
   if [ -e "${target_path}" ]; then
     echo "✓ Target found: ${target_path}"
   else
     echo "✗ Error: Path not found"
     exit 1
   fi
   ```

3. **Detect tech stack**:
   ```bash
   # Scan for common frameworks
   grep -r "import numpy" ${target_path} && echo "- NumPy detected"
   grep -r "import jax" ${target_path} && echo "- JAX detected"
   grep -r "import torch" ${target_path} && echo "- PyTorch detected"
   grep -r "using " ${target_path}/*.jl && echo "- Julia detected"

   # Check package files
   [ -f "requirements.txt" ] && echo "- Python project"
   [ -f "package.json" ] && echo "- Node.js project"
   ```

4. **Quick pattern analysis** (grep for common anti-patterns):
   ```bash
   # Look for optimization opportunities
   echo "Scanning for quick wins..."

   # Python loops that could be vectorized
   grep -n "for.*in range" ${target_path}/**/*.py | head -5

   # Repeated function calls (caching candidates)
   grep -n "def.*(" ${target_path}/**/*.py | sort | uniq -c | sort -rn | head -3

   # pandas.apply (vectorization opportunity)
   grep -n "\.apply(" ${target_path}/**/*.py
   ```

5. **Invoke analysis agents** (if available):
   ```
   Query available agents:
   - systems-architect (primary)
   - hpc-numerical-coordinator (if NumPy/SciPy detected)
   - jax-pro (if JAX detected)

   For each available agent, use Task tool with:
   - subagent_type: "<agent-name>"
   - prompt: "Analyze ${target_path} for optimization opportunities.
             Focus on quick wins (high impact, low effort).
             Provide specific line numbers and expected speedups."
   ```

6. **Generate scan report**:
   ```
   Create file: .optimization/$(basename ${target_path})-scan-$(date +%Y-%m-%d).json
   Format:
   {
     "target": "${target_path}",
     "stack_detected": [...],
     "quick_wins": [
       {
         "file": "...",
         "line": ...,
         "issue": "...",
         "fix": "...",
         "expected_speedup": "...",
         "confidence": "..."
       }
     ],
     "agents_used": [...],
     "agents_unavailable": [...]
   }

   Display: Formatted summary with emoji indicators (🚀 high impact, ⚡ medium, 💡 low)
   ```

**Example Execution**:
```
Optimization Scan: src/analytics/
Stack Detected: Python 3.11 + NumPy 1.24 + Pandas 2.0

Quick Wins (High Impact, Low Effort):
🚀 1. Vectorize for-loop in compute_correlation() [line 145]
     → Expected: 50x speedup | Effort: 10 min | Confidence: 95%
🚀 2. Replace pandas.apply() with vectorized ops [line 203]
     → Expected: 20x speedup | Effort: 15 min | Confidence: 90%
🚀 3. Add @lru_cache to expensive_computation() [line 87]
     → Expected: 5x speedup (repeated calls) | Effort: 2 min | Confidence: 99%

Medium Impact (5 more found) | Low Impact (7 more found)

Available Agents: 4/8
✅ multi-agent-orchestrator, systems-architect, hpc-numerical-coordinator
⚠️  jax-pro unavailable (install jax-implementation for GPU optimizations)

Run: /multi-agent-optimize src/analytics/ --mode=analyze
```

### Mode: analyze (Deep Multi-Agent Analysis)
1. **Discover agents**: Query available agents, check plugin dependencies
2. **Trigger conditional agents**: Based on stack detection and patterns
3. **Launch parallel execution**: All agents analyze simultaneously
4. **Collect results**: Stream findings to orchestrator as agents complete
5. **Synthesize findings**:
   - Deduplicate (same bottleneck found by multiple agents)
   - Resolve conflicts (speed vs memory tradeoffs)
   - Rank by composite score
6. **Generate artifacts**:
   - `.optimization/<target>-report-YYYY-MM-DD.md` (comprehensive report)
   - `.optimization/patches/*.patch` (code patches ready to apply)
   - `.optimization/benchmarks/before-after.json` (expected improvements)

**Agent Coordination Protocol**:
```
┌─ Orchestrator ─────────────────────────────────┐
│ 1. Stack Detection → Python + NumPy + JAX     │
│ 2. Agent Selection → systems-architect +       │
│                      hpc-numerical-coordinator │
│                      + jax-pro                 │
├────────────────────────────────────────────────┤
│ Parallel Execution:                            │
│  ├─ systems-architect ███████░░ 80%            │
│  ├─ hpc-numerical    █████████░ 90%            │
│  └─ jax-pro          ████████░░ 75%            │
├────────────────────────────────────────────────┤
│ Synthesis:                                     │
│  • Found 3 convergent recommendations          │
│  • Found 2 complementary optimizations         │
│  • Resolved 1 conflict (memory vs speed)       │
│  • Total expected improvement: 200-500x        │
└────────────────────────────────────────────────┘
```

### Mode: apply (Safe Optimization Application)
1. **Load recommendations**: From previous analyze run
2. **User review**: Display patches, expected improvements, risks
3. **Create backup**: Save original files to `.optimization/backups/`
4. **Apply patches**: Sequentially with validation between each
5. **Run validation**:
   - Execute existing tests (must pass)
   - Run benchmarks (verify speedup claims)
   - Check numerical accuracy (scientific code)
   - Verify model performance (ML code)
6. **Auto-rollback on failure**: Restore from backup if validation fails
7. **Commit on success**: Git commit with optimization metadata

**Validation Gates**:
- ✅ All tests pass (no regressions)
- ✅ Performance improved or unchanged
- ✅ Numerical accuracy within tolerance (for scientific code)
- ✅ Model metrics unchanged (for ML code)
- ✅ Memory usage not increased >20%

## Optimization Pattern Library

**Comprehensive Patterns**: [Optimization Patterns Guide](../../docs/optimization-patterns.md)

### Quick Reference
- **Vectorization**: NumPy/Pandas loops → vectorized operations (10-100x)
  - Before: `for i in range(n): result[i] = data[i] * 2`
  - After: `result = data * 2`

- **JIT Compilation**: JAX @jit, Numba @njit decorators (5-50x)
  - Pattern: Wrap pure functions with `@jax.jit` or `@numba.njit`

- **Caching**: @lru_cache, @functools.cache (2-10x for repeated calls)
  - Pattern: Cache expensive pure functions

- **Parallelization**: multiprocessing, concurrent.futures (Nx speedup, N=cores)
  - Pattern: Independent iterations → parallel execution

- **GPU Acceleration**: JAX/CuPy/PyTorch CUDA (10-1000x)
  - Pattern: NumPy operations → JAX equivalents with @jit

**Domain-Specific Patterns**:
- **Scientific Computing**: [scientific-patterns.md](../../docs/scientific-patterns.md)
- **ML Optimization**: [ml-optimization.md](../../docs/ml-optimization.md)
- **Web Performance**: [web-performance.md](../../docs/web-performance.md)

## Common Workflows

### Workflow 1: Scientific Computing Optimization
```bash
# 1. Quick scan for opportunities (5 min)
/multi-agent-optimize src/simulation/ --mode=scan

# 2. Deep analysis with scientific agents (20 min)
/multi-agent-optimize src/simulation/ --mode=analyze --focus=scientific --parallel

# 3. Review report: .optimization/simulation-report-YYYY-MM-DD.md

# 4. Apply top 3 recommendations (10 min)
/multi-agent-optimize src/simulation/ --mode=apply --top=3

# 5. Benchmark improvements
pytest benchmarks/ --benchmark-only
```

### Workflow 2: Iterative Optimization (Safest)
```bash
# Start with low-risk quick wins
/multi-agent-optimize src/ --mode=apply --quick-wins

# Verify improvements
pytest && python benchmarks/run.py

# Deep dive on remaining bottlenecks
/multi-agent-optimize src/ --mode=analyze --deep
```

### Workflow 3: Focused Optimization
```bash
# Target specific domain (e.g., only JAX optimizations)
/multi-agent-optimize src/ --mode=analyze --agents=jax-pro,hpc-numerical-coordinator

# Or focus on specific concern
/multi-agent-optimize src/ --mode=analyze --focus=quality
```

## Output Artifacts

All artifacts timestamped and versioned:

- **Scan Report**: `.optimization/<target>-scan-YYYY-MM-DD.json`
  - Quick wins list with impact estimates
  - Medium/low priority optimizations
  - Stack detection results

- **Analysis Report**: `.optimization/<target>-report-YYYY-MM-DD.md`
  - Comprehensive findings from all agents
  - Convergent recommendations (multiple agents agree)
  - Complementary strategies (layered optimizations)
  - Conflict resolutions with rationale
  - Risk assessment and mitigation strategies

- **Code Patches**: `.optimization/patches/<file>-optimized.patch`
  - Ready-to-apply git patches
  - Before/after code comparison
  - Expected improvement and confidence

- **Benchmarks**: `.optimization/benchmarks/<target>-before-after.json`
  - Performance measurements
  - Memory profiling
  - Speedup verification

- **Agent Logs**: `.optimization/logs/<agent>-YYYY-MM-DD.log`
  - Detailed agent analysis
  - Decision rationale
  - Warnings and caveats

## Success Metrics

Optimization is successful when:
- ✅ Performance improves by ≥20% (measured, not estimated)
- ✅ All existing tests still pass
- ✅ No regressions in functionality
- ✅ Code remains maintainable (complexity doesn't increase significantly)
- ✅ Documentation updated with optimization notes

**Detailed Metrics**: [Success Metrics Guide](../../docs/success-metrics.md)

## Best Practices

1. **Always start with scan**: Understand opportunities before deep analysis
2. **Use parallel execution**: Maximize agent efficiency with `--parallel`
3. **Validate incrementally**: Apply one optimization at a time for complex code
4. **Benchmark everything**: Verify claimed speedups with real measurements
5. **Version control**: Commit after each successful optimization
6. **Monitor production**: Watch for unexpected behavior post-deployment

**Full Best Practices**: [Best Practices Guide](../../docs/best-practices.md)

## Troubleshooting

**Issue**: "Agent X not available" warning
**Solution**: Install required plugin or proceed with available agents
```bash
# Check plugin installation
ls ~/.claude/plugins/jax-implementation/

# Install if missing (example)
claude-code plugin install jax-implementation
```

**Issue**: Validation fails after optimization
**Solution**: Automatic rollback engaged, review validation logs
```bash
cat .optimization/logs/validation-YYYY-MM-DD.log
```

**Issue**: Speedup claims not realized in practice
**Solution**: Profile with realistic data, check for I/O bottlenecks
```bash
python -m cProfile -o profile.stats src/main.py
```

**More troubleshooting**: [Troubleshooting Guide](../../docs/troubleshooting.md)

## Examples with Real Metrics

- [Optimizing MD Simulation (200x speedup)](../../docs/examples/md-simulation-optimization.md)
- [JAX Training Pipeline (50x speedup)](../../docs/examples/jax-training-optimization.md)
- [API Performance Enhancement (10x throughput)](../../docs/examples/api-performance-optimization.md)

## Additional Resources

- **Patterns**: [Complete Optimization Patterns](../../docs/optimization-patterns.md)
- **Scientific**: [NumPy/SciPy/JAX Patterns](../../docs/scientific-patterns.md)
- **ML**: [PyTorch/TensorFlow Optimization](../../docs/ml-optimization.md)
- **Tools**: [Profiling & Benchmarking](../../docs/profiling-tools.md)
- **Theory**: [Performance Engineering Guide](../../docs/performance-engineering.md)

Target: $ARGUMENTS
