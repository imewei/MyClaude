# Metrics Guide

Complete reference for understanding validation metrics.

## Performance Metrics

### Execution Time
**What it measures**: Time to complete a validation scenario
**Unit**: Seconds
**Target**: Varies by scenario (15-45 minutes typical)
**Good/Bad**:
- Good: ≤ baseline + 10%
- Bad: > baseline + 25%

**Interpretation**:
- Compare to baseline
- Look for 2x+ improvements in optimization scenarios
- Consistent execution time = stable performance

### Memory Usage
**What it measures**: RAM consumed during validation
**Unit**: Megabytes (MB)
**Target**: < 2GB for most scenarios
**Good/Bad**:
- Good: Steady, no growth over time
- Bad: Linear growth (memory leak)

**Interpretation**:
- Peak memory vs average
- Large spike = inefficient operation
- Growing over time = memory leak

### CPU Utilization
**What it measures**: Processor usage percentage
**Unit**: Percentage (0-100%)
**Target**: 50-80% average
**Good/Bad**:
- Good: 50-80% (efficient use)
- Bad: >95% (bottleneck) or <20% (idle)

**Interpretation**:
- High + fast = CPU-bound (good)
- High + slow = thrashing (bad)
- Low = I/O bound or idle

### Disk I/O
**What it measures**: Disk read/write operations
**Unit**: Megabytes (MB)
**Target**: Varies by operation

**Metrics**:
- `disk_read_mb`: Data read from disk
- `disk_write_mb`: Data written to disk

**Interpretation**:
- High read = file scanning
- High write = code generation
- Compare to baseline for regressions

### Cache Hit Rate
**What it measures**: Percentage of cache hits vs misses
**Unit**: Percentage (0-100%)
**Target**: ≥75%
**Good/Bad**:
- Good: >75% (efficient caching)
- Bad: <50% (poor cache utilization)

**Interpretation**:
- High = efficient reuse
- Low = cache misses or cold cache
- Improving over time = warming up

## Quality Metrics

### Overall Quality Score
**What it measures**: Composite code quality score
**Unit**: 0-100
**Target**: ≥70
**Rating Scale**:
- 90-100: Excellent
- 70-89: Good
- 50-69: Fair
- <50: Poor

**Components** (weighted):
- Complexity: 20%
- Maintainability: 20%
- Test Coverage: 20%
- Documentation: 15%
- Security: 15%
- Style: 10%

**Interpretation**:
- Compare before/after
- Target ≥20% improvement
- <50 = needs immediate attention

### Complexity Score
**What it measures**: Code complexity analysis
**Unit**: 0-100 (higher = simpler)
**Target**: ≥75

**Based on**:
- Cyclomatic complexity
- Nesting depth
- Function length

**Interpretation**:
- 90-100: Simple, maintainable
- 70-89: Moderate complexity
- <70: High complexity, refactor recommended

### Maintainability Index
**What it measures**: How easy code is to maintain
**Unit**: 0-100
**Target**: ≥65
**Rating**:
- 85-100: Highly maintainable
- 65-84: Moderately maintainable
- <65: Difficult to maintain

**Factors**:
- Complexity
- Documentation
- Code style
- Duplication

### Test Coverage
**What it measures**: Percentage of code covered by tests
**Unit**: Percentage (0-100%)
**Target**: ≥80%
**Industry Standards**:
- 80-100%: Excellent
- 60-79%: Good
- 40-59%: Fair
- <40%: Poor

**Types**:
- Line coverage
- Branch coverage
- Function coverage

**Interpretation**:
- ≥80% = well-tested
- <60% = risky for refactoring
- 100% not always necessary

### Documentation Coverage
**What it measures**: Percentage of functions with docstrings
**Unit**: Percentage (0-100%)
**Target**: ≥70%
**Rating**:
- 85-100%: Well documented
- 70-84%: Adequately documented
- <70%: Under-documented

**Counts**:
- Public functions
- Classes
- Modules

**Interpretation**:
- API code needs ≥90%
- Internal code ≥70% acceptable
- Generated code may be lower

### Security Score
**What it measures**: Absence of security vulnerabilities
**Unit**: 0-100 (higher = more secure)
**Target**: ≥80

**Detects**:
- Use of eval/exec
- SQL injection risks
- Unsafe deserialization
- Shell injection
- Hardcoded secrets

**Severity Levels**:
- Critical: -25 points each
- High: -10 points each
- Medium: -5 points each
- Low: -2 points each

**Interpretation**:
- 100 = No issues found
- 80-99 = Minor issues
- <80 = Security review needed
- <60 = Critical issues present

### Code Smells
**What it measures**: Number of code quality issues
**Unit**: Count
**Target**: <10 per 1000 LOC

**Common Smells**:
- Long functions (>50 lines)
- Long lines (>120 chars)
- Deep nesting (>4 levels)
- TODO comments
- Dead code
- Duplicate code

**Interpretation**:
- 0-10: Clean code
- 11-50: Normal
- >50: Needs refactoring

### Duplication Percentage
**What it measures**: Percentage of duplicated code
**Unit**: Percentage (0-100%)
**Target**: <5%
**Rating**:
- 0-3%: Excellent
- 3-5%: Good
- 5-10%: Fair
- >10%: Significant duplication

**Interpretation**:
- Some duplication is acceptable
- >10% = extract to functions/classes
- >20% = major refactoring needed

## Improvement Metrics

### Quality Improvement Percentage
**What it measures**: Change in quality score from baseline
**Unit**: Percentage
**Target**: ≥20%
**Calculation**: `(current - baseline) / baseline * 100`

**Interpretation**:
- Positive = improvement
- ≥20% = successful optimization
- <0% = regression (investigate!)

### Performance Improvement
**What it measures**: Speed improvement from baseline
**Unit**: Multiplier (e.g., 2x, 3x)
**Target**: ≥2x for optimization scenarios
**Calculation**: `baseline_time / current_time`

**Examples**:
- 2x = twice as fast
- 0.5x = half as fast (regression!)

**Interpretation**:
- ≥2x = excellent optimization
- 1.5x-2x = good improvement
- 1.0x-1.5x = minor improvement
- <1.0x = regression

### Complexity Reduction
**What it measures**: Decrease in code complexity
**Unit**: Percentage
**Target**: ≥15%

**Achieved through**:
- Simplifying logic
- Extracting functions
- Reducing nesting

**Interpretation**:
- ≥20% = significant simplification
- 10-20% = moderate improvement
- <10% = minor changes

## Success Metrics

### Commands Successful/Failed
**What it measures**: Number of commands that succeeded/failed
**Unit**: Count
**Target**: 100% successful

**Interpretation**:
- All successful = validation passed
- Any failed = investigate logs
- >10% failed = systemic issue

### Regression Detection
**What it measures**: Number of performance regressions
**Unit**: Count
**Target**: 0

**Severity**:
- Critical: >50% degradation
- High: >25% degradation
- Medium: >10% degradation
- Low: >5% degradation

**Interpretation**:
- 0 = no regressions (good!)
- 1-2 low/medium = acceptable
- Any high/critical = block deployment

### Test Pass Rate
**What it measures**: Percentage of tests that passed
**Unit**: Percentage (0-100%)
**Target**: 100%
**Acceptable**: ≥95%

**Interpretation**:
- 100% = all tests pass (ideal)
- 95-99% = investigate failures
- <95% = significant issues

## Metric Relationships

### Quality ↔ Complexity
- Inversely related
- Simpler code = higher quality
- Reduce complexity → improve quality

### Coverage ↔ Confidence
- Direct relationship
- Higher coverage = more confidence
- But 100% coverage ≠ bug-free

### Performance ↔ Memory
- Trade-off relationship
- Caching improves performance but uses memory
- Balance based on constraints

### Documentation ↔ Maintainability
- Strong positive correlation
- Better docs = easier maintenance
- Invest in docs for complex code

## Using Metrics for Decisions

### When to Refactor
- Complexity score <70
- Maintainability <65
- Duplication >10%

### When to Optimize
- Performance <2x baseline
- Memory usage growing
- CPU utilization >95%

### When to Test More
- Coverage <80%
- Critical code <95%
- Recent bugs in area

### When to Document
- Documentation <70%
- Complex code (complexity >10)
- Public APIs <90%

## Metric Thresholds

### Production Readiness
- Overall quality: ≥70
- Test coverage: ≥80%
- Security score: ≥85
- Documentation: ≥70%
- No critical regressions

### Excellent Code
- Overall quality: ≥85
- Test coverage: ≥90%
- Security score: ≥95
- Documentation: ≥85%
- Complexity: ≤5 average

### Needs Improvement
- Overall quality: <60
- Test coverage: <60%
- Security score: <70
- Documentation: <50%
- Complexity: >10 average

## Advanced Analysis

### Trend Analysis
Track metrics over time:
- Improving = positive trajectory
- Stable = consistent quality
- Degrading = investigate immediately

### Correlation Analysis
Find relationships:
- High complexity + low coverage = risky
- Low documentation + high complexity = difficult
- High duplication + many smells = needs refactoring

### Anomaly Detection
Watch for:
- Sudden metric changes
- Outlier projects
- Inconsistent patterns

## Troubleshooting Metrics

### Unexpectedly Low Scores
1. Check baseline validity
2. Verify measurement tools
3. Review code changes
4. Compare to similar projects

### Metrics Not Collected
1. Check tool availability (pytest, coverage, etc.)
2. Verify project structure
3. Review logs for errors
4. Run metric collection manually

### Inconsistent Metrics
1. Cache invalidation issue?
2. Non-deterministic code?
3. External dependencies?
4. Measurement timing?