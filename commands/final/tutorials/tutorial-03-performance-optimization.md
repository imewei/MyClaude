# Tutorial 03: Performance Optimization

**Duration**: 60 minutes | **Level**: Intermediate | **Prerequisites**: Tutorials 01-02

---

## Learning Objectives

By the end of this tutorial, you'll be able to:
- Profile applications to identify performance bottlenecks
- Optimize algorithms and data structures
- Fix memory leaks and optimize memory usage
- Benchmark and validate performance improvements
- Set up performance regression testing

---

## Overview

Performance optimization is critical for production applications. This tutorial teaches systematic performance improvement using the command executor framework's optimization capabilities.

**What You'll Build**:
- Profile and optimize a slow web API
- Improve response time from 2000ms to 200ms (10x faster)
- Reduce memory usage by 60%
- Set up automated performance monitoring

---

## Setup

### Prerequisites Check
```bash
# Verify system is ready
/check-code-quality --version
/optimize --help
/run-all-tests --help
```

### Sample Project
```bash
# Clone slow API example
cd ~/projects
git clone https://github.com/claude-code/slow-api-example
cd slow-api-example

# Install dependencies
pip install -r requirements.txt

# Verify it's slow
python -m pytest tests/test_performance.py --benchmark
# Expected: ~2000ms per request ❌
```

---

## Part 1: Performance Profiling (15 minutes)

### Step 1: Baseline Performance Measurement

```bash
# Profile the application
/optimize --profile --category=all src/

# Expected output:
# ⚡ Performance Analysis
# ══════════════════════════════════════════════════
#
# 🔍 Bottlenecks Detected:
#   1. database.py:45 - query_users() - 1500ms (75% of time)
#   2. api.py:23 - serialize_response() - 400ms (20% of time)
#   3. cache.py:12 - get_from_cache() - 100ms (5% of time)
#
# 💡 Optimization Opportunities:
#   - N+1 query problem in query_users()
#   - Inefficient JSON serialization
#   - Missing cache indexes
```

**What Just Happened?**
- The `--profile` flag runs performance profiling
- Identifies functions taking the most time
- Provides specific line numbers for investigation

### Step 2: Analyze Memory Usage

```bash
# Memory profiling
/optimize --profile --category=memory src/

# Expected output:
# 💾 Memory Analysis
# ══════════════════════════════════════════════════
#
# High Memory Usage:
#   - data_processor.py:67 - Loads entire dataset into memory (1.2GB)
#   - cache.py:89 - Unbounded cache growth (500MB+)
#
# Memory Leaks Detected:
#   - websocket.py:34 - Unclosed connections
```

### Step 3: Understand the Issues

**Key Findings**:
1. **N+1 Query Problem**: Making 1000 database queries instead of 1
2. **Inefficient Serialization**: Using naive JSON encoding
3. **Memory Bloat**: Loading unnecessary data
4. **Cache Issues**: No expiration policy

---

## Part 2: Algorithm Optimization (15 minutes)

### Step 4: Fix the N+1 Query Problem

**Before** (database.py:45):
```python
def get_users_with_posts():
    users = User.query.all()  # 1 query
    result = []
    for user in users:
        posts = Post.query.filter_by(user_id=user.id).all()  # N queries!
        result.append({"user": user, "posts": posts})
    return result
```

**Fix with AI Assistance**:
```bash
# Let AI optimize the database query
/optimize --implement --category=algorithm database.py

# AI automatically rewrites to:
```

**After** (database.py:45):
```python
def get_users_with_posts():
    # Single query with JOIN - O(1) database calls
    users = User.query.options(
        joinedload(User.posts)
    ).all()
    return [{"user": user, "posts": user.posts} for user in users]
```

**Impact**: 1000 queries → 1 query = **1000x database improvement**

### Step 5: Optimize Data Structures

**Before** (api.py:78):
```python
def find_user_by_email(email):
    users = get_all_users()  # Returns list of 100,000 users
    for user in users:  # O(n) search
        if user.email == email:
            return user
    return None
```

**Fix**:
```bash
/optimize --implement --category=algorithm api.py
```

**After**:
```python
def find_user_by_email(email):
    # Use database index - O(1) lookup
    return User.query.filter_by(email=email).first()
```

**Impact**: O(n) → O(1) = **100,000x improvement for lookups**

### Step 6: Optimize JSON Serialization

```bash
# Optimize serialization code
/optimize --implement --category=io api.py

# AI automatically adds:
# - orjson (faster JSON library)
# - Schema validation caching
# - Lazy loading for large fields
```

**Impact**: 400ms → 40ms = **10x serialization improvement**

---

## Part 3: Memory Optimization (15 minutes)

### Step 7: Fix Memory Bloat

**Before** (data_processor.py:67):
```python
def process_data():
    data = load_entire_dataset()  # Loads 1.2GB into memory!
    results = []
    for item in data:
        results.append(process_item(item))
    return results
```

**Fix**:
```bash
/optimize --implement --category=memory data_processor.py
```

**After**:
```python
def process_data():
    # Stream processing - only loads one item at a time
    for item in stream_dataset():
        yield process_item(item)
```

**Impact**: 1.2GB → 10MB memory usage = **120x memory reduction**

### Step 8: Implement Cache Expiration

**Before** (cache.py:89):
```python
cache = {}  # Unbounded cache, grows forever

def get_cached(key):
    if key in cache:
        return cache[key]
    value = expensive_operation(key)
    cache[key] = value  # Never expires!
    return value
```

**Fix**:
```bash
/optimize --implement --category=memory cache.py
```

**After**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)  # Bounded cache with LRU eviction
def get_cached(key):
    return expensive_operation(key)
```

**Impact**: Prevents unbounded memory growth

---

## Part 4: Benchmarking and Validation (15 minutes)

### Step 9: Run Performance Tests

```bash
# Benchmark before and after
/run-all-tests --benchmark --scientific

# Expected output:
# 📊 Performance Benchmarks
# ══════════════════════════════════════════════════
#
# API Response Time:
#   Before: 2000ms  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   After:   200ms  ━━━  (10x faster) ✅
#
# Memory Usage:
#   Before: 1.2GB   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   After:  480MB   ━━━━━━━━━━━  (60% reduction) ✅
#
# Database Queries:
#   Before: 1001 queries
#   After:  1 query     (1000x reduction) ✅
```

### Step 10: Performance Regression Testing

```bash
# Set up automated performance monitoring
/ci-setup --platform=github --monitoring --performance-gates

# Creates .github/workflows/performance.yml:
```

**Generated Workflow**:
```yaml
name: Performance Monitoring

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: /run-all-tests --benchmark --report

      - name: Check performance regression
        run: |
          # Fail if response time > 250ms
          # Fail if memory usage > 600MB
          /double-check --validate performance
```

### Step 11: Create Performance Dashboard

```bash
# Generate performance report
/optimize --report=html --format=html src/

# Creates performance_report.html with:
# - Response time trends
# - Memory usage graphs
# - Query count tracking
# - Bottleneck heatmap
```

---

## Key Concepts Explained

### 1. Performance Profiling
**What**: Measuring where your code spends time
**Why**: Can't optimize what you don't measure
**How**: Use `--profile` flag to identify bottlenecks

### 2. Big O Notation
**O(1)**: Constant time (best)
**O(log n)**: Logarithmic (excellent)
**O(n)**: Linear (acceptable)
**O(n²)**: Quadratic (problematic for large n)

### 3. N+1 Query Problem
**Problem**: Making N+1 database queries instead of 1
**Solution**: Use JOINs or eager loading
**Impact**: Can be 100x-1000x slower

### 4. Memory Management
**Stack vs Heap**: Understand memory allocation
**Memory Leaks**: Resources not released
**Streaming**: Process data incrementally

### 5. Caching Strategies
**LRU**: Least Recently Used eviction
**TTL**: Time To Live expiration
**Bounded Cache**: Prevent unbounded growth

---

## Practice Projects

### Project 1: Optimize Database Queries
**Scenario**: API with 50 database queries per request
**Goal**: Reduce to 5 queries or less
**Time**: 20 minutes

```bash
cd ~/projects/db-heavy-api
/optimize --profile --category=algorithm .
/optimize --implement --category=algorithm .
/run-all-tests --benchmark
```

### Project 2: Fix Memory Leak
**Scenario**: Application memory grows from 100MB to 2GB over 1 hour
**Goal**: Keep memory under 200MB
**Time**: 15 minutes

```bash
cd ~/projects/memory-leak-app
/optimize --profile --category=memory .
/debug --issue=memory --monitor
/optimize --implement --category=memory .
```

### Project 3: Algorithm Optimization
**Scenario**: Data processing taking 10 minutes
**Goal**: Reduce to under 1 minute
**Time**: 25 minutes

```bash
cd ~/projects/slow-processor
/optimize --profile --category=all .
/optimize --implement --category=algorithm .
/run-all-tests --benchmark --validate
```

---

## Troubleshooting

### Issue: "No bottlenecks detected"
**Cause**: Application is already optimized or profiling disabled
**Solution**:
```bash
# Enable detailed profiling
/optimize --profile --detailed --category=all .
```

### Issue: "Optimization breaks tests"
**Cause**: Optimization changed behavior
**Solution**:
```bash
# Use safe optimization only
/optimize --implement --safe-only .
/run-all-tests --auto-fix
```

### Issue: "Can't reproduce performance issue"
**Cause**: Need production-like data
**Solution**:
```bash
# Use production-scale test data
/generate-tests --type=performance --scale=production .
```

---

## Summary

### What You Learned ✅
- ✅ How to profile applications to find bottlenecks
- ✅ Optimizing algorithms (O(n²) → O(n log n))
- ✅ Fixing N+1 query problems
- ✅ Memory optimization techniques
- ✅ Benchmarking and validation
- ✅ Performance regression testing

### Results Achieved 📊
- **Response Time**: 2000ms → 200ms (10x faster)
- **Memory Usage**: 1.2GB → 480MB (60% reduction)
- **Database Queries**: 1001 → 1 (1000x improvement)
- **Automated Monitoring**: Performance CI/CD pipeline

### Key Takeaways 💡
1. Always measure before optimizing
2. Focus on algorithmic improvements first
3. Database queries are often the bottleneck
4. Memory leaks compound over time
5. Automate performance testing

---

## Next Steps

### Continue Learning
- **Tutorial 04**: [Workflow Automation](tutorial-04-workflows.md) - Automate optimization workflows
- **Tutorial 06**: [Agent System](tutorial-06-agents.md) - Use agents for deeper analysis

### Advanced Topics
- Distributed system performance
- GPU optimization
- Microservice optimization
- Real-time system optimization

### Real-World Application
Apply these techniques to your projects:
```bash
cd ~/my-project
/optimize --profile --category=all .
/optimize --implement --dry-run .  # Preview first
/optimize --implement --backup --rollback .  # Safe implementation
```

---

## Additional Resources

- [Performance Optimization Guide](../docs/USER_GUIDE.md#performance)
- [Profiling Best Practices](../docs/DEVELOPER_GUIDE.md#profiling)
- [Benchmark Suite Documentation](../docs/API_REFERENCE.md#benchmarking)

---

**Estimated Time**: 60 minutes | **Completion**: Achieved 10x performance improvement ✅

**Next**: [Tutorial 04: Workflow Automation →](tutorial-04-workflows.md)