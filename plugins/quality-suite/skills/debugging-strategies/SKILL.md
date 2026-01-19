---
name: debugging-strategies
version: "1.0.0"
description: Systematic debugging methodologies using the scientific method, profiling, and root cause analysis across any stack.
---

# Debugging Strategies

Expert guide for identifying and resolving complex software defects.

## 1. The Scientific Method for Debugging

1.  **Observe**: Document the actual vs. expected behavior.
2.  **Hypothesize**: List potential causes based on data.
3.  **Experiment**: Test one hypothesis at a time by isolating variables.
4.  **Analyze**: Verify if the change resolved the issue.
5.  **Repeat**: Refine the hypothesis until the root cause is identified.

## 2. Tools & Techniques

- **Differential Debugging**: Compare behavior between working (Dev) and broken (Prod) environments.
- **Git Bisect**: Perform a binary search through history to find the commit that introduced the bug.
- **Profiling**: Use `cProfile` (Python), `perf` (Systems), or Chrome DevTools (JS) to identify performance regressions.
- **Tracing**: Use distributed tracing (OpenTelemetry) to track requests across service boundaries.

## 3. Troubleshooting Checklist

- [ ] **Reproducibility**: Is there a minimal, reliable reproduction case?
- [ ] **Recent Changes**: What changed in the environment or code recently?
- [ ] **Logs**: Are there relevant error logs or stack traces?
- [ ] **Assumptions**: Are there any hidden assumptions that might be false?
- [ ] **Root Cause**: Is the fix addressing the cause or just the symptom?
