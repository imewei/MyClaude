---
name: interop-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Cross-Language
description: Master cross-language integration with PythonCall.jl, RCall.jl, and CxxWrap.jl. Use when calling Python/R libraries from Julia or minimizing data transfer overhead.
---

# Julia Interoperability Patterns

Cross-language integration with Python, R, and C++.

---

## PythonCall.jl (Python)

```julia
using PythonCall

np = pyimport("numpy")
pd = pyimport("pandas")

# Call Python functions
py_array = np.array([1, 2, 3, 4, 5])
py_result = np.sum(py_array)

# Convert between Julia and Python
jl_array = pyconvert(Vector, py_array)  # Python → Julia
py_data = Py(jl_array)                  # Julia → Python
```

---

## RCall.jl (R)

```julia
using RCall

# Execute R code
R"library(ggplot2)"

# Transfer data
jl_vector = [1, 2, 3, 4, 5]
@rput jl_vector
R"r_squared <- jl_vector^2"
@rget r_squared

# Inline R
result = R"mean(1:10)"
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Use PythonCall.jl | Not PyCall.jl (modern) |
| Minimize transfers | Data transfers are expensive |
| Zero-copy when possible | Use views, not copies |
| Profile mixed code | Find language boundary bottlenecks |
| Consider Julia alternatives | Pure Julia often faster |

---

## Checklist

- [ ] PythonCall.jl for Python interop
- [ ] Data transfers minimized
- [ ] Zero-copy views used where possible
- [ ] Performance profiled across boundaries

---

**Version**: 1.0.5
