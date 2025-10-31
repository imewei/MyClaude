---
name: interop-patterns
description: Master cross-language integration with PythonCall.jl, RCall.jl, and CxxWrap.jl for calling Python, R, and C++ libraries from Julia. Use when calling Python libraries (.jl files with pyimport, PyArray for zero-copy), using R packages with RCall.jl (@rput, @rget for data transfer), integrating C++ code with CxxWrap.jl, converting data between Julia and Python (pyconvert, Py()), executing R code from Julia (R"..." string macro), leveraging existing libraries from other languages, minimizing data transfer overhead with zero-copy views, or building multi-language scientific workflows. Essential for accessing specialized libraries not available in Julia and integrating with existing codebases in other languages.
---

# Interoperability Patterns

Master cross-language integration in Julia with PythonCall.jl (Python), RCall.jl (R), and CxxWrap.jl (C++).

## When to use this skill

- Calling Python libraries from Julia (NumPy, Pandas, Matplotlib, scikit-learn)
- Using PythonCall.jl with pyimport and zero-copy PyArray
- Executing R code from Julia with RCall.jl (R"..." macro)
- Transferring data between Julia and R (@rput, @rget)
- Integrating C++ code with CxxWrap.jl for performance
- Converting data types between Julia and Python (pyconvert, Py())
- Minimizing data transfer overhead with zero-copy views
- Leveraging specialized libraries not available in Julia native
- Building multi-language scientific computing workflows
- Calling existing Python/R codebases from Julia projects
- Comparing Julia vs Python/R performance for specific tasks

## Python Interop (PythonCall.jl)

```julia
using PythonCall

# Import Python libraries
np = pyimport("numpy")
pd = pyimport("pandas")
plt = pyimport("matplotlib.pyplot")

# Call Python functions
py_array = np.array([1, 2, 3, 4, 5])
py_result = np.sum(py_array)

# Convert between Julia and Python
jl_array = pyconvert(Vector, py_array)  # Python → Julia
py_data = Py(jl_array)                  # Julia → Python (zero-copy when possible)

# Use Python objects
df = pd.DataFrame(pydict(Dict("x" => 1:5, "y" => rand(5))))
```

## R Interop (RCall.jl)

```julia
using RCall

# Execute R code
R"library(ggplot2)"
R"data <- data.frame(x=1:10, y=rnorm(10))"

# Transfer data: Julia → R
jl_vector = [1, 2, 3, 4, 5]
@rput jl_vector
R"r_squared <- jl_vector^2"

# Transfer data: R → Julia
@rget r_squared

# Inline R code
result = R"mean(1:10)"
```

## Best Practices

- Use PythonCall.jl (not PyCall.jl) for modern Python integration
- Minimize data transfers between languages (expensive)
- Use zero-copy views when possible
- Profile mixed-language code for performance bottlenecks
- Consider pure Julia alternatives when available

## Resources

- **PythonCall.jl**: https://juliapy.github.io/PythonCall.jl/
- **RCall.jl**: https://juliainterop.github.io/RCall.jl/
