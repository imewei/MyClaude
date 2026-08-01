# PySINDy and Symbolic Regression — Full Code Reference

`--mode deep` content for **equation-discovery**. Covers the Python PySINDy API and Julia `SymbolicRegression.jl`.

## PySINDy (Python)

```python
import pysindy as ps
import numpy as np

# Basic SINDy
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=3),
    feature_names=["x", "y"]
)
model.fit(X, t=t)
model.print()  # Display discovered equations

# Custom library with GeneralizedLibrary
lib = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree=2),
     ps.FourierLibrary(n_frequencies=3),
     ps.CustomLibrary(library_functions=[lambda x: np.exp(-x)],
                       function_names=[lambda x: f"exp(-{x})"])]
)
model = ps.SINDy(feature_library=lib)
model.fit(X, t=t)
```

PySINDy ships `STLSQ`, `SR3`, `SSR`, `FROLS`, `ConstrainedSR3`, `MIOSR` optimizers; feature libraries include `Polynomial`, `Fourier`, `Custom`, `PDE`, `WeakForm`, `Generalized`, `Tensored`; differentiation methods cover finite difference, smoothed FD, spectral, Savitzky-Golay, and Kalman. Supports control inputs (SINDyc), implicit dynamics, trapping theorem, and ensemble/bagging methods for UQ. NumPy/scikit-learn based — **not JAX-native**.

### Related Python Packages

| Package | Role | Notes |
|---------|------|-------|
| **PySINDy** | Sparse regression (SINDy, PDE/weak-form, ensemble) | NumPy/sklearn — see above |
| **PyDMD** | DMD family (exact, FbDMD, CDMD, MrDMD, Hankel, EDMD, DMDc, BOPDMD, PiDMD) | Koopman operator approximation; complementary to SINDy |
| **PySR** | Symbolic regression via Julia `SymbolicRegression.jl` backend | sklearn-compat; exports to SymPy/LaTeX/JAX/PyTorch via `model.jax()` |
| **gplearn** | Classical genetic-programming symbolic regression | NumPy; `SymbolicRegressor`/`Classifier`/`Transformer` |

> **No mature JAX-native SINDy library exists.** For a JAX-first workflow, hand-roll STLSQ via `jax.lax.scan` over polynomial libraries — the regression step is trivially vectorizable. PySR's `model.jax()` exporter is the cleanest bridge into a JAX pipeline.

## Symbolic Regression (Julia)

When SINDy's predefined basis is too restrictive, use evolutionary symbolic regression:

```julia
using SymbolicRegression

# Search for symbolic expressions
options = SymbolicRegression.Options(
    binary_operators=[+, -, *, /],
    unary_operators=[sin, cos, exp, sqrt],
    populations=30,
    maxsize=25
)

hall_of_fame = equation_search(X, y;
    options=options,
    niterations=100
)

# Pareto front: complexity vs accuracy
for member in hall_of_fame
    println("Complexity: $(member.complexity), Loss: $(member.loss)")
    println("  Equation: $(member.equation)")
end
```

> **Rule:** Use the Pareto front (complexity vs loss) to select models. Prefer the simplest equation whose loss is within 5% of the best.
