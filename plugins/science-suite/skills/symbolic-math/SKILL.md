---
name: symbolic-math
description: "Perform symbolic mathematics with SymPy including algebraic manipulation, symbolic differentiation/integration, equation solving, matrix algebra, and code generation from symbolic expressions. Use when deriving analytical solutions, simplifying expressions, or generating numerical code from symbolic formulas."
---

# Symbolic Mathematics

Derive analytical solutions, manipulate expressions, and generate numerical code.

## Expert Agent

For Python systems engineering and code generation workflows, delegate to the expert agent:

- **`python-pro`**: Python systems specialist for type-driven design, packaging, and performance optimization.
  - *Location*: `plugins/science-suite/agents/python-pro.md`
  - *Capabilities*: Modern Python patterns, Rust extensions (PyO3), structured logging, testing.

## SymPy Basics

```python
import sympy as sp

# Define symbols with assumptions
x, y, z = sp.symbols("x y z", real=True)
t = sp.Symbol("t", positive=True)
n = sp.Symbol("n", integer=True, positive=True)

# Create expressions
expr = x**2 + 2*x*y + y**2
factored = sp.factor(expr)       # (x + y)**2
expanded = sp.expand(factored)   # x**2 + 2*x*y + y**2
simplified = sp.simplify(sp.sin(x)**2 + sp.cos(x)**2)  # 1

# Substitution
result = expr.subs([(x, 1), (y, 2)])  # 9
```

## Symbolic Calculus

```python
# Differentiation
f = sp.sin(x) * sp.exp(-x**2)
df_dx = sp.diff(f, x)                 # First derivative
d2f_dx2 = sp.diff(f, x, 2)           # Second derivative
partial = sp.diff(x**2 * y**3, x, y)  # Mixed partial d^2f/dxdy

# Integration
integral = sp.integrate(sp.exp(-x**2), x)           # erf
definite = sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))  # sqrt(pi)/2
double = sp.integrate(x*y, (x, 0, 1), (y, 0, 1))      # 1/4

# Limits
lim = sp.limit(sp.sin(x) / x, x, 0)    # 1
lim_inf = sp.limit((1 + 1/n)**n, n, sp.oo)  # E

# Series expansion
series = sp.series(sp.exp(x), x, 0, n=5)
# 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)
```

## Equation Solving

```python
# Algebraic equations
solutions = sp.solve(x**3 - 6*x**2 + 11*x - 6, x)  # [1, 2, 3]

# Systems of equations
system = [x + y - 5, 2*x - y - 1]
sol = sp.solve(system, [x, y])  # {x: 2, y: 3}

# Nonlinear systems
sol_nl = sp.solve([x**2 + y**2 - 1, x - y], [x, y])

# Differential equations
f = sp.Function("f")
ode = sp.Eq(f(x).diff(x, 2) + f(x), 0)
sol_ode = sp.dsolve(ode, f(x))
# f(x) = C1*sin(x) + C2*cos(x)

# With initial conditions
ics = {f(0): 1, f(x).diff(x).subs(x, 0): 0}
sol_ivp = sp.dsolve(ode, f(x), ics=ics)
# f(x) = cos(x)
```

## Matrix Algebra

```python
# Symbolic matrices
A = sp.Matrix([[1, x], [y, 2]])
B = sp.Matrix([[x, 0], [0, y]])

# Operations
product = A * B
determinant = A.det()           # 2 - x*y
inverse = A.inv()
eigenvals = A.eigenvals()       # {eigenvalue: multiplicity}
eigenvects = A.eigenvects()     # [(eigenval, mult, [eigenvec]), ...]

# Characteristic polynomial
lam = sp.Symbol("lambda")
char_poly = (A - lam * sp.eye(2)).det()

# Diagonalization
P, D = A.diagonalize()  # A = P * D * P^(-1)

# Matrix exponential (useful for linear ODEs)
exp_A = sp.exp(A * t)
```

## Assumptions System

```python
# Assumptions guide simplification
a = sp.Symbol("a", positive=True)
b = sp.Symbol("b", real=True)

# sqrt(a**2) simplifies to a (because a > 0)
sp.sqrt(a**2)  # a

# Without assumptions, result is Abs(b)
sp.sqrt(b**2)  # Abs(b)

# Query assumptions
sp.ask(sp.Q.positive(a))  # True
sp.ask(sp.Q.real(b))      # True

# Refine with assumptions
expr = sp.Abs(x)
sp.refine(expr, sp.Q.positive(x))  # x
```

## Code Generation

```python
from sympy.utilities.lambdify import lambdify
from sympy.printing.numpy import NumPyPrinter

# lambdify: SymPy expression -> NumPy function
expr = sp.sin(x)**2 + sp.cos(y)**2
f_numpy = lambdify([x, y], expr, modules="numpy")
# Now f_numpy(1.0, 2.0) uses NumPy

# Generate C code
from sympy.utilities.codegen import codegen
code = codegen(("my_func", expr), language="C", header=False)

# Print as LaTeX
latex_str = sp.latex(expr)
# \sin^{2}{\left(x \right)} + \cos^{2}{\left(y \right)}

# Common subexpression elimination for efficient code
from sympy.codegen.rewriting import optimize, optims_c99
cse_exprs = sp.cse([expr, sp.diff(expr, x)])
```

## LaTeX Output

```python
# Pretty printing
sp.pprint(expr)  # Unicode terminal output

# LaTeX for publications
sp.latex(sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo)))
# \int\limits_{-\infty}^{\infty} e^{- x^{2}}\, dx

# Matrix LaTeX
sp.latex(A)  # \left[\begin{matrix}1 & x\\y & 2\end{matrix}\right]
```

## Symbolic Math Checklist

- [ ] Define symbols with correct assumptions (real, positive, integer)
- [ ] Use `simplify()` sparingly -- prefer `expand()`, `factor()`, `trigsimp()`
- [ ] Verify symbolic solutions numerically at test points
- [ ] Use `lambdify` with `modules="numpy"` for numerical evaluation
- [ ] Apply CSE (common subexpression elimination) before code generation
- [ ] Check branch cuts for complex-valued functions
- [ ] Use `Rational` instead of floats for exact arithmetic: `sp.Rational(1, 3)`
- [ ] Document all variable assumptions in docstrings
