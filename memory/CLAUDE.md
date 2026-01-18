# MyClaude Project Guidelines

## 0. Quick Start & Commands
**Environment:** Python 3.12+ | Manager: `uv` | Framework: JAX (NumPyro, Equinox)

- **Install:** `uv sync` (Updates `uv.lock`)
- **Run:** `uv run python <script.py>`
- **Test:** `uv run pytest`
- **Lint:** `uv run ruff check .`
- **Type Check:** `uv run mypy .`
- **Docs:** `uv run sphinx-build docs docs/_build`

---

## 1. Critical Priorities (Non-Negotiable)
1.  **Correctness:** Math must be theoretically exact.
2.  **Integrity:** No silent data loss/truncation.
3.  **Reproducibility:** Explicit seeds, version locking.
4.  **Performance:** JAX/JIT compilation.
5.  **UX:** Responsive PyQt/PySide6 interfaces.

**Prohibited:**
- Silent downsampling or random SVD.
- `from module import *` (wildcard imports).
- Global/user site-packages (local `.venv` only).
- Non-JIT-safe interpolation (Use `interpax`).

---

## 2. Computational Architecture
- **Core:** JAX-first. Minimize Host↔Device transfers.
- **Optimization:**
  - **NLSQ:** GPU-accelerated non-linear least squares.
  - **Solvers:** `optimistix` (root-finding).
  - **Schedule:** `optax`.
- **Bayesian Pipeline:**
  - **Engine:** NumPyro (preferred).
  - **Workflow:** NLSQ (warm-start) → NUTS/CMC.
  - **Diagnostics:** Mandatory ArviZ (R-hat, ESS, BFMI).

---

## 3. Data & Validation
- **I/O Boundaries:** Runtime validation for shape, dtype, NaN, and monotonicity.
- **Streaming:** Only allowed if mathematically equivalent to full-batch.
- **Subsampling:** Off by default. Must be explicitly logged if enabled.

---

## 4. UI & Visualization
- **Framework:** PyQt/PySide6 (View layer only).
- **Logic:** Decoupled. Numerical logic stays in JAX.
- **Plotting:** PyQtGraph (Interactive) / Matplotlib (Publication).
- **Theming:** System-aware Light/Dark modes.

---

## 5. Development Standards
- **Lockfile:** `uv.lock` is single source of truth.
- **Typing:** Strict hints at API boundaries and config objects.
- **Logging:** Structured logging for convergence and diagnostics.
- **Tone:** Factual, technical documentation only.
