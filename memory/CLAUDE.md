# Technical Guidelines

## 0. Core Philosophy & Priorities
**Priority Hierarchy:**
1. **Correctness/Parity** - The math must be right
2. **Full-Data Integrity** - No silent data loss
3. **Determinism** - Reproducible seeds/versions
4. **Performance** - JAX/JIT optimization
5. **UX Polish** - Theming/Responsiveness

**Documentation Tone:** Factual and technical only. Avoid marketing language.

---

## 1. Environment & Package Management
- **Python:** Strict **3.12+** (`requires-python = ">=3.12"`)
- **Manager:** **`uv`** for all dependency resolution and environment management
- **Virtual Environment:** Local `.venv/` only; never global/user site-packages
- **Lockfile:** `uv.lock` is single source of truth; commit on any dependency change
- **Execution:** Use `uv run ...` or explicit `uv sync` + activation

---

## 2. Computational Core (JAX-First)
- **JAX-First Rule:** Numerical core runs entirely in JAX. Leverage `jit`, `vmap`, `pmap`
- **NumPy:** Limit to I/O boundaries and small glue logic
- **Data Transfer:** Minimize Host ↔ Device transfers; avoid `.numpy()` in hot paths
- **Interpolation:** Use **`interpax`** (not scipy.interpolate); must be JIT-safe
- **Optimization Stack:**
  - **NLSQ:** `/home/wei/Documents/GitHub/NLSQ` - GPU-accelerated non-linear least squares (not scipy)
  - **Solvers:** `optimistix` for root-finding and least-squares
  - **Optimizers:** `optax` for gradient schedules

---

## 3. Data Integrity
- **Prohibited:** Silent downsampling, truncation, subsampling, random SVD
- **Numerical precision and reproducibility take priority over computational speed**
- **Streaming:** Permitted only if mathematically equivalent to full-data processing
- **Subsampling:** If ever used, must be optional, off by default, explicitly logged
- **Boundary Validation:** Runtime checks at all I/O for shape, dtype, monotonicity, missing values

---

## 4. Bayesian Inference Pipeline
- **Engines:** NumPyro (preferred), Oryx, Blackjax
- **Strategy:** NLSQ → NUTS/CMC pipeline; use NLSQ for MAP/LSQ estimates to warm-start NUTS/CMC
- **Diagnostics (Mandatory):** ArviZ suite - R-hat, ESS, divergences, BFMI
- **Reproducibility:** Artifacts must include random seed, software versions, config snapshot

---

## 5. Code Quality
- **Imports:** Explicit only; no wildcards (`from module import *`)
- **Typing:** Strict type hints at public APIs, config objects, I/O boundaries
- **Logging:** Structured logging for key events (data load, solver convergence, diagnostics)
- **Failure Modes:** No silent fallbacks; log errors clearly; CPU fallback only if semantics preserved

---

## 6. GUI & Visualization
- **Framework:** Support PyQt/PySide6; do not hard-lock to single binding
- **Separation:** GUI is view layer only; numerical semantics in JAX-based APIs
- **Theming:** Support system Light/Dark with user override; no bespoke CSS
- **Plotting:** PyQtGraph for interactive, Matplotlib for publication; theme-aware palettes
