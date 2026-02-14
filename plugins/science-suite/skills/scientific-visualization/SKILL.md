---
name: scientific-visualization
version: "2.2.1"
description: Create publication-quality scientific visualizations across physics, biology, chemistry, and climate science. Supports uncertainty quantification, multi-dimensional data, and domain-specific plots in both Python and Julia.
---

# Scientific Visualization

Expert guide for creating research-grade figures that meet international publication standards (Nature, Science, etc.).

## Expert Agent

For creating publication-quality figures, complex multi-dimensional plots, and interactive visualizations, delegate to the expert agent:

- **`research-expert`**: Unified specialist for Scientific Visualization and Communication.
  - *Location*: `plugins/science-suite/agents/research-expert.md`
  - *Capabilities*: Matplotlib/Makie styling, domain-specific plotting, and adherence to publication standards.

## 1. Uncertainty & Statistical Visualization

Always include uncertainty quantification to ensure scientific integrity.

### Python (Matplotlib/Seaborn)
```python
import matplotlib.pyplot as plt
import numpy as np

# Error bars and confidence bands
plt.errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=3)
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='±1σ')
```

### Julia (Plots.jl/Makie.jl)
```julia
using Plots
plot(x, y, ribbon=y_std, fillalpha=0.3, label="Data ± σ")
```

## 2. Domain-Specific Visualizations

### Physics & Engineering (Vector Fields)
- **Python**: Use `plt.streamplot` for streamlines and `plt.quiver` for vector fields.
- **Julia**: Use `Makie.streamplot` or `Plots.quiver`.

### Molecular Biology & Chemistry
- **Structures**: Use `biopython` or `ASE` in Python; `BioStructures.jl` in Julia.
- **Networks**: Use `networkx` (Python) or `Graphs.jl` (Julia).

### Climate & Geospatial
- **Python**: Use `Cartopy` for map projections and `GeoPandas`.
- **Julia**: Use `GeoMakie.jl`.

## 3. Publication Standards Checklist

- [ ] **DPI**: 300 DPI for print (PDF/EPS), 600 DPI for high-res raster (PNG).
- [ ] **Fonts**: Use standard serif (Times New Roman) or sans-serif (Arial/Helvetica) as required by the journal.
- [ ] **Colorblind Safety**: Use perceptually uniform colormaps like `viridis`, `magma`, or IBM design palettes. Avoid "rainbow" or "jet".
- [ ] **Sizing**: Match figure width to journal column specs (typically 3.5" for single, 7" for double).
- [ ] **Units**: Ensure all axes have clear labels with SI units in parentheses.

## 4. High-Performance Rendering

For large datasets (>10^6 points):
- **Python**: Use `Datashader` or `VisPy` for GPU-accelerated rendering.
- **Julia**: Use `Makie.jl` for native GL-based interactive visualization.
