---
name: scientific-data-visualization
version: "1.0.7"
maturity: "5-Expert"
specialization: Publication-Quality Figures
description: Create domain-specific scientific visualizations with uncertainty quantification (error bars, confidence bands), multi-dimensional data (parallel coordinates, heatmaps), domain plots (vector fields, molecular structures, climate maps, spectroscopy), and publication standards (Nature/Science specs, colorblind-friendly palettes). Use for research papers, presentations, and technical reports.
---

# Scientific Data Visualization

Publication-quality figures across physics, biology, chemistry, and climate science.

---

## Domain Selection

| Domain | Visualization Types |
|--------|---------------------|
| Physics/Engineering | Vector fields, streamlines, quiver plots |
| Molecular Biology | Protein structures, trajectory, networks |
| Climate Science | Geospatial maps, temperature anomalies |
| Spectroscopy | UV-Vis, FTIR, NMR spectra |
| Statistics | Distributions, time series, correlations |

---

## Uncertainty Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 20)
y_mean, y_std = 2*x + 5, 0.5*np.sqrt(x)

# Error bars
plt.errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=3)

# Confidence bands
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='±1σ')
plt.fill_between(x, y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.15, label='±2σ')
```

---

## Vector Fields (Physics/Engineering)

```python
Y, X = np.mgrid[-3:3:20j, -3:3:20j]
U, V = -1 - X**2 + Y, 1 + X - Y**2
speed = np.sqrt(U**2 + V**2)

# Streamplot
plt.streamplot(X, Y, U, V, color=speed, cmap='viridis', linewidth=2)

# Quiver plot
plt.quiver(X, Y, U, V, speed, cmap='plasma')
```

---

## Climate/Geospatial

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

im = ax.contourf(LON, LAT, temperature, levels=20,
                 transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True)
plt.colorbar(im, label='Temperature Anomaly (°C)')
```

---

## Spectroscopy

```python
# UV-Vis spectra
wavelengths = np.linspace(200, 800, 600)
for compound, color in zip(compounds, colors):
    absorption = gaussian_peaks(wavelengths)
    plt.plot(wavelengths, absorption, label=compound, color=color)

# FTIR (inverted x-axis convention)
plt.gca().invert_xaxis()
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Transmittance (%)')
```

---

## Network Visualization (Biology)

```python
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([
    ('Gene A', 'Protein X', {'type': 'transcription'}),
    ('Protein X', 'Enzyme Z', {'type': 'activation'}),
])

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3000)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)
```

---

## Publication Standards

```python
def setup_publication_figure(width='single'):
    """Journal-ready figure configuration."""
    width_inches = {'single': 3.5, 'double': 7.0, 'full': 7.5}[width]

    plt.rcParams.update({
        'figure.figsize': (width_inches, width_inches * 0.75),
        'figure.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 8,
        'axes.labelsize': 9,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
    })

# Save for publication
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure.png', dpi=600, bbox_inches='tight')
```

---

## Colorblind-Friendly Palettes

```python
# IBM Design colorblind-safe
cb_palette = {
    'blue': '#648FFF',
    'orange': '#FE6100',
    'purple': '#B56576',
    'red': '#DC267F',
    'yellow': '#FFB000'
}
```

---

## Julia Visualization

```julia
using Plots

# Time-series analysis
plot(t, signal, label="Raw", alpha=0.6)
plot!(t, smoothed, linewidth=2, label="Smoothed")

# 3D structure
plot3d(x, y, z, linewidth=3, color=:plasma,
       xlabel="X (Å)", ylabel="Y (Å)", zlabel="Z (Å)")
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Uncertainty | Always show error bars or confidence bands |
| Colorblind-safe | Use tested palettes (IBM, viridis) |
| Publication specs | Match journal requirements (DPI, fonts, sizes) |
| Consistent styling | Use rcParams or theme for all figures |
| Vector formats | PDF for print, PNG for web (600 DPI) |

---

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Missing uncertainty | Results appear more precise than they are |
| Rainbow colormap | Colorblind-unfriendly, perceptual issues |
| Low DPI | Pixelated figures in publications |
| Cluttered plots | Too much data, poor readability |
| Wrong aspect ratio | Distorted data relationships |

---

## Checklist

- [ ] Uncertainty shown (error bars, bands)
- [ ] Colorblind-friendly palette
- [ ] Publication DPI (300+ for print)
- [ ] Appropriate figure width (journal specs)
- [ ] Axis labels with units
- [ ] Legend if multiple series
- [ ] Vector format for publication

---

**Version**: 1.0.5
