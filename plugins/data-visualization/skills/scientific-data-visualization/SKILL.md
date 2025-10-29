---
name: scientific-data-visualization
description: Domain-specific scientific data visualization techniques for physics, biology, chemistry, climate science, and engineering with uncertainty quantification, multi-dimensional data, and publication standards
tools: Read, Write, python, julia, ParaView, VMD, matplotlib
integration: Use for creating publication-quality scientific visualizations across domains
---

# Scientific Data Visualization Mastery

Comprehensive techniques for visualizing complex scientific data across domains with proper uncertainty representation, accessibility, and publication standards.

## Core Scientific Visualization Principles

### 1. Uncertainty Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Error bars with confidence intervals
x = np.linspace(0, 10, 20)
y_mean = 2 * x + 5
y_std = 0.5 * np.sqrt(x)  # Heteroscedastic error

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Standard error bars
axs[0].errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=3,
                label='Data ± σ', color='#2C7BB6')
axs[0].set_title('(a) Error Bars')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Confidence bands
axs[1].plot(x, y_mean, 'b-', linewidth=2, label='Mean')
axs[1].fill_between(x, y_mean - y_std, y_mean + y_std,
                     alpha=0.3, label='±1σ')
axs[1].fill_between(x, y_mean - 2*y_std, y_mean + 2*y_std,
                     alpha=0.2, label='±2σ')
axs[1].set_title('(b) Confidence Bands')
axs[1].legend()

# Violin plots for distributions
from scipy import stats
data_dist = [stats.norm.rvs(y_mean[i], y_std[i], 100)
             for i in range(len(x))]

parts = axs[2].violinplot(data_dist, positions=x, widths=0.5,
                           showmeans=True, showmedians=True)
axs[2].set_title('(c) Distribution Violin Plots')

plt.tight_layout()
plt.savefig('uncertainty_visualization.png', dpi=300)
```

### 2. Multi-Dimensional Data Representation

```python
# Parallel coordinates for high-dimensional data
from pandas.plotting import parallel_coordinates
import pandas as pd

# Generate multi-dimensional scientific data
n_samples = 200
df = pd.DataFrame({
    'Temperature': np.random.uniform(20, 100, n_samples),
    'Pressure': np.random.uniform(1, 10, n_samples),
    'pH': np.random.uniform(3, 11, n_samples),
    'Conductivity': np.random.uniform(0.1, 5.0, n_samples),
    'Yield': np.random.uniform(0, 100, n_samples),
    'Condition': np.random.choice(['A', 'B', 'C'], n_samples)
})

fig, ax = plt.subplots(figsize=(12, 6))
parallel_coordinates(df, 'Condition', ax=ax,
                    color=['#E41A1C', '#377EB8', '#4DAF4A'])
ax.set_title('Multi-Dimensional Parameter Space')
ax.legend(loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('parallel_coordinates.png', dpi=300)
```

## Domain-Specific Visualizations

### Physics & Engineering

```python
# Vector field visualization (fluid dynamics, electromagnetics)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Velocity field
Y, X = np.mgrid[-3:3:20j, -3:3:20j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U**2 + V**2)

# Streamplot
strm = axs[0].streamplot(X, Y, U, V, color=speed, cmap='viridis',
                         linewidth=2, arrowsize=1.5, density=1.5)
axs[0].set_title('Velocity Field Streamlines')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
fig.colorbar(strm.lines, ax=axs[0], label='Speed')

# Quiver plot
Q = axs[1].quiver(X, Y, U, V, speed, cmap='plasma',
                  scale=50, width=0.003)
axs[1].set_title('Vector Field')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
fig.colorbar(Q, ax=axs[1], label='Magnitude')

plt.tight_layout()
plt.savefig('vector_fields.png', dpi=300)
```

### Molecular & Structural Biology

```julia
# Julia example for molecular visualization
using Plots, LinearAlgebra

# Generate protein backbone coordinates
n_residues = 50
t = range(0, 4π, length=n_residues)

# Helix parametrization
α = 2.3  # radius
pitch = 5.4  # pitch per turn
x = α .* cos.(t)
y = α .* sin.(t)
z = pitch .* t ./ (2π)

# 3D protein backbone
plot3d(x, y, z,
       linewidth=3,
       marker=:circle,
       markersize=4,
       color=:plasma,
       line_z=1:n_residues,
       colorbar_title="Residue",
       xlabel="X (Å)",
       ylabel="Y (Å)",
       zlabel="Z (Å)",
       title="Protein α-Helix Structure",
       camera=(30, 30))

savefig("protein_helix.png")
```

### Climate & Environmental Science

```python
# Geospatial climate data visualization
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Generate synthetic climate data
lon = np.linspace(-180, 180, 180)
lat = np.linspace(-90, 90, 90)
LON, LAT = np.meshgrid(lon, lat)

# Temperature anomaly simulation
temperature = (np.sin(np.radians(LAT)) * 20 +
               np.cos(np.radians(LON)) * 5 +
               np.random.randn(*LAT.shape) * 2)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# Plot temperature data
im = ax.contourf(LON, LAT, temperature, levels=20,
                 transform=ccrs.PlateCarree(),
                 cmap='RdBu_r', extend='both')

# Add geographic features
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.colorbar(im, ax=ax, orientation='horizontal',
             pad=0.05, label='Temperature Anomaly (°C)')
plt.title('Global Temperature Anomaly Distribution', pad=20)
plt.tight_layout()
plt.savefig('climate_map.png', dpi=300, bbox_inches='tight')
```

### Spectroscopy & Analytical Chemistry

```python
# Multi-spectrum visualization
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# UV-Vis spectra
wavelengths = np.linspace(200, 800, 600)
compounds = ['Compound A', 'Compound B', 'Compound C']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (compound, color) in enumerate(zip(compounds, colors)):
    # Simulate absorption peaks
    absorption = (np.exp(-((wavelengths - 300 - i*100)/50)**2) * 0.8 +
                 np.exp(-((wavelengths - 450 - i*50)/30)**2) * 0.5)

    axs[0].plot(wavelengths, absorption, label=compound,
                linewidth=2, color=color)

axs[0].set_xlabel('Wavelength (nm)')
axs[0].set_ylabel('Absorbance (AU)')
axs[0].set_title('UV-Vis Absorption Spectra')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# IR spectra (inverted)
wavenumbers = np.linspace(4000, 500, 500)

for i, (compound, color) in enumerate(zip(compounds, colors)):
    # Simulate IR peaks
    transmittance = (100 -
                    40 * np.exp(-((wavenumbers - 3000)/100)**2) -
                    30 * np.exp(-((wavenumbers - 1700 - i*50)/80)**2) -
                    25 * np.exp(-((wavenumbers - 1000)/60)**2))

    axs[1].plot(wavenumbers, transmittance, label=compound,
                linewidth=2, color=color)

axs[1].set_xlabel('Wavenumber (cm⁻¹)')
axs[1].set_ylabel('Transmittance (%)')
axs[1].set_title('FTIR Spectra')
axs[1].invert_xaxis()  # IR convention
axs[1].set_ylim(0, 105)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectroscopy.png', dpi=300)
```

## Advanced Techniques

### Time-Series Scientific Data

```julia
using Plots, Statistics, DSP

# Generate time-series with trends and seasonality
t = 0:0.1:100
trend = 0.5 .* t
seasonal = 10 .* sin.(2π .* t ./ 10)
noise = randn(length(t)) .* 2
signal = trend .+ seasonal .+ noise

# Multi-panel time-series analysis
p1 = plot(t, signal,
          xlabel="Time",
          ylabel="Signal",
          title="(a) Original Signal",
          label="Raw Data",
          alpha=0.6)

# Moving average
window = 20
smoothed = [mean(signal[max(1, i-window):min(end, i+window)])
            for i in 1:length(signal)]
plot!(p1, t, smoothed, linewidth=2, label="Smoothed", color=:red)

# Autocorrelation
p2 = plot(autocor(signal, 0:50),
          xlabel="Lag",
          ylabel="Autocorrelation",
          title="(b) Autocorrelation Function",
          marker=:circle,
          legend=false)
hline!(p2, [0], linestyle=:dash, color=:black)

# Power spectrum
p3 = plot(abs.(fft(signal)[1:length(signal)÷2]),
          xlabel="Frequency",
          ylabel="Power",
          title="(c) Frequency Domain",
          yscale=:log10,
          legend=false)

# Residuals
residuals = signal .- smoothed
p4 = scatter(t, residuals,
             xlabel="Time",
             ylabel="Residuals",
             title="(d) Residual Analysis",
             markersize=2,
             alpha=0.5,
             legend=false)
hline!(p4, [0], linestyle=:dash, color=:red)

plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
savefig("timeseries_analysis.png")
```

### Network/Graph Visualization (Systems Biology)

```python
import networkx as nx
from matplotlib.patches import FancyBboxPatch

# Create biological network
G = nx.DiGraph()

# Add nodes (proteins/genes)
nodes = ['Gene A', 'Gene B', 'Protein X', 'Protein Y',
         'Enzyme Z', 'Metabolite M']
G.add_nodes_from(nodes)

# Add edges (interactions)
interactions = [
    ('Gene A', 'Protein X', {'type': 'transcription'}),
    ('Gene B', 'Protein Y', {'type': 'transcription'}),
    ('Protein X', 'Enzyme Z', {'type': 'activation'}),
    ('Protein Y', 'Enzyme Z', {'type': 'inhibition'}),
    ('Enzyme Z', 'Metabolite M', {'type': 'catalysis'}),
]
G.add_edges_from([(u, v, d) for u, v, d in interactions])

# Layout
pos = nx.spring_layout(G, k=1, iterations=50)

fig, ax = plt.subplots(figsize=(12, 10))

# Node types and colors
node_colors = {
    'Gene': '#8DD3C7',
    'Protein': '#FFFFB3',
    'Enzyme': '#BEBADA',
    'Metabolite': '#FB8072'
}

colors = [node_colors.get(n.split()[0], '#FFFFFF') for n in nodes]

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=colors,
                       node_size=3000, alpha=0.9,
                       linewidths=2, edgecolors='black', ax=ax)

nx.draw_networkx_labels(G, pos, font_size=10,
                        font_weight='bold', ax=ax)

# Edge styles by interaction type
edge_colors = []
edge_styles = []
for u, v in G.edges():
    itype = G[u][v]['type']
    if itype == 'activation':
        edge_colors.append('#2CA02C')
        edge_styles.append('-')
    elif itype == 'inhibition':
        edge_colors.append('#D62728')
        edge_styles.append('-')
    else:
        edge_colors.append('#7F7F7F')
        edge_styles.append('--')

nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                       width=2, arrows=True, arrowsize=20,
                       arrowstyle='->', connectionstyle='arc3,rad=0.1',
                       ax=ax)

ax.set_title('Gene Regulatory Network', fontsize=16, pad=20)
ax.axis('off')

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#2CA02C', lw=2, label='Activation'),
    plt.Line2D([0], [0], color='#D62728', lw=2, label='Inhibition'),
    plt.Line2D([0], [0], color='#7F7F7F', lw=2, ls='--', label='Other')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('biological_network.png', dpi=300, bbox_inches='tight')
```

## Publication Standards

### Journal-Ready Figure Formatting

```python
def setup_publication_figure(width='single', height_ratio=0.75):
    """
    Setup figure with journal specifications.

    Parameters
    ----------
    width : str
        'single' (3.5"), 'double' (7.0"), 'full' (7.5")
    height_ratio : float
        Height as fraction of width
    """
    width_inches = {
        'single': 3.5,
        'double': 7.0,
        'full': 7.5
    }[width]

    plt.rcParams.update({
        'figure.figsize': (width_inches, width_inches * height_ratio),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })

# Example usage
setup_publication_figure('single')

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='Sin(x)')
ax.plot(x, np.cos(x), label='Cos(x)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('publication_figure.pdf', format='pdf', bbox_inches='tight')
plt.savefig('publication_figure.png', dpi=600, bbox_inches='tight')
```

### Colorblind-Friendly Palettes

```python
# IBM Design colorblind-safe palette
cb_palette = {
    'blue': '#648FFF',
    'orange': '#FE6100',
    'green': '#00B4D8',
    'purple': '#B56576',
    'teal': '#06FFA5',
    'red': '#DC267F',
    'yellow': '#FFB000'
}

# Apply to plot
colors = list(cb_palette.values())

fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    ax.plot(np.random.randn(100).cumsum(), color=color,
            label=f'Series {i+1}', linewidth=2)

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Colorblind-Friendly Palette')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('colorblind_friendly.png', dpi=300)
```

## Integration with Scientific Tools

### ParaView/VTK Python Integration

```python
from vtk import *
from vtk.util import numpy_support

# Create 3D scalar field
nx, ny, nz = 50, 50, 50
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
z = np.linspace(-2, 2, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Generate 3D data (e.g., electron density)
data = np.exp(-(X**2 + Y**2 + Z**2))

# Create VTK image data
image_data = vtkImageData()
image_data.SetDimensions(nx, ny, nz)
image_data.SetSpacing(x[1]-x[0], y[1]-y[0], z[1]-z[0])
image_data.SetOrigin(x[0], y[0], z[0])

# Add scalar data
vtk_data = numpy_support.numpy_to_vtk(data.ravel(), deep=True)
vtk_data.SetName('Density')
image_data.GetPointData().SetScalars(vtk_data)

# Write to file for ParaView
writer = vtkXMLImageDataWriter()
writer.SetFileName('3d_density.vti')
writer.SetInputData(image_data)
writer.Write()

print("3D data written to 3d_density.vti - open with ParaView")
```

This skill provides domain-specific visualization techniques for scientific research across physics, biology, chemistry, climate science, and engineering!
