---
name: python-julia-visualization
version: "1.0.7"
maturity: "5-Expert"
specialization: Scientific Data Visualization
description: Implement production-ready scientific visualizations using Python (Matplotlib, Seaborn, Plotly, Bokeh) and Julia (Plots.jl, Makie.jl) ecosystems. Use when creating publication-quality static plots, interactive dashboards, real-time streaming visualizations, 3D surfaces, or configuring export settings (300 DPI, vector formats, colorblind-friendly palettes).
---

# Python and Julia Visualization

Production scientific visualization with modern Python and Julia ecosystems.

---

## Library Selection

| Library | Language | Best For | Performance |
|---------|----------|----------|-------------|
| Matplotlib | Python | Publication figures | Static |
| Seaborn | Python | Statistical plots | Static |
| Plotly | Python | Interactive 3D/web | Interactive |
| Bokeh | Python | Large datasets, streaming | 10k+ points |
| Plots.jl | Julia | Unified API, backends | Backend-dependent |
| Makie.jl | Julia | GPU-accelerated, real-time | High performance |

---

## Matplotlib - Publication Quality

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Publication setup
rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 12, 'axes.titlesize': 14,
    'figure.dpi': 300, 'savefig.dpi': 300
})

# Multi-panel figure
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

# Line with error bars
x, y = np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50))
axs[0, 0].errorbar(x, y, yerr=0.1, fmt='o-', capsize=3, label='Data')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].legend()

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax3d = fig.add_subplot(122, projection='3d')
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))
ax3d.plot_surface(X, Y, Z, cmap='viridis')

# Export
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')  # Vector
plt.savefig('figure.png', dpi=300, bbox_inches='tight')       # Raster
```

---

## Seaborn - Statistical Visualization

```python
import seaborn as sns
import pandas as pd

sns.set_theme(style='whitegrid', palette='Set2')
sns.set_context('paper', font_scale=1.2)

# Violin with swarm overlay
sns.violinplot(data=df, x='group', y='value', hue='condition', inner='quartile')

# Line with confidence interval
sns.lineplot(data=df, x='time', y='response', hue='treatment',
             err_style='band', ci=95, markers=True)

# FacetGrid for multi-dimensional
g = sns.FacetGrid(df, col='category', row='group', hue='category', height=4)
g.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.6)
g.add_legend()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
```

---

## Plotly - Interactive 3D

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 3D surface
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Dashboard with subplots
fig = make_subplots(rows=2, cols=2,
    subplot_titles=('Time Series', 'Histogram', 'Scatter', 'Box'))

fig.add_trace(go.Scatter(x=t, y=signal, mode='lines'), row=1, col=1)
fig.add_trace(go.Histogram(x=data, nbinsx=30), row=1, col=2)

# Animation
frames = [go.Frame(data=[go.Scatter(x=x, y=np.sin(x - t*0.1))], name=f't{t}')
          for t in range(50)]
fig = go.Figure(data=[go.Scatter(x=x, y=np.sin(x))], frames=frames)

# Export
fig.write_html('interactive.html')
fig.write_image('static.png', width=1200, height=800, scale=2)
```

---

## Bokeh - Large Datasets & Streaming

```python
from bokeh.plotting import figure, show, save
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256

# Large scatter (10k+ points)
source = ColumnDataSource(data=dict(x=x, y=y, colors=colors, labels=labels))

p = figure(title='Large Dataset', width=800, height=600,
           tools='pan,wheel_zoom,box_zoom,reset,lasso_select')

mapper = linear_cmap('colors', Viridis256, low=min(colors), high=max(colors))
p.circle('x', 'y', source=source, fill_color=mapper, size=8, alpha=0.6)

hover = HoverTool(tooltips=[('X', '@x{0.00}'), ('Y', '@y{0.00}')])
p.add_tools(hover)

save(p, 'large_scatter.html')
```

### Streaming Updates

```python
from bokeh.server.server import Server

def streaming_app(doc):
    source = ColumnDataSource(data=dict(time=[], value=[]))
    p = figure(x_axis_type='datetime', width=800, height=400)
    p.line('time', 'value', source=source)

    def update():
        new_data = dict(time=[datetime.now()], value=[np.random.randn()])
        source.stream(new_data, rollover=100)

    doc.add_periodic_callback(update, 100)
    doc.add_root(p)
```

---

## Julia Plots.jl

```julia
using Plots
gr()  # Fast backend; use plotlyjs() for interactive

# Publication style
default(fontfamily="Computer Modern", titlefontsize=14, guidefontsize=12, dpi=300)

# Multi-panel
p1 = plot(0:0.1:10, [sin, cos], label=["sin" "cos"], lw=2, xlabel="x", ylabel="f(x)")
p2 = scatter(x, y, yerror=err, xlabel="X", ylabel="Y", markersize=6)
p3 = histogram(data, bins=30, normalize=:pdf, xlabel="Value", ylabel="Density")
p4 = boxplot(groups, values, xlabel="Group", ylabel="Response")

plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
savefig("julia_figure.pdf")

# 3D surface
surface(x, y, z, xlabel="X", ylabel="Y", zlabel="Z", c=:viridis, camera=(30, 60))
```

---

## Julia Makie.jl - GPU Accelerated

```julia
using GLMakie

fig = Figure(resolution=(1200, 800))

# Observable for reactive updates
t = Observable(0.0)
z = Observable(zeros(100, 100))

# 3D surface
ax = Axis3(fig[1, 1], title="Real-Time Surface")
surface!(ax, x, y, z, colormap=:viridis)

# Animation
record(fig, "animation.mp4", 1:100; framerate=30) do frame
    t[] = frame * 0.1
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
        z[][i, j] = sin(sqrt(xi^2 + yj^2) - t[])
    end
    notify(z)
end

# Dashboard
ax1 = Axis(fig[1, 1], title="Scatter")
scatter!(ax1, x_data, y_data, color=colors, colormap=:plasma, markersize=10)

ax2 = Axis(fig[2, 1], title="Histogram")
hist!(ax2, data, bins=30, color=(:blue, 0.5))

save("dashboard.png", fig)
```

---

## Interactive Notebooks

### Jupyter + ipywidgets

```python
import ipywidgets as widgets

@widgets.interact(freq=(0.1, 5.0), amp=(0.1, 2.0))
def plot_wave(freq=1.0, amp=1.0):
    x = np.linspace(0, 10, 1000)
    plt.plot(x, amp * np.sin(2 * np.pi * freq * x))
    plt.ylim(-2.5, 2.5)
    plt.show()
```

### Pluto.jl

```julia
using PlutoUI, Plots

@bind frequency Slider(0.1:0.1:5.0, default=1.0, show_value=true)
@bind amplitude Slider(0.1:0.1:2.0, default=1.0, show_value=true)

begin
    x = 0:0.01:10
    y = @. amplitude * sin(2π * frequency * x)
    plot(x, y, lw=2, ylims=(-2.5, 2.5), legend=false)
end
```

---

## Publication Standards

| Parameter | Standard | Example |
|-----------|----------|---------|
| Resolution | 300 DPI | `dpi=300` |
| Font | Serif | Computer Modern, Times |
| Single column | 3.5" | 89mm |
| Double column | 7" | 178mm |
| Format | Vector | PDF, SVG |
| Colors | Colorblind-safe | Viridis, Cividis |

### Colormap Selection

| Type | Use Case | Examples |
|------|----------|----------|
| Sequential | Ordered data | viridis, plasma, inferno |
| Diverging | Centered data (zero) | RdBu, coolwarm, BrBG |
| Qualitative | Categories | Set1, Set2, tab10 |

**Avoid**: jet, rainbow (not perceptually uniform)

---

## Best Practices

| Area | Practice |
|------|----------|
| **Resolution** | 300 DPI for print, 150 for web |
| **Format** | PDF/SVG for scalability, PNG for web |
| **Colors** | Colorblind-friendly palettes |
| **Large data** | Bokeh/Makie for 10k+ points |
| **Interactive** | Plotly for web, Makie for performance |
| **Labels** | Clear axis labels with units |
| **Real-time** | Streaming with periodic callbacks |

---

## Checklist

- [ ] Resolution set to 300 DPI for publication
- [ ] Vector format (PDF/SVG) for scalability
- [ ] Colorblind-friendly palette selected
- [ ] Axis labels include units
- [ ] Legend positioned clearly
- [ ] Font consistent with target journal
- [ ] Figure fits column width requirements

---

**Version**: 1.0.5
