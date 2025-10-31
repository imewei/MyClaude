---
name: python-julia-visualization
description: Implement production-ready scientific visualizations using Python (Matplotlib, Seaborn, Plotly, Bokeh) and Julia (Plots.jl, Makie.jl, Gadfly.jl) ecosystems. Use this skill when writing or editing Python files (.py, .ipynb) or Julia files (.jl) that create data visualizations, implementing matplotlib publication-quality static plots with rcParams configuration and multi-panel figures, creating seaborn statistical plots (violin plots, FacetGrid, joint distributions, correlation heatmaps), building plotly interactive 3D visualizations with animations and real-time streaming dashboards, developing bokeh large-scale scatter plots with HoverTool and ColumnDataSource for 10k+ data points, implementing Julia Plots.jl unified plotting interface with multiple backends (GR, PlotlyJS), creating Makie.jl GPU-accelerated real-time visualizations with Observable patterns, building interactive Jupyter notebooks with ipywidgets sliders and interactive_output, creating reactive Pluto.jl notebooks with @bind for parameter exploration, implementing real-time streaming data visualization with periodic callbacks, creating 3D surface plots and contour visualizations, building custom colormaps for perceptually uniform color encoding, integrating Python-Julia workflows using PyCall, exporting visualizations to multiple formats (PNG, PDF, SVG, HTML, JSON), or applying publication standards (300 DPI, serif fonts, colorblind-friendly palettes, journal column widths).
tools: Read, Write, MultiEdit, Bash, Glob, Grep, python, julia, jupyter
integration: Use for scientific visualization development with Python/Julia ecosystems
---

# Python and Julia Visualization Mastery

Complete framework for building publication-quality visualizations, interactive dashboards, and real-time scientific plots using Python and Julia ecosystems.

## When to Use This Skill

- Writing or editing Python visualization scripts (.py files) using matplotlib, seaborn, plotly, or bokeh
- Working with Jupyter notebooks (.ipynb files) that create scientific plots or interactive visualizations
- Creating Julia visualization code (.jl files) using Plots.jl, Makie.jl, or Gadfly.jl
- Implementing publication-quality static plots with matplotlib (multi-panel figures, error bars, custom styling)
- Building statistical visualizations with seaborn (violin plots, FacetGrid, regression plots, distribution analysis)
- Creating interactive 3D visualizations with plotly (surface plots, animations, real-time dashboards)
- Developing large-scale data visualizations with bokeh (10k+ points with HoverTool and interactive tools)
- Implementing high-performance GPU-accelerated plots with Makie.jl for real-time data streaming
- Creating unified plotting interfaces with Julia Plots.jl supporting multiple backends (GR, PlotlyJS, PyPlot)
- Building interactive Jupyter notebooks with ipywidgets for parameter exploration and real-time updates
- Developing reactive Pluto.jl notebooks with @bind syntax for dynamic scientific visualizations
- Implementing real-time streaming data visualization with periodic callbacks and ColumnDataSource updates
- Creating 3D surface plots, contour visualizations, and volume rendering for scientific data
- Designing custom colormaps (sequential, diverging, qualitative) for perceptually uniform data encoding
- Configuring publication standards (300 DPI resolution, serif fonts, colorblind-friendly palettes)
- Formatting figures for journal specifications (single/double column widths, Nature/Science standards)
- Integrating Python and Julia workflows using PyCall for cross-language visualization
- Exporting visualizations to multiple formats (PNG, PDF, SVG for vector graphics, HTML for web embedding)
- Implementing animations and time-series visualizations with frames and temporal data
- Building multi-panel scientific figures with consistent styling across subplots

## Python Visualization Ecosystem

### 1. Matplotlib - Publication-Quality Static Plots

#### Core Plotting Patterns
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Publication-quality setup
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16
rcParams['figure.dpi'] = 300  # High DPI for publications
rcParams['savefig.dpi'] = 300

# Scientific figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

# Top-left: Line plot with error bars
x = np.linspace(0, 10, 50)
y = np.sin(x)
error = 0.1 * np.random.rand(len(x))

axs[0, 0].errorbar(x, y, yerr=error, fmt='o-', capsize=3,
                   label='Experimental data', color='#2C7BB6')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].set_title('(a) Time Series with Error Bars')
axs[0, 0].legend(loc='best')
axs[0, 0].grid(True, alpha=0.3)

# Top-right: Scatter plot with trend
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5

axs[0, 1].scatter(x_scatter, y_scatter, alpha=0.6, s=30, color='#D7191C')
z = np.polyfit(x_scatter, y_scatter, 1)
p = np.poly1d(z)
x_trend = np.linspace(x_scatter.min(), x_scatter.max(), 100)
axs[0, 1].plot(x_trend, p(x_trend), 'k--', linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
axs[0, 1].set_xlabel('X Variable')
axs[0, 1].set_ylabel('Y Variable')
axs[0, 1].set_title('(b) Correlation Analysis')
axs[0, 1].legend()

# Bottom-left: Histogram with KDE
data = np.random.normal(100, 15, 1000)

axs[1, 0].hist(data, bins=30, density=True, alpha=0.7,
               color='#FDAE61', edgecolor='black')

# Add KDE
from scipy import stats
kde = stats.gaussian_kde(data)
x_kde = np.linspace(data.min(), data.max(), 200)
axs[1, 0].plot(x_kde, kde(x_kde), 'k-', linewidth=2, label='KDE')
axs[1, 0].axvline(data.mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean = {data.mean():.1f}')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title('(c) Distribution Analysis')
axs[1, 0].legend()

# Bottom-right: Box plot comparison
groups = {
    'Control': np.random.normal(10, 2, 50),
    'Treatment A': np.random.normal(12, 2.5, 50),
    'Treatment B': np.random.normal(15, 3, 50)
}

positions = range(1, len(groups) + 1)
box_plot = axs[1, 1].boxplot(groups.values(), positions=positions,
                              patch_artist=True, notch=True)

# Color boxes
colors = ['#ABD9E9', '#FDAE61', '#D7191C']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

axs[1, 1].set_xticklabels(groups.keys())
axs[1, 1].set_ylabel('Response')
axs[1, 1].set_title('(d) Treatment Comparison')
axs[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Scientific Data Analysis Dashboard', fontsize=16, y=0.995)
plt.savefig('scientific_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('scientific_analysis.pdf', format='pdf', bbox_inches='tight')  # Vector format
plt.show()
```

#### 3D Surface Plots for Scientific Data
```python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(12, 5))

# Left: 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R) / R

surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True, alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Surface: Sinc Function')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Right: Contour plot with filled contours
ax2 = fig.add_subplot(122)

contour_filled = ax2.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
contour_lines = ax2.contour(X, Y, Z, levels=10, colors='black',
                             linewidths=0.5, alpha=0.4)
ax2.clabel(contour_lines, inline=True, fontsize=8)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('2D Contour Representation')
fig.colorbar(contour_filled, ax=ax2)

plt.tight_layout()
plt.savefig('3d_visualization.png', dpi=300)
plt.show()
```

#### Custom Colormaps for Scientific Data
```python
from matplotlib.colors import LinearSegmentedColormap

# Create perceptually uniform colormap
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
          '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
n_bins = 100
cmap_name = 'scientific_diverging'
cm_scientific = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Apply to heatmap
data_matrix = np.random.randn(20, 20)

plt.figure(figsize=(10, 8))
im = plt.imshow(data_matrix, cmap=cm_scientific, aspect='auto')
plt.colorbar(im, label='Intensity')
plt.title('Heatmap with Custom Colormap')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# Add annotations for significant values
for i in range(data_matrix.shape[0]):
    for j in range(data_matrix.shape[1]):
        if abs(data_matrix[i, j]) > 2:
            plt.text(j, i, f'{data_matrix[i, j]:.1f}',
                    ha='center', va='center', color='white', fontsize=8)

plt.tight_layout()
plt.savefig('heatmap_custom_cmap.png', dpi=300)
plt.show()
```

### 2. Seaborn - Statistical Data Visualization

#### Advanced Statistical Plots
```python
import seaborn as sns
import pandas as pd

# Set publication style
sns.set_theme(style='whitegrid', palette='Set2')
sns.set_context('paper', font_scale=1.2)

# Generate sample data
np.random.seed(42)
df = pd.DataFrame({
    'experiment': np.repeat(['Control', 'Treatment A', 'Treatment B'], 100),
    'time': np.tile(np.repeat([0, 1, 2, 3, 4], 20), 3),
    'response': np.concatenate([
        np.random.normal(10, 2, 100),
        np.random.normal(12 + np.repeat([0, 1, 2, 3, 4], 20) * 0.5, 2.5, 100),
        np.random.normal(15 + np.repeat([0, 1, 2, 3, 4], 20) * 1, 3, 100)
    ]),
    'subject_id': np.tile(range(20), 15)
})

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Violin plot with swarm overlay
ax1 = sns.violinplot(data=df, x='time', y='response', hue='experiment',
                     split=False, ax=axs[0, 0], inner='quartile')
axs[0, 0].set_title('Response Over Time by Treatment')
axs[0, 0].set_xlabel('Time Point')
axs[0, 0].set_ylabel('Response Value')

# 2. Line plot with confidence intervals
ax2 = sns.lineplot(data=df, x='time', y='response', hue='experiment',
                   err_style='band', markers=True, dashes=False,
                   ax=axs[0, 1], ci=95)
axs[0, 1].set_title('Longitudinal Response (95% CI)')
axs[0, 1].set_xlabel('Time Point')
axs[0, 1].set_ylabel('Response Value')

# 3. Joint distribution plot
# Create subset for joint plot
df_subset = df[df['time'] == 4]
g = sns.JointGrid(data=df_subset, x='subject_id', y='response',
                  hue='experiment', height=5)
g.plot_joint(sns.scatterplot, alpha=0.6)
g.plot_marginals(sns.kdeplot, fill=True, alpha=0.5)
g.figure.suptitle('Final Time Point Distribution', y=1.02)

# 4. Correlation heatmap
pivot_data = df.pivot_table(values='response', index='subject_id',
                             columns='time', aggfunc='mean')
ax4 = sns.heatmap(pivot_data.corr(), annot=True, fmt='.2f',
                  cmap='coolwarm', center=0, ax=axs[1, 1],
                  square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
axs[1, 1].set_title('Time Point Correlations')

plt.tight_layout()
plt.savefig('seaborn_statistical_plots.png', dpi=300)
plt.show()
```

#### FacetGrid for Multi-dimensional Analysis
```python
# Create complex dataset
df_complex = pd.DataFrame({
    'x': np.random.randn(1000),
    'y': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'group': np.random.choice(['Group 1', 'Group 2'], 1000)
})

# Create FacetGrid
g = sns.FacetGrid(df_complex, col='category', row='group',
                  hue='category', height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.6, s=30)
g.map_dataframe(sns.kdeplot, x='x', y='y', levels=5, alpha=0.5)

g.add_legend()
g.set_axis_labels('X Variable', 'Y Variable')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.suptitle('Multi-dimensional Data Exploration', y=1.01)

plt.savefig('facetgrid_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3. Plotly - Interactive Web-Based Visualizations

#### Interactive 3D Scientific Visualization
```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Create 3D surface with interactive controls
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Viridis',
    name='Sinc Function',
    showscale=True,
    colorbar=dict(title='Amplitude', x=1.1)
)])

# Add contour projections
fig.add_trace(go.Contour(
    x=x, y=y, z=Z,
    showscale=False,
    colorscale='Viridis',
    opacity=0.4,
    contours_z=dict(show=True, usecolormap=True, project_z=True)
))

fig.update_layout(
    title='Interactive 3D Surface Visualization',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    width=900,
    height=700
)

fig.write_html('interactive_3d_surface.html')
fig.show()
```

#### Real-Time Streaming Data Dashboard
```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create dashboard with multiple subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Time Series', 'Histogram', 'Scatter Plot', 'Box Plot'),
    specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
           [{'type': 'scatter'}, {'type': 'box'}]]
)

# Generate streaming data simulation
time_points = np.linspace(0, 10, 100)
signal = np.sin(time_points) + 0.1 * np.random.randn(100)

# Time series
fig.add_trace(
    go.Scatter(x=time_points, y=signal, mode='lines+markers',
               name='Signal', line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

# Histogram
fig.add_trace(
    go.Histogram(x=signal, nbinsx=30, name='Distribution',
                 marker_color='#ff7f0e'),
    row=1, col=2
)

# Scatter with trend
x_data = np.random.randn(100)
y_data = 2 * x_data + np.random.randn(100) * 0.5

fig.add_trace(
    go.Scatter(x=x_data, y=y_data, mode='markers',
               name='Data Points', marker=dict(size=8, color='#2ca02c')),
    row=2, col=1
)

# Box plots
categories = ['A', 'B', 'C']
for i, cat in enumerate(categories):
    data = np.random.normal(10 + i*2, 2, 50)
    fig.add_trace(
        go.Box(y=data, name=cat, marker_color=px.colors.qualitative.Set1[i]),
        row=2, col=2
    )

fig.update_layout(
    title_text='Real-Time Data Monitoring Dashboard',
    showlegend=True,
    height=800,
    width=1200
)

fig.write_html('streaming_dashboard.html')
fig.show()
```

#### Animated Time Series
```python
# Create animation of evolving data
frames = []
time_steps = 50

for t in range(time_steps):
    x = np.linspace(0, 10, 100)
    y = np.sin(x - t * 0.2) * np.exp(-t * 0.02)

    frames.append(go.Frame(
        data=[go.Scatter(x=x, y=y, mode='lines',
                        line=dict(color='blue', width=3))],
        name=f'frame_{t}'
    ))

fig = go.Figure(
    data=[go.Scatter(x=x, y=np.sin(x), mode='lines',
                    line=dict(color='blue', width=3))],
    frames=frames,
    layout=go.Layout(
        title='Animated Wave Propagation',
        xaxis=dict(title='Position', range=[0, 10]),
        yaxis=dict(title='Amplitude', range=[-1.5, 1.5]),
        updatemenus=[dict(
            type='buttons',
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=50, redraw=True),
                                     fromcurrent=True)]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode='immediate')])
            ]
        )]
    )
)

fig.write_html('animated_wave.html')
fig.show()
```

### 4. Bokeh - Large Dataset Visualization

#### Interactive Large-Scale Scatter Plot
```python
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import HoverTool, ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from bokeh.layouts import column, row

# Generate large dataset
n_points = 10000
x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = np.random.rand(n_points)
sizes = np.random.randint(5, 20, n_points)

source = ColumnDataSource(data=dict(
    x=x, y=y, colors=colors, sizes=sizes,
    labels=[f'Point {i}' for i in range(n_points)]
))

# Create figure with tools
output_file('bokeh_large_scatter.html')

p = figure(title='Interactive Large-Scale Scatter Plot',
           width=800, height=600,
           tools='pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,save')

# Color mapper
mapper = linear_cmap(field_name='colors', palette=Viridis256,
                     low=min(colors), high=max(colors))

# Add scatter with color mapping
scatter = p.circle('x', 'y', size='sizes', source=source,
                   fill_color=mapper, fill_alpha=0.6,
                   line_color=None, legend_label='Data Points')

# Add hover tool
hover = HoverTool(tooltips=[
    ('Label', '@labels'),
    ('X', '@x{0.00}'),
    ('Y', '@y{0.00}'),
    ('Value', '@colors{0.000}')
])
p.add_tools(hover)

# Styling
p.xaxis.axis_label = 'X Coordinate'
p.yaxis.axis_label = 'Y Coordinate'
p.legend.location = 'top_right'

# Add color bar
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,
                     location=(0, 0), title='Value')
p.add_layout(color_bar, 'right')

save(p)
show(p)
```

#### Real-Time Streaming with Bokeh Server
```python
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import DatetimeTickFormatter
import numpy as np
from datetime import datetime

def streaming_app(doc):
    """Bokeh server application for real-time data streaming."""

    # Initial data
    source = ColumnDataSource(data=dict(
        time=[], value=[], baseline=[]
    ))

    # Create figure
    p = figure(title='Real-Time Data Stream',
               x_axis_type='datetime',
               width=800, height=400,
               tools='pan,box_zoom,reset,save')

    p.line('time', 'value', source=source, line_width=2,
           color='blue', legend_label='Signal')
    p.line('time', 'baseline', source=source, line_width=1,
           color='red', line_dash='dashed', legend_label='Baseline')

    p.xaxis.formatter = DatetimeTickFormatter(
        seconds='%H:%M:%S',
        minutes='%H:%M',
        hours='%H:%M'
    )
    p.yaxis.axis_label = 'Value'
    p.legend.location = 'top_left'

    def update():
        """Update data callback."""
        new_time = datetime.now()
        new_value = np.sin(len(source.data['time']) * 0.1) + \
                    0.1 * np.random.randn()
        baseline = 0.0

        # Update source data
        new_data = dict(
            time=source.data['time'] + [new_time],
            value=source.data['value'] + [new_value],
            baseline=source.data['baseline'] + [baseline]
        )

        # Keep last 100 points
        if len(new_data['time']) > 100:
            new_data = {key: values[-100:] for key, values in new_data.items()}

        source.data = new_data

    # Add periodic callback (update every 100ms)
    doc.add_periodic_callback(update, 100)
    doc.add_root(p)

# To run: bokeh serve --show script_name.py
# Then access at http://localhost:5006/script_name
```

## Julia Visualization Ecosystem

### 1. Plots.jl - Unified Plotting Interface

#### Publication-Quality Plots
```julia
using Plots, Statistics, Distributions

# Set backend (GR for speed, PlotlyJS for interactivity)
gr()  # or plotlyjs()

# Publication style
default(
    fontfamily="Computer Modern",
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    legendfontsize=10,
    dpi=300,
    size=(800, 600)
)

# Multi-panel scientific figure
p1 = plot(
    0:0.1:10, [sin, cos, x -> sin(x) * cos(x)],
    label=["sin(x)" "cos(x)" "sin(x)cos(x)"],
    xlabel="x",
    ylabel="f(x)",
    title="(a) Trigonometric Functions",
    lw=2,
    legend=:topright,
    grid=true
)

# Scatter with error bars
x = 1:10
y = @. 2x + randn() * 2
yerr = rand(length(x)) .* 2

p2 = scatter(
    x, y,
    yerror=yerr,
    xlabel="X Variable",
    ylabel="Y Variable",
    title="(b) Data with Error Bars",
    markersize=6,
    markerstrokewidth=1.5,
    label="Measurements",
    legend=:topleft
)

# Add trend line
coeffs = [ones(length(x)) x] \ y
plot!(p2, x, coeffs[1] .+ coeffs[2] .* x,
      lw=2, ls=:dash, label="Linear Fit", color=:red)

# Histogram with KDE
data = randn(1000) .* 10 .+ 50

p3 = histogram(
    data,
    bins=30,
    normalize=:pdf,
    alpha=0.6,
    xlabel="Value",
    ylabel="Density",
    title="(c) Distribution Analysis",
    label="Histogram"
)

# Add KDE
using KernelDensity
kde_result = kde(data)
plot!(p3, kde_result.x, kde_result.density,
      lw=2, label="KDE", color=:black)

# Box plot
groups = ["Control", "Treatment A", "Treatment B"]
group_data = [randn(50) .+ 10, randn(50) .+ 12, randn(50) .+ 15]

p4 = boxplot(
    repeat(groups, inner=50),
    vcat(group_data...),
    xlabel="Group",
    ylabel="Response",
    title="(d) Group Comparison",
    fillalpha=0.75,
    linewidth=1.5,
    legend=false
)

# Combine into figure
plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
savefig("julia_scientific_figure.png")
savefig("julia_scientific_figure.pdf")
```

#### 3D Surface and Contour Plots
```julia
using Plots

# 3D surface data
x = y = range(-5, 5, length=100)
f(x, y) = sin(√(x^2 + y^2)) / √(x^2 + y^2 + 0.1)
z = [f(xi, yi) for xi in x, yi in y]

# 3D surface plot
p1 = surface(
    x, y, z,
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
    title="3D Surface: Sinc Function",
    c=:viridis,
    camera=(30, 60),
    colorbar_title="Amplitude"
)

# Contour plot
p2 = contourf(
    x, y, z,
    xlabel="X",
    ylabel="Y",
    title="2D Contour Representation",
    levels=20,
    c=:RdYlBu,
    colorbar_title="Value"
)

# Add contour lines
contour!(p2, x, y, z, levels=10, color=:black, lw=0.5, alpha=0.5)

plot(p1, p2, layout=(1, 2), size=(1400, 600))
savefig("julia_3d_visualization.png")
```

### 2. Makie.jl - High-Performance GPU-Accelerated Visualization

#### Interactive Real-Time Visualization
```julia
using GLMakie

# Create figure
fig = Figure(resolution=(1200, 800))

# Time-varying 3D surface
n = 100
x = y = range(-3, 3, length=n)
z = Observable(zeros(n, n))

# Initial surface
t = Observable(0.0)
for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
    z[][i, j] = sin(√(xi^2 + yj^2) - t[])
end

# 3D surface plot
ax1 = Axis3(fig[1, 1:2], title="Real-Time 3D Surface",
            xlabel="X", ylabel="Y", zlabel="Z")
surface!(ax1, x, y, z, colormap=:viridis, shading=true)

# 2D heatmap
ax2 = Axis(fig[2, 1], title="Heatmap View",
           xlabel="X", ylabel="Y")
heatmap!(ax2, x, y, z, colormap=:plasma)

# Line plot showing slice
slice_y_idx = Observable(n ÷ 2)
slice_data = Observable(z[][: slice_y_idx[]])

ax3 = Axis(fig[2, 2], title="Y-Slice",
           xlabel="X", ylabel="Z")
lines!(ax3, x, slice_data, color=:blue, linewidth=2)

# Animation loop
record(fig, "makie_animation.mp4", 1:100; framerate=30) do frame
    t[] = frame * 0.1

    # Update surface
    for (i, xi) in enumerate(x), (j, yj) in enumerate(y)
        z[][i, j] = sin(√(xi^2 + yj^2) - t[])
    end
    notify(z)

    # Update slice
    slice_data[] = z[][: slice_y_idx[]]
    notify(slice_data)
end

display(fig)
```

#### Scientific Data Dashboard
```julia
using GLMakie, Statistics

# Create interactive dashboard
fig = Figure(resolution=(1600, 1000), fontsize=14)

# Generate sample data
n_points = 1000
x_data = randn(n_points)
y_data = 2 .* x_data .+ randn(n_points) .* 0.5
colors = (x_data .^ 2 .+ y_data .^ 2) .^ 0.5

# Scatter plot with color mapping
ax1 = Axis(fig[1, 1], title="Scatter Plot with Density",
           xlabel="X Variable", ylabel="Y Variable")
scatter!(ax1, x_data, y_data, color=colors,
         colormap=:plasma, markersize=10, alpha=0.6)
Colorbar(fig[1, 2], limits=(minimum(colors), maximum(colors)),
         colormap=:plasma, label="Distance from Origin")

# Histograms
ax2 = Axis(fig[2, 1], title="X Distribution",
           xlabel="X", ylabel="Frequency")
hist!(ax2, x_data, bins=30, color=(:blue, 0.5),
      strokewidth=1, strokecolor=:black)

ax3 = Axis(fig[2, 2], title="Y Distribution",
           xlabel="Y", ylabel="Frequency")
hist!(ax3, y_data, bins=30, color=(:red, 0.5),
      strokewidth=1, strokecolor=:black)

# 2D density plot
ax4 = Axis(fig[3, 1:2], title="2D Density Estimation",
           xlabel="X", ylabel="Y")
hexbin!(ax4, x_data, y_data, bins=30, colormap=:thermal)

# Statistics panel
stats_text = """
Statistics Summary:
X: μ = $(round(mean(x_data), digits=3)), σ = $(round(std(x_data), digits=3))
Y: μ = $(round(mean(y_data), digits=3)), σ = $(round(std(y_data), digits=3))
Correlation: r = $(round(cor(x_data, y_data), digits=3))
N = $n_points
"""

Label(fig[1:3, 3], stats_text, tellwidth=false,
      justification=:left, padding=(10, 10, 10, 10))

save("makie_dashboard.png", fig)
display(fig)
```

### 3. Interactive Notebooks - Jupyter and Pluto.jl

#### Jupyter Notebook with ipywidgets (Python)
```python
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

# Create interactive plot
@widgets.interact(
    frequency=(0.1, 5.0, 0.1),
    amplitude=(0.1, 2.0, 0.1),
    phase=(0, 2*np.pi, 0.1)
)
def interactive_wave(frequency=1.0, amplitude=1.0, phase=0.0):
    x = np.linspace(0, 10, 1000)
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Wave: A={amplitude:.1f}, f={frequency:.1f}Hz, φ={phase:.2f}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-2.5, 2.5])
    plt.show()

# Dashboard with multiple widgets
frequency_slider = widgets.FloatSlider(
    value=1.0, min=0.1, max=5.0, step=0.1,
    description='Frequency:', continuous_update=False
)

amplitude_slider = widgets.FloatSlider(
    value=1.0, min=0.1, max=2.0, step=0.1,
    description='Amplitude:', continuous_update=False
)

phase_slider = widgets.FloatSlider(
    value=0.0, min=0.0, max=2*np.pi, step=0.1,
    description='Phase:', continuous_update=False
)

output = widgets.interactive_output(
    interactive_wave,
    {'frequency': frequency_slider,
     'amplitude': amplitude_slider,
     'phase': phase_slider}
)

display(widgets.VBox([frequency_slider, amplitude_slider,
                      phase_slider, output]))
```

#### Pluto.jl Reactive Notebook (Julia)
```julia
### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity.
using PlutoUI
using Plots

md"""
# Interactive Scientific Visualization

Adjust parameters to explore the wave function:
"""

@bind frequency Slider(0.1:0.1:5.0, default=1.0, show_value=true)
@bind amplitude Slider(0.1:0.1:2.0, default=1.0, show_value=true)
@bind phase Slider(0:0.1:2π, default=0.0, show_value=true)

# Reactive plot (updates automatically when sliders change)
begin
    x = 0:0.01:10
    y = @. amplitude * sin(2π * frequency * x + phase)

    plot(x, y,
         xlabel="Time (s)",
         ylabel="Amplitude",
         title="Wave Function: A=$(round(amplitude, digits=2)), " *
               "f=$(round(frequency, digits=2))Hz, " *
               "φ=$(round(phase, digits=2))",
         lw=2,
         legend=false,
         ylims=(-2.5, 2.5),
         size=(800, 400))
end

md"""
## Statistics
"""

begin
    mean_val = mean(y)
    std_val = std(y)
    max_val = maximum(y)
    min_val = minimum(y)

    md"""
    - Mean: $(round(mean_val, digits=3))
    - Std Dev: $(round(std_val, digits=3))
    - Max: $(round(max_val, digits=3))
    - Min: $(round(min_val, digits=3))
    """
end
```

## Best Practices

### Publication-Quality Standards

1. **Resolution**: Always use 300 DPI for publications
2. **Fonts**: Use serif fonts (Computer Modern, Times New Roman)
3. **Colors**: Use colorblind-friendly palettes (ColorBrewer, Viridis)
4. **Size**: Match journal column width (usually 3.5" or 7" wide)
5. **Format**: Save as vector (PDF, SVG) for scalability
6. **Labels**: Clear axis labels with units
7. **Legends**: Descriptive and well-positioned

### Performance Optimization

1. **Large datasets**: Use Bokeh or Makie.jl for 100k+ points
2. **Real-time**: Limit update frequency, use downsampling
3. **3D**: Use GPU acceleration (Makie.jl, Three.js)
4. **Interactive**: Lazy loading, progressive rendering
5. **Memory**: Stream data, don't load everything at once

### Color Theory for Science

```python
# Recommended colormaps by data type

# Sequential (single hue, increasing intensity)
# Use for: ordered data, heat maps
sequential = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

# Diverging (two hues with neutral center)
# Use for: data with meaningful center (zero, average)
diverging = ['RdBu', 'RdYlBu', 'BrBG', 'PiYG', 'coolwarm']

# Qualitative (distinct colors)
# Use for: categorical data, different groups
qualitative = ['Set1', 'Set2', 'Set3', 'Paired', 'tab10']

# Avoid jet/rainbow - not perceptually uniform
```

## Integration Examples

### Python-Julia Bridge for Visualization
```julia
# Call Python from Julia using PyCall
using PyCall

# Import matplotlib
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")

# Create data in Julia
x = range(0, 10, length=100)
y = sin.(x)

# Plot with matplotlib
plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Julia Data, Python Plot")
plt.grid(true)
plt.savefig("julia_python_plot.png", dpi=300)
plt.show()
```

### Export for Web Integration
```python
# Create interactive Plotly figure
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])

# Export as HTML (embeddable in web pages)
fig.write_html('plot.html', include_plotlyjs='cdn')

# Export as JSON (for custom JavaScript integration)
fig.write_json('plot.json')

# Export as static image
fig.write_image('plot.png', width=1200, height=800, scale=2)
```

This skill provides comprehensive tools for creating professional scientific visualizations in both Python and Julia ecosystems!
