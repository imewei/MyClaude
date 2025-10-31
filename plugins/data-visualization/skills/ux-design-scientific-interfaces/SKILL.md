---
name: ux-design-scientific-interfaces
description: Design intuitive, accessible, and user-centered interfaces for scientific tools, research applications, and data analysis platforms. Use this skill when designing or building interactive dashboards with Plotly Dash (Python) or Streamlit for data exploration, creating reactive Pluto.jl notebooks for Julia-based scientific workflows, implementing WCAG 2.1 AA accessibility standards (color contrast ratios, keyboard navigation, screen reader support with ARIA labels), designing Jupyter notebook widgets with ipywidgets for parameter exploration, building command-line interfaces (CLI) for scientific tools with argparse or Click, creating Figma prototypes or wireframes for scientific applications, implementing usability testing frameworks to measure task success rates and user performance, designing progressive disclosure patterns that minimize cognitive load for researchers, implementing reproducibility features (save/load parameter sets, export capabilities in CSV/JSON/HDF5), creating help documentation with inline examples and scientific tooltips, designing responsive layouts that work on tablets for field research, implementing keyboard shortcuts for power users, designing clear information hierarchies for complex scientific data, creating batch processing workflows for multiple datasets, implementing undo/redo functionality for safe experimentation, or when working with dashboard application files (app.py for Dash, streamlit_app.py for Streamlit) that require user-friendly design for non-technical researchers.
tools: Read, Write, python, julia, Figma, streamlit, dash
integration: Use for designing scientist-friendly interfaces that minimize cognitive load
---

# UX Design for Scientific Interfaces

Complete framework for designing intuitive, accessible, and efficient user interfaces for scientific applications, dashboards, and interactive tools.

## When to use this skill

- Designing or building interactive Plotly Dash applications (app.py files) for scientific data exploration
- Creating Streamlit applications (streamlit_app.py files) for rapid prototyping of research tools
- Developing reactive Pluto.jl notebooks with @bind widgets for Julia-based scientific workflows
- Implementing WCAG 2.1 AA accessibility standards for scientific web applications
- Designing color schemes with proper contrast ratios (4.5:1 for normal text, 3:1 for large text)
- Implementing keyboard navigation and focus indicators for accessible scientific tools
- Adding ARIA labels and screen reader support for complex scientific visualizations
- Building Jupyter notebook widgets using ipywidgets for interactive parameter exploration
- Designing command-line interfaces (CLI) for scientific tools with argparse, Click, or Typer
- Creating Figma prototypes or wireframes for scientific applications before implementation
- Conducting usability testing with researchers to measure task success rates and completion times
- Designing progressive disclosure patterns that show simple options first, complexity when needed
- Implementing reproducibility features (save/load analysis states, export parameter configurations)
- Creating export functionality supporting multiple formats (CSV, JSON, HDF5, PDF reports)
- Designing clear information hierarchies that prioritize critical scientific data
- Implementing batch processing workflows for applying analysis to multiple datasets
- Adding undo/redo functionality for safe experimentation without data loss
- Creating inline help documentation with scientific tooltips and example usage
- Designing responsive layouts that work on tablets and mobile devices for field research
- Implementing keyboard shortcuts (Alt+key combinations) for power users
- Designing dashboards with tabs or panels for different analysis views
- Creating control panels with sliders, dropdowns, and input fields for parameter adjustment
- Implementing real-time feedback showing results as parameters change
- Designing error prevention with input validation and sensible default values
- Creating user testing frameworks to track metrics (duration, success rate, error counts)
- Implementing consistent visual patterns across scientific tools for familiarity
- Designing interfaces that support common scientific workflows (load → analyze → visualize → export)

## Core UX Principles for Scientists

### 1. Minimize Cognitive Load
- **Information hierarchy**: Most important data visible first
- **Progressive disclosure**: Hide complexity until needed
- **Consistent patterns**: Reuse familiar interface elements
- **Clear labeling**: Scientific terminology with tooltips
- **Visual grouping**: Related controls together

### 2. Support Scientific Workflows
- **Reproducibility**: Save/load parameter sets
- **Batch processing**: Apply to multiple datasets
- **Export capabilities**: Multiple formats (CSV, HDF5, JSON)
- **Undo/redo**: Safe experimentation
- **Documentation**: Inline help and examples

## Interactive Dashboards

### Dash (Python) - Web-Based Scientific Dashboards

```python
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Initialize app
app = dash.Dash(__name__)

# Layout with scientific workflow
app.layout = html.Div([
    html.H1('Scientific Data Analysis Dashboard',
            style={'textAlign': 'center', 'color': '#2C3E50'}),

    # Control panel
    html.Div([
        html.H3('Parameters'),

        # Parameter inputs
        html.Label('Sample Size:'),
        dcc.Slider(id='sample-size', min=10, max=1000, step=10,
                   value=100, marks={i: str(i) for i in range(0, 1001, 200)},
                   tooltip={'placement': 'bottom', 'always_visible': True}),

        html.Label('Noise Level (σ):'),
        dcc.Input(id='noise-level', type='number', value=0.1,
                  min=0, max=1, step=0.01,
                  style={'width': '100%', 'margin': '10px 0'}),

        html.Label('Distribution:'),
        dcc.Dropdown(
            id='distribution',
            options=[
                {'label': 'Normal', 'value': 'normal'},
                {'label': 'Exponential', 'value': 'exponential'},
                {'label': 'Uniform', 'value': 'uniform'}
            ],
            value='normal',
            clearable=False
        ),

        # Action buttons
        html.Div([
            html.Button('Generate Data', id='generate-btn',
                       n_clicks=0,
                       style={'marginRight': '10px',
                             'backgroundColor': '#3498DB',
                             'color': 'white',
                             'padding': '10px 20px',
                             'border': 'none',
                             'cursor': 'pointer'}),

            html.Button('Export CSV', id='export-btn',
                       n_clicks=0,
                       style={'backgroundColor': '#27AE60',
                             'color': 'white',
                             'padding': '10px 20px',
                             'border': 'none',
                             'cursor': 'pointer'}),
        ], style={'marginTop': '20px'}),

    ], style={'width': '25%', 'display': 'inline-block',
             'verticalAlign': 'top', 'padding': '20px',
             'backgroundColor': '#ECF0F1'}),

    # Visualization panel
    html.Div([
        # Tabs for different views
        dcc.Tabs(id='tabs', value='histogram', children=[
            dcc.Tab(label='Distribution', value='histogram'),
            dcc.Tab(label='Q-Q Plot', value='qq'),
            dcc.Tab(label='Statistics', value='stats')
        ]),

        # Graph output
        html.Div(id='tabs-content'),

    ], style={'width': '70%', 'display': 'inline-block',
             'padding': '20px'}),

    # Hidden div to store data
    dcc.Store(id='data-store'),

    # Download component
    dcc.Download(id='download-data')
])

@app.callback(
    Output('data-store', 'data'),
    Input('generate-btn', 'n_clicks'),
    State('sample-size', 'value'),
    State('noise-level', 'value'),
    State('distribution', 'value')
)
def generate_data(n_clicks, n, noise, dist):
    """Generate scientific data based on parameters."""
    if dist == 'normal':
        data = np.random.normal(0, 1, n) + noise * np.random.randn(n)
    elif dist == 'exponential':
        data = np.random.exponential(1, n) + noise * np.random.randn(n)
    else:
        data = np.random.uniform(-1, 1, n) + noise * np.random.randn(n)

    return {'data': data.tolist(), 'n': n, 'noise': noise, 'dist': dist}

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value'),
    Input('data-store', 'data')
)
def render_content(tab, stored_data):
    """Render content based on selected tab."""
    if stored_data is None:
        return html.Div('Click "Generate Data" to start')

    data = np.array(stored_data['data'])

    if tab == 'histogram':
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Data',
                                   marker_color='#3498DB'))

        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_values = kde(x_range)

        # Scale KDE to histogram
        hist, bins = np.histogram(data, bins=30, density=False)
        kde_scaled = kde_values * (hist.max() / kde_values.max())

        fig.add_trace(go.Scatter(x=x_range, y=kde_scaled,
                                mode='lines', name='KDE',
                                line=dict(color='red', width=2)))

        fig.update_layout(
            title='Data Distribution',
            xaxis_title='Value',
            yaxis_title='Frequency',
            hovermode='closest'
        )

        return dcc.Graph(figure=fig)

    elif tab == 'qq':
        # Q-Q plot
        from scipy import stats
        fig = go.Figure()

        # Theoretical quantiles
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm')

        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                                name='Data', marker=dict(size=6)))

        # Theoretical line
        fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept,
                                mode='lines', name='Theoretical',
                                line=dict(color='red', dash='dash')))

        fig.update_layout(
            title=f'Q-Q Plot (R² = {r**2:.4f})',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles'
        )

        return dcc.Graph(figure=fig)

    else:  # stats tab
        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        return html.Div([
            html.H3('Descriptive Statistics'),
            html.Table([
                html.Tr([html.Td('Mean:'), html.Td(f'{mean:.4f}')]),
                html.Tr([html.Td('Median:'), html.Td(f'{median:.4f}')]),
                html.Tr([html.Td('Std Dev:'), html.Td(f'{std:.4f}')]),
                html.Tr([html.Td('Skewness:'), html.Td(f'{skew:.4f}')]),
                html.Tr([html.Td('Kurtosis:'), html.Td(f'{kurt:.4f}')]),
                html.Tr([html.Td('Sample Size:'), html.Td(f'{len(data)}')]),
            ], style={'width': '50%', 'fontSize': '16px'})
        ])

@app.callback(
    Output('download-data', 'data'),
    Input('export-btn', 'n_clicks'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def export_data(n_clicks, stored_data):
    """Export data to CSV."""
    if stored_data is None:
        return None

    df = pd.DataFrame({'value': stored_data['data']})
    return dcc.send_data_frame(df.to_csv, 'data_export.csv', index=False)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

### Streamlit - Rapid Prototyping

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title='Scientific Analysis Tool',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for scientific look
st.markdown("""
<style>
    .main {background-color: #F8F9FA;}
    h1 {color: #2C3E50; text-align: center;}
    .stButton>button {
        background-color: #3498DB;
        color: white;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title('🔬 Scientific Data Analysis Platform')

# Sidebar controls
with st.sidebar:
    st.header('⚙️ Parameters')

    # File upload
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

    if uploaded_file is None:
        st.subheader('Generate Sample Data')

        n_samples = st.slider('Sample Size', 10, 1000, 100)
        noise_level = st.number_input('Noise Level (σ)', 0.0, 1.0, 0.1, 0.01)

        distribution = st.selectbox(
            'Distribution',
            ['Normal', 'Exponential', 'Uniform']
        )

        if st.button('Generate Data'):
            if distribution == 'Normal':
                data = np.random.normal(0, 1, n_samples)
            elif distribution == 'Exponential':
                data = np.random.exponential(1, n_samples)
            else:
                data = np.random.uniform(-1, 1, n_samples)

            data += noise_level * np.random.randn(n_samples)
            st.session_state['data'] = data
            st.success(f'Generated {n_samples} samples!')

    else:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df.iloc[:, 0].values
        st.success(f'Loaded {len(df)} samples!')

# Main content area
if 'data' in st.session_state:
    data = st.session_state['data']

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(['📊 Distribution', '📈 Analysis', '📋 Statistics'])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(data, bins=30, density=True, alpha=0.7,
                   color='#3498DB', edgecolor='black')

            # KDE overlay
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title('Distribution with KDE')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        with col2:
            # Box plot and violin plot
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))

            axs[0].boxplot(data, vert=True)
            axs[0].set_ylabel('Value')
            axs[0].set_title('Box Plot')
            axs[0].grid(True, alpha=0.3)

            parts = axs[1].violinplot([data], showmeans=True, showmedians=True)
            axs[1].set_ylabel('Value')
            axs[1].set_title('Violin Plot')
            axs[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Q-Q plot
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(data, dist='norm', plot=ax)
            ax.set_title('Q-Q Plot')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            # Autocorrelation
            from statsmodels.graphics.tsaplots import plot_acf

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_acf(data, lags=40, ax=ax)
            ax.set_title('Autocorrelation Function')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    with tab3:
        # Statistics table
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric('Mean', f'{np.mean(data):.4f}')
            st.metric('Median', f'{np.median(data):.4f}')

        with col2:
            st.metric('Std Dev', f'{np.std(data):.4f}')
            st.metric('Variance', f'{np.var(data):.4f}')

        with col3:
            st.metric('Skewness', f'{stats.skew(data):.4f}')
            st.metric('Kurtosis', f'{stats.kurtosis(data):.4f}')

        # Detailed statistics
        st.subheader('Detailed Statistics')

        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [
                len(data),
                np.mean(data),
                np.std(data),
                np.min(data),
                np.percentile(data, 25),
                np.percentile(data, 50),
                np.percentile(data, 75),
                np.max(data)
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

        # Export button
        csv = pd.DataFrame({'value': data}).to_csv(index=False)
        st.download_button(
            label='📥 Download Data (CSV)',
            data=csv,
            file_name='analysis_data.csv',
            mime='text/csv'
        )

else:
    st.info('👈 Configure parameters and generate/upload data to begin analysis')

    # Show example
    st.subheader('Example Usage')
    st.code("""
    1. Set sample size and noise level in sidebar
    2. Choose distribution type
    3. Click 'Generate Data'
    4. Explore visualizations in tabs
    5. Export results as needed
    """)
```

### Pluto.jl - Julia Reactive Notebooks

```julia
### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils
using PlutoUI
using Plots, Statistics, Distributions, StatsPlots

md"""
# 🔬 Scientific Data Explorer

Interactive notebook for analyzing experimental data
"""

md"""
## Data Generation

Configure your dataset:
"""

@bind n_samples Slider(10:10:1000, default=100, show_value=true)
@bind noise_level Slider(0.0:0.01:1.0, default=0.1, show_value=true)
@bind dist_type Select(["Normal", "Exponential", "Uniform"], default="Normal")

# Generate data reactively
data = begin
    if dist_type == "Normal"
        randn(n_samples) .+ noise_level .* randn(n_samples)
    elseif dist_type == "Exponential"
        rand(Exponential(1), n_samples) .+ noise_level .* randn(n_samples)
    else
        rand(Uniform(-1, 1), n_samples) .+ noise_level .* randn(n_samples)
    end
end

md"""
## Visualization Dashboard
"""

begin
    # Multi-panel visualization
    p1 = histogram(data, bins=30, normalize=:pdf, alpha=0.6,
                   label="Data", xlabel="Value", ylabel="Density",
                   title="Distribution")

    # Overlay KDE
    kde_x = range(minimum(data), maximum(data), length=200)
    kde_y = pdf(kde(data), kde_x)
    plot!(p1, kde_x, kde_y, linewidth=2, label="KDE", color=:red)

    # Box and violin plots
    p2 = boxplot([data], label="", ylabel="Value", title="Box Plot")

    p3 = violin([data], label="", ylabel="Value", title="Violin Plot")

    # Q-Q plot
    using StatsPlots
    p4 = qqplot(Normal, data, title="Q-Q Plot",
                xlabel="Theoretical", ylabel="Sample")

    plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
end

md"""
## Statistical Summary
"""

begin
    summary_md = md"""
    | Statistic | Value |
    |-----------|-------|
    | **Sample Size** | $(length(data)) |
    | **Mean** | $(round(mean(data), digits=4)) |
    | **Median** | $(round(median(data), digits=4)) |
    | **Std Dev** | $(round(std(data), digits=4)) |
    | **Skewness** | $(round(skewness(data), digits=4)) |
    | **Kurtosis** | $(round(kurtosis(data), digits=4)) |
    | **Min** | $(round(minimum(data), digits=4)) |
    | **Max** | $(round(maximum(data), digits=4)) |
    """
    summary_md
end

md"""
## Export Options
"""

@bind export_button Button("📥 Export Data")

begin
    export_button  # Trigger on click

    using CSV, DataFrames
    df = DataFrame(value=data)
    CSV.write("exported_data.csv", df)

    md"✅ Data exported to `exported_data.csv`"
end
```

## Accessibility Standards (WCAG 2.1 AA)

### Color Contrast Ratios

```python
def check_contrast_ratio(fg_hex, bg_hex):
    """
    Calculate WCAG contrast ratio between foreground and background colors.

    WCAG AA requires:
    - 4.5:1 for normal text
    - 3:1 for large text (18pt+ or 14pt+ bold)
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def relative_luminance(rgb):
        def adjust(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * adjust(rgb[0]) + 0.7152 * adjust(rgb[1]) + 0.0722 * adjust(rgb[2])

    fg_lum = relative_luminance(hex_to_rgb(fg_hex))
    bg_lum = relative_luminance(hex_to_rgb(bg_hex))

    lighter = max(fg_lum, bg_lum)
    darker = min(fg_lum, bg_lum)

    ratio = (lighter + 0.05) / (darker + 0.05)

    return ratio

# Test color combinations
test_pairs = [
    ('#FFFFFF', '#000000'),  # Black on white
    ('#2C3E50', '#ECF0F1'),  # Dark gray on light gray
    ('#3498DB', '#FFFFFF'),  # Blue on white
]

for fg, bg in test_pairs:
    ratio = check_contrast_ratio(fg, bg)
    aa_normal = '✓' if ratio >= 4.5 else '✗'
    aa_large = '✓' if ratio >= 3.0 else '✗'
    print(f"{fg} on {bg}: {ratio:.2f}:1 | Normal: {aa_normal} | Large: {aa_large}")
```

### Keyboard Navigation

```python
# Dash example with keyboard shortcuts
app.layout = html.Div([
    html.Button('Generate (Alt+G)', id='gen-btn',
                accessKey='g',  # Alt+G shortcut
                n_clicks=0),

    html.Button('Export (Alt+E)', id='export-btn',
                accessKey='e',  # Alt+E shortcut
                n_clicks=0),

    # Focus indicators via CSS
    html.Style("""
        button:focus {
            outline: 3px solid #3498DB;
            outline-offset: 2px;
        }

        input:focus, select:focus {
            border-color: #3498DB;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.25);
        }
    """)
])
```

### Screen Reader Support

```python
# ARIA labels for scientific interfaces
app.layout = html.Div([
    html.Label('Sample Size', id='sample-label'),
    dcc.Slider(
        id='sample-size',
        min=10, max=1000, value=100,
        **{'aria-labelledby': 'sample-label',
           'aria-valuemin': 10,
           'aria-valuemax': 1000,
           'aria-valuenow': 100}
    ),

    html.Div(
        id='status-message',
        role='status',  # Announces changes to screen readers
        **{'aria-live': 'polite'}
    )
])
```

## Usability Testing Framework

### User Testing Protocol

```python
class UsabilityTest:
    """Framework for conducting usability tests."""

    def __init__(self, task_list):
        self.tasks = task_list
        self.results = []

    def record_task(self, user_id, task_id, duration, success, errors):
        """Record task completion metrics."""
        self.results.append({
            'user': user_id,
            'task': task_id,
            'duration_sec': duration,
            'success': success,
            'errors': errors,
            'timestamp': pd.Timestamp.now()
        })

    def analyze_results(self):
        """Analyze usability metrics."""
        df = pd.DataFrame(self.results)

        metrics = {
            'success_rate': df['success'].mean() * 100,
            'avg_duration': df['duration_sec'].mean(),
            'avg_errors': df['errors'].mean(),
            'total_tests': len(df)
        }

        # Task-level analysis
        task_metrics = df.groupby('task').agg({
            'success': 'mean',
            'duration_sec': 'mean',
            'errors': 'mean'
        })

        return metrics, task_metrics

# Example usage
test = UsabilityTest([
    'Load dataset',
    'Generate visualization',
    'Export results',
    'Adjust parameters'
])

# Simulate user testing
test.record_task(user_id=1, task_id='Load dataset',
                duration=15.3, success=True, errors=0)
test.record_task(user_id=1, task_id='Generate visualization',
                duration=8.7, success=True, errors=1)

metrics, task_metrics = test.analyze_results()
print(f"Success Rate: {metrics['success_rate']:.1f}%")
print(f"Average Duration: {metrics['avg_duration']:.1f}s")
```

## Best Practices Summary

1. **Progressive Disclosure**: Start simple, reveal complexity gradually
2. **Immediate Feedback**: Show results as parameters change
3. **Error Prevention**: Validate inputs, provide sensible defaults
4. **Clear Labels**: Use scientific terminology with tooltips
5. **Export Options**: Multiple formats (CSV, JSON, HDF5, PDF)
6. **Keyboard Shortcuts**: Power users appreciate speed
7. **Responsive Design**: Works on tablets for field work
8. **Help Documentation**: Inline examples and tutorials
9. **Reproducibility**: Save/load complete analysis state
10. **Accessibility**: WCAG 2.1 AA minimum compliance

This skill provides comprehensive UX design patterns for scientific interfaces that balance power with usability!
