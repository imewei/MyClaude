---
name: ux-design-scientific-interfaces
version: "1.0.7"
maturity: "5-Expert"
specialization: Scientific Interface Design
description: Design intuitive, accessible interfaces for scientific tools and research applications. Use when building Plotly Dash or Streamlit dashboards, creating Pluto.jl notebooks, implementing WCAG 2.1 AA accessibility, designing Jupyter widgets, building CLIs for scientific tools, implementing progressive disclosure patterns, or creating reproducibility features.
---

# UX Design for Scientific Interfaces

User-centered design patterns for scientific dashboards and research tools.

---

## UX Principles for Scientists

| Principle | Implementation |
|-----------|----------------|
| Minimize Cognitive Load | Most important data visible first |
| Progressive Disclosure | Hide complexity until needed |
| Reproducibility | Save/load parameter sets |
| Batch Processing | Apply to multiple datasets |
| Export Options | CSV, HDF5, JSON, PDF |
| Undo/Redo | Safe experimentation |

---

## Dashboard Selection

| Tool | Use Case | Strengths |
|------|----------|-----------|
| Dash (Python) | Production apps | Full control, callbacks |
| Streamlit | Rapid prototyping | Simple, fast iteration |
| Pluto.jl | Julia workflows | Reactive, notebook-based |
| Jupyter Widgets | Exploratory | Integrated with notebooks |

---

## Dash Dashboard Pattern

```python
import dash
from dash import dcc, html, Input, Output, State

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Scientific Data Analysis'),

    # Control panel (sidebar)
    html.Div([
        html.Label('Sample Size:'),
        dcc.Slider(id='sample-size', min=10, max=1000, value=100,
                   tooltip={'placement': 'bottom', 'always_visible': True}),

        html.Label('Distribution:'),
        dcc.Dropdown(id='distribution', options=[
            {'label': 'Normal', 'value': 'normal'},
            {'label': 'Exponential', 'value': 'exponential'},
        ], value='normal'),

        html.Button('Generate', id='generate-btn', n_clicks=0),
        html.Button('Export CSV', id='export-btn', n_clicks=0),
    ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px'}),

    # Visualization panel
    html.Div([
        dcc.Tabs(id='tabs', children=[
            dcc.Tab(label='Distribution', value='histogram'),
            dcc.Tab(label='Q-Q Plot', value='qq'),
            dcc.Tab(label='Statistics', value='stats'),
        ]),
        html.Div(id='tabs-content'),
    ], style={'width': '70%', 'display': 'inline-block'}),

    dcc.Store(id='data-store'),
    dcc.Download(id='download-data'),
])

@app.callback(Output('data-store', 'data'),
              Input('generate-btn', 'n_clicks'),
              State('sample-size', 'value'),
              State('distribution', 'value'))
def generate_data(n_clicks, n, dist):
    import numpy as np
    data = np.random.normal(0, 1, n) if dist == 'normal' else np.random.exponential(1, n)
    return {'data': data.tolist()}
```

---

## Streamlit Pattern

```python
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title='Scientific Analysis', layout='wide')
st.title('🔬 Scientific Data Analysis')

# Sidebar controls
with st.sidebar:
    st.header('⚙️ Parameters')
    n_samples = st.slider('Sample Size', 10, 1000, 100)
    distribution = st.selectbox('Distribution', ['Normal', 'Exponential'])

    if st.button('Generate Data'):
        data = np.random.normal(0, 1, n_samples) if distribution == 'Normal' else np.random.exponential(1, n_samples)
        st.session_state['data'] = data

# Main content
if 'data' in st.session_state:
    data = st.session_state['data']

    tab1, tab2, tab3 = st.tabs(['📊 Distribution', '📈 Analysis', '📋 Statistics'])

    with tab1:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(data, bins=30, density=True, alpha=0.7)
        st.pyplot(fig)

    with tab3:
        col1, col2 = st.columns(2)
        col1.metric('Mean', f'{np.mean(data):.4f}')
        col2.metric('Std Dev', f'{np.std(data):.4f}')

        # Export
        csv = pd.DataFrame({'value': data}).to_csv(index=False)
        st.download_button('📥 Download CSV', csv, 'data.csv', 'text/csv')
```

---

## Pluto.jl Pattern

```julia
using PlutoUI, Plots, Statistics, Distributions

md"## Data Generation"

@bind n_samples Slider(10:10:1000, default=100, show_value=true)
@bind dist_type Select(["Normal", "Exponential"], default="Normal")

data = dist_type == "Normal" ? randn(n_samples) : rand(Exponential(1), n_samples)

# Multi-panel visualization
begin
    p1 = histogram(data, bins=30, normalize=:pdf, label="Data", title="Distribution")
    p2 = boxplot([data], label="", title="Box Plot")
    plot(p1, p2, layout=(1, 2), size=(800, 400))
end

# Statistics table
md"""
| Statistic | Value |
|-----------|-------|
| Mean | $(round(mean(data), digits=4)) |
| Std Dev | $(round(std(data), digits=4)) |
"""
```

---

## Accessibility (WCAG 2.1 AA)

### Color Contrast

```python
def check_contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """WCAG requires 4.5:1 for normal text, 3:1 for large text."""
    def hex_to_lum(hex_color):
        rgb = [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
        rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    l1, l2 = hex_to_lum(fg_hex), hex_to_lum(bg_hex)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

# Test: check_contrast_ratio('#2C3E50', '#FFFFFF') >= 4.5
```

### Keyboard Navigation

```python
html.Button('Generate (Alt+G)', id='gen-btn', accessKey='g'),
html.Button('Export (Alt+E)', id='export-btn', accessKey='e'),

# Focus indicators via CSS
html.Style("""
    button:focus { outline: 3px solid #3498DB; outline-offset: 2px; }
    input:focus { border-color: #3498DB; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.25); }
""")
```

### Screen Reader Support

```python
dcc.Slider(
    id='sample-size',
    **{'aria-labelledby': 'sample-label',
       'aria-valuemin': 10,
       'aria-valuemax': 1000}
),
html.Div(id='status', role='status', **{'aria-live': 'polite'})
```

---

## Usability Testing

```python
class UsabilityTest:
    def __init__(self):
        self.results = []

    def record_task(self, user_id: int, task: str, duration: float, success: bool, errors: int):
        self.results.append({'user': user_id, 'task': task, 'duration': duration, 'success': success, 'errors': errors})

    def analyze(self) -> dict:
        import pandas as pd
        df = pd.DataFrame(self.results)
        return {
            'success_rate': f"{df['success'].mean():.1%}",
            'avg_duration': f"{df['duration'].mean():.1f}s",
            'avg_errors': f"{df['errors'].mean():.1f}",
        }

# Target metrics: >90% success rate, <30s avg duration, <2 avg errors
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Progressive Disclosure | Simple options first, complexity when needed |
| Immediate Feedback | Show results as parameters change |
| Error Prevention | Validate inputs, sensible defaults |
| Clear Labels | Scientific terminology with tooltips |
| Export Options | CSV, JSON, HDF5, PDF |
| Keyboard Shortcuts | Alt+key for power users |
| Responsive Design | Works on tablets for field research |
| Reproducibility | Save/load complete analysis state |
| WCAG 2.1 AA | Minimum accessibility compliance |

---

## Checklist

- [ ] Controls grouped logically in sidebar
- [ ] Tabs/panels for different analysis views
- [ ] Real-time feedback on parameter changes
- [ ] Export to multiple formats
- [ ] Save/load parameter sets
- [ ] Color contrast ratio ≥4.5:1
- [ ] Keyboard navigation works
- [ ] ARIA labels for screen readers
- [ ] Tooltips for scientific terms
- [ ] Error messages are actionable

---

**Version**: 1.0.5
