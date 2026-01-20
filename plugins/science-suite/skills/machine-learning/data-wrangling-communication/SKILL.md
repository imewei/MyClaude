---
name: data-wrangling-communication
version: "1.0.7"
maturity: "5-Expert"
specialization: Data Cleaning & Visualization
description: Comprehensive data wrangling, cleaning, feature engineering, and visualization workflows using pandas, NumPy, Matplotlib, Seaborn, and Plotly. Use when cleaning messy datasets, handling missing values, dealing with outliers, engineering features, performing EDA, creating statistical visualizations, building interactive dashboards with Plotly Dash or Streamlit, or presenting data-driven insights to stakeholders.
---

# Data Wrangling and Communication

Transform raw data into actionable insights through systematic cleaning, feature engineering, and visualization.

---

## Data Cleaning Workflow

### Initial Inspection
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
print(df.info())
print(df.describe())
print(f"Missing:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
```

### Missing Value Strategy

| Pattern | Strategy | When to Use |
|---------|----------|-------------|
| Random, <5% | Drop rows | MCAR |
| Random, >5% | Imputation | MAR |
| Systematic | Domain logic | MNAR |
| Column >50% missing | Drop column | Too sparse |

```python
# Imputation
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Group-based imputation
df['age'] = df['age'].fillna(df.groupby('occupation')['age'].transform('median'))

# Time series forward fill
df['value'] = df['value'].ffill()
```

### Outlier Treatment

```python
# IQR method
Q1, Q3 = df['value'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Treatment options
df_clean = df[(df['value'] >= lower) & (df['value'] <= upper)]  # Remove
df['value_capped'] = df['value'].clip(lower, upper)              # Cap
df['value_log'] = np.log1p(df['value'])                          # Transform

# Z-score (for normal distributions)
from scipy import stats
z_scores = np.abs(stats.zscore(df['value']))
df_clean = df[z_scores < 3]
```

---

## Feature Engineering

### Numerical Transformations
```python
# Derived features
df['total'] = df['quantity'] * df['unit_price']
df['ratio'] = df['price'] / df['area']

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['Youth', 'Young', 'Middle', 'Senior'])

# Log transform
df['price_log'] = np.log1p(df['price'])
```

### Categorical Encoding

| Method | Use Case | Code |
|--------|----------|------|
| One-Hot | Nominal, low cardinality | `pd.get_dummies(df, columns=['cat'], drop_first=True)` |
| Label | Ordinal | `LabelEncoder().fit_transform(df['size'])` |
| Frequency | High cardinality | `df['cat'].map(df['cat'].value_counts(normalize=True))` |
| Target | Classification | `df['cat'].map(df.groupby('cat')['target'].mean())` |

### Time Series Features
```python
# Components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag and rolling
df['lag_1'] = df['sales'].shift(1)
df['rolling_mean_7'] = df['sales'].rolling(7).mean()
```

### Feature Scaling

| Scaler | Formula | Use Case |
|--------|---------|----------|
| StandardScaler | (x - μ) / σ | Algorithms sensitive to scale |
| MinMaxScaler | (x - min) / (max - min) | Neural networks [0, 1] |
| RobustScaler | (x - median) / IQR | Outlier-robust |

---

## EDA Visualizations

### Distribution Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram with KDE
sns.histplot(data=df, x='price', kde=True, bins=50)

# Box plot by category
sns.boxplot(data=df, x='category', y='price')

# Correlation heatmap
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
```

### Relationship Analysis
```python
# Scatter with regression
sns.regplot(data=df, x='area', y='price', scatter_kws={'alpha': 0.3})

# Pair plot
sns.pairplot(df[['price', 'area', 'bedrooms']], diag_kind='kde')
```

### Interactive with Plotly
```python
import plotly.express as px

# Scatter with hover
fig = px.scatter(df, x='area', y='price', color='category',
                 size='bedrooms', hover_data=['address'])

# Time series
fig = px.line(df, x='date', y='sales', color='category')

# 3D scatter
fig = px.scatter_3d(df, x='area', y='price', z='age', color='category')
```

---

## Dashboard Patterns

### Streamlit
```python
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard", layout="wide")

# Sidebar filters
with st.sidebar:
    categories = st.multiselect("Category", df['category'].unique())

df_filtered = df[df['category'].isin(categories)] if categories else df

# KPI metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${df_filtered['sales'].sum():,.0f}")
col2.metric("Avg Transaction", f"${df_filtered['sales'].mean():.2f}")
col3.metric("Orders", f"{len(df_filtered):,}")

# Charts
fig = px.line(df_filtered.groupby('date')['sales'].sum().reset_index(), x='date', y='sales')
st.plotly_chart(fig, use_container_width=True)
```

---

## Automated Insights

```python
def generate_insights(df):
    insights = []

    # Trend
    recent = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]['sales'].sum()
    previous = df[(df['date'] >= df['date'].max() - pd.Timedelta(days=60)) &
                  (df['date'] < df['date'].max() - pd.Timedelta(days=30))]['sales'].sum()
    change = (recent - previous) / previous * 100
    insights.append(f"Sales {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}%")

    # Top performer
    top = df.groupby('category')['sales'].sum().idxmax()
    insights.append(f"{top} is the top category")

    return insights
```

---

## Pandas Quick Reference

```python
# Loading
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# Selection
df['col'], df[['col1', 'col2']]
df.loc[rows, cols], df.iloc[row_int, col_int]
df.query('age > 30')

# Manipulation
df.drop(columns=['col'])
df.rename(columns={'old': 'new'})
df.sort_values('col')
df.groupby('col').agg({'col2': 'mean'})
df.merge(df2, on='key')
df.pivot_table(index='A', columns='B', values='C')
```

---

## Visualization Quick Reference

| Chart | Seaborn | Plotly |
|-------|---------|--------|
| Line | `sns.lineplot(data=df, x='x', y='y')` | `px.line(df, x='x', y='y')` |
| Scatter | `sns.scatterplot(data=df, x='x', y='y')` | `px.scatter(df, x='x', y='y')` |
| Histogram | `sns.histplot(data=df, x='x')` | `px.histogram(df, x='x')` |
| Box | `sns.boxplot(data=df, x='cat', y='num')` | `px.box(df, x='cat', y='num')` |
| Heatmap | `sns.heatmap(corr_matrix)` | `px.imshow(corr_matrix)` |

---

## Data Quality Checklist

- [ ] **Completeness**: Missing values handled
- [ ] **Consistency**: Data types correct, formats standardized
- [ ] **Accuracy**: Outliers detected and treated
- [ ] **Uniqueness**: Duplicates removed
- [ ] **Validity**: Values within expected ranges
- [ ] **Timeliness**: Date ranges validated

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Inspect first | `df.info()`, `df.describe()`, check nulls |
| Document assumptions | Comment imputation and outlier decisions |
| Preserve originals | Create new columns for transformations |
| Validate transformations | Check distributions before/after |
| Use appropriate charts | Match chart type to data and message |
| Tell a story | Context → Analysis → Insight → Action |

---

**Version**: 1.0.5
