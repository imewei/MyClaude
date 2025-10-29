---
name: data-wrangling-communication
description: Comprehensive data wrangling, cleaning, feature engineering, and visualization workflows for preparing data for analysis and communicating insights effectively. Use when cleaning messy datasets, handling missing values, engineering features, creating visualizations, building dashboards, or presenting data-driven insights to stakeholders.
---

# Data Wrangling and Communication

Transform raw data into actionable insights through systematic data cleaning, feature engineering, visualization, and compelling storytelling.

---

## When to Use

- Cleaning and preprocessing raw datasets
- Handling missing values, outliers, and inconsistencies
- Feature engineering for machine learning
- Exploratory data analysis (EDA)
- Creating visualizations and dashboards
- Communicating insights to stakeholders
- Building data narratives for business decisions

---

## Data Cleaning Workflows

### 1. Initial Data Inspection

```python
import pandas as pd
import numpy as np

# Load and inspect data
df = pd.read_csv('data.csv')

# Quick overview
print(df.info())  # Data types, non-null counts
print(df.describe())  # Statistical summary
print(df.head())

# Check for issues
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Data types:\n{df.dtypes}")
```

### 2. Handling Missing Values

**Decision Framework:**

| Pattern | Strategy | When to Use |
|---------|----------|-------------|
| Random, <5% | Drop rows | MCAR (Missing Completely At Random) |
| Random, >5% | Imputation | MAR (Missing At Random) |
| Systematic | Domain logic | MNAR (Missing Not At Random) |
| Entire column >50% | Drop column | Too much missing data |

**Implementation:**

```python
# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows with missing values in specific columns
df_clean = df.dropna(subset=['important_col1', 'important_col2'])

# Drop columns with >50% missing
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)

# Imputation strategies
from sklearn.impute import SimpleImputer

# Mean/median for numerical
imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Mode for categorical
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

# Forward fill for time series
df['value'] = df['value'].fillna(method='ffill')

# Custom logic
df['age'] = df['age'].fillna(df.groupby('occupation')['age'].transform('median'))
```

### 3. Outlier Detection and Treatment

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Box plot
sns.boxplot(data=df, y='value', ax=axes[0])

# Histogram
df['value'].hist(bins=50, ax=axes[1])

# Statistical detection: IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Treatment options:
# 1. Remove outliers
df_no_outliers = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# 2. Cap outliers (winsorization)
df['value_capped'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

# 3. Transform (log for right-skewed)
df['value_log'] = np.log1p(df['value'])

# Z-score method (for normal distributions)
from scipy import stats
z_scores = np.abs(stats.zscore(df['value']))
df_no_outliers = df[z_scores < 3]
```

### 4. Data Type Conversions

```python
# Convert to appropriate types
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Parse complex strings
df['year'] = df['date_string'].str.extract(r'(\d{4})')
df['email_domain'] = df['email'].str.split('@').str[1]

# Boolean conversions
df['is_active'] = df['status'].map({'active': True, 'inactive': False})
```

---

## Feature Engineering

### 1. Creating New Features

**Numerical Transformations:**

```python
# Mathematical operations
df['total_price'] = df['quantity'] * df['unit_price']
df['price_per_sqft'] = df['price'] / df['area']

# Binning continuous variables
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 18, 35, 50, 100],
                         labels=['Youth', 'Young Adult', 'Middle Age', 'Senior'])

# Polynomial features
df['area_squared'] = df['area'] ** 2
df['price_log'] = np.log1p(df['price'])

# Interaction features
df['bedroom_bathroom_ratio'] = df['bedrooms'] / df['bathrooms']
```

**Categorical Encoding:**

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding (ordinal)
label_encoder = LabelEncoder()
df['size_encoded'] = label_encoder.fit_transform(df['size'])
# small=0, medium=1, large=2

# One-hot encoding (nominal)
df_encoded = pd.get_dummies(df, columns=['category', 'color'], drop_first=True)

# Or using sklearn
encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['category']])

# Frequency encoding
freq_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(freq_map)

# Target encoding (for high-cardinality categorical)
target_mean = df.groupby('category')['target'].mean()
df['category_target_enc'] = df['category'].map(target_mean)
```

**Time Series Features:**

```python
# Extract datetime components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Cyclical encoding (preserve cyclicity)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)

# Rolling statistics
df['sales_rolling_mean_7d'] = df['sales'].rolling(window=7).mean()
df['sales_rolling_std_7d'] = df['sales'].rolling(window=7).std()

# Time since event
df['days_since_purchase'] = (pd.Timestamp.now() - df['purchase_date']).dt.days
```

### 2. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (mean=0, std=1) - for algorithms sensitive to scale
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Min-Max scaling (range [0, 1]) - for neural networks
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Robust scaling (uses median, IQR) - robust to outliers
scaler = RobustScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

### 3. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Univariate selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Mutual information
mi_scores = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)

# Feature importance (tree-based)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Correlation-based filtering
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_features = [col for col in upper_triangle.columns
                      if any(upper_triangle[col] > 0.95)]
X_filtered = X.drop(columns=high_corr_features)
```

---

## Exploratory Data Analysis

### 1. Statistical Summaries

```python
# Comprehensive summary
print(df.describe(include='all'))  # All columns
print(df.describe(percentiles=[.1, .25, .5, .75, .9]))

# Group statistics
print(df.groupby('category').agg({
    'price': ['mean', 'median', 'std'],
    'quantity': 'sum',
    'customer_id': 'nunique'
}))

# Correlation analysis
corr_matrix = df[numerical_cols].corr()
print(corr_matrix)
```

### 2. Visualization Patterns

**Distribution Analysis:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Histogram with KDE
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x='price', kde=True, bins=50, ax=ax)
ax.set_title('Price Distribution')

# Box plots by category
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='category', y='price', ax=ax)
plt.xticks(rotation=45)

# Violin plot (distribution + density)
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=df, x='category', y='price', ax=ax)
```

**Relationship Analysis:**

```python
# Scatter plot with regression line
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=df, x='area', y='price', scatter_kws={'alpha':0.3}, ax=ax)

# Pair plot (multiple variables)
sns.pairplot(df[['price', 'area', 'bedrooms', 'age']], diag_kind='kde')

# Heatmap (correlation matrix)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap')
```

**Categorical Analysis:**

```python
# Count plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x='category', order=df['category'].value_counts().index, ax=ax)
plt.xticks(rotation=45)

# Bar plot with aggregation
fig, ax = plt.subplots(figsize=(12, 6))
df.groupby('category')['price'].mean().sort_values().plot(kind='barh', ax=ax)
ax.set_title('Average Price by Category')
```

**Time Series Visualization:**

```python
# Line plot with trend
fig, ax = plt.subplots(figsize=(14, 6))
df.groupby('date')['sales'].sum().plot(ax=ax)
ax.set_title('Sales Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')

# Multiple series
fig, ax = plt.subplots(figsize=(14, 6))
for category in df['category'].unique():
    data = df[df['category'] == category].groupby('date')['sales'].sum()
    ax.plot(data.index, data.values, label=category)
ax.legend()
ax.set_title('Sales by Category Over Time')
```

### 3. Interactive Visualizations with Plotly

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter plot
fig = px.scatter(df, x='area', y='price', color='category',
                 size='bedrooms', hover_data=['address'],
                 title='House Prices vs Area')
fig.show()

# Interactive time series
fig = px.line(df, x='date', y='sales', color='category',
              title='Sales Trends by Category')
fig.show()

# 3D scatter
fig = px.scatter_3d(df, x='area', y='price', z='age',
                    color='category', size='bedrooms')
fig.show()

# Animated scatter (over time)
fig = px.scatter(df, x='metric1', y='metric2',
                 animation_frame='year',
                 size='population', color='country',
                 hover_name='country', range_x=[0, 100], range_y=[0, 100])
fig.show()
```

---

## Dashboard Creation

### 1. Streamlit Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Category", df['category'].unique())
date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

# Filter data
if categories:
    df_filtered = df[df['category'].isin(categories)]
else:
    df_filtered = df

# Main dashboard
st.title("📊 Sales Analytics Dashboard")

# KPI metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${df_filtered['sales'].sum():,.0f}")
col2.metric("Avg Transaction", f"${df_filtered['sales'].mean():.2f}")
col3.metric("Total Orders", f"{len(df_filtered):,}")
col4.metric("Unique Customers", f"{df_filtered['customer_id'].nunique():,}")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Over Time")
    fig = px.line(df_filtered.groupby('date')['sales'].sum().reset_index(),
                  x='date', y='sales')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sales by Category")
    fig = px.pie(df_filtered, names='category', values='sales')
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.subheader("Detailed Data")
st.dataframe(df_filtered, use_container_width=True)
```

### 2. Jupyter Dashboard with ipywidgets

```python
import ipywidgets as widgets
from IPython.display import display

# Interactive widgets
category_dropdown = widgets.Dropdown(
    options=df['category'].unique(),
    description='Category:'
)

date_slider = widgets.SelectionRangeSlider(
    options=pd.date_range(df['date'].min(), df['date'].max(), freq='D').tolist(),
    description='Date Range',
    layout=widgets.Layout(width='500px')
)

output = widgets.Output()

def update_plot(category, date_range):
    with output:
        output.clear_output(wait=True)

        # Filter data
        mask = (df['category'] == category) & \
               (df['date'] >= date_range[0]) & \
               (df['date'] <= date_range[1])
        df_filtered = df[mask]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_filtered.groupby('date')['sales'].sum().plot(ax=ax)
        ax.set_title(f'Sales for {category}')
        plt.show()

# Connect widgets to update function
interactive_plot = widgets.interactive(update_plot,
                                       category=category_dropdown,
                                       date_range=date_slider)
display(interactive_plot, output)
```

---

## Communicating Insights

### 1. Storytelling Framework

**Structure:**
1. **Context**: What's the business problem?
2. **Analysis**: What did you find?
3. **Insight**: What does it mean?
4. **Action**: What should we do?

**Example Report:**

```python
# Generate automated insights
def generate_insights(df):
    insights = []

    # Trend analysis
    recent_sales = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]['sales'].sum()
    previous_sales = df[(df['date'] >= df['date'].max() - pd.Timedelta(days=60)) &
                        (df['date'] < df['date'].max() - pd.Timedelta(days=30))]['sales'].sum()
    change = (recent_sales - previous_sales) / previous_sales * 100

    insights.append(f"Sales {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}% in the last 30 days.")

    # Top performers
    top_category = df.groupby('category')['sales'].sum().idxmax()
    insights.append(f"{top_category} is the top-performing category.")

    # Anomalies
    avg_daily = df.groupby('date')['sales'].sum().mean()
    recent_avg = df[df['date'] >= df['date'].max() - pd.Timedelta(days=7)].groupby('date')['sales'].sum().mean()

    if recent_avg > avg_daily * 1.5:
        insights.append(f"⚠️ Recent sales spike detected: {(recent_avg/avg_daily - 1)*100:.0f}% above average.")

    return insights

# Display insights
for insight in generate_insights(df):
    print(f"• {insight}")
```

### 2. Executive Summary Template

```python
from datetime import datetime

def create_executive_summary(df, output_file='summary.md'):
    """Generate markdown executive summary."""

    summary = f"""
# Executive Summary - Sales Analysis
**Report Date:** {datetime.now().strftime('%Y-%m-%d')}

## Key Metrics
- **Total Revenue:** ${df['sales'].sum():,.2f}
- **Total Orders:** {len(df):,}
- **Average Order Value:** ${df['sales'].mean():.2f}
- **Active Customers:** {df['customer_id'].nunique():,}

## Top Findings

### 1. Revenue Trends
{generate_trend_insight(df)}

### 2. Category Performance
{generate_category_insight(df)}

### 3. Customer Behavior
{generate_customer_insight(df)}

## Recommendations

1. **Short-term (0-3 months):** [Based on immediate findings]
2. **Medium-term (3-6 months):** [Strategic initiatives]
3. **Long-term (6+ months):** [Transformational changes]

## Next Steps
- [ ] Implement top recommendation
- [ ] Monitor key metrics weekly
- [ ] Schedule follow-up analysis in 30 days
"""

    with open(output_file, 'w') as f:
        f.write(summary)

    return summary

# Generate report
summary = create_executive_summary(df)
print(summary)
```

### 3. Presentation Best Practices

**Visualization Guidelines:**
```python
# Clean, publication-ready plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')  # Larger fonts for presentations
sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(12, 7))

# Main plot
df.groupby('category')['sales'].sum().sort_values().plot(
    kind='barh',
    color='steelblue',
    ax=ax
)

# Formatting
ax.set_title('Revenue by Product Category', fontsize=20, weight='bold', pad=20)
ax.set_xlabel('Total Sales ($)', fontsize=14)
ax.set_ylabel('Category', fontsize=14)

# Add data labels
for i, v in enumerate(df.groupby('category')['sales'].sum().sort_values()):
    ax.text(v + 1000, i, f'${v:,.0f}', va='center', fontsize=12)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('category_revenue.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Data Quality Checklist

Use this checklist for every dataset:

- [ ] **Completeness**: Missing values handled appropriately
- [ ] **Consistency**: Data types correct, formats standardized
- [ ] **Accuracy**: Outliers detected and treated
- [ ] **Uniqueness**: Duplicates identified and removed
- [ ] **Timeliness**: Date ranges validated
- [ ] **Validity**: Values within expected ranges
- [ ] **Integrity**: Referential integrity maintained

---

## Quick Reference

### Pandas Cheat Sheet

```python
# Data loading
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_sql('SELECT * FROM table', conn)

# Inspection
df.shape, df.info(), df.describe()
df.head(), df.tail(), df.sample(10)

# Selection
df['col'], df[['col1', 'col2']]
df.loc[row_indexer, col_indexer]
df.iloc[row_int, col_int]
df.query('age > 30')

# Manipulation
df.drop(columns=['col']), df.rename(columns={'old': 'new'})
df.sort_values('col'), df.groupby('col').agg({'col2': 'mean'})
df.merge(df2, on='key'), df.pivot_table(index='A', columns='B', values='C')

# Cleaning
df.dropna(), df.fillna(value)
df.drop_duplicates()
df['col'].astype('int')
```

### Common Visualizations

| Task | Matplotlib | Seaborn | Plotly |
|------|-----------|---------|--------|
| Line plot | `plt.plot(x, y)` | `sns.lineplot(data=df, x='x', y='y')` | `px.line(df, x='x', y='y')` |
| Scatter | `plt.scatter(x, y)` | `sns.scatterplot(data=df, x='x', y='y')` | `px.scatter(df, x='x', y='y')` |
| Histogram | `plt.hist(x)` | `sns.histplot(data=df, x='x')` | `px.histogram(df, x='x')` |
| Box plot | `plt.boxplot(x)` | `sns.boxplot(data=df, x='cat', y='num')` | `px.box(df, x='cat', y='num')` |
| Heatmap | `plt.imshow(data)` | `sns.heatmap(corr_matrix)` | `px.imshow(corr_matrix)` |

---

*Transform raw data into actionable insights through systematic wrangling, compelling visualization, and effective storytelling.*
