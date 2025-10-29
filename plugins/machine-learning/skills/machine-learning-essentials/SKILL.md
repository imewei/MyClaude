---
name: machine-learning-essentials
description: Core machine learning workflows including classical ML algorithms (regression, trees, ensembles), neural networks, model evaluation, hyperparameter tuning, and deployment. Use when building predictive models, performing model selection, or deploying ML systems to production.
---

# Machine Learning Essentials

Practical frameworks for building, evaluating, and deploying machine learning models across classical and deep learning paradigms.

## When to Use

- Building predictive models (classification, regression, clustering)
- Selecting and comparing ML algorithms
- Performing hyperparameter tuning and model optimization
- Evaluating model performance with appropriate metrics
- Understanding model predictions and feature importance
- Deploying models to production environments
- Handling imbalanced datasets and model diagnostics

## Classical Machine Learning

### 1. Algorithm Selection Guide

| Task | Algorithm | When to Use |
|------|-----------|-------------|
| Linear relationships | Linear/Logistic Regression | Interpretability needed, baseline model |
| Non-linear, tabular | XGBoost, LightGBM | High performance on structured data |
| High-dimensional | Ridge, Lasso, Elastic Net | Feature selection, regularization |
| Clustering | K-Means, DBSCAN | Customer segmentation, anomaly detection |
| Dimensionality reduction | PCA, t-SNE, UMAP | Visualization, feature engineering |

### 2. Supervised Learning Workflow

**Classification Example (XGBoost):**
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(
    model, param_grid, cv=5,
    scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

**Regression Example (Random Forest):**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
```

### 3. Unsupervised Learning

**Clustering (K-Means):**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method for optimal k
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Fit final model
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Analyze clusters
df['cluster'] = clusters
print(df.groupby('cluster').mean())
```

**Dimensionality Reduction (PCA):**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_reduced = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_reduced.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

## Deep Learning

### 1. Neural Network with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Training loop
model = FeedForwardNN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.FloatTensor(y_train)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train
model.train()
for epoch in range(50):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
```

### 2. Transfer Learning

```python
from torchvision import models, transforms

# Load pretrained model
model = models.resnet18(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Fine-tune
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

## Model Evaluation

### 1. Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
```

### 2. Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
```

### 3. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')

print(f"CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Model Interpretability

### 1. Feature Importance

```python
import shap

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

### 2. Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# Plot partial dependence
features = ['feature1', 'feature2']
PartialDependenceDisplay.from_estimator(
    model, X_train, features, kind='both'
)
```

## Hyperparameter Tuning

### 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### 2. Bayesian Optimization

```python
from skopt import BayesSearchCV

search_spaces = {
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'n_estimators': (50, 300)
}

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)
```

## Handling Imbalanced Data

### 1. Class Weighting

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)

# Apply in model
model = XGBClassifier(scale_pos_weight=class_weights[1]/class_weights[0])
```

### 2. Resampling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SMOTE (oversampling minority class)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Combine over and under sampling
from imblearn.combine import SMOTETomek
sampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
```

## Model Deployment

### 1. Model Serialization

```python
import joblib
import pickle

# Save model
joblib.dump(model, 'model.joblib')

# Load model
loaded_model = joblib.load('model.joblib')
```

### 2. API Endpoint (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model at startup
model = joblib.load('model.joblib')

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        "prediction": int(prediction),
        "probability": probability.tolist()
    }
```

## Quick Reference

### Model Selection Checklist

- [ ] Understand problem type (classification, regression, clustering)
- [ ] Check data size (small: linear models, large: tree ensembles/deep learning)
- [ ] Assess interpretability needs (high: linear/trees, low: ensembles/neural nets)
- [ ] Consider training time constraints
- [ ] Evaluate deployment requirements (latency, memory)

### Evaluation Metrics by Task

**Classification:**
- Balanced classes: Accuracy, F1-score
- Imbalanced classes: Precision, Recall, ROC-AUC
- Multi-class: Macro/Micro-averaged metrics

**Regression:**
- General: R², RMSE
- Outliers present: MAE
- Percentage error: MAPE

---

*Build, evaluate, and deploy machine learning models effectively across classical and deep learning paradigms.*
