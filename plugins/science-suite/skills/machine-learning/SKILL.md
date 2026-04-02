---
name: machine-learning
description: Comprehensive Classical Machine Learning suite. Covers scikit-learn, XGBoost, LightGBM, and MLOps pipelines. Focuses on tabular data, feature engineering, production deployment, algorithm selection, cross-validation, hyperparameter tuning (GridSearch, Optuna), handling imbalanced data (SMOTE), model evaluation, SHAP interpretability, and deployment.
---

# Machine Learning & MLOps

Complete workflow for building, training, and deploying classical ML models.

## Expert Agent

For classical ML workflows, MLOps, and deployment, delegate to the expert agent:

- **`ml-expert`**: Unified specialist for MLOps, Infrastructure, and Classical ML.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: Pipeline orchestration, Model Serving, A/B Testing, Data Engineering, model selection, hyperparameter tuning (Optuna), and interpretability (SHAP).
- **`julia-ml-hpc`**: Julia ML pipelines with MLJ.jl and DrWatson.jl.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Julia skill*: See `julia-ml-pipelines` for Julia-specific ML workflows.

## Core Skills

### [Advanced ML Systems](./advanced-ml-systems/SKILL.md)
Boosting (XGBoost/LightGBM) and deep tabular models.

### [Data Wrangling & Communication](./data-wrangling-communication/SKILL.md)
Preprocessing, feature engineering, and reporting.

### [Statistical Analysis Fundamentals](./statistical-analysis-fundamentals/SKILL.md)
Hypothesis testing and uncertainty quantification in ML.

### [ML Pipeline Workflow](./ml-pipeline-workflow/SKILL.md)
Orchestration and reproducibility with Airflow/Dagster.

### [ML Engineering & Production](./ml-engineering-production/SKILL.md)
Serialization, scaling, and high-performance inference.

### [Model Deployment & Serving](./model-deployment-serving/SKILL.md)
FastAPI, Triton, and cloud-native deployment.

### [DevOps & ML Infrastructure](./devops-ml-infrastructure/SKILL.md)
Docker, Kubernetes, and cluster management.

## 1. Algorithm Selection

| Task | Algorithm | When to Use |
|------|-----------|-------------|
| Baseline | Linear/Logistic Regression | Interpretability, fast |
| Tabular (best) | XGBoost, LightGBM | Structured data performance |
| High-dimensional | Ridge, Lasso, Elastic Net | Regularization needed |
| Clustering | K-Means, DBSCAN | Segmentation, anomaly |
| Dimensionality | PCA, t-SNE, UMAP | Visualization, features |

## 2. Classification Workflow

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

grid = GridSearchCV(XGBClassifier(random_state=42), param_grid,
                    cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.best_estimator_.predict(X_test)
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

## 3. Regression Workflow

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
```

## 4. Evaluation Metrics

### Classification

| Metric | Use Case |
|--------|----------|
| Accuracy | Balanced classes |
| Precision/Recall | Imbalanced, cost-sensitive |
| F1-Score | Balance precision/recall |
| ROC-AUC | Ranking, threshold-independent |

### Regression

| Metric | Use Case |
|--------|----------|
| R² | General goodness-of-fit |
| RMSE | Penalize large errors |
| MAE | Robust to outliers |
| MAPE | Percentage interpretation |

## 5. Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Option 1: SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Option 2: Class weights
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model = XGBClassifier(scale_pos_weight=weights[1]/weights[0])
```

## 6. Hyperparameter Tuning

### Bayesian Optimization (Optuna)

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300)
    }
    model = XGBClassifier(**params, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)
```

## 7. Feature Engineering & Selection

- **Preprocessing**: Scaling (Standard/MinMax), Imputation, Encoding (One-hot/Target).
- **Selection**: Recursive Feature Elimination (RFE), Feature Importance, SHAP.
- **Dimensionality Reduction**: PCA, t-SNE, UMAP.

## 8. Model Interpretability (SHAP)

```python
import shap

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

## 9. Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"CV: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## 10. MLOps & Production

### Pipeline Orchestration
- **Tools**: Airflow, Dagster, Kubeflow, Prefect.
- **Goals**: Reproducibility, scheduling, data validation.

### Model Deployment
```python
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Save/load
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')

# API endpoint
app = FastAPI()

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    return {"prediction": int(model.predict(X)[0]),
            "probability": model.predict_proba(X)[0].tolist()}
```

### Monitoring
- Drift detection, performance metrics.
- **Serialization**: `joblib`, `ONNX`.
- **Serving**: FastAPI, Triton Inference Server.

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Baseline first | Start with simple models |
| Stratified splits | For classification |
| Cross-validation | 5-fold minimum |
| Feature scaling | StandardScaler for linear models |
| Early stopping | Prevent overfitting |
| SHAP values | Model interpretability |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Data leakage | Fit scaler on train only |
| Class imbalance | SMOTE or class weights |
| Overfitting | Cross-validation, regularization |
| Wrong metric | Match metric to business goal |
| No baseline | Compare against simple model |

## Checklist

- [ ] Problem type identified (classification/regression)
- [ ] Baseline model established
- [ ] Data split with stratification (if classification)
- [ ] Cross-validation performed
- [ ] Hyperparameters tuned
- [ ] Model evaluated on held-out test set
- [ ] Interpretability assessed (SHAP)
- [ ] Model serialized for deployment
- [ ] Data leakage verified (preprocessing fit on train only)
- [ ] Imbalance handled (SMOTE or class weights)
- [ ] Metrics matched to scientific objective (ROC-AUC, F1, RMSE, MAE)
