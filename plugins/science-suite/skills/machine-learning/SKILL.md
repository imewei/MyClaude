---
name: machine-learning
version: "2.1.0"
description: Comprehensive Classical Machine Learning suite. Covers scikit-learn, XGBoost, LightGBM, and MLOps pipelines. Focuses on tabular data, feature engineering, and production deployment.
---

# Machine Learning & MLOps

Complete workflow for building, training, and deploying classical ML models.

## Expert Agent

For classical ML workflows, MLOps, and deployment, delegate to the expert agent:

- **`ml-expert`**: Unified specialist for MLOps, Infrastructure, and Classical ML.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: Pipeline orchestration, Model Serving, A/B Testing, and Data Engineering.

## Core Skills

### [Machine Learning Essentials](./machine-learning-essentials/SKILL.md)
Foundational algorithms: Linear models, Trees, and Ensembles.

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

## 1. Core Machine Learning (Tabular Data)

### Algorithm Selection
- **Linear/Logistic Regression**: For baselines and interpretability.
- **XGBoost / LightGBM**: State-of-the-art for structured/tabular data.
- **Random Forests**: Robust ensembles for high-dimensional data.
- **Clustering**: K-Means, DBSCAN, Hierarchical.

### Workflow Example (Scikit-learn & XGBoost)
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
```

## 2. Feature Engineering & Selection

- **Preprocessing**: Scaling (Standard/MinMax), Imputation, Encoding (One-hot/Target).
- **Selection**: Recursive Feature Elimination (RFE), Feature Importance, SHAP.
- **Dimensionality Reduction**: PCA, t-SNE, UMAP.

## 3. MLOps & Production

### Pipeline Orchestration
- **Tools**: Airflow, Dagster, Kubeflow, Prefect.
- **Goals**: Reproducibility, scheduling, data validation.

### Model Deployment
- **Serialization**: `joblib`, `ONNX`.
- **Serving**: FastAPI, Triton Inference Server.
- **Monitoring**: Drift detection, performance metrics.

## 4. Validation & Interpretability Checklist

- [ ] **Data Leakage**: Ensure preprocessing parameters are fit ONLY on the training set.
- [ ] **Imbalance**: Use SMOTE or class weights for highly skewed target variables.
- [ ] **SHAP Values**: Use SHAP for post-hoc model interpretability and feature importance.
- [ ] **Metrics**: Match metrics (ROC-AUC, F1, RMSE, MAE) to the scientific objective.
