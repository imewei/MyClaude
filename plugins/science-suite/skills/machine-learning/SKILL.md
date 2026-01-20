---
name: machine-learning
version: "1.1.0"
description: Comprehensive Machine Learning and Deep Learning suite. Covers core ML (scikit-learn, XGBoost), Deep Learning (PyTorch, JAX), experiment tracking, and model optimization.
---

# Machine Learning & Deep Learning

Complete workflow for building, training, and deploying ML/DL models in scientific research.

## Expert Agent

For advanced machine learning architectures, MLOps pipelines, and scientific ML, delegate to the expert agent:

- **`ml-expert`**: Unified specialist for Scientific Machine Learning (SciML), Deep Learning, and MLOps.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: PINNs, Neural Operators, Distributed Training (DDP/FSDP), and Model Deployment.

## 1. Core Machine Learning (Tabular Data)

### Algorithm Selection
- **Linear/Logistic Regression**: For baselines and interpretability.
- **XGBoost / LightGBM**: State-of-the-art for structured/tabular data.
- **Random Forests**: Robust ensembles for high-dimensional data.

### Workflow Example (Scikit-learn & XGBoost)
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
```

## 2. Deep Learning & Experimentation

### Frameworks
- **PyTorch**: Flexible, widely used for research and production.
- **JAX**: High-performance, functional, and differentiable programming.

### Systematic Experimentation
- **Reproducibility**: Always set seeds for `random`, `numpy`, `torch`, and `jax`.
- **Tracking**: Use `Weights & Biases` or `MLflow` to log metrics, hyperparameters, and artifacts.
- **Ablation Studies**: Systematically remove components to identify critical model features.

## 3. Advanced Optimization

### Hyperparameter Tuning
- **Optuna**: Bayesian optimization for efficient search spaces.
- **Learning Rate Range Test**: Find optimal LR by observing loss descent.

### Model Deployment
- **Serialization**: Use `joblib` or `ONNX` for traditional ML; `torch.jit` or `SavedModel` for DL.
- **FastAPI**: Serve models via REST APIs for integration into scientific workflows.

## 4. Validation & Interpretability Checklist

- [ ] **Data Leakage**: Ensure preprocessing parameters are fit ONLY on the training set.
- [ ] **Imbalance**: Use SMOTE or class weights for highly skewed target variables.
- [ ] **SHAP Values**: Use SHAP for post-hoc model interpretability and feature importance.
- [ ] **Metrics**: Match metrics (ROC-AUC, F1, RMSE, MAE) to the scientific objective.
