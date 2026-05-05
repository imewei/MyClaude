"""ML pipeline with scikit-learn and MLflow experiment tracking."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build column transformer for mixed feature types."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Train a gradient boosting classifier with cross-validation and MLflow logging."""
    X = df[numeric_features + categorical_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=random_state)),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.01, 0.1],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name="gbc_grid_search"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="roc_auc")

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("cv_auc_mean", np.mean(cv_scores))
        mlflow.log_metric("cv_auc_std", np.std(cv_scores))
        mlflow.sklearn.log_model(best_model, "model")

        print(classification_report(y_test, y_pred))

    return best_model
