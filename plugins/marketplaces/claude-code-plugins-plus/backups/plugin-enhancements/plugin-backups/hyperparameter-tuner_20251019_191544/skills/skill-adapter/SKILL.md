---
name: Tuning Hyperparameters
description: |
  This skill enables Claude to optimize machine learning model hyperparameters using grid search, random search, or Bayesian optimization. It is used when the user requests hyperparameter tuning, model optimization, or improvement of model performance. The skill analyzes the current context, generates code for the specified search strategy, handles data validation and errors, and provides performance metrics. Trigger terms include "tune hyperparameters," "optimize model," "grid search," "random search," and "Bayesian optimization."
---

## Overview

This skill empowers Claude to fine-tune machine learning models by automatically searching for the optimal hyperparameter configurations. It leverages different search strategies (grid, random, Bayesian) to efficiently explore the hyperparameter space and identify settings that maximize model performance.

## How It Works

1. **Analyzing Requirements**: Claude analyzes the user's request to determine the model, the hyperparameters to tune, the search strategy, and the evaluation metric.
2. **Generating Code**: Claude generates Python code using appropriate ML libraries (e.g., scikit-learn, Optuna) to implement the specified hyperparameter search. The code includes data loading, preprocessing, model training, and evaluation.
3. **Executing Search**: The generated code is executed to perform the hyperparameter search. The plugin iterates through different hyperparameter combinations, trains the model with each combination, and evaluates its performance.
4. **Reporting Results**: Claude reports the best hyperparameter configuration found during the search, along with the corresponding performance metrics. It also provides insights into the search process and potential areas for further optimization.

## When to Use This Skill

This skill activates when you need to:
- Optimize the performance of a machine learning model.
- Automatically search for the best hyperparameter settings.
- Compare different hyperparameter search strategies.
- Improve model accuracy, precision, recall, or other relevant metrics.

## Examples

### Example 1: Optimizing a Random Forest Model

User request: "Tune hyperparameters of a Random Forest model using grid search to maximize accuracy on the iris dataset. Consider n_estimators and max_depth."

The skill will:
1. Generate code to perform a grid search over the specified hyperparameters (n_estimators, max_depth) of a Random Forest model using the iris dataset.
2. Execute the grid search and report the best hyperparameter combination and the corresponding accuracy score.

### Example 2: Using Bayesian Optimization

User request: "Optimize a Gradient Boosting model using Bayesian optimization with Optuna to minimize the root mean squared error on the Boston housing dataset."

The skill will:
1. Generate code to perform Bayesian optimization using Optuna to find the best hyperparameters for a Gradient Boosting model on the Boston housing dataset.
2. Execute the optimization and report the best hyperparameter combination and the corresponding RMSE.

## Best Practices

- **Define Search Space**: Clearly define the range and type of values for each hyperparameter to be tuned.
- **Choose Appropriate Strategy**: Select the hyperparameter search strategy (grid, random, Bayesian) based on the complexity of the hyperparameter space and the available computational resources. Bayesian optimization is generally more efficient for complex spaces.
- **Use Cross-Validation**: Implement cross-validation to ensure the robustness of the evaluation metric and prevent overfitting.

## Integration

This skill integrates seamlessly with other Claude Code plugins that involve machine learning tasks, such as data analysis, model training, and deployment. It can be used in conjunction with data visualization tools to gain insights into the impact of different hyperparameter settings on model performance.