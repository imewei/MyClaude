# Error Budget Management and Burn Rate Detection

Comprehensive guide to error budget calculation, burn rate monitoring, and budget-based engineering decisions for SLO-driven reliability practices.

## Documentation Modules

### [1. Concepts & Fundamentals](./error-budgets/concept.md)
*   **Error Budget Fundamentals**: Definition, philosophy, and strategic uses.
*   **Burn Rate Concepts**: Understanding burn rate ratios and consumption speeds.

### [2. Mathematics & Calculations](./error-budgets/calculation.md)
*   **Basic Formulas**: Total error budget, consumed budget, and time-based calculations.
*   **Burn Rate Math**: Formulas for burn rate and time-to-exhaustion.
*   **Reference**: Common SLO targets and their corresponding downtime budgets.

### [3. Tracking & Analysis](./error-budgets/tracking.md)
*   **Budget Consumption Tracking**: Python implementation for tracking consumption events.
*   **Projected Exhaustion**: Algorithms to predict when the budget will run out.
*   **Historical Analysis**: Analyzing trends, patterns, and anomalies in budget consumption.

### [4. Status Determination](./error-budgets/status.md)
*   **Status Classification**: Defining Healthy, Attention, Warning, Critical, and Exhausted states.
*   **Logic**: Determining status based on remaining budget, burn rate, and projection.

### [5. Alerting & Burn Rate](./error-budgets/alerting.md)
*   **Standard Thresholds**: Google SRE burn rate thresholds (14.4x, 6x, etc.).
*   **Multi-Window Detection**: Reducing false positives with short/long window analysis.
*   **Alert Generation**: Prometheus rules and Python generators for actionable alerts.

### [6. Policies & Decisions](./error-budgets/policies.md)
*   **Release Gates**: Automated decision making for releases based on budget status.
*   **Policy Documents**: Example YAML configuration for error budget policies.
*   **Action Plans**: What to do when budget is low or exhausted.

### [7. Visualization](./error-budgets/visualization.md)
*   **Grafana Dashboards**: Complete JSON and panel configurations for visualizing:
    *   Budget remaining
    *   Current burn rate
    *   Time to exhaustion
    *   Historical trends

### [8. Examples & Implementation](./error-budgets/examples.md)
*   **Scenarios**: Walkthrough of different burn rate scenarios (Normal, On-Track, Outage).
*   **Reference Implementation**: Complete `ErrorBudgetManager` Python class for production use.

### [9. Advanced Patterns](./error-budgets/advanced.md)
*   **Composite Budgets**: Managing error budgets across multiple SLIs with weighted composition.
