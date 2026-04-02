---
name: julia-ml-pipelines
description: Build ML pipelines in Julia with MLJ.jl for model selection, tuning, and evaluation, plus DrWatson.jl for experiment management and reproducibility. Covers learning networks, composable pipelines, hyperparameter tuning (Grid/Random/Latin), cross-validation, and scientific project organization. Use when building end-to-end ML workflows in Julia.
---

# Julia ML Pipelines

## Expert Agent

For end-to-end ML workflows and experiment management in Julia, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for pipeline design and optimization.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: MLJ pipelines, hyperparameter tuning, experiment tracking.

## MLJ Model Interface

```julia
using MLJ

# Search available models
models(matching(X, y))                    # Models compatible with data
models(m -> m.is_supervised && m.prediction_type == :probabilistic)

# Load and instantiate
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
rf = RandomForestClassifier(n_trees=100, max_depth=10)

# Machine (model + data binding)
mach = machine(rf, X, y)
fit!(mach; verbosity=0)
y_pred = predict(mach, X_test)
y_class = predict_mode(mach, X_test)   # Hard classification
```

## Composable Pipelines

### Simple Pipeline with `|>`

```julia
using MLJ

# Standardize -> PCA -> Classifier
pipe = Standardizer() |> PCA(maxoutdim=10) |> RandomForestClassifier(n_trees=200)

mach = machine(pipe, X, y)
fit!(mach)
y_pred = predict(mach, X_test)
```

### Learning Networks with `@from_network`

```julia
using MLJ

# Define a reusable composite model
Xs = source()
ys = source()

# Preprocessing
stand = machine(Standardizer(), Xs)
X_stand = transform(stand, Xs)

# Feature selection
pca = machine(PCA(maxoutdim=15), X_stand)
X_pca = transform(pca, X_stand)

# Model
clf = machine(RandomForestClassifier(), X_pca, ys)
y_hat = predict(clf, X_pca)

# Export as reusable model type
@from_network mach = PCAForest(
    standardizer = stand,
    pca = pca,
    classifier = clf
) <= y_hat
```

## Hyperparameter Tuning

```julia
using MLJ

model = RandomForestClassifier()

# Define search space
r_trees = range(model, :n_trees; lower=50, upper=500)
r_depth = range(model, :max_depth; lower=3, upper=20)

# Grid search
tuned_model = TunedModel(
    model = model,
    tuning = Grid(resolution=10),
    range = [r_trees, r_depth],
    measure = LogLoss(),
    resampling = CV(nfolds=5)
)

# Random search (more efficient for large spaces)
tuned_model = TunedModel(
    model = model,
    tuning = RandomSearch(rng=42),
    range = [r_trees, r_depth],
    n = 50,                          # Number of random samples
    measure = LogLoss()
)

# Latin hypercube (space-filling, best coverage)
tuned_model = TunedModel(
    model = model,
    tuning = LatinHypercube(),
    range = [r_trees, r_depth],
    n = 30,
    measure = LogLoss()
)

mach = machine(tuned_model, X, y)
fit!(mach)
report(mach).best_model              # Best hyperparameters
```

## Evaluation

```julia
using MLJ

# Single evaluation with cross-validation
result = evaluate!(
    mach,
    resampling = CV(nfolds=5, shuffle=true, rng=42),
    measures = [LogLoss(), Accuracy(), AreaUnderCurve()],
    verbosity = 0
)

# Stratified CV (preserves class balance)
result = evaluate!(
    mach,
    resampling = StratifiedCV(nfolds=10),
    measures = [LogLoss(), Accuracy()]
)

# Access results
result.measurement          # Mean scores
result.per_fold             # Per-fold scores
```

## Model Comparison

```julia
using MLJ

models = [
    ("RF", RandomForestClassifier(n_trees=200)),
    ("XGB", @load XGBoostClassifier()),
    ("LR", LogisticClassifier())
]

results = Dict()
for (name, model) in models
    mach = machine(model, X, y)
    result = evaluate!(mach,
        resampling = StratifiedCV(nfolds=5, rng=42),
        measures = [LogLoss(), Accuracy()],
        verbosity = 0
    )
    results[name] = (
        logloss = result.measurement[1],
        accuracy = result.measurement[2]
    )
end
```

## DrWatson Project Structure

```julia
using DrWatson

# Initialize project
initialize_project("MyMLExperiment";
    authors = "Name",
    force = true
)

# Project structure created:
# MyMLExperiment/
# ├── _research/       # Exploratory scripts
# ├── data/            # Data files
# │   ├── exp_raw/
# │   ├── exp_pro/     # Processed data
# │   └── sims/        # Simulation outputs
# ├── notebooks/
# ├── papers/
# ├── plots/
# ├── scripts/
# ├── src/             # Source code
# ├── Project.toml
# └── Manifest.toml

# Activate project
@quickactivate "MyMLExperiment"
```

## Experiment Management

```julia
using DrWatson

# Define parameter sweep
all_params = dict_list(Dict(
    "n_trees" => [100, 200, 500],
    "max_depth" => [5, 10, 20],
    "learning_rate" => [0.01, 0.1]
))
# Returns 18 parameter combinations

# Run and save experiments
for params in all_params
    result = run_experiment(params)

    # Save with auto-generated filename from params
    @tagsave(
        datadir("sims", savename(params, "jld2")),
        merge(params, result)   # Combines params + results
    )
end

# Load or produce (caching)
data, file = produce_or_load(params, datadir("sims")) do params
    result = expensive_computation(params)
    return Dict("result" => result, "params" => params)
end
```

## DataFrames Integration

```julia
using DrWatson, DataFrames

# Collect all results into a DataFrame
df = collect_results(datadir("sims"))

# Filter and analyze
best = sort(df, :accuracy; rev=true)[1, :]

# Save processed results
CSV.write(datadir("exp_pro", "summary.csv"), df)
```

## Reproducibility Checklist

- [ ] Use `DrWatson.initialize_project` for standardized project structure
- [ ] Pin all dependencies with `Manifest.toml` (committed to version control)
- [ ] Set explicit RNG seeds in `evaluate!` and `TunedModel` calls
- [ ] Use `@tagsave` to record git commit hash with every result
- [ ] Use `produce_or_load` to cache expensive computations
- [ ] Define search ranges explicitly (not hardcoded values)
- [ ] Use `StratifiedCV` for classification to preserve class balance
- [ ] Log all hyperparameter configurations and evaluation metrics
