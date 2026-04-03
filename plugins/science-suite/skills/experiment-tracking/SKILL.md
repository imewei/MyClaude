---
name: experiment-tracking
description: "Track ML experiments with MLflow, Weights & Biases, and DVC including metric logging, artifact management, model registry, hyperparameter sweeps, and experiment comparison. Use when setting up experiment tracking, comparing model runs, or managing ML artifacts and model versions."
---

# Experiment Tracking

Track, compare, and manage ML experiments with reproducibility.

## Expert Agent

For ML pipeline design and experiment methodology, delegate to the expert agent:

- **`ml-expert`**: Classical and applied ML specialist for pipeline design, model selection, and evaluation.
  - *Location*: `plugins/science-suite/agents/ml-expert.md`
  - *Capabilities*: Experiment design, hyperparameter optimization, model comparison, evaluation strategies.

## MLflow Setup

```python
import mlflow
from mlflow.tracking import MlflowClient

# Configure tracking server
mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Local SQLite
mlflow.set_experiment("my-experiment")

# Log a training run
with mlflow.start_run(run_name="baseline-v1") as run:
    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam",
        "architecture": "resnet50",
    })

    # Log metrics (step-wise)
    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_loss = evaluate(model, val_loader)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch)

    # Log final metrics
    mlflow.log_metrics({
        "test_accuracy": 0.92,
        "test_f1": 0.89,
    })

    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

## Weights & Biases Integration

```python
import wandb

def init_wandb_run(config: dict, project: str = "my-project") -> wandb.Run:
    """Initialize a W&B run with configuration."""
    run = wandb.init(
        project=project,
        config=config,
        tags=["baseline", "v1"],
        notes="Initial baseline experiment",
    )
    return run

# Training loop with W&B logging
wandb.init(project="cv-experiment", config={"lr": 0.001, "epochs": 50})

for epoch in range(50):
    metrics = train_and_evaluate(model, train_loader, val_loader)
    wandb.log({
        "train/loss": metrics["train_loss"],
        "val/loss": metrics["val_loss"],
        "val/accuracy": metrics["val_acc"],
        "epoch": epoch,
    })

wandb.finish()
```

## DVC for Data Versioning

```bash
# Initialize DVC in a git repo
# dvc init
# dvc remote add -d storage s3://my-bucket/dvc-store

# Track large data files
# dvc add data/train.csv
# git add data/train.csv.dvc data/.gitignore
# git commit -m "Track training data v1"

# Create reproducible pipeline
# dvc.yaml defines stages:
```

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - data/raw/
      - preprocess.py
    outs:
      - data/processed/
  train:
    cmd: python train.py
    deps:
      - data/processed/
      - train.py
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/best.pt
    metrics:
      - metrics.json:
          cache: false
```

## Experiment Comparison

```python
def compare_runs(experiment_name: str, metric: str = "test_accuracy") -> pd.DataFrame:
    """Compare all runs in an experiment by a specific metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )
    records = []
    for run in runs:
        records.append({
            "run_name": run.info.run_name,
            "run_id": run.info.run_id[:8],
            metric: run.data.metrics.get(metric),
            **{k: v for k, v in run.data.params.items()},
        })
    return pd.DataFrame(records)
```

## Experiment Tracking Checklist

- [ ] Log all hyperparameters before training starts
- [ ] Track metrics at every epoch (not just final)
- [ ] Version training data (DVC or artifact store)
- [ ] Save model checkpoints as artifacts
- [ ] Log environment info (Python version, package versions, GPU)
- [ ] Tag runs with meaningful labels (baseline, ablation, final)
- [ ] Set up alerts for metric regressions
- [ ] Use model registry for production promotion workflow
