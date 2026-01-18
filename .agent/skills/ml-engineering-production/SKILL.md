---
name: ml-engineering-production
version: "1.0.7"
maturity: "5-Expert"
specialization: Production ML Engineering Practices
description: Software and data engineering best practices for production ML. Type-safe code, pytest testing, pre-commit hooks, pandas/SQL pipelines, and modern project structure. Use when building maintainable ML systems.
---

# ML Engineering Production Practices

Software and data engineering practices for scalable, maintainable ML systems.

---

## Project Structure

```
ml-project/
├── src/
│   ├── models/        # Model definitions (Flax)
│   ├── data/          # Data loaders, preprocessing
│   ├── training/      # Trainer, callbacks
│   └── utils/         # Metrics, helpers
├── tests/             # pytest tests
├── configs/           # YAML configurations
├── scripts/           # Training/evaluation scripts
├── pyproject.toml     # Project config
└── README.md
```

---

## Type-Safe ML Code (JAX/Flax)

```python
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import nnx
import optax

@dataclass
class ModelConfig:
    hidden_dims: list[int]
    dropout_rate: float = 0.1
    learning_rate: float = 1e-3

class MLModel(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        layers = []
        for dim in config.hidden_dims:
            layers.append(nnx.Linear(dim, dim, rngs=rngs))
            layers.append(nnx.relu)
            layers.append(nnx.Dropout(config.dropout_rate, rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)

class Trainer:
    def __init__(self, model: MLModel, optimizer: nnx.Optimizer):
        self.model = model
        self.optimizer = optimizer

    @nnx.jit
    def train_step(self, batch_x: jax.Array, batch_y: jax.Array) -> jax.Array:
        def loss_fn(model):
            logits = model(batch_x)
            return optax.l2_loss(logits, batch_y).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(grads)
        return loss

    @nnx.jit
    def predict(self, x: jax.Array) -> jax.Array:
        return self.model(x)
```

---

## Testing ML Code

```python
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

class TestModel:
    @pytest.fixture
    def model(self):
        config = ModelConfig(hidden_dims=[64, 32])
        return MLModel(config, rngs=nnx.Rngs(0))

    def test_forward_shape(self, model):
        x = jnp.ones((4, 64)) # Batch size 4
        output = model(x)
        assert output.shape == (4, 32)

    def test_gradients_flow(self, model):
        x = jnp.ones((4, 64))
        y = jnp.ones((4, 32))

        def loss_fn(m):
            return ((m(x) - y) ** 2).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)

        # Check gradients exist and are finite
        assert not jnp.isnan(loss)
        # Verify gradients for first layer weights exist
        first_layer_grads = grads.net.layers[0].kernel
        assert first_layer_grads is not None
        assert not jnp.isnan(first_layer_grads).any()

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_batch_sizes(self, model, batch_size):
        x = jnp.ones((batch_size, 64))
        assert model(x).shape[0] == batch_size
```

---

## Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
```

---

## Data Pipeline (ETL)

```python
import pandas as pd
from pathlib import Path
from typing import Iterator

class DataPipeline:
    def __init__(self, source: str, destination: str, chunk_size: int = 10000):
        self.source = source
        self.destination = destination
        self.chunk_size = chunk_size

    def extract(self) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(self.source, chunksize=self.chunk_size):
            yield chunk

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df

    def load(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.destination, compression='snappy', index=False)

    def run(self) -> None:
        chunks = [self.transform(chunk) for chunk in self.extract()]
        final_df = pd.concat(chunks, ignore_index=True)
        self.load(final_df)
```

---

## SQL Feature Extraction

```python
from sqlalchemy import create_engine
import pandas as pd

class SQLDataLoader:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def load_features(self, table: str, feature_cols: list[str],
                      target_col: str, start_date: str = None) -> tuple:
        query = f"SELECT {', '.join(feature_cols)}, {target_col} FROM {table}"
        if start_date:
            query += f" WHERE date >= '{start_date}'"
        df = pd.read_sql(query, self.engine)
        return df[feature_cols], df[target_col]
```

---

## Data Format Comparison

| Format | Write (1M rows) | Read | Size | Use Case |
|--------|-----------------|------|------|----------|
| CSV | 4.2s | 1.9s | 73 MB | Exchange, debugging |
| Parquet (uncompressed) | 0.4s | 0.15s | 38 MB | Fast I/O |
| Parquet (snappy) | 0.5s | 0.16s | 27 MB | Production |

---

## Experiment Tracking

```python
import wandb
import orbax.checkpoint as ocp

class ExperimentTracker:
    def __init__(self, project: str, config: dict):
        self.run = wandb.init(project=project, config=config)
        self.checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def log_model(self, model_state, path: str, step: int):
        # Async checkpointing with Orbax
        self.checkpointer.save(path, model_state, step=step)
        artifact = wandb.Artifact(f'model-step-{step}', type='model')
        artifact.add_dir(path)
        self.run.log_artifact(artifact)

    def finish(self):
        wandb.finish()
```

---

## Best Practices Summary

| Area | Practice |
|------|----------|
| **Code Quality** | Type hints, pytest, PEP 8, pre-commit, docstrings |
| **Data** | Parquet format, chunked processing, validation |
| **Project** | Separate concerns, config files, modular components |
| **Collaboration** | Clear commits, feature branches, code review |

---

## Testing Commands

```bash
pytest tests/                          # Run all tests
pytest --cov=src tests/                # With coverage
pytest tests/test_models.py::TestModel # Specific test
```

---

## Git Workflow

```bash
git checkout -b feature/new-feature
git commit -m "feat: add new feature"
git fetch origin && git rebase origin/main
```

---

## .gitignore for ML

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints
*.h5, *.pkl, *.pth, *.onnx
checkpoints/, logs/, wandb/, mlruns/
data/, datasets/
*.csv, *.parquet
.env
```

---

## ML Engineering Checklist

- [ ] Type hints on all functions
- [ ] Unit and integration tests (≥80% coverage)
- [ ] Pre-commit hooks configured
- [ ] Parquet for large datasets
- [ ] Modular project structure
- [ ] Experiment tracking enabled
- [ ] Configuration files (not hardcoded)
- [ ] Documentation up to date

---

**Version**: 1.0.5
