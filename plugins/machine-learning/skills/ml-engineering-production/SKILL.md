# ML Engineering Production Practices

Expert guidance on software engineering, data engineering, and professional practices for production ML systems. Use when implementing robust ML workflows, building data pipelines, or establishing engineering best practices.

## Overview

This skill covers the software engineering and data engineering practices essential for ML engineers to build scalable, maintainable, and production-ready machine learning systems.

## Core Topics

### 1. Software Engineering Fundamentals

#### Python Best Practices

**Type Hints and Modern Python**
```python
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

@dataclass
class ModelConfig:
    """Configuration for ML model."""
    model_name: str
    hidden_dims: List[int]
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 32

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.dropout >= 0 and self.dropout <= 1
        assert self.learning_rate > 0
        assert self.batch_size > 0

class MLModel:
    """Production ML model with type hints."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: Optional[torch.nn.Module] = None

    def train(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """Train model and return metrics history."""
        history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': []
        }
        # Training logic here
        return history

    def predict(self, data: torch.Tensor) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model(data).detach().numpy()
```

**Project Structure**
```
ml-project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── transformer.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
├── configs/
│   ├── base.yaml
│   └── experiment.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── exploration.ipynb
├── pyproject.toml
├── requirements.txt
└── README.md
```

#### Testing ML Code

**Unit Tests with pytest**
```python
import pytest
import torch
import numpy as np
from src.models.transformer import TransformerModel
from src.data.preprocessing import normalize_features

class TestTransformerModel:
    """Test suite for Transformer model."""

    @pytest.fixture
    def model_config(self):
        """Fixture for model configuration."""
        return {
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1
        }

    @pytest.fixture
    def sample_batch(self):
        """Fixture for sample input batch."""
        batch_size, seq_len, d_model = 4, 10, 512
        return torch.randn(batch_size, seq_len, d_model)

    def test_model_initialization(self, model_config):
        """Test model initializes correctly."""
        model = TransformerModel(**model_config)
        assert model.d_model == 512
        assert model.nhead == 8

    def test_forward_pass_shape(self, model_config, sample_batch):
        """Test output shape is correct."""
        model = TransformerModel(**model_config)
        output = model(sample_batch)
        assert output.shape == sample_batch.shape

    def test_forward_pass_gradients(self, model_config, sample_batch):
        """Test gradients flow correctly."""
        model = TransformerModel(**model_config)
        output = model(sample_batch)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 5),
        (4, 10),
        (8, 20)
    ])
    def test_different_batch_sizes(self, model_config, batch_size, seq_len):
        """Test model handles different batch sizes."""
        model = TransformerModel(**model_config)
        x = torch.randn(batch_size, seq_len, model_config['d_model'])
        output = model(x)
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len

class TestDataPreprocessing:
    """Test suite for data preprocessing."""

    def test_normalize_features_mean_std(self):
        """Test normalization produces correct statistics."""
        data = np.random.randn(100, 10) * 5 + 3
        normalized = normalize_features(data)

        assert np.allclose(normalized.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(normalized.std(axis=0), 1, atol=1e-6)

    def test_normalize_features_invariance(self):
        """Test normalization is consistent."""
        data = np.random.randn(100, 10)
        normalized1 = normalize_features(data)
        normalized2 = normalize_features(data)

        assert np.allclose(normalized1, normalized2)

    @pytest.mark.parametrize("invalid_input", [
        None,
        [],
        np.array([]),
        "invalid"
    ])
    def test_normalize_features_invalid_input(self, invalid_input):
        """Test error handling for invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            normalize_features(invalid_input)
```

**Integration Tests**
```python
import pytest
import tempfile
from pathlib import Path
from src.training.trainer import Trainer
from src.models.transformer import TransformerModel
from src.data.loaders import get_dataloader

@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for full training pipeline."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_training_run(self, temp_checkpoint_dir):
        """Test complete training workflow."""
        # Setup
        model = TransformerModel(d_model=256, nhead=4, num_layers=2)
        train_loader = get_dataloader('train', batch_size=8)
        val_loader = get_dataloader('val', batch_size=8)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=temp_checkpoint_dir,
            max_epochs=2
        )

        # Train
        history = trainer.train()

        # Assertions
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 2
        assert history['train_loss'][-1] < history['train_loss'][0]  # Loss decreased

        # Check checkpoint saved
        checkpoint_files = list(temp_checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0
```

**Property-Based Testing**
```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    data=npst.arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),  # batch_size
            st.integers(min_value=1, max_value=50)     # features
        )
    )
)
def test_normalization_properties(data):
    """Property-based test for normalization."""
    # Skip degenerate cases
    if data.shape[1] == 0 or np.all(data == data[0, 0]):
        return

    normalized = normalize_features(data)

    # Properties that should always hold
    assert normalized.shape == data.shape
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()

    # Statistical properties (with tolerance for small samples)
    if data.shape[0] > 5:
        assert np.allclose(normalized.mean(axis=0), 0, atol=0.1)
```

#### Version Control Best Practices

**Git Workflow**
```bash
# Feature branch workflow
git checkout -b feature/new-model-architecture
git add src/models/new_architecture.py
git commit -m "feat: add new transformer architecture with sparse attention"

# Keep branch updated
git fetch origin
git rebase origin/main

# Push feature branch
git push origin feature/new-model-architecture

# After PR approval, squash merge to main
git checkout main
git merge --squash feature/new-model-architecture
git commit -m "feat: add new transformer architecture

- Implement sparse attention mechanism
- Add tests for new architecture
- Update documentation"
```

**.gitignore for ML Projects**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints
*.ipynb

# ML artifacts
*.h5
*.pkl
*.pth
*.onnx
*.pb
checkpoints/
logs/
runs/
wandb/
mlruns/

# Data
data/
datasets/
*.csv
*.parquet
*.arrow

# Models
models/
saved_models/
*.safetensors

# Environment
.env
.env.local
```

**Pre-commit Hooks**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 2. Data Engineering Basics

#### Data Pipeline Architecture

**ETL Pipeline with Python**
```python
from typing import Iterator, Dict, Any
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataPipeline:
    """ETL pipeline for ML data processing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.source = config['source']
        self.destination = config['destination']

    def extract(self) -> Iterator[pd.DataFrame]:
        """Extract data from source in chunks."""
        logger.info(f"Extracting data from {self.source}")

        # Handle different source types
        if self.source.endswith('.csv'):
            # Process large CSV in chunks
            for chunk in pd.read_csv(
                self.source,
                chunksize=self.config.get('chunk_size', 10000)
            ):
                yield chunk
        elif self.source.endswith('.parquet'):
            # Parquet is more efficient for large data
            df = pd.read_parquet(self.source)
            yield df
        else:
            raise ValueError(f"Unsupported source format: {self.source}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for ML training."""
        logger.info("Transforming data")

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

        # Feature engineering
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

        # Outlier removal (IQR method)
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def load(self, df: pd.DataFrame) -> None:
        """Load transformed data to destination."""
        logger.info(f"Loading data to {self.destination}")

        dest_path = Path(self.destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in efficient format
        if dest_path.suffix == '.parquet':
            df.to_parquet(dest_path, compression='snappy', index=False)
        elif dest_path.suffix == '.csv':
            df.to_csv(dest_path, index=False)
        else:
            raise ValueError(f"Unsupported destination format: {dest_path.suffix}")

    def run(self) -> None:
        """Execute full ETL pipeline."""
        logger.info("Starting ETL pipeline")

        all_chunks = []
        for chunk in self.extract():
            transformed = self.transform(chunk)
            all_chunks.append(transformed)

        # Combine all chunks
        final_df = pd.concat(all_chunks, ignore_index=True)

        # Final validation
        assert not final_df.empty, "Pipeline produced empty dataset"
        assert not final_df.isnull().all().any(), "Pipeline produced all-null columns"

        self.load(final_df)
        logger.info(f"Pipeline complete. Processed {len(final_df)} rows")

# Usage
config = {
    'source': 'data/raw/large_dataset.csv',
    'destination': 'data/processed/training_data.parquet',
    'chunk_size': 50000
}

pipeline = DataPipeline(config)
pipeline.run()
```

#### SQL for ML Engineers

**Data Extraction**
```sql
-- Feature extraction with SQL
SELECT
    user_id,
    COUNT(*) as total_orders,
    SUM(order_value) as lifetime_value,
    AVG(order_value) as avg_order_value,
    MAX(order_date) as last_order_date,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) as days_since_last_order,
    COUNT(DISTINCT product_category) as unique_categories,
    STDDEV(order_value) as order_value_std
FROM orders
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 365 DAY)
GROUP BY user_id
HAVING total_orders >= 3;

-- Time series features
SELECT
    date,
    product_id,
    SUM(quantity) as daily_sales,
    AVG(SUM(quantity)) OVER (
        PARTITION BY product_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as sales_7day_ma,
    LAG(SUM(quantity), 7) OVER (
        PARTITION BY product_id
        ORDER BY date
    ) as sales_same_day_last_week
FROM sales
GROUP BY date, product_id;

-- Training/validation split by time
-- Training set: first 80% of time period
SELECT *
FROM features
WHERE date <= (
    SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY date)
    FROM features
);

-- Validation set: remaining 20%
SELECT *
FROM features
WHERE date > (
    SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY date)
    FROM features
);
```

**Python + SQL Integration**
```python
import sqlalchemy as sa
from sqlalchemy import create_engine
import pandas as pd

class SQLDataLoader:
    """Load ML training data from SQL database."""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def load_features(
        self,
        table: str,
        feature_cols: List[str],
        target_col: str,
        date_col: str = 'date',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and target from database."""

        query = f"""
        SELECT {', '.join(feature_cols)}, {target_col}
        FROM {table}
        WHERE 1=1
        """

        if start_date:
            query += f" AND {date_col} >= '{start_date}'"
        if end_date:
            query += f" AND {date_col} <= '{end_date}'"

        df = pd.read_sql(query, self.engine)

        X = df[feature_cols]
        y = df[target_col]

        return X, y

    def load_in_batches(
        self,
        query: str,
        batch_size: int = 10000
    ) -> Iterator[pd.DataFrame]:
        """Load data in batches for memory efficiency."""

        offset = 0
        while True:
            batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            df = pd.read_sql(batch_query, self.engine)

            if df.empty:
                break

            yield df
            offset += batch_size

# Usage
loader = SQLDataLoader("postgresql://user:pass@localhost:5432/mldb")
X_train, y_train = loader.load_features(
    table='user_features',
    feature_cols=['age', 'tenure', 'avg_order_value', 'total_orders'],
    target_col='will_churn',
    start_date='2024-01-01',
    end_date='2024-10-01'
)
```

#### Data Format Optimization

**Parquet vs CSV Performance**
```python
import pandas as pd
import numpy as np
import time
from pathlib import Path

def benchmark_formats():
    """Compare different data formats for ML."""

    # Create sample data
    n_rows = 1_000_000
    data = {
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randn(n_rows),
        'feature3': np.random.randint(0, 100, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'target': np.random.randint(0, 2, n_rows)
    }
    df = pd.DataFrame(data)

    results = {}

    # CSV
    start = time.time()
    df.to_csv('data.csv', index=False)
    csv_write_time = time.time() - start
    csv_size = Path('data.csv').stat().st_size / 1024 / 1024  # MB

    start = time.time()
    pd.read_csv('data.csv')
    csv_read_time = time.time() - start

    results['CSV'] = {
        'write_time': csv_write_time,
        'read_time': csv_read_time,
        'size_mb': csv_size
    }

    # Parquet (uncompressed)
    start = time.time()
    df.to_parquet('data_uncompressed.parquet', compression=None, index=False)
    parquet_write_time = time.time() - start
    parquet_size = Path('data_uncompressed.parquet').stat().st_size / 1024 / 1024

    start = time.time()
    pd.read_parquet('data_uncompressed.parquet')
    parquet_read_time = time.time() - start

    results['Parquet (uncompressed)'] = {
        'write_time': parquet_write_time,
        'read_time': parquet_read_time,
        'size_mb': parquet_size
    }

    # Parquet (snappy compression)
    start = time.time()
    df.to_parquet('data.parquet', compression='snappy', index=False)
    parquet_snappy_write_time = time.time() - start
    parquet_snappy_size = Path('data.parquet').stat().st_size / 1024 / 1024

    start = time.time()
    pd.read_parquet('data.parquet')
    parquet_snappy_read_time = time.time() - start

    results['Parquet (snappy)'] = {
        'write_time': parquet_snappy_write_time,
        'read_time': parquet_snappy_read_time,
        'size_mb': parquet_snappy_size
    }

    # Print comparison
    print("\nFormat Comparison (1M rows):")
    print("-" * 70)
    print(f"{'Format':<25} {'Write (s)':<12} {'Read (s)':<12} {'Size (MB)':<12}")
    print("-" * 70)
    for format_name, metrics in results.items():
        print(f"{format_name:<25} "
              f"{metrics['write_time']:<12.3f} "
              f"{metrics['read_time']:<12.3f} "
              f"{metrics['size_mb']:<12.1f}")

    # Typical output:
    # Format                    Write (s)    Read (s)     Size (MB)
    # ----------------------------------------------------------------------
    # CSV                       4.231        1.892        73.2
    # Parquet (uncompressed)    0.412        0.143        38.4
    # Parquet (snappy)          0.523        0.156        26.7
```

#### Streaming Data with Kafka

```python
from kafka import KafkaProducer, KafkaConsumer
import json
import torch
from typing import Generator

class StreamingMLPipeline:
    """Real-time ML inference with Kafka."""

    def __init__(self, model: torch.nn.Module, kafka_broker: str):
        self.model = model
        self.model.eval()

        self.consumer = KafkaConsumer(
            'input-data',
            bootstrap_servers=[kafka_broker],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def process_stream(self) -> Generator[Dict, None, None]:
        """Process streaming data in real-time."""

        for message in self.consumer:
            # Extract features
            data = message.value
            features = torch.tensor([
                data['feature1'],
                data['feature2'],
                data['feature3']
            ]).float().unsqueeze(0)

            # Inference
            with torch.no_grad():
                prediction = self.model(features)
                prob = torch.sigmoid(prediction).item()

            # Send result
            result = {
                'id': data['id'],
                'prediction': float(prediction.item()),
                'probability': prob,
                'timestamp': data['timestamp']
            }

            self.producer.send('predictions', value=result)

            yield result

    def close(self):
        """Clean up resources."""
        self.consumer.close()
        self.producer.close()
```

### 3. Code Optimization and Design

#### Design Principles for ML Code

**Modular Architecture**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn

class BaseModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

class ModelFactory:
    """Factory for creating models."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register model classes."""
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create model by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}")
        return cls._registry[name](**kwargs)

# Usage
@ModelFactory.register('transformer')
class TransformerModel(BaseModel, nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

    def get_config(self) -> Dict[str, Any]:
        return {
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers
        }

# Create models dynamically
model = ModelFactory.create('transformer', d_model=512, nhead=8, num_layers=6)
```

**Scalable Training Framework**
```python
from dataclasses import dataclass
from typing import Callable, Optional, List
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging

@dataclass
class TrainingConfig:
    """Configuration for training."""
    max_epochs: int
    gradient_clip: Optional[float] = None
    checkpoint_every: int = 1
    validate_every: int = 1
    early_stopping_patience: Optional[int] = None

class Trainer:
    """Scalable training framework."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
        scheduler: Optional[_LRScheduler] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.logger = logging.getLogger(__name__)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> Dict[str, List[float]]:
        """Execute training loop."""
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)

            # Validate
            if epoch % self.config.validate_every == 0:
                val_loss = self.validate()
                history['val_loss'].append(val_loss)

                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if self.config.early_stopping_patience:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stopping_patience:
                            self.logger.info("Early stopping triggered")
                            break

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Callbacks
            for callback in self.callbacks:
                callback(epoch, self.model, history)

        return history
```

### 4. Collaboration and Soft Skills

#### Code Documentation

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for sequence modeling.

    This implementation follows "Attention Is All You Need" (Vaswani et al., 2017)
    with optional dropout and layer normalization.

    Args:
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)

    Attributes:
        attention_weights: Last computed attention weights for visualization

    Example:
        >>> layer = AttentionLayer(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)  # (batch, seq_len, d_model)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([32, 10, 512])

    Note:
        The input dimension d_model must be divisible by num_heads.

    References:
        Vaswani et al. (2017). Attention Is All You Need.
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                  where 1 indicates positions to attend to and 0 to mask

        Returns:
            Output tensor of same shape as input

        Raises:
            ValueError: If input dimensions don't match expected d_model
        """
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input dimension {x.size(-1)} doesn't match "
                f"expected d_model {self.d_model}"
            )

        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        # Shape: (batch, seq_len, num_heads, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose for attention: (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        self.attention_weights = attention  # Store for visualization
        attention = self.dropout(attention)

        # Apply attention to values
        output = torch.matmul(attention, V)

        # Concatenate heads and apply final linear
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output
```

#### Experiment Tracking

```python
import wandb
from pathlib import Path
from typing import Dict, Any

class ExperimentTracker:
    """
    Track ML experiments with Weights & Biases.

    Handles model checkpointing, metric logging, and artifact management
    for reproducible experiments.
    """

    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize experiment tracker.

        Args:
            project: W&B project name
            config: Experiment configuration dictionary
            name: Optional run name
            tags: Optional tags for run organization
        """
        self.run = wandb.init(
            project=project,
            config=config,
            name=name,
            tags=tags
        )
        self.config = config

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to W&B."""
        wandb.log(metrics, step=step)

    def log_model(
        self,
        model: nn.Module,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model as W&B artifact."""
        artifact = wandb.Artifact(name, type='model', metadata=metadata)

        # Save model
        model_path = Path(f"models/{name}.pth")
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)

        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish experiment tracking."""
        wandb.finish()

# Usage in training loop
tracker = ExperimentTracker(
    project="my-ml-project",
    config={
        'learning_rate': 1e-3,
        'batch_size': 32,
        'model': 'transformer',
        'd_model': 512
    },
    name="experiment-001",
    tags=['baseline', 'transformer']
)

for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()

    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    }, step=epoch)

    if val_loss < best_val_loss:
        tracker.log_model(
            model,
            name=f"best_model_epoch_{epoch}",
            metadata={'val_loss': val_loss}
        )

tracker.finish()
```

## Best Practices Summary

### Code Quality
1. Use type hints for all function signatures
2. Write comprehensive unit and integration tests
3. Follow PEP 8 style guidelines
4. Use pre-commit hooks for automatic formatting
5. Document all public APIs with docstrings

### Data Engineering
1. Use Parquet for large datasets (10x smaller, 5x faster than CSV)
2. Process data in chunks for memory efficiency
3. Validate data at pipeline boundaries
4. Use SQL for efficient feature extraction
5. Implement proper error handling and logging

### Project Organization
1. Separate concerns: models, data, training, evaluation
2. Use configuration files for hyperparameters
3. Version control everything except data/models
4. Implement modular, reusable components
5. Track experiments systematically

### Collaboration
1. Write clear commit messages
2. Use feature branches for development
3. Document design decisions
4. Share experiment results
5. Review code before merging

## Quick Reference

### Testing Commands
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models.py::TestTransformer

# Run property-based tests
pytest -v tests/test_properties.py
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Commit with conventional commit format
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update readme"
git commit -m "refactor: improve code structure"

# Rebase before merge
git fetch origin
git rebase origin/main
```

### Data Pipeline
```bash
# Convert CSV to Parquet
python -c "import pandas as pd; pd.read_csv('data.csv').to_parquet('data.parquet')"

# Check Parquet schema
python -c "import pyarrow.parquet as pq; print(pq.read_schema('data.parquet'))"
```

## When to Use This Skill

Use this skill when you need to:
- Set up proper testing infrastructure for ML code
- Design ETL pipelines for ML data
- Implement scalable training frameworks
- Establish code quality standards
- Integrate SQL databases with ML workflows
- Track and manage ML experiments
- Collaborate effectively on ML projects
- Optimize data storage and processing
- Build production-ready ML systems

This skill bridges the gap between data science and software engineering, ensuring ML systems are maintainable, scalable, and production-ready.
