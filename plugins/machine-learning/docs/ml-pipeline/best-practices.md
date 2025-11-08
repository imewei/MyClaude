# ML Pipeline Best Practices

Production readiness checklist and best practices for enterprise machine learning systems.

## Production Readiness Checklist

### Data Pipeline
- [ ] Data validation with Great Expectations or similar
- [ ] Schema versioning and evolution strategy
- [ ] Data quality monitoring and alerting
- [ ] Data lineage tracking
- [ ] PII handling and compliance (GDPR, HIPAA)
- [ ] Backup and disaster recovery procedures
- [ ] Data versioning with DVC or lakeFS
- [ ] Incremental data loading implemented

### Model Training
- [ ] Reproducible training with seed management
- [ ] Hyperparameter optimization automated
- [ ] Experiment tracking (MLflow/W&B)
- [ ] Model versioning in registry
- [ ] Training/validation/test split documented
- [ ] Cross-validation implemented
- [ ] Feature importance analysis
- [ ] Model interpretability (SHAP/LIME)

### Model Deployment
- [ ] Containerized with Docker
- [ ] Health check endpoints implemented
- [ ] Graceful shutdown handling
- [ ] Resource limits defined
- [ ] Horizontal auto-scaling configured
- [ ] Load testing completed
- [ ] Rollback procedure tested
- [ ] Blue-green or canary deployment strategy

### Monitoring & Observability
- [ ] Prediction latency monitoring
- [ ] Error rate tracking
- [ ] Data drift detection automated
- [ ] Concept drift detection
- [ ] Model performance dashboards
- [ ] Alerting configured (PagerDuty/Slack)
- [ ] Distributed tracing enabled
- [ ] Structured logging implemented

### Security
- [ ] Secrets management (Vault/AWS Secrets Manager)
- [ ] TLS/SSL for all endpoints
- [ ] Authentication and authorization
- [ ] Rate limiting implemented
- [ ] Input validation and sanitization
- [ ] Dependency vulnerability scanning
- [ ] Container image scanning
- [ ] Least privilege access control

### Testing
- [ ] Unit tests for data transformations
- [ ] Integration tests for pipeline
- [ ] Model quality tests (invariance, directional)
- [ ] Load/stress testing
- [ ] Smoke tests for deployment
- [ ] Regression tests for model performance
- [ ] CI/CD pipeline with automated testing

### Documentation
- [ ] Model card created
- [ ] Architecture diagram
- [ ] Deployment runbook
- [ ] Incident response procedures
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Training data documentation
- [ ] Feature documentation

---

## Code Quality Standards

### Type Hints and Documentation
```python
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def engineer_features(
    raw_data: pd.DataFrame,
    feature_config: Dict[str, any]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer features from raw customer data.

    Args:
        raw_data: Raw customer dataframe with columns [customer_id, signup_date, ...]
        feature_config: Configuration dict with feature engineering parameters

    Returns:
        Tuple of (feature_dataframe, feature_names)

    Raises:
        ValueError: If required columns missing from raw_data
        TypeError: If feature_config not a dictionary

    Example:
        >>> raw_data = pd.DataFrame({'customer_id': [1, 2], 'signup_date': ['2023-01-01', '2023-02-01']})
        >>> config = {'window_days': 30}
        >>> features, names = engineer_features(raw_data, config)
        >>> len(features) == len(raw_data)
        True
    """
    # Implementation
    pass
```

### Error Handling
```python
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

class ModelNotFoundError(Exception):
    """Raised when model cannot be loaded"""
    pass

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load data with comprehensive error handling"""

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Data file is empty: {file_path}")
        raise DataValidationError(f"Empty data file: {file_path}")

    # Validate required columns
    required_columns = ['customer_id', 'signup_date', 'status']
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    # Validate data types
    if not pd.api.types.is_datetime64_any_dtype(data['signup_date']):
        try:
            data['signup_date'] = pd.to_datetime(data['signup_date'])
        except Exception as e:
            raise DataValidationError(f"Invalid date format in signup_date: {e}")

    return data
```

### Logging Standards
```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Usage
logger.info(
    "model_prediction_made",
    customer_id="12345",
    prediction="churn",
    confidence=0.85,
    latency_ms=42.3,
    model_version="v2.1.3"
)

logger.warning(
    "low_confidence_prediction",
    customer_id="67890",
    confidence=0.52,
    threshold=0.70
)

logger.error(
    "prediction_failed",
    customer_id="11111",
    error=str(exception),
    stack_trace=traceback.format_exc()
)
```

---

## Testing Strategies

### Unit Tests for Transformations
```python
import pytest
import pandas as pd
from src.features import calculate_customer_tenure, calculate_total_spend

def test_customer_tenure_zero_days():
    """Test tenure calculation for same-day signup"""
    signup_date = "2024-01-15"
    current_date = "2024-01-15"

    tenure = calculate_customer_tenure(signup_date, current_date)

    assert tenure == 0, "Same day signup should have 0 tenure"

def test_customer_tenure_one_year():
    """Test tenure calculation for one year"""
    signup_date = "2023-01-15"
    current_date = "2024-01-15"

    tenure = calculate_customer_tenure(signup_date, current_date)

    assert tenure == 365, f"Expected 365 days, got {tenure}"

def test_total_spend_empty_transactions():
    """Test total spend with no transactions"""
    transactions = pd.DataFrame(columns=['amount', 'date'])

    total_spend = calculate_total_spend(transactions, window_days=30)

    assert total_spend == 0, "Empty transactions should return 0 spend"
```

### Integration Tests
```python
def test_end_to_end_prediction_pipeline():
    """Test complete prediction workflow"""

    # Load test data
    test_data = load_test_dataset()

    # Extract features
    features = feature_pipeline.transform(test_data)

    # Load model
    model = load_model("test_model_v1.0.0")

    # Predict
    predictions = model.predict(features)

    # Validate predictions
    assert len(predictions) == len(test_data), "Prediction count mismatch"
    assert all(p in [0, 1] for p in predictions), "Invalid prediction values"
```

### Model Quality Tests
```python
def test_model_invariance_to_feature_order():
    """Ensure model predictions don't depend on feature order"""

    X_test = load_test_features()
    model = load_model()

    predictions_1 = model.predict(X_test)
    predictions_2 = model.predict(X_test[X_test.columns[::-1]])  # Reverse order

    assert (predictions_1 == predictions_2).all(), "Model not invariant to feature order"

def test_model_directional_expectation():
    """Test model responds correctly to known patterns"""

    # High-spend, long-tenure customer should have low churn probability
    high_value_customer = create_test_sample(
        tenure_days=1000,
        total_spend=50000,
        days_since_last_login=1
    )

    churn_prob = model.predict_proba(high_value_customer)[0][1]

    assert churn_prob < 0.3, f"High-value customer churn prob too high: {churn_prob}"

    # Zero-spend, new customer should have high churn probability
    low_value_customer = create_test_sample(
        tenure_days=7,
        total_spend=0,
        days_since_last_login=30
    )

    churn_prob = model.predict_proba(low_value_customer)[0][1]

    assert churn_prob > 0.5, f"Low-value customer churn prob too low: {churn_prob}"
```

---

## Performance Optimization

### Batch Prediction
```python
import numpy as np

def batch_predict(model, features, batch_size=1000):
    """Batch predictions for better throughput"""

    n_samples = len(features)
    predictions = np.zeros(n_samples)

    for i in range(0, n_samples, batch_size):
        batch = features[i:i+batch_size]
        predictions[i:i+batch_size] = model.predict(batch)

    return predictions
```

### Feature Caching
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def get_cached_features(customer_id: str, date: str):
    """Cache frequently accessed features"""

    cache_key = hashlib.md5(f"{customer_id}_{date}".encode()).hexdigest()

    # Try to get from cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute features
    features = compute_features(customer_id, date)

    # Store in cache with 1-hour TTL
    redis_client.setex(cache_key, 3600, json.dumps(features))

    return features
```

---

## Security Best Practices

### Secrets Management
```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str) -> dict:
    """Retrieve secrets from AWS Secrets Manager"""

    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager')

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        logger.error(f"Failed to retrieve secret: {e}")
        raise

# Usage
db_credentials = get_secret("ml-pipeline/database")
connection = connect_to_database(
    host=db_credentials['host'],
    user=db_credentials['username'],
    password=db_credentials['password']
)
```

### Input Validation
```python
from pydantic import BaseModel, validator, Field
from typing import List

class PredictionRequest(BaseModel):
    customer_id: str = Field(..., regex=r'^[a-zA-Z0-9\-]+$', max_length=50)
    features: dict

    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v:
            raise ValueError("customer_id cannot be empty")
        return v

    @validator('features')
    def validate_features(cls, v):
        required_features = ['tenure_days', 'total_spend', 'days_since_last_login']
        missing = set(required_features) - set(v.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Validate feature ranges
        if v['tenure_days'] < 0:
            raise ValueError("tenure_days must be non-negative")

        return v

# FastAPI automatically validates with Pydantic
@app.post("/predict")
async def predict(request: PredictionRequest):  # Validation happens here
    # Safe to use request.features
    pass
```

---

## Disaster Recovery

### Backup Strategy
```python
import boto3
from datetime import datetime

def backup_model_artifacts(model_version: str):
    """Backup model and metadata to S3"""

    s3_client = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Backup model file
    s3_client.upload_file(
        f'models/{model_version}/model.pkl',
        'ml-model-backups',
        f'{model_version}/model_{timestamp}.pkl'
    )

    # Backup feature engineering code
    s3_client.upload_file(
        'src/features.py',
        'ml-model-backups',
        f'{model_version}/features_{timestamp}.py'
    )

    # Backup model metadata
    metadata = {
        'model_version': model_version,
        'backup_timestamp': timestamp,
        'training_date': get_training_date(model_version),
        'metrics': get_model_metrics(model_version)
    }

    s3_client.put_object(
        Bucket='ml-model-backups',
        Key=f'{model_version}/metadata_{timestamp}.json',
        Body=json.dumps(metadata)
    )
```

### Rollback Procedure
```bash
#!/bin/bash
# rollback_model.sh

MODEL_VERSION=$1

echo "Rolling back to model version: $MODEL_VERSION"

# Update Kubernetes deployment
kubectl set image deployment/ml-model-serving \
    ml-model=ml-model:$MODEL_VERSION

# Wait for rollout
kubectl rollout status deployment/ml-model-serving

# Verify health
HEALTH=$(curl -s http://ml-model-service/health | jq -r '.status')

if [ "$HEALTH" = "healthy" ]; then
    echo "Rollback successful"
    exit 0
else
    echo "Rollback failed, reverting"
    kubectl rollout undo deployment/ml-model-serving
    exit 1
fi
```

---

## Best Practices Summary

1. **Code Quality**: Type hints, comprehensive error handling, structured logging
2. **Testing**: Unit, integration, and model quality tests with >80% coverage
3. **Security**: Secrets management, input validation, least privilege access
4. **Performance**: Batch predictions, caching, optimized data loading
5. **Monitoring**: Comprehensive metrics, drift detection, alerting
6. **Documentation**: Model cards, API docs, runbooks
7. **Disaster Recovery**: Regular backups, tested rollback procedures
8. **Reproducibility**: Version everything (code, data, models, infrastructure)
