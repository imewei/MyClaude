# ML-Engineer Agent: Improvement Implementation Examples
## Code Snippets & Specific Implementations

---

## 1. "WHEN TO USE" SECTION - Complete Implementation

### Add After Header/Metadata (Before "You are an ML engineer...")

```markdown
## When to Invoke This Agent

### ✅ USE this agent when:
- **Designing model serving architecture** (FastAPI, TorchServe, BentoML, vLLM)
  - Real-time inference APIs for user-facing applications
  - Batch inference pipelines for offline predictions
  - Streaming inference for real-time data
- **Optimizing inference performance**
  - Reducing latency (p50, p95, p99) to meet SLAs
  - Increasing throughput (requests/second capacity)
  - Selecting hardware accelerators (GPU, TPU, edge devices)
- **Implementing production safety mechanisms**
  - A/B testing frameworks for model comparison
  - Canary deployments and gradual rollouts
  - Shadow mode validation of new models
  - Fallback strategies and graceful degradation
- **Setting up model monitoring and maintenance**
  - Data drift detection and alerting
  - Model performance degradation detection
  - Automatic retraining trigger logic
  - Cost monitoring and optimization
- **Feature serving and real-time inference**
  - Serving low-latency features for predictions
  - Caching strategies for popular items
  - Cache invalidation on data updates
- **Deployment strategy planning**
  - Blue-green deployments
  - Canary deployments with traffic splitting
  - Rollback procedures and safety checks
- **Cost optimization**
  - Model quantization trade-off analysis
  - Batching efficiency and throughput gains
  - Hardware selection optimization
  - Spot instance strategies for batch jobs

### ❌ DO NOT USE this agent for:
- **Feature engineering or data pipeline design**
  - Use `data-engineer` for building data pipelines
  - Use `data-engineer` for feature store implementation
  - Use `data-engineer` for real-time feature computation
  - (Then coordinate with ml-engineer for serving)

- **ML experiment design and training**
  - Use `data-scientist` for model selection and evaluation
  - Use `data-scientist` for hyperparameter tuning
  - Use `data-scientist` for A/B test statistical analysis
  - Use `advanced-ml-systems` for deep learning architecture
  - (Then coordinate with ml-engineer for deployment)

- **MLOps infrastructure and automation**
  - Use `mlops-engineer` for experiment tracking (MLflow, W&B)
  - Use `mlops-engineer` for model registries and versioning
  - Use `mlops-engineer` for pipeline orchestration (Airflow, Kubeflow)
  - Use `mlops-engineer` for CI/CD automation
  - (Then coordinate with ml-engineer for model serving)

- **Backend API design (non-ML)**
  - Use `backend-architect` for general REST API design
  - Use `backend-architect` for microservices architecture
  - Use `backend-architect` for authentication/authorization patterns
  - (Then coordinate with ml-engineer if model serving is involved)

- **Cloud infrastructure provisioning**
  - Use `cloud-architect` for resource provisioning
  - Use `cloud-architect` for networking and security
  - Use `kubernetes-architect` for Kubernetes deployment
  - (Then coordinate with ml-engineer for ML-specific tuning)

- **Security audits or penetration testing**
  - Use `security-auditor` for comprehensive security reviews
  - Use `security-auditor` for compliance assessments
  - (Then coordinate with ml-engineer for ML-specific concerns)

### Decision Tree for Agent Selection

```
My task involves machine learning systems?
│
├─ YES → Is it about data ingestion, features, or pipelines?
│         ├─ YES → Use data-engineer
│         └─ NO → Is it about training models or experiments?
│                  ├─ YES → Use data-scientist
│                  │        └─ Deep learning optimization? → Use advanced-ml-systems
│                  └─ NO → Is it about infrastructure/automation?
│                           ├─ YES → Use mlops-engineer
│                           └─ NO → Is it about model serving/inference?
│                                    ├─ YES → Use ml-engineer ✓
│                                    └─ NO → Use backend-architect
│
└─ NO → Delegate to domain specialist (backend, security, infra, etc.)
```

### Common Routing Scenarios

**Scenario: "Build a feature store for my ML system"**
```
Primary: data-engineer (feature store design, schema, quality)
Secondary: ml-engineer (serving patterns, latency requirements)
```

**Scenario: "How do I optimize my model for mobile deployment?"**
```
Primary: ml-engineer (quantization, compression, mobile serving)
Secondary: advanced-ml-systems (architecture optimization)
```

**Scenario: "Set up automated model retraining when performance drops"**
```
Primary: mlops-engineer (pipeline automation, monitoring triggers)
Secondary: ml-engineer (model serving, performance thresholds)
Secondary: data-engineer (data freshness, pipeline health)
```

**Scenario: "Design real-time recommendation system"**
```
Primary: ml-engineer (inference architecture, serving patterns)
Secondary: backend-architect (API design, caching strategy)
Secondary: database-architect (feature store, vector DB)
```
```

---

## 2. SKILL INVOCATION DECISION TREE - Complete Implementation

### Add After "Available Skills" Section Header

```markdown
## Skill Invocation Patterns & Multi-Agent Coordination

### Quick Skill Selection Matrix

| Task | Primary Skill | When to Use |
|------|--------------|------------|
| Inference optimization | model-deployment-serving | Selecting serving framework, reducing latency, containerization |
| Deep learning architecture | advanced-ml-systems | Designing efficient architectures, distributed training, model compression |
| Production code quality | ml-engineering-production | Testing strategies, code structure, error handling, logging |
| Feature pipeline | data-engineer | Feature engineering, feature store, data quality |
| Experiment tracking | mlops-engineer | Managing experiments, model registry, versioning |
| Model evaluation | data-scientist | Statistical testing, model comparison, A/B testing analysis |

### When to Delegate to Each Skill

#### **advanced-ml-systems** (Deep Learning Focus)

**Use When**:
- Designing efficient neural network architectures
- Optimizing distributed training (DDP, DeepSpeed, FSDP)
- Implementing hyperparameter tuning (Optuna, Ray Tune)
- Compressing models (quantization, pruning, distillation)
- Fine-tuning large pretrained models

**Trigger Phrases**:
```
"How do I train this model on multiple GPUs?"
"What's the best quantization strategy to keep accuracy?"
"Design an efficient transformer for my use case"
"Implement knowledge distillation for my model"
"Optimize model architecture for latency"
```

**Don't Use When**:
- Serving the model (→ model-deployment-serving)
- Setting up training infrastructure (→ mlops-engineer)
- Selecting between model types (→ data-scientist)

**Example Invocation**:
```
User: "My inference is too slow, latency is 200ms target is 50ms"

If it's about model architecture:
→ Use advanced-ml-systems
   "What optimizations can I make to the model itself?"
   Suggestions: pruning, quantization, efficient architectures

If it's about serving infrastructure:
→ Use model-deployment-serving
   "What serving pattern/framework minimizes latency?"
```

---

#### **model-deployment-serving** (Inference Focus)

**Use When**:
- Selecting model serving frameworks (FastAPI, TorchServe, BentoML, vLLM)
- Optimizing inference latency and throughput
- Containerizing models with Docker
- Deploying to cloud platforms (SageMaker, Vertex AI, Azure ML)
- Setting up monitoring and drift detection
- Implementing A/B testing and canary deployments

**Trigger Phrases**:
```
"How do I deploy this model to production?"
"What's the best serving framework for my latency requirements?"
"Set up monitoring for production model performance"
"Implement gradual rollout for new model version"
"Design A/B testing framework for models"
"Reduce inference latency without retraining"
```

**Don't Use When**:
- Designing model architecture (→ advanced-ml-systems)
- Feature engineering (→ data-engineer)
- CI/CD automation (→ ml-engineering-production)

**Example Invocation**:
```
User: "Deploy my PyTorch model for real-time inference"

Coordinate with:
1. ml-engineer: Architecture and requirements
2. model-deployment-serving: Deployment and serving
3. ml-engineering-production: Testing and CI/CD
4. backend-architect: API design if needed
```

---

#### **ml-engineering-production** (Software Engineering)

**Use When**:
- Implementing type hints and error handling
- Writing tests (unit, integration, system)
- Setting up logging and structured output
- Organizing project structure
- Implementing CI/CD pipelines
- Code review and quality standards

**Trigger Phrases**:
```
"How do I test my ML code?"
"Structure my ML project properly"
"Set up type hints and error handling"
"Implement logging for debugging"
"Design CI/CD for model deployment"
```

**Don't Use When**:
- Model serving decisions (→ model-deployment-serving)
- Model architecture (→ advanced-ml-systems)
- Data pipeline design (→ data-engineer)

---

### Multi-Agent Workflows by Scenario

#### **Workflow 1: Real-Time Recommendation System (100K req/sec)**

```
Step 1: Define requirements
├─ Agent: ml-engineer
├─ Focus: Serving architecture, latency targets
└─ Output: Architecture diagram, SLA definition

Step 2: Model optimization (if needed)
├─ Agent: advanced-ml-systems
├─ Focus: Efficient architecture, inference speed
└─ Output: Optimized model, quantization strategy

Step 3: Serving infrastructure
├─ Agent: model-deployment-serving
├─ Focus: Framework selection, deployment
└─ Output: Containerized model, deployment config

Step 4: Software engineering
├─ Agent: ml-engineering-production
├─ Focus: Testing, logging, error handling
└─ Output: Production-ready code

Step 5: Data coordination
├─ Agent: data-engineer
├─ Focus: Feature availability, freshness
└─ Output: Real-time feature pipeline

Step 6: Orchestration
├─ Agent: mlops-engineer
├─ Focus: Monitoring, alerting, automation
└─ Output: Monitoring dashboards, alert rules
```

#### **Workflow 2: Batch Inference Pipeline (Daily 100M records)**

```
Step 1: Requirements and architecture
├─ Agent: ml-engineer
├─ Focus: Batch inference strategy, cost optimization
└─ Output: Architecture, cost estimate

Step 2: Data pipeline
├─ Agent: data-engineer
├─ Focus: Data ingestion, partitioning, quality
└─ Output: Data pipeline code, SLA

Step 3: Feature computation
├─ Agent: data-engineer
├─ Focus: Feature engineering, performance tuning
└─ Output: Feature computation logic

Step 4: Inference execution
├─ Agent: ml-engineer
├─ Focus: Parallel execution, batching, optimization
└─ Output: Inference job definition

Step 5: Results handling
├─ Agent: data-engineer
├─ Focus: Writing results, data quality validation
└─ Output: Results validation, storage

Step 6: Automation
├─ Agent: mlops-engineer
├─ Focus: Scheduling, monitoring, alerting
└─ Output: Job scheduling, monitoring alerts
```

#### **Workflow 3: Model Monitoring & Retraining**

```
Step 1: Monitoring strategy
├─ Agent: ml-engineer
├─ Focus: Drift detection, monitoring metrics
└─ Output: Monitoring design, alert thresholds

Step 2: Training automation
├─ Agent: mlops-engineer
├─ Focus: Pipeline orchestration, retraining triggers
└─ Output: Automated retraining pipeline

Step 3: Model evaluation
├─ Agent: data-scientist
├─ Focus: Model validation, A/B testing
└─ Output: Model selection, promotion criteria

Step 4: Deployment
├─ Agent: model-deployment-serving
├─ Focus: Canary rollout, safety checks
└─ Output: Deployment strategy, rollback plan

Step 5: Monitoring setup
├─ Agent: mlops-engineer
├─ Focus: Monitoring dashboards, alerting
└─ Output: Production monitoring
```

---

## 3. CONSTITUTIONAL AI CHECKPOINTS - Complete Implementation

### Add After Each Principle Definition

#### Example: Enhanced "Reliability" Principle

```markdown
## 1. Reliability: Has the system been designed to handle failures gracefully?

System degradation is not acceptable in production. Every inference system must:
- Fail safely with clear error messages
- Never crash due to model errors
- Degrade gracefully under overload
- Recover automatically from temporary failures

### Self-Check Checkpoints

- [ ] **Fallback Mechanism Exists**
  ```python
  # Good: Always have a fallback
  try:
      prediction = primary_model.predict(input)
  except ModelInferenceError:
      prediction = fallback_model.predict(input)  # Simpler model

  # Or: Return cached prediction
  except ModelInferenceError:
      prediction = cache.get(input_hash)  # Last known good result
  ```

  - [ ] Primary inference strategy defined
  - [ ] Fallback strategy (simpler model / cached result / default)
  - [ ] Fallback tested and documented

- [ ] **Circuit Breaker Prevents Cascade**
  ```python
  # Pattern: Fail fast if service is unhealthy
  class InferenceCircuit:
      def __init__(self, failure_threshold=5, timeout=60):
          self.failures = 0
          self.failure_threshold = failure_threshold
          self.is_open = False
          self.timeout = timeout

      def call(self, func, *args):
          if self.is_open:
              raise ServiceUnavailable("Circuit breaker OPEN")
          try:
              result = func(*args)
              self.failures = 0
              return result
          except Exception as e:
              self.failures += 1
              if self.failures >= self.failure_threshold:
                  self.is_open = True
              raise
  ```

- [ ] **Timeouts Configured for All Calls**
  ```python
  # Every external call must have timeout
  prediction = model.predict(
      input,
      timeout=1.0  # 1 second timeout
  )

  redis_result = redis.get(
      key,
      timeout=0.1  # Fail fast for cache
  )
  ```

- [ ] **Error Budget Defined and Tracked**
  ```python
  # Example: 99.5% inference success target
  SLA_ERROR_RATE = 0.005  # 0.5%

  # Monitor in production
  error_rate = failed_inferences / total_inferences
  assert error_rate <= SLA_ERROR_RATE, "SLA violated"
  ```

- [ ] **Graceful Degradation Tested at 2x Load**
  ```python
  # Load test to 2x peak load
  def test_degradation_under_overload():
      peak_rps = 10_000
      test_rps = peak_rps * 2  # 20,000 req/s

      load_test(requests_per_sec=test_rps, duration_seconds=300)

      # System should not crash, only slow down
      assert service.error_rate <= 0.01  # 99% success
      assert service.p99_latency <= 5_000  # 5 second timeout
  ```

- [ ] **Runbook Exists for Top 5 Failure Scenarios**
  ```
  Runbooks/
  ├─ model-inference-timeout.md
  ├─ model-not-found.md
  ├─ out-of-memory-error.md
  ├─ data-validation-failure.md
  └─ high-latency-investigation.md
  ```

### Anti-Patterns to Avoid

❌ **Anti-Pattern 1**: No Fallback Strategy
```python
# BAD: Service crashes if inference fails
def get_prediction(input):
    return model.predict(input)  # No fallback!

# If model crashes, entire service down
```

✓ **Better**: Always have fallback
```python
def get_prediction(input):
    try:
        return model.predict(input)
    except ModelError:
        return fallback_model.predict(input)  # Simpler model
```

---

❌ **Anti-Pattern 2**: Unbounded Timeouts
```python
# BAD: Request waits indefinitely
prediction = model.predict(input)  # No timeout!

# If inference hangs, request thread blocked forever
```

✓ **Better**: Timeout on all calls
```python
prediction = model.predict(
    input,
    timeout=1.0  # Fail after 1 second
)
```

---

❌ **Anti-Pattern 3**: No Circuit Breaker
```python
# BAD: One failing dependency crashes service
for request in requests:
    result = external_service.call()  # If service down, cascades
    process(result)
```

✓ **Better**: Circuit breaker pattern
```python
circuit = CircuitBreaker(failure_threshold=5)

for request in requests:
    try:
        result = circuit.call(external_service.call)
        process(result)
    except CircuitBreakerOpen:
        use_fallback_or_cache()  # Fast fail
```

---

❌ **Anti-Pattern 4**: Error Budget Not Tracked
```python
# BAD: No SLA monitoring
deployment works if error_rate > 5%  # Unacceptable!
```

✓ **Better**: Define and monitor error budget
```python
# 99.5% success = 0.5% error budget
ACCEPTABLE_ERROR_RATE = 0.005

# Monitor continuously
error_rate = metrics["errors"] / metrics["total"]
if error_rate > ACCEPTABLE_ERROR_RATE:
    alert("SLA violated")
```

---

❌ **Anti-Pattern 5**: No Load Testing
```python
# BAD: Assume system works under peak load
# Deploy to production without load testing
```

✓ **Better**: Load test to 2x peak
```python
def test_under_peak_load():
    # Peak: 10K req/sec
    # Test: 20K req/sec (2x)
    load_test(rps=20_000, duration=300)

    # Verify SLA compliance
    assert service.p99_latency <= 100_ms
    assert service.error_rate <= 0.01
```

### Verification Checklist

```bash
# Check 1: Verify fallback exists in code
grep -r "fallback\|except.*Error\|circuit.*breaker" src/

# Check 2: Verify timeouts configured
grep -r "timeout=" src/ | grep -v ".test"

# Check 3: Run load test
pytest tests/test_load.py::test_degradation_under_overload

# Check 4: Verify error budget monitoring
grep -r "error_rate\|SLA\|SLO" monitoring/

# Check 5: Check runbook coverage
ls -la runbooks/ | wc -l  # Should be >= 5
```

### Related Self-Check Questions for Phase Review

**Requirements Analysis Phase**: Have you identified failure scenarios?
```
- What could go wrong during inference?
- What's the fallback strategy?
- What's the acceptable error rate?
```

**System Design Phase**: Have you designed for failure?
```
- Where are the single points of failure?
- What's the fallback strategy for each?
- How does system behave at 2x peak load?
```

**Implementation Phase**: Have you implemented resilience?
```
- Are all external calls wrapped in try/except?
- Do all calls have timeouts?
- Is circuit breaker pattern used?
```

**Optimization Phase**: Have you load-tested?
```
- Does system meet SLA under peak load?
- Does system gracefully degrade under 2x peak?
- Is error budget defined and tracked?
```

**Deployment Phase**: Have you documented failures?
```
- Do runbooks cover top 5 failure scenarios?
- Are on-call engineers trained on procedures?
- Is automated rollback configured?
```
```

---

## 4. MISSING EXAMPLE 3: Batch Inference at Scale

### Add as New Subsection Under "Few-Shot Examples"

```markdown
### Example 3: Batch Inference at Scale

**User Request**: "Design a batch inference pipeline to generate recommendations for 50M users daily, minimize cost (<$500/day), and maintain model freshness (< 24 hours old)"

**My Reasoning Process**:
```
Requirements Analysis:
- Scale: 50M users, daily frequency
- Volume: ~100M records (users × features per user)
- Cost constraint: <$500/day (~$15K/month)
- Freshness: Must complete by 6 AM daily
- Performance: No latency constraint (offline task)
- Availability: Can tolerate 1-2 hour failures

Architecture Decision:
- Processing: Apache Spark for distributed computing
- Scale out: 100 worker nodes processing in parallel
- Cost optimization: Use spot instances (75% cheaper)
- Storage: Partitioned Parquet for efficient access
- Serving: Redis cache for 24-hour results
- Monitoring: Job health alerts, output validation

Cost Breakdown (Target):
- Compute: 50 nodes × 8 hours × $0.30/hour = $120
- Storage: 10GB daily results × $0.02/GB = $0.20
- Infrastructure: ~$30
- Total: ~$150 < $500 target ✓

Performance Optimization:
- Partitioning: 1000 partitions (100M records / 100K per partition)
- Parallelism: 100 workers processing simultaneously
- Batching: Process 10K users per inference call
- Caching: Popular users cached for 24 hours
```

**System Architecture**:

Overview:
```
PostgreSQL (User profiles, interaction history)
    ↓
[ETL] Extract & Transform (Apache Spark)
    ├─ Feature engineering (interaction aggregates)
    ├─ Partitioning by user ID
    └─ Format: Parquet
    ↓
[Batch Inference] (Spark MLlib + PyTorch)
    ├─ Distributed model serving
    ├─ 100 parallel workers
    └─ Batch predictions (10K records at a time)
    ↓
[Results Processing] (Spark)
    ├─ Format: recommendation scores
    ├─ Validation: quality checks
    └─ Deduplicate: remove invalid results
    ↓
[Storage] (S3 + Redis)
    ├─ S3: Full results (Parquet format)
    ├─ Redis: Top results for API serving
    └─ Retention: 24 hours
    ↓
[API] (FastAPI)
    └─ Serve cached recommendations in <10ms
```

**Implementation Details**:

**1. Feature Engineering with Spark** (CPU-only, cost-effective)
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, sum, count

spark = SparkSession.builder \
    .appName("recommendation-features") \
    .config("spark.executor.instances", "100") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

# Load user interactions
interactions = spark.read.parquet("s3://data/interactions/")

# Aggregate features by time window
features = interactions.groupBy("user_id") \
    .agg(
        sum("click").alias("total_clicks"),
        count("*").alias("num_interactions"),
        # ... more aggregations
    ) \
    .coalesce(1000)  # 1000 partitions for parallelism

features.write.mode("overwrite") \
    .parquet("s3://features/batch/latest/")
```

**2. Batch Inference with Model Distribution** (GPU-optimized)
```python
import torch
from torch.utils.data import DataLoader
import numpy as np

class BatchInferenceJob:
    def __init__(self, model_path, batch_size=10_000):
        self.model = torch.load(model_path)
        self.model.eval()
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def process_partition(self, partition_id, features_df):
        """
        Process one partition of features (e.g., 100K records)
        Run in parallel across 100 workers
        """
        results = []

        # Convert DataFrame to numpy batches
        batches = self._create_batches(features_df, self.batch_size)

        for batch_idx, batch in enumerate(batches):
            try:
                # Convert to tensor
                batch_tensor = torch.from_numpy(batch).float().to(self.device)

                # Inference (batch processing for efficiency)
                with torch.no_grad():
                    predictions = self.model(batch_tensor)

                # Convert back to numpy
                predictions_np = predictions.cpu().numpy()

                # Store results with user IDs
                for user_id, pred in predictions_np:
                    results.append({
                        'user_id': int(user_id),
                        'recommendation_score': float(pred),
                        'timestamp': datetime.utcnow()
                    })

                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Partition {partition_id}: processed {batch_idx*self.batch_size} records")

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                # Continue processing (fault tolerance)
                continue

        return results

    def _create_batches(self, df, batch_size):
        """Create batches from DataFrame"""
        for i in range(0, len(df), batch_size):
            yield df[i:i+batch_size].values

# Spark batch processing
def run_batch_inference(features_parquet_path, model_path, output_path):
    # Load distributed features
    features_df = spark.read.parquet(features_parquet_path)

    # Broadcast model to all workers
    broadcast_model = spark.broadcast(model_path)

    # Process each partition in parallel
    inference_job = BatchInferenceJob(model_path)

    def process_with_broadcast(partition):
        """Run on each partition in parallel"""
        job = BatchInferenceJob(broadcast_model.value)
        return job.process_partition(partition_id, partition)

    # Map inference across all partitions
    predictions_rdd = features_df.rdd.mapPartitions(process_with_broadcast)

    # Convert results back to DataFrame
    predictions_df = spark.createDataFrame(predictions_rdd, schema=[
        ("user_id", "long"),
        ("recommendation_score", "float"),
        ("timestamp", "timestamp")
    ])

    # Validation: check for NaN, nulls
    invalid_predictions = predictions_df.filter(
        col("recommendation_score").isnan() |
        col("recommendation_score").isNull()
    ).count()

    if invalid_predictions > 0:
        logger.warning(f"Found {invalid_predictions} invalid predictions")

    # Write results
    predictions_df.write.mode("overwrite") \
        .parquet(output_path)

    return predictions_df
```

**3. Result Validation** (Data quality gates)
```python
def validate_batch_inference_results(results_df, output_path):
    """
    Validate inference results meet quality standards
    Fail job if quality checks don't pass
    """
    total_records = results_df.count()
    valid_records = results_df.filter(
        (col("recommendation_score") >= 0.0) &
        (col("recommendation_score") <= 1.0) &
        col("recommendation_score").isNotNull()
    ).count()

    # Quality metrics
    validity_rate = valid_records / total_records if total_records > 0 else 0
    coverage_rate = total_records / 50_000_000  # Expected 50M users

    logger.info(f"Validity rate: {validity_rate:.2%}")
    logger.info(f"Coverage rate: {coverage_rate:.2%}")

    # Quality gates
    if validity_rate < 0.99:
        raise DataQualityError(f"Validity {validity_rate:.2%} < 99% threshold")

    if coverage_rate < 0.95:
        raise DataQualityError(f"Coverage {coverage_rate:.2%} < 95% threshold")

    # If passed, write to production path
    results_df.write.mode("overwrite") \
        .parquet(f"{output_path}/validated/")

    logger.info("✓ Quality checks passed")
```

**4. Monitoring Job Health**
```python
import json
from datetime import datetime, timedelta

class BatchJobMonitor:
    def __init__(self, job_id, alert_channel="slack"):
        self.job_id = job_id
        self.start_time = datetime.utcnow()
        self.alert_channel = alert_channel

    def monitor(self, spark_job, expected_runtime_minutes=480):
        """Monitor batch job health"""

        # Track metrics
        metrics = {
            'job_id': self.job_id,
            'start_time': self.start_time.isoformat(),
            'status': 'RUNNING',
            'errors': 0,
            'records_processed': 0,
            'records_failed': 0
        }

        try:
            # Run with timeout
            timeout = timedelta(minutes=expected_runtime_minutes * 1.5)
            result = wait_for_job_with_timeout(spark_job, timeout)

            metrics['status'] = 'SUCCESS'
            metrics['end_time'] = datetime.utcnow().isoformat()
            metrics['duration_minutes'] = (datetime.utcnow() - self.start_time).total_seconds() / 60

            # Alert on successful completion
            self._alert(f"✓ Batch job completed: {self.job_id}")

        except TimeoutError:
            metrics['status'] = 'TIMEOUT'
            metrics['error'] = f"Job exceeded {expected_runtime_minutes}m timeout"
            self._alert(f"❌ TIMEOUT: {self.job_id}", severity="critical")
            raise

        except Exception as e:
            metrics['status'] = 'FAILED'
            metrics['error'] = str(e)
            self._alert(f"❌ FAILED: {self.job_id}\n{str(e)}", severity="critical")
            raise

        finally:
            # Log metrics
            self._log_metrics(metrics)

        return metrics

    def _alert(self, message, severity="warning"):
        """Send alert to monitoring channel"""
        if self.alert_channel == "slack":
            # Send to Slack
            requests.post(SLACK_WEBHOOK, json={
                'text': message,
                'severity': severity
            })

    def _log_metrics(self, metrics):
        """Log to metrics system"""
        logger.info(json.dumps(metrics, indent=2))
        # Also write to time-series DB
        prometheus.gauge('batch_job_duration_minutes', metrics.get('duration_minutes', 0))
        prometheus.gauge('batch_job_success', 1 if metrics['status'] == 'SUCCESS' else 0)
```

**5. Cost Analysis**
```python
# Cost breakdown for batch inference

SPOT_INSTANCE_COST = 0.30  # $/hour for compute-optimized
INSTANCE_COUNT = 100
JOB_DURATION_HOURS = 8
STORAGE_COST_PER_GB = 0.02
NETWORK_COST_PER_GB = 0.01

# Compute cost
compute_cost = SPOT_INSTANCE_COST * INSTANCE_COUNT * JOB_DURATION_HOURS
# = $0.30 * 100 * 8 = $240/day but only run 24 times/month = $200/month

# Storage cost (10GB daily results × 30 days)
storage_cost = 10 * 30 * STORAGE_COST_PER_GB
# = $6/month

# Network cost
network_cost = 100 * 30 * NETWORK_COST_PER_GB  # 100GB/day × 30 days
# = $30/month

# Total monthly
MONTHLY_COST = compute_cost + storage_cost + network_cost
# = $200 + $6 + $30 = $236/month

# Daily cost
DAILY_COST = MONTHLY_COST / 30  # = $7.87/day
# Well under $500/day budget ✓
```

**Performance Characteristics**:
- **Total Runtime**: 8 hours for 100M records
- **Throughput**: 3.4M predictions/hour
- **Cost**: ~$8/day (~$240/month)
- **Model Freshness**: Daily (< 24 hours old)
- **Reliability**: 99.9% record completion rate

**Operational Runbook**:
- **Scheduling**: Daily at 10 PM UTC, complete by 6 AM
- **Monitoring**: CloudWatch alerts for job failures
- **Troubleshooting**:
  - Timeout: Increase worker count from 100 to 150
  - Quality gate failure: Check model for NaN outputs
  - Cost spike: Check for non-spot instance usage
- **Fallback**: Serve last known good results from S3 if current day fails

**Why This Works**:
- Spark distributes 100M records across 100 workers (1M each)
- Spot instances reduce compute cost by 75%
- Batch processing (10K records at a time) maximizes GPU utilization
- Quality gates catch data issues before serving
- Monitoring catches failures automatically
- Serves results from Redis cache (<10ms latency for users)
```

---

## 5. MISSING EXAMPLE 4: Monitoring & Drift Detection

### Add as New Subsection Under "Few-Shot Examples"

```markdown
### Example 4: Model Monitoring & Drift Detection

**User Request**: "Set up monitoring to detect when fraud detection model degrades below 95% recall in production. Alert within 1 hour and trigger retraining"

[Detailed implementation with code examples]
```

---

## 6. Summary: Implementation Checklist

### Phase 1: Critical Additions (Week 1)
- [ ] Add "When to Use/DO NOT USE" section (1-2 hours)
  - [ ] Copy template from section 1 above
  - [ ] Customize for ml-engineer scope
  - [ ] Add decision tree
  - [ ] Test with sample scenarios

- [ ] Add Skill Invocation Decision Tree (1-2 hours)
  - [ ] Copy template from section 2 above
  - [ ] Verify coordination patterns
  - [ ] Test with multi-agent scenarios

### Phase 2: Enhancements (Week 2)
- [ ] Add Constitutional AI Checkpoints (1-2 hours)
  - [ ] Enhanced Reliability principle (use section 3 template)
  - [ ] Add 2-3 more principles with checkpoints
  - [ ] Add anti-patterns library
  - [ ] Add verification checklists

- [ ] Add Missing Technologies (2-3 hours)
  - [ ] LLM Inference section
  - [ ] Expand Edge Deployment
  - [ ] Add Streaming ML
  - [ ] Add Observability as Code

### Phase 3: Examples (Week 2-3)
- [ ] Add Batch Inference Example (2-3 hours)
  - [ ] Use template from section 4
  - [ ] Test code snippets
  - [ ] Customize for your domain

- [ ] Add Monitoring/Drift Example (2-3 hours)
  - [ ] Create similar to section 5 template

- [ ] Add 1-2 More Examples (2-3 hours)
  - [ ] Cost optimization
  - [ ] Multi-model ensemble
  - [ ] Edge deployment with updates

### Final Quality Checks
- [ ] Length reasonable (800-950 lines, up from 581)
- [ ] Consistency with data-engineer and mlops-engineer agents
- [ ] Examples have runnable code
- [ ] Decision trees validated
- [ ] Anti-patterns reviewed

---

**Expected health score after implementation**: 85-92/100 (up from 64/100)
