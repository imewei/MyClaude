# DevOps and ML Infrastructure

Expert guidance on CI/CD pipelines, infrastructure automation, and deployment orchestration for ML systems. Use when building automated training pipelines, setting up ML infrastructure as code, or implementing deployment workflows with GitHub Actions, Terraform, and Kubernetes.

## Overview

This skill covers comprehensive DevOps practices for ML systems, from automated CI/CD pipelines to infrastructure provisioning and deployment optimization.

## Core Topics

### 1. CI/CD Pipelines for ML

#### GitHub Actions for ML Workflows

**Complete ML Pipeline with GitHub Actions**
```.github/workflows/ml-pipeline.yml
name: ML Training and Deployment Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'data/**'
      - 'models/**'
  pull_request:
    branches: [main]
  schedule:
    # Run training weekly
    - cron: '0 0 * * 0'
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: false

env:
  PYTHON_VERSION: '3.12'
  AWS_REGION: us-east-1
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install black flake8 mypy pytest pytest-cov

      - name: Code formatting check
        run: black --check src/

      - name: Lint code
        run: flake8 src/ --max-line-length=100

      - name: Type checking
        run: mypy src/ --ignore-missing-imports

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml --cov-report=term

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  data-validation:
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install great-expectations pandas

      - name: Validate data quality
        run: |
          python scripts/validate_data.py \
            --data-path data/train.parquet \
            --expectations-suite expectations/train_data.json

      - name: Check data drift
        run: |
          python scripts/check_data_drift.py \
            --reference-data data/reference.parquet \
            --current-data data/train.parquet \
            --threshold 0.05

  train-model:
    runs-on: ubuntu-latest
    needs: [code-quality, data-validation]
    strategy:
      matrix:
        model: [baseline, advanced]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Download training data from S3
        run: |
          aws s3 cp s3://ml-bucket/data/train.parquet data/train.parquet
          aws s3 cp s3://ml-bucket/data/val.parquet data/val.parquet

      - name: Train model
        run: |
          python scripts/train.py \
            --config configs/${{ matrix.model }}.yaml \
            --experiment-name github-actions-${{ github.run_id }} \
            --output-dir models/${{ matrix.model }}

      - name: Evaluate model
        id: evaluate
        run: |
          python scripts/evaluate.py \
            --model-path models/${{ matrix.model }}/model.pth \
            --test-data data/val.parquet \
            --output metrics.json

      - name: Check performance threshold
        run: |
          python scripts/check_performance.py \
            --metrics metrics.json \
            --min-accuracy 0.85 \
            --min-f1 0.80

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: model-${{ matrix.model }}
          path: models/${{ matrix.model }}/

      - name: Register model in MLflow
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/register_model.py \
            --model-path models/${{ matrix.model }} \
            --model-name ${{ matrix.model }}-model \
            --run-id ${{ github.run_id }}

  model-testing:
    runs-on: ubuntu-latest
    needs: train-model
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifacts
        uses: actions/download-artifact@v3
        with:
          name: model-baseline

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt pytest

      - name: Test model inference
        run: |
          pytest tests/test_model_inference.py -v

      - name: Test model robustness
        run: |
          python tests/test_adversarial.py \
            --model-path model.pth

      - name: Benchmark latency
        run: |
          python tests/benchmark_latency.py \
            --model-path model.pth \
            --batch-sizes 1,8,32 \
            --num-iterations 100

  build-container:
    runs-on: ubuntu-latest
    needs: model-testing
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifact
        uses: actions/download-artifact@v3
        with:
          name: model-baseline
          path: models/

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Amazon ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            ${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }}
            ${{ secrets.ECR_REGISTRY }}/ml-model:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image for vulnerabilities
        run: |
          docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image \
            --severity HIGH,CRITICAL \
            ${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }}

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-container
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Update kubeconfig
        run: |
          aws eks update-kubeconfig \
            --name ml-cluster-staging \
            --region ${{ env.AWS_REGION }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-model \
            ml-model=${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }} \
            -n staging

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ml-model -n staging --timeout=5m

      - name: Smoke test
        run: |
          python tests/smoke_test.py \
            --endpoint https://ml-staging.example.com

  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Update kubeconfig
        run: |
          aws eks update-kubeconfig \
            --name ml-cluster-prod \
            --region ${{ env.AWS_REGION }}

      - name: Canary deployment (10%)
        run: |
          kubectl set image deployment/ml-model-canary \
            ml-model=${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }} \
            -n production

      - name: Monitor canary metrics
        run: |
          python scripts/monitor_canary.py \
            --duration 300 \
            --error-threshold 0.01 \
            --latency-threshold-p99 500

      - name: Full production deployment
        run: |
          kubectl set image deployment/ml-model \
            ml-model=${{ secrets.ECR_REGISTRY }}/ml-model:${{ github.sha }} \
            -n production

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/ml-model -n production --timeout=10m

      - name: Create GitHub release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ github.run_number }}
          release_name: Model Release v${{ github.run_number }}
          body: |
            Automated model deployment
            - Commit: ${{ github.sha }}
            - Run ID: ${{ github.run_id }}
```

**Data Validation Script**
```python
# scripts/validate_data.py
import argparse
import pandas as pd
import great_expectations as gx
from pathlib import Path
import sys

def validate_data(data_path: str, expectations_suite: str) -> bool:
    """Validate data using Great Expectations."""

    # Load data
    df = pd.read_parquet(data_path)

    # Create data context
    context = gx.get_context()

    # Create or load expectation suite
    suite = context.get_expectation_suite(expectations_suite)

    # Create batch
    batch = context.sources.pandas_default.read_dataframe(df)

    # Validate
    results = batch.validate(suite)

    # Check results
    if not results.success:
        print("Data validation failed:")
        for result in results.results:
            if not result.success:
                print(f"  - {result.expectation_config.expectation_type}")
                print(f"    {result.result}")
        return False

    print("✓ Data validation passed")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--expectations-suite", required=True)
    args = parser.parse_args()

    success = validate_data(args.data_path, args.expectations_suite)
    sys.exit(0 if success else 1)
```

**Performance Threshold Check**
```python
# scripts/check_performance.py
import argparse
import json
import sys

def check_performance(
    metrics_path: str,
    min_accuracy: float,
    min_f1: float
) -> bool:
    """Check if model performance meets thresholds."""

    with open(metrics_path) as f:
        metrics = json.load(f)

    passed = True

    # Check accuracy
    if metrics['accuracy'] < min_accuracy:
        print(f"❌ Accuracy {metrics['accuracy']:.4f} below threshold {min_accuracy}")
        passed = False
    else:
        print(f"✓ Accuracy {metrics['accuracy']:.4f} meets threshold")

    # Check F1
    if metrics['f1'] < min_f1:
        print(f"❌ F1 {metrics['f1']:.4f} below threshold {min_f1}")
        passed = False
    else:
        print(f"✓ F1 {metrics['f1']:.4f} meets threshold")

    return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--min-f1", type=float, required=True)
    args = parser.parse_args()

    passed = check_performance(args.metrics, args.min_accuracy, args.min_f1)
    sys.exit(0 if passed else 1)
```

### 2. Infrastructure as Code with Terraform

#### Complete ML Infrastructure on AWS

**Main Terraform Configuration**
```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }

  backend "s3" {
    bucket         = "ml-terraform-state"
    key            = "ml-infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "ml-platform"
      ManagedBy   = "terraform"
    }
  }
}

# VPC for ML workloads
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "5.1.2"

  name = "ml-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Enable VPC endpoints for cost savings
  enable_s3_endpoint       = true
  enable_ecr_api_endpoint  = true
  enable_ecr_dkr_endpoint  = true
}

# EKS Cluster for ML workloads
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "19.16.0"

  cluster_name    = "ml-cluster-${var.environment}"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Enable cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Node groups
  eks_managed_node_groups = {
    # CPU node group for general workloads
    cpu_nodes = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      instance_types = ["m5.2xlarge"]
      capacity_type  = "ON_DEMAND"

      labels = {
        workload = "cpu"
      }

      taints = []
    }

    # GPU node group for training
    gpu_nodes = {
      min_size     = 0
      max_size     = 5
      desired_size = 0

      instance_types = ["p3.2xlarge"]
      capacity_type  = "SPOT"

      labels = {
        workload = "gpu"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NoSchedule"
      }]
    }
  }

  # Cluster security group rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }
}

# S3 bucket for ML artifacts
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "ml-artifacts-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  rule {
    id     = "archive_old_models"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# ECR repository for ML models
resource "aws_ecr_repository" "ml_models" {
  name                 = "ml-models-${var.environment}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "ml_models" {
  repository = aws_ecr_repository.ml_models.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# RDS for MLflow tracking
module "mlflow_db" {
  source = "terraform-aws-modules/rds/aws"
  version = "6.1.1"

  identifier = "mlflow-${var.environment}"

  engine               = "postgres"
  engine_version       = "15.3"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 500

  db_name  = "mlflow"
  username = var.mlflow_db_username
  password = var.mlflow_db_password
  port     = 5432

  vpc_security_group_ids = [aws_security_group.mlflow_db.id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"
}

# SageMaker execution role
resource "aws_iam_role" "sagemaker_execution" {
  name = "sagemaker-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# CloudWatch log group for ML pipelines
resource "aws_cloudwatch_log_group" "ml_pipelines" {
  name              = "/aws/ml-pipelines/${var.environment}"
  retention_in_days = var.log_retention_days
}

# Outputs
output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.ml_models.repository_url
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.id
}

output "mlflow_db_endpoint" {
  description = "MLflow database endpoint"
  value       = module.mlflow_db.db_instance_endpoint
}

data "aws_caller_identity" "current" {}
```

**Variables**
```hcl
# terraform/variables.tf
variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production"
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "mlflow_db_username" {
  description = "MLflow database username"
  type        = string
  sensitive   = true
}

variable "mlflow_db_password" {
  description = "MLflow database password"
  type        = string
  sensitive   = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}
```

**Deployment Commands**
```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan changes
terraform plan -var-file="environments/production.tfvars"

# Apply changes
terraform apply -var-file="environments/production.tfvars"

# Destroy infrastructure (use with caution!)
terraform destroy -var-file="environments/production.tfvars"
```

### 3. Model Deployment Optimization

#### Inference Optimization Strategies

**Optimized FastAPI Service**
```python
# src/optimized_serving.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import Response
import torch
import torch.nn as nn
from typing import List, Dict, Any
import asyncio
from collections import deque
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time: float = 0.01  # 10ms

class BatchProcessor:
    """Dynamic batching for improved throughput."""

    def __init__(self, model: nn.Module, config: BatchConfig):
        self.model = model
        self.config = config
        self.queue: deque = deque()
        self.lock = asyncio.Lock()
        self.processing = False

    async def add_request(self, data: torch.Tensor) -> torch.Tensor:
        """Add request to batch queue."""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append((data, future))

            # Start processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests as batch."""
        self.processing = True

        # Wait for batch to accumulate
        await asyncio.sleep(self.config.max_wait_time)

        async with self.lock:
            if not self.queue:
                self.processing = False
                return

            # Extract batch
            batch_size = min(len(self.queue), self.config.max_batch_size)
            batch_items = [self.queue.popleft() for _ in range(batch_size)]

        # Process batch
        try:
            inputs = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]

            # Stack inputs
            batch_tensor = torch.stack(inputs)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)

            # Distribute results
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result(outputs[i])

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for _, future in batch_items:
                if not future.done():
                    future.set_exception(e)

        finally:
            self.processing = False

            # Process remaining items if any
            if self.queue:
                asyncio.create_task(self._process_batch())

class ModelCache:
    """LRU cache for model predictions."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Any] = {}
        self.access_order: deque = deque()
        self.max_size = max_size

    def get(self, key: str) -> Any:
        """Get cached prediction."""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Cache prediction."""
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.popleft()
            del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

# Initialize optimized service
app = FastAPI(title="Optimized ML Service")

model = YourModel()
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Compile model for PyTorch 2.x speedup
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")

# Initialize batch processor and cache
batch_processor = BatchProcessor(model, BatchConfig())
prediction_cache = ModelCache()

# Thread pool for CPU-bound preprocessing
executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict/optimized")
async def predict_optimized(features: List[float]):
    """Optimized prediction endpoint with batching and caching."""

    # Generate cache key
    cache_key = str(hash(tuple(features)))

    # Check cache
    cached_result = prediction_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    # Prepare input tensor
    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # Add to batch processor
    output = await batch_processor.add_request(input_tensor)

    # Format result
    result = {
        'prediction': float(output.item()),
        'cached': False
    }

    # Cache result
    prediction_cache.put(cache_key, result)

    return result

@app.get("/metrics/performance")
async def performance_metrics():
    """Get performance metrics."""
    return {
        'cache_size': len(prediction_cache.cache),
        'queue_size': len(batch_processor.queue),
        'processing': batch_processor.processing
    }
```

**Load Testing Script**
```python
# tests/load_test.py
import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    features: List[float]
) -> Dict:
    """Send single prediction request."""
    start = time.time()

    async with session.post(url, json={'features': features}) as response:
        result = await response.json()
        latency = time.time() - start

    return {
        'latency': latency,
        'status': response.status,
        'result': result
    }

async def load_test(
    url: str,
    num_requests: int,
    concurrency: int
) -> None:
    """Run load test."""

    # Generate test data
    test_features = [
        np.random.randn(10).tolist()
        for _ in range(num_requests)
    ]

    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    ) as session:

        # Send requests concurrently
        start_time = time.time()

        tasks = [
            send_request(session, url, features)
            for features in test_features
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

    # Analyze results
    successful = [r for r in results if isinstance(r, dict) and r['status'] == 200]
    failed = len(results) - len(successful)

    latencies = [r['latency'] for r in successful]

    print(f"\nLoad Test Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {num_requests / total_time:.2f} req/s")
    print(f"\nLatency Statistics:")
    print(f"  Min: {np.min(latencies)*1000:.2f}ms")
    print(f"  Mean: {np.mean(latencies)*1000:.2f}ms")
    print(f"  Median: {np.median(latencies)*1000:.2f}ms")
    print(f"  P95: {np.percentile(latencies, 95)*1000:.2f}ms")
    print(f"  P99: {np.percentile(latencies, 99)*1000:.2f}ms")
    print(f"  Max: {np.max(latencies)*1000:.2f}ms")

if __name__ == "__main__":
    asyncio.run(load_test(
        url="http://localhost:8000/predict/optimized",
        num_requests=1000,
        concurrency=50
    ))
```

## Best Practices Summary

### CI/CD for ML
1. Automate data validation before training
2. Check model performance thresholds in pipeline
3. Use matrix builds for multiple model variants
4. Implement canary deployments for production
5. Maintain separate staging and production environments

### Infrastructure as Code
1. Use modules for reusable infrastructure components
2. Enable state locking with DynamoDB
3. Use workspaces for multi-environment management
4. Tag all resources for cost tracking
5. Implement lifecycle policies for cost optimization

### Deployment Optimization
1. Use dynamic batching for improved throughput
2. Implement prediction caching for repeated requests
3. Compile models with PyTorch 2.x torch.compile
4. Use connection pooling for database and APIs
5. Profile and optimize bottlenecks

### Monitoring
1. Track deployment success rates
2. Monitor rollout progress and health checks
3. Implement automated rollback on failures
4. Set up alerts for performance degradation
5. Track cost metrics alongside performance

## Quick Reference

### GitHub Actions Commands
```bash
# Manually trigger workflow
gh workflow run ml-pipeline.yml

# View workflow runs
gh run list --workflow=ml-pipeline.yml

# View run logs
gh run view <run-id> --log

# Cancel run
gh run cancel <run-id>
```

### Terraform Commands
```bash
# Format code
terraform fmt -recursive

# Validate configuration
terraform validate

# Show planned changes
terraform plan

# Apply with auto-approve (use carefully)
terraform apply -auto-approve

# Refresh state
terraform refresh

# Import existing resource
terraform import aws_s3_bucket.ml_artifacts bucket-name
```

### Performance Optimization
```python
# Benchmark model inference
python tests/benchmark_latency.py \
  --model-path model.pth \
  --batch-sizes 1,8,16,32 \
  --device cuda

# Run load test
python tests/load_test.py \
  --url http://localhost:8000 \
  --num-requests 1000 \
  --concurrency 50
```

## When to Use This Skill

Use this skill when you need to:
- Set up CI/CD pipelines for ML with GitHub Actions
- Provision ML infrastructure with Terraform
- Optimize model serving for throughput and latency
- Implement automated training and deployment
- Build data validation pipelines
- Set up multi-environment deployments
- Implement canary and blue-green deployments
- Optimize inference with batching and caching
- Load test ML services
- Automate infrastructure provisioning

This skill provides complete DevOps and infrastructure automation for production ML systems.
