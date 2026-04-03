---
name: cloud-provider-patterns
description: Design cloud-native architectures across AWS, GCP, and Azure including serverless (Lambda/Cloud Functions), managed services, IaC (Terraform/Pulumi), cost optimization, and multi-cloud strategies. Use when provisioning cloud resources, designing serverless functions, or writing Terraform/Pulumi configurations.
---

# Cloud Provider Patterns

## Expert Agent

For cloud architecture, infrastructure as code, and managed service selection, delegate to:

- **`devops-architect`**: Designs cloud-native platform architecture with IaC, serverless, and multi-cloud strategies.
  - *Location*: `plugins/dev-suite/agents/devops-architect.md`


## Service Comparison

| Capability | AWS | GCP | Azure |
|------------|-----|-----|-------|
| Compute (serverless) | Lambda | Cloud Functions | Azure Functions |
| Container orchestration | EKS / Fargate | GKE / Cloud Run | AKS / Container Apps |
| Object storage | S3 | Cloud Storage | Blob Storage |
| Relational DB | RDS / Aurora | Cloud SQL / AlloyDB | Azure SQL / Cosmos DB |
| NoSQL | DynamoDB | Firestore / Bigtable | Cosmos DB |
| Message queue | SQS / SNS | Pub/Sub | Service Bus |
| Cache | ElastiCache | Memorystore | Azure Cache for Redis |
| CDN | CloudFront | Cloud CDN | Azure CDN / Front Door |
| IaC native | CloudFormation | Deployment Manager | ARM / Bicep |


## Serverless Patterns

### AWS Lambda (Python)

```python
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """Process API Gateway event."""
    try:
        body = json.loads(event.get("body", "{}"))
        result = process_request(body)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return {"statusCode": 400, "body": json.dumps({"error": str(exc)})}
    except Exception as exc:
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": "Internal error"})}

def process_request(body: dict) -> dict:
    return {"message": "processed", "input": body}
```

### Serverless Design Rules

- Keep functions small and single-purpose
- Externalize state to managed services (DynamoDB, S3)
- Set memory/timeout based on profiling, not guessing
- Use provisioned concurrency for latency-sensitive paths
- Implement idempotency for retried invocations


## Infrastructure as Code (Terraform)

### Module Structure

```
infra/
  main.tf
  variables.tf
  outputs.tf
  providers.tf
  modules/
    networking/
    compute/
    database/
    monitoring/
```

### VPC + Lambda Example

```hcl
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  tags = { Name = "${var.project}-vpc" }
}

resource "aws_lambda_function" "api" {
  function_name = "${var.project}-api"
  runtime       = "python3.12"
  handler       = "handler.handler"
  filename      = data.archive_file.lambda_zip.output_path
  role          = aws_iam_role.lambda_exec.arn
  memory_size   = 256
  timeout       = 30

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.main.name
    }
  }

  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}
```

### Terraform Best Practices

- [ ] Use remote state (S3 + DynamoDB lock)
- [ ] Pin provider versions
- [ ] Use modules for reusable components
- [ ] Tag all resources with project, environment, owner
- [ ] Use `terraform plan` in CI before apply
- [ ] Store secrets in Vault or AWS Secrets Manager, not in tfvars


## IAM Patterns

### Least Privilege Template

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/my-table"
    }
  ]
}
```

## Cost Optimization

| Strategy | Savings | Effort |
|----------|---------|--------|
| Right-size instances | 20-40% | Low |
| Reserved/Savings Plans | 30-60% | Medium |
| Spot/Preemptible instances | 60-90% | Medium |
| Auto-scaling | Variable | Medium |
| S3 lifecycle policies | 30-50% on storage | Low |
| Delete unused resources | Immediate | Low |

### Cost Monitoring

- Enable billing alerts at 50%, 80%, 100% of budget
- Use AWS Cost Explorer / GCP Billing / Azure Cost Management
- Tag resources for per-team cost allocation
- Review monthly for unused or oversized resources


## Architecture Checklist

- [ ] Services selected based on workload fit, not habit
- [ ] IaC manages all infrastructure (no manual console changes)
- [ ] IAM follows least privilege
- [ ] Multi-AZ or multi-region for critical services
- [ ] Cost alerts and tagging in place
- [ ] Secrets stored in managed secret service
- [ ] Backup and disaster recovery plan documented
