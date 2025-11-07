# Terraform CI/CD Integration

**Version**: 1.0.3
**Command**: `/workflow-automate`
**Category**: CI/CD Automation

## Overview

Comprehensive Terraform CI/CD integration patterns covering remote state management, PR plan previews, cost estimation, policy validation, drift detection, and multi-environment deployments.

---

## Table of Contents

1. [GitHub Actions Terraform Workflow](#1-github-actions-terraform-workflow)
2. [GitLab CI Terraform Pipeline](#2-gitlab-ci-terraform-pipeline)
3. [Multi-Environment Deployment](#3-multi-environment-deployment)

---

## 1. GitHub Actions Terraform Workflow

Complete Terraform workflow for GitHub Actions with plan comments, cost estimation, and automated apply.

### `.github/workflows/terraform.yml`

```yaml
name: Terraform CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'terraform/**'
      - '.github/workflows/terraform.yml'
  pull_request:
    branches: [main]
    paths:
      - 'terraform/**'
  workflow_dispatch:

env:
  TF_VERSION: '1.6.6'
  TF_WORKING_DIR: './terraform'
  AWS_REGION: 'us-east-1'

permissions:
  contents: read
  pull-requests: write
  id-token: write  # For OIDC authentication

jobs:
  # ===== FORMAT CHECK =====
  fmt:
    name: Terraform Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Format Check
        id: fmt
        run: terraform fmt -check -recursive
        working-directory: ${{ env.TF_WORKING_DIR }}
        continue-on-error: true

      - name: Comment Format Errors
        if: steps.fmt.outcome == 'failure' && github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '⚠️ **Terraform Format Check Failed**\n\nRun `terraform fmt -recursive` to fix formatting issues.'
            })

  # ===== VALIDATE =====
  validate:
    name: Terraform Validate
    runs-on: ubuntu-latest
    needs: fmt
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Terraform Init
        run: terraform init -backend=false
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Terraform Validate
        run: terraform validate
        working-directory: ${{ env.TF_WORKING_DIR }}

  # ===== SECURITY SCAN =====
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: validate
    steps:
      - uses: actions/checkout@v4

      - name: tfsec
        uses: aquasecurity/tfsec-action@v1
        with:
          working_directory: ${{ env.TF_WORKING_DIR }}
          soft_fail: true

      - name: Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: ${{ env.TF_WORKING_DIR }}
          framework: terraform
          output_format: sarif
          output_file_path: checkov-report.sarif
          soft_fail: true

      - name: Upload Checkov Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: checkov-report.sarif

  # ===== PLAN =====
  plan:
    name: Terraform Plan
    runs-on: ubuntu-latest
    needs: [validate, security]
    outputs:
      plan-exitcode: ${{ steps.plan.outputs.exitcode }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="bucket=${{ secrets.TF_STATE_BUCKET }}" \
            -backend-config="key=prod/terraform.tfstate" \
            -backend-config="region=${{ env.AWS_REGION }}" \
            -backend-config="dynamodb_table=${{ secrets.TF_LOCK_TABLE }}"
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Terraform Plan
        id: plan
        run: |
          terraform plan \
            -detailed-exitcode \
            -out=tfplan \
            -no-color | tee plan-output.txt
        working-directory: ${{ env.TF_WORKING_DIR }}
        continue-on-error: true

      - name: Upload Plan
        uses: actions/upload-artifact@v4
        with:
          name: tfplan
          path: ${{ env.TF_WORKING_DIR }}/tfplan
          retention-days: 7

      - name: Comment Plan on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const planOutput = fs.readFileSync('${{ env.TF_WORKING_DIR }}/plan-output.txt', 'utf8');

            // Truncate if too long
            const maxLength = 65000;
            const truncatedPlan = planOutput.length > maxLength
              ? planOutput.substring(0, maxLength) + '\n\n...(truncated)'
              : planOutput;

            const output = `#### Terraform Plan 📖
            <details><summary>Show Plan</summary>

            \`\`\`terraform
            ${truncatedPlan}
            \`\`\`

            </details>

            *Pusher: @${{ github.actor }}, Action: \`${{ github.event_name }}\`, Workflow: \`${{ github.workflow }}\`*`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: output
            });

  # ===== COST ESTIMATION =====
  cost:
    name: Cost Estimation
    runs-on: ubuntu-latest
    needs: plan
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="bucket=${{ secrets.TF_STATE_BUCKET }}" \
            -backend-config="key=prod/terraform.tfstate" \
            -backend-config="region=${{ env.AWS_REGION }}"
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Setup Infracost
        uses: infracost/actions/setup@v2
        with:
          api-key: ${{ secrets.INFRACOST_API_KEY }}

      - name: Generate Infracost Estimate
        run: |
          infracost breakdown --path ${{ env.TF_WORKING_DIR }} \
            --format json \
            --out-file /tmp/infracost.json

      - name: Post Infracost Comment
        run: |
          infracost comment github \
            --path /tmp/infracost.json \
            --repo $GITHUB_REPOSITORY \
            --github-token ${{ secrets.GITHUB_TOKEN }} \
            --pull-request ${{ github.event.pull_request.number }} \
            --behavior update

  # ===== POLICY VALIDATION =====
  policy:
    name: Policy Validation
    runs-on: ubuntu-latest
    needs: plan
    steps:
      - uses: actions/checkout@v4

      - name: Download Plan
        uses: actions/download-artifact@v4
        with:
          name: tfplan
          path: ${{ env.TF_WORKING_DIR }}

      - name: Setup OPA
        uses: open-policy-agent/setup-opa@v2

      - name: Convert Plan to JSON
        run: |
          terraform show -json tfplan > tfplan.json
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Run OPA Tests
        run: |
          opa test policies/ -v
        continue-on-error: true

      - name: Evaluate Policies
        run: |
          opa eval \
            --data policies/ \
            --input ${{ env.TF_WORKING_DIR }}/tfplan.json \
            --format pretty \
            "data.terraform.deny"

  # ===== APPLY =====
  apply:
    name: Terraform Apply
    runs-on: ubuntu-latest
    needs: [plan, policy]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment:
      name: production
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="bucket=${{ secrets.TF_STATE_BUCKET }}" \
            -backend-config="key=prod/terraform.tfstate" \
            -backend-config="region=${{ env.AWS_REGION }}" \
            -backend-config="dynamodb_table=${{ secrets.TF_LOCK_TABLE }}"
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Download Plan
        uses: actions/download-artifact@v4
        with:
          name: tfplan
          path: ${{ env.TF_WORKING_DIR }}

      - name: Terraform Apply
        run: terraform apply -auto-approve tfplan
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Output Summary
        run: terraform output -json > outputs.json
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Upload Outputs
        uses: actions/upload-artifact@v4
        with:
          name: terraform-outputs
          path: ${{ env.TF_WORKING_DIR }}/outputs.json
```

**Key Features**:
- ✅ OIDC authentication (no static credentials)
- ✅ Automated PR plan comments
- ✅ Infracost for cost estimation
- ✅ tfsec and Checkov security scanning
- ✅ OPA policy validation
- ✅ Remote state with locking (S3 + DynamoDB)
- ✅ Production environment gate

---

## 2. GitLab CI Terraform Pipeline

Complete GitLab CI pipeline for Terraform with caching, plan artifacts, and MR comments.

### `.gitlab-ci.yml`

```yaml
# GitLab CI/CD for Terraform

variables:
  TF_VERSION: "1.6.6"
  TF_ROOT: "${CI_PROJECT_DIR}/terraform"
  TF_STATE_BUCKET: "my-terraform-state"
  TF_LOCK_TABLE: "terraform-locks"
  AWS_REGION: "us-east-1"

image:
  name: hashicorp/terraform:${TF_VERSION}
  entrypoint: [""]

cache:
  key: terraform-${CI_COMMIT_REF_SLUG}
  paths:
    - ${TF_ROOT}/.terraform
    - ${TF_ROOT}/.terraform.lock.hcl

stages:
  - validate
  - plan
  - apply

before_script:
  - cd ${TF_ROOT}
  - terraform --version
  - |
    terraform init \
      -backend-config="bucket=${TF_STATE_BUCKET}" \
      -backend-config="key=${CI_ENVIRONMENT_NAME}/terraform.tfstate" \
      -backend-config="region=${AWS_REGION}" \
      -backend-config="dynamodb_table=${TF_LOCK_TABLE}"

# ===== VALIDATE =====
fmt:
  stage: validate
  script:
    - terraform fmt -check -recursive
  allow_failure: false

validate:
  stage: validate
  script:
    - terraform validate
  artifacts:
    paths:
      - ${TF_ROOT}/.terraform.lock.hcl
    expire_in: 1 day

security-scan:
  stage: validate
  image: aquasec/tfsec:latest
  script:
    - tfsec ${TF_ROOT} --format json --out tfsec-report.json
  artifacts:
    reports:
      sast: tfsec-report.json
  allow_failure: true

# ===== PLAN =====
plan:staging:
  stage: plan
  variables:
    CI_ENVIRONMENT_NAME: staging
  script:
    # Create plan
    - terraform plan -out=tfplan -input=false

    # Convert to JSON for analysis
    - terraform show -json tfplan > tfplan.json

    # Generate human-readable plan
    - terraform show -no-color tfplan > plan-output.txt

    # Create MR comment
    - |
      if [ -n "$CI_MERGE_REQUEST_IID" ]; then
        PLAN_OUTPUT=$(cat plan-output.txt | head -c 60000)
        COMMENT="## Terraform Plan - Staging\n\n\`\`\`terraform\n${PLAN_OUTPUT}\n\`\`\`\n\n---\n*Pusher: @${GITLAB_USER_LOGIN}*"

        curl --request POST \
          --header "PRIVATE-TOKEN: ${CI_JOB_TOKEN}" \
          --data-urlencode "body=${COMMENT}" \
          "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/merge_requests/${CI_MERGE_REQUEST_IID}/notes"
      fi
  artifacts:
    paths:
      - ${TF_ROOT}/tfplan
      - ${TF_ROOT}/tfplan.json
      - ${TF_ROOT}/plan-output.txt
    expire_in: 1 week
    reports:
      terraform: ${TF_ROOT}/tfplan.json
  environment:
    name: staging
    action: prepare
  only:
    - merge_requests
    - develop

plan:production:
  stage: plan
  variables:
    CI_ENVIRONMENT_NAME: production
  script:
    - terraform plan -out=tfplan -input=false
    - terraform show -json tfplan > tfplan.json
    - terraform show -no-color tfplan > plan-output.txt
  artifacts:
    paths:
      - ${TF_ROOT}/tfplan
      - ${TF_ROOT}/tfplan.json
    expire_in: 1 week
    reports:
      terraform: ${TF_ROOT}/tfplan.json
  environment:
    name: production
    action: prepare
  only:
    - main

# ===== COST ESTIMATION =====
cost-estimate:
  stage: plan
  image: infracost/infracost:latest
  variables:
    INFRACOST_API_KEY: ${INFRACOST_API_KEY}
  script:
    - |
      infracost breakdown \
        --path ${TF_ROOT} \
        --format json \
        --out-file /tmp/infracost.json

    - |
      infracost comment gitlab \
        --path /tmp/infracost.json \
        --repo $CI_PROJECT_PATH \
        --merge-request $CI_MERGE_REQUEST_IID \
        --gitlab-token $GITLAB_TOKEN \
        --behavior update
  only:
    - merge_requests

# ===== APPLY =====
apply:staging:
  stage: apply
  variables:
    CI_ENVIRONMENT_NAME: staging
  dependencies:
    - plan:staging
  script:
    - terraform apply -auto-approve tfplan
    - terraform output -json > outputs.json
  artifacts:
    paths:
      - ${TF_ROOT}/outputs.json
    expire_in: 30 days
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop
  when: on_success

apply:production:
  stage: apply
  variables:
    CI_ENVIRONMENT_NAME: production
  dependencies:
    - plan:production
  script:
    - terraform apply -auto-approve tfplan
    - terraform output -json > outputs.json
  artifacts:
    paths:
      - ${TF_ROOT}/outputs.json
    expire_in: 30 days
  environment:
    name: production
    url: https://example.com
  only:
    - main
  when: manual
```

**Key Features**:
- ✅ Environment-specific plans (staging/production)
- ✅ Automated MR plan comments
- ✅ Infracost integration
- ✅ tfsec security scanning
- ✅ Terraform plan artifacts in MR UI
- ✅ Manual approval for production

---

## 3. Multi-Environment Deployment

Advanced multi-environment workflow with workspace management and drift detection.

### `.github/workflows/terraform-multi-env.yml`

```yaml
name: Terraform Multi-Environment

on:
  push:
    branches: [main, develop, staging]
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Daily drift detection
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
          - dev
          - staging
          - production

env:
  TF_VERSION: '1.6.6'

jobs:
  # ===== DETERMINE ENVIRONMENT =====
  setup:
    name: Setup Environment
    runs-on: ubuntu-latest
    outputs:
      environment: ${{ steps.env.outputs.environment }}
      workspace: ${{ steps.env.outputs.workspace }}
    steps:
      - id: env
        run: |
          if [ "${{ github.event_name }}" == "workflow_dispatch" ]; then
            echo "environment=${{ inputs.environment }}" >> $GITHUB_OUTPUT
          elif [ "${{ github.ref }}" == "refs/heads/main" ]; then
            echo "environment=production" >> $GITHUB_OUTPUT
          elif [ "${{ github.ref }}" == "refs/heads/staging" ]; then
            echo "environment=staging" >> $GITHUB_OUTPUT
          else
            echo "environment=dev" >> $GITHUB_OUTPUT
          fi

  # ===== DRIFT DETECTION =====
  drift-detection:
    name: Drift Detection
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    strategy:
      matrix:
        environment: [dev, staging, production]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: us-east-1

      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="key=${{ matrix.environment }}/terraform.tfstate"
        working-directory: ./terraform

      - name: Select Workspace
        run: |
          terraform workspace select ${{ matrix.environment }} || \
          terraform workspace new ${{ matrix.environment }}
        working-directory: ./terraform

      - name: Detect Drift
        id: plan
        run: |
          terraform plan -detailed-exitcode -no-color | tee drift-output.txt
        working-directory: ./terraform
        continue-on-error: true

      - name: Report Drift
        if: steps.plan.outputs.exitcode == '2'
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "⚠️ Terraform Drift Detected in ${{ matrix.environment }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Terraform Drift Alert*\n*Environment:* ${{ matrix.environment }}\n*Status:* Configuration drift detected\n*Action:* Review and reconcile infrastructure"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  # ===== PLAN =====
  plan:
    name: Plan (${{ needs.setup.outputs.environment }})
    runs-on: ubuntu-latest
    needs: setup
    if: github.event_name != 'schedule'
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: us-east-1

      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="key=${{ needs.setup.outputs.environment }}/terraform.tfstate"
        working-directory: ./terraform

      - name: Select Workspace
        run: |
          terraform workspace select ${{ needs.setup.outputs.environment }} || \
          terraform workspace new ${{ needs.setup.outputs.environment }}
        working-directory: ./terraform

      - name: Terraform Plan
        run: |
          terraform plan \
            -var-file="environments/${{ needs.setup.outputs.environment }}.tfvars" \
            -out=tfplan
        working-directory: ./terraform

      - name: Upload Plan
        uses: actions/upload-artifact@v4
        with:
          name: tfplan-${{ needs.setup.outputs.environment }}
          path: terraform/tfplan

  # ===== APPLY =====
  apply:
    name: Apply (${{ needs.setup.outputs.environment }})
    runs-on: ubuntu-latest
    needs: [setup, plan]
    if: |
      github.event_name == 'push' ||
      github.event_name == 'workflow_dispatch'
    environment:
      name: ${{ needs.setup.outputs.environment }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          aws-region: us-east-1

      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="key=${{ needs.setup.outputs.environment }}/terraform.tfstate"
        working-directory: ./terraform

      - name: Select Workspace
        run: |
          terraform workspace select ${{ needs.setup.outputs.environment }}
        working-directory: ./terraform

      - name: Download Plan
        uses: actions/download-artifact@v4
        with:
          name: tfplan-${{ needs.setup.outputs.environment }}
          path: terraform/

      - name: Terraform Apply
        run: terraform apply -auto-approve tfplan
        working-directory: ./terraform
```

**Key Features**:
- ✅ Multi-environment support (dev/staging/production)
- ✅ Workspace-based environment isolation
- ✅ Scheduled drift detection (daily)
- ✅ Environment-specific tfvars
- ✅ Slack notifications for drift
- ✅ Manual workflow dispatch with environment selection

---

## Best Practices

### 1. Remote State Configuration

**S3 Backend with DynamoDB Locking:**

```hcl
# terraform/backend.tf
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
```

**Required AWS Resources:**

```hcl
# S3 bucket for state
resource "aws_s3_bucket" "terraform_state" {
  bucket = "my-terraform-state"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
```

### 2. OPA Policy Example

**policies/terraform.rego:**

```rego
package terraform

import input as tfplan

# Deny if encryption is not enabled for S3 buckets
deny[msg] {
  resource := tfplan.resource_changes[_]
  resource.type == "aws_s3_bucket"
  not resource.change.after.server_side_encryption_configuration

  msg := sprintf("S3 bucket '%s' must have encryption enabled", [resource.address])
}

# Deny if RDS instances are not encrypted
deny[msg] {
  resource := tfplan.resource_changes[_]
  resource.type == "aws_db_instance"
  resource.change.after.storage_encrypted == false

  msg := sprintf("RDS instance '%s' must have encryption enabled", [resource.address])
}

# Deny if public access to S3 buckets
deny[msg] {
  resource := tfplan.resource_changes[_]
  resource.type == "aws_s3_bucket_acl"
  resource.change.after.acl == "public-read"

  msg := sprintf("S3 bucket '%s' must not have public-read ACL", [resource.address])
}
```

---

For related documentation, see:
- [github-actions-reference.md](github-actions-reference.md)
- [gitlab-ci-reference.md](gitlab-ci-reference.md)
- [security-automation-workflows.md](security-automation-workflows.md)
