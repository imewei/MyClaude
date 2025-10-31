---
name: secrets-management
description: Implement secure secrets management for CI/CD pipelines using HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, Google Secret Manager, or platform-native solutions (GitHub Secrets, GitLab CI/CD Variables) with encryption, rotation, and access control. Use when storing sensitive credentials like API keys, database passwords, or TLS certificates in CI/CD workflows, implementing automated secret rotation policies for enhanced security, integrating HashiCorp Vault with GitHub Actions or GitLab CI for centralized secrets, retrieving secrets from AWS Secrets Manager in deployment pipelines, configuring Azure Key Vault integration for secure credential storage, using Google Secret Manager for GCP-based applications and infrastructure, managing GitHub repository secrets, organization secrets, or environment-specific secrets, configuring GitLab CI/CD masked and protected variables for branch protection, implementing External Secrets Operator for Kubernetes secret synchronization, setting up secret scanning with TruffleHog, GitGuardian, or git-secrets to prevent leaks, establishing least-privilege access controls with IAM roles and service accounts, implementing secret versioning and audit logging for compliance, handling dynamic secrets with temporary credentials and short-lived tokens, managing TLS/SSL certificates and private keys securely, configuring OIDC authentication for keyless workflows (Workload Identity), implementing secret rotation automation with Lambda or Cloud Functions, or preventing secrets from being exposed in logs, artifacts, or error messages. Use this skill when working with GitHub Actions secrets context, Vault integration actions, AWS CLI secret retrieval, environment variables in CI/CD, or any scenarios requiring secure credential management.
---

# Secrets Management

Secure secrets management practices for CI/CD pipelines using Vault, AWS Secrets Manager, and other tools.

## When to use this skill

- When storing API keys, access tokens, or authentication credentials in CI/CD pipelines
- When managing database connection strings, passwords, or service account keys
- When handling TLS/SSL certificates, private keys, or SSH keys in automated workflows
- When implementing automated secret rotation policies for enhanced security posture
- When integrating HashiCorp Vault with GitHub Actions, GitLab CI, or other CI/CD platforms
- When retrieving secrets from AWS Secrets Manager in deployment or infrastructure pipelines
- When configuring Azure Key Vault for storing and accessing sensitive credentials
- When using Google Secret Manager for GCP-based applications and infrastructure automation
- When managing GitHub repository secrets, organization secrets, or environment-specific secrets
- When setting up GitLab CI/CD variables (masked, protected, or file-type variables)
- When implementing External Secrets Operator in Kubernetes for secret synchronization
- When setting up pre-commit hooks with TruffleHog or git-secrets to prevent secret leaks
- When configuring secret scanning in CI/CD pipelines to detect exposed credentials
- When establishing least-privilege access using IAM roles, service accounts, or RBAC policies
- When implementing secret versioning and maintaining audit trails for compliance requirements
- When handling dynamic secrets with temporary credentials or time-limited access tokens
- When configuring OIDC/Workload Identity for keyless authentication to cloud providers
- When automating secret rotation with AWS Lambda, Azure Functions, or scheduled jobs
- When ensuring secrets are masked in CI/CD logs, job outputs, and error messages
- When migrating from hardcoded secrets to secure secret management solutions
- When implementing compliance requirements (SOC2, HIPAA, PCI-DSS) for secret handling
- When managing secrets across multi-cloud or hybrid cloud environments
- When setting up developer access to secrets for local development and testing
- When configuring emergency secret revocation and incident response procedures
- When integrating third-party secret management tools or enterprise solutions

## Secrets Management Tools

### HashiCorp Vault
- Centralized secrets management
- Dynamic secrets generation
- Secret rotation
- Audit logging
- Fine-grained access control

### AWS Secrets Manager
- AWS-native solution
- Automatic rotation
- Integration with RDS
- CloudFormation support

### Azure Key Vault
- Azure-native solution
- HSM-backed keys
- Certificate management
- RBAC integration

### Google Secret Manager
- GCP-native solution
- Versioning
- IAM integration

## HashiCorp Vault Integration

### Setup Vault

```bash
# Start Vault dev server
vault server -dev

# Set environment
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='root'

# Enable secrets engine
vault secrets enable -path=secret kv-v2

# Store secret
vault kv put secret/database/config username=admin password=secret
```

### GitHub Actions with Vault

```yaml
name: Deploy with Vault Secrets

on: [push]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Import Secrets from Vault
      uses: hashicorp/vault-action@v2
      with:
        url: https://vault.example.com:8200
        token: ${{ secrets.VAULT_TOKEN }}
        secrets: |
          secret/data/database username | DB_USERNAME ;
          secret/data/database password | DB_PASSWORD ;
          secret/data/api key | API_KEY

    - name: Use secrets
      run: |
        echo "Connecting to database as $DB_USERNAME"
        # Use $DB_PASSWORD, $API_KEY
```

### GitLab CI with Vault

```yaml
deploy:
  image: vault:latest
  before_script:
    - export VAULT_ADDR=https://vault.example.com:8200
    - export VAULT_TOKEN=$VAULT_TOKEN
    - apk add curl jq
  script:
    - |
      DB_PASSWORD=$(vault kv get -field=password secret/database/config)
      API_KEY=$(vault kv get -field=key secret/api/credentials)
      echo "Deploying with secrets..."
      # Use $DB_PASSWORD, $API_KEY
```

**Reference:** See `references/vault-setup.md`

## AWS Secrets Manager

### Store Secret

```bash
aws secretsmanager create-secret \
  --name production/database/password \
  --secret-string "super-secret-password"
```

### Retrieve in GitHub Actions

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-west-2

- name: Get secret from AWS
  run: |
    SECRET=$(aws secretsmanager get-secret-value \
      --secret-id production/database/password \
      --query SecretString \
      --output text)
    echo "::add-mask::$SECRET"
    echo "DB_PASSWORD=$SECRET" >> $GITHUB_ENV

- name: Use secret
  run: |
    # Use $DB_PASSWORD
    ./deploy.sh
```

### Terraform with AWS Secrets Manager

```hcl
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "production/database/password"
}

resource "aws_db_instance" "main" {
  allocated_storage    = 100
  engine              = "postgres"
  instance_class      = "db.t3.large"
  username            = "admin"
  password            = jsondecode(data.aws_secretsmanager_secret_version.db_password.secret_string)["password"]
}
```

## GitHub Secrets

### Organization/Repository Secrets

```yaml
- name: Use GitHub secret
  run: |
    echo "API Key: ${{ secrets.API_KEY }}"
    echo "Database URL: ${{ secrets.DATABASE_URL }}"
```

### Environment Secrets

```yaml
deploy:
  runs-on: ubuntu-latest
  environment: production
  steps:
  - name: Deploy
    run: |
      echo "Deploying with ${{ secrets.PROD_API_KEY }}"
```

**Reference:** See `references/github-secrets.md`

## GitLab CI/CD Variables

### Project Variables

```yaml
deploy:
  script:
    - echo "Deploying with $API_KEY"
    - echo "Database: $DATABASE_URL"
```

### Protected and Masked Variables
- Protected: Only available in protected branches
- Masked: Hidden in job logs
- File type: Stored as file

## Best Practices

1. **Never commit secrets** to Git
2. **Use different secrets** per environment
3. **Rotate secrets regularly**
4. **Implement least-privilege access**
5. **Enable audit logging**
6. **Use secret scanning** (GitGuardian, TruffleHog)
7. **Mask secrets in logs**
8. **Encrypt secrets at rest**
9. **Use short-lived tokens** when possible
10. **Document secret requirements**

## Secret Rotation

### Automated Rotation with AWS

```python
import boto3
import json

def lambda_handler(event, context):
    client = boto3.client('secretsmanager')

    # Get current secret
    response = client.get_secret_value(SecretId='my-secret')
    current_secret = json.loads(response['SecretString'])

    # Generate new password
    new_password = generate_strong_password()

    # Update database password
    update_database_password(new_password)

    # Update secret
    client.put_secret_value(
        SecretId='my-secret',
        SecretString=json.dumps({
            'username': current_secret['username'],
            'password': new_password
        })
    )

    return {'statusCode': 200}
```

### Manual Rotation Process

1. Generate new secret
2. Update secret in secret store
3. Update applications to use new secret
4. Verify functionality
5. Revoke old secret

## External Secrets Operator

### Kubernetes Integration

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: production
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "production"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: database-credentials
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: database/config
      property: username
  - secretKey: password
    remoteRef:
      key: database/config
      property: password
```

## Secret Scanning

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for secrets with TruffleHog
docker run --rm -v "$(pwd):/repo" \
  trufflesecurity/trufflehog:latest \
  filesystem --directory=/repo

if [ $? -ne 0 ]; then
  echo "❌ Secret detected! Commit blocked."
  exit 1
fi
```

### CI/CD Secret Scanning

```yaml
secret-scan:
  stage: security
  image: trufflesecurity/trufflehog:latest
  script:
    - trufflehog filesystem .
  allow_failure: false
```

## Reference Files

- `references/vault-setup.md` - HashiCorp Vault configuration
- `references/github-secrets.md` - GitHub Secrets best practices

## Related Skills

- `github-actions-templates` - For GitHub Actions integration
- `gitlab-ci-patterns` - For GitLab CI integration
- `deployment-pipeline-design` - For pipeline architecture
