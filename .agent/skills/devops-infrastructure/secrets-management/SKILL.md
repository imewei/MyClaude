---
name: secrets-management
version: "1.0.7"
maturity: "5-Expert"
specialization: Credential Security
description: Implement secrets management with HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, or platform-native solutions with encryption, rotation, and access control. Use when storing API keys, database passwords, TLS certificates, or implementing secret rotation.
---

# Secrets Management

Secure credential storage with rotation and access control.

---

## Tool Selection

| Tool | Best For | Features |
|------|----------|----------|
| HashiCorp Vault | Multi-cloud, dynamic secrets | Rotation, audit, fine-grained access |
| AWS Secrets Manager | AWS-native | RDS integration, auto-rotation |
| Azure Key Vault | Azure-native | HSM-backed, RBAC |
| GitHub Secrets | GitHub Actions | Environment scopes |
| GitLab CI Variables | GitLab CI | Masked, protected |

---

## HashiCorp Vault

### GitHub Actions Integration

```yaml
- name: Import Secrets from Vault
  uses: hashicorp/vault-action@v2
  with:
    url: https://vault.example.com:8200
    token: ${{ secrets.VAULT_TOKEN }}
    secrets: |
      secret/data/database username | DB_USERNAME ;
      secret/data/database password | DB_PASSWORD

- run: echo "Connecting as $DB_USERNAME"
```

### CLI Usage

```bash
vault kv put secret/database username=admin password=secret
vault kv get -field=password secret/database
```

---

## AWS Secrets Manager

### GitHub Actions

```yaml
- uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

- name: Get secret
  run: |
    SECRET=$(aws secretsmanager get-secret-value \
      --secret-id production/database \
      --query SecretString --output text)
    echo "::add-mask::$SECRET"
    echo "DB_PASSWORD=$SECRET" >> $GITHUB_ENV
```

### Terraform

```hcl
data "aws_secretsmanager_secret_version" "db" {
  secret_id = "production/database"
}

resource "aws_db_instance" "main" {
  password = jsondecode(data.aws_secretsmanager_secret_version.db.secret_string)["password"]
}
```

---

## Platform-Native Secrets

### GitHub Secrets

```yaml
- run: |
    echo "API Key: ${{ secrets.API_KEY }}"
    echo "DB URL: ${{ secrets.DATABASE_URL }}"

# Environment-specific
deploy:
  environment: production
  steps:
    - run: echo "${{ secrets.PROD_API_KEY }}"
```

### GitLab CI Variables

```yaml
deploy:
  script:
    - echo "Deploying with $API_KEY"
  # Variables: Protected (protected branches only), Masked (hidden in logs)
```

---

## External Secrets Operator (K8s)

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
  target:
    name: database-credentials
  data:
  - secretKey: password
    remoteRef:
      key: database/config
      property: password
```

---

## Secret Rotation

### Automated (AWS Lambda)

```python
def lambda_handler(event, context):
    client = boto3.client('secretsmanager')
    new_password = generate_strong_password()
    update_database_password(new_password)
    client.put_secret_value(
        SecretId='my-secret',
        SecretString=json.dumps({'password': new_password})
    )
```

### Manual Process

1. Generate new secret
2. Update in secret store
3. Deploy applications
4. Verify functionality
5. Revoke old secret

---

## Secret Scanning

### Pre-commit Hook

```bash
#!/bin/bash
docker run --rm -v "$(pwd):/repo" \
  trufflesecurity/trufflehog filesystem /repo

if [ $? -ne 0 ]; then
  echo "Secret detected! Commit blocked."
  exit 1
fi
```

### CI Pipeline

```yaml
secret-scan:
  image: trufflesecurity/trufflehog:latest
  script: trufflehog filesystem .
  allow_failure: false
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Never commit secrets | Use .gitignore, scanning |
| Environment separation | Different secrets per env |
| Rotate regularly | Automate rotation |
| Least privilege | Minimal access per service |
| Audit logging | Track secret access |
| Mask in logs | Use ::add-mask:: |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Secrets in code | Use secret stores |
| Same secret everywhere | Per-environment secrets |
| No rotation | Automate rotation |
| Overly broad access | Apply least privilege |
| Secrets in logs | Mask all sensitive values |

---

## Checklist

- [ ] Central secret store selected
- [ ] Secrets not committed to Git
- [ ] Environment-specific secrets
- [ ] Rotation policy defined
- [ ] Secret scanning in CI
- [ ] Audit logging enabled
- [ ] Least-privilege access

---

**Version**: 1.0.5
