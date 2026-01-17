# Security Deep Dive Reference

Comprehensive security validation checklist and best practices across all dimensions.

## Table of Contents

1. [OWASP Top 10 (2021/2025)](#owasp-top-10)
2. [Authentication & Authorization](#authentication--authorization)
3. [Input Validation & Sanitization](#input-validation--sanitization)
4. [Cryptography Best Practices](#cryptography-best-practices)
5. [Dependency Security](#dependency-security)
6. [Secret Management](#secret-management)
7. [Language-Specific Security](#language-specific-security)
8. [API Security](#api-security)
9. [Infrastructure Security](#infrastructure-security)

---

## OWASP Top 10

### A01:2021 – Broken Access Control

**What it is**: Failures in enforcing proper access restrictions.

**Checklist**:
- [ ] All endpoints require authentication (no unprotected routes)
- [ ] Authorization checks happen on the server, not client
- [ ] Users can only access their own resources (no IDOR vulnerabilities)
- [ ] Principle of least privilege applied
- [ ] Role-based access control (RBAC) properly implemented
- [ ] No direct object references without authorization checks

**Code Examples**:

**❌ Bad (Python/FastAPI)**:
```python
@app.get("/user/{user_id}/profile")
def get_profile(user_id: int):
    # Missing authorization check!
    return db.get_user(user_id)
```

**✅ Good**:
```python
@app.get("/user/{user_id}/profile")
def get_profile(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(403, "Access denied")
    return db.get_user(user_id)
```

### A02:2021 – Cryptographic Failures

**What it is**: Failures in protecting sensitive data through cryptography.

**Checklist**:
- [ ] Sensitive data encrypted at rest and in transit
- [ ] TLS 1.2+ used for all connections
- [ ] Strong cipher suites configured
- [ ] No hardcoded keys or secrets
- [ ] Proper key rotation implemented
- [ ] PII is hashed or encrypted

**Examples**:

**❌ Bad**:
```javascript
// Storing plaintext password
db.users.insert({ username, password: password })
```

**✅ Good**:
```javascript
const bcrypt = require('bcrypt');
const hash = await bcrypt.hash(password, 10);
db.users.insert({ username, password: hash });
```

### A03:2021 – Injection

**What it is**: Untrusted data sent to an interpreter as part of a command or query.

**Checklist**:
- [ ] All user inputs validated and sanitized
- [ ] Parameterized queries used (no string concatenation in SQL)
- [ ] ORM properly used
- [ ] Command injection prevented
- [ ] LDAP injection prevented
- [ ] NoSQL injection prevented

**Examples**:

**❌ Bad (SQL Injection)**:
```python
# Vulnerable to SQL injection
query = f"SELECT * FROM users WHERE username = '{username}'"
db.execute(query)
```

**✅ Good**:
```python
# Parameterized query
query = "SELECT * FROM users WHERE username = %s"
db.execute(query, (username,))
```

**❌ Bad (Command Injection)**:
```python
import os
os.system(f"convert {user_filename} output.pdf")  # Dangerous!
```

**✅ Good**:
```python
import subprocess
subprocess.run(["convert", user_filename, "output.pdf"], check=True)
```

### A04:2021 – Insecure Design

**Checklist**:
- [ ] Threat modeling performed
- [ ] Security requirements defined early
- [ ] Fail securely (fail closed, not open)
- [ ] Rate limiting on sensitive endpoints
- [ ] Account lockout after failed attempts
- [ ] Security design patterns used

### A05:2021 – Security Misconfiguration

**Checklist**:
- [ ] No default credentials used
- [ ] Unnecessary features disabled
- [ ] Error messages don't leak sensitive info
- [ ] Security headers configured (CSP, HSTS, etc.)
- [ ] CORS properly configured
- [ ] File upload restrictions enforced

### A06:2021 – Vulnerable and Outdated Components

**Checklist**:
- [ ] All dependencies up to date
- [ ] Automated dependency scanning (Dependabot, Renovate)
- [ ] Known vulnerabilities addressed
- [ ] Unused dependencies removed
- [ ] Software Bill of Materials (SBOM) maintained

### A07:2021 – Identification and Authentication Failures

**Checklist**:
- [ ] Strong password policy enforced
- [ ] Multi-factor authentication available
- [ ] Session management secure
- [ ] Account enumeration prevented
- [ ] Brute force protection implemented
- [ ] Password reset secure

### A08:2021 – Software and Data Integrity Failures

**Checklist**:
- [ ] Code signing implemented
- [ ] Integrity checks on artifacts
- [ ] CI/CD pipeline secured
- [ ] Dependency verification (lock files, checksums)
- [ ] Auto-update mechanism secure

### A09:2021 – Security Logging and Monitoring Failures

**Checklist**:
- [ ] All authentication attempts logged
- [ ] Failed access attempts logged
- [ ] Security events trigger alerts
- [ ] Logs protected from tampering
- [ ] Log retention policy defined
- [ ] Sensitive data not logged

### A10:2021 – Server-Side Request Forgery (SSRF)

**Checklist**:
- [ ] User-supplied URLs validated
- [ ] Whitelist allowed domains
- [ ] Internal resources not accessible via URL
- [ ] Network segmentation in place

---

## Authentication & Authorization

### OAuth 2.0 / OpenID Connect

**Security Checklist**:
- [ ] Authorization code flow with PKCE used
- [ ] State parameter used to prevent CSRF
- [ ] Nonce used in ID tokens
- [ ] Tokens stored securely (httpOnly cookies or secure storage)
- [ ] Refresh tokens rotated
- [ ] Short-lived access tokens (<15 min)

**Example (Python/FastAPI)**:
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(401, "Invalid token")
        return get_user(user_id)
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
```

### JWT Best Practices

**Checklist**:
- [ ] Use HS256 or RS256 algorithm
- [ ] Never trust client-provided algorithm
- [ ] Validate signature before using claims
- [ ] Check exp (expiration) claim
- [ ] Use short expiration times
- [ ] Don't store sensitive data in payload

---

## Input Validation & Sanitization

### General Principles

1. **Whitelist > Blacklist**: Define what's allowed, not what's forbidden
2. **Validate on server**: Never trust client-side validation alone
3. **Canonicalize input**: Normalize before validation
4. **Fail securely**: Reject invalid input

### By Input Type

#### Email Validation

```python
import re

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

def validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email))
```

#### File Upload Validation

```python
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file_upload(filename: str, file_size: int) -> bool:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {ext} not allowed")
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large")
    return True
```

#### URL Validation

```python
from urllib.parse import urlparse

ALLOWED_SCHEMES = {'http', 'https'}
ALLOWED_DOMAINS = {'example.com', 'trusted-site.com'}

def validate_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    return True
```

---

## Cryptography Best Practices

### Password Hashing

**Use**: Argon2, bcrypt, or scrypt (in that order of preference)
**Don't use**: MD5, SHA1, plain SHA256

**Python Example (bcrypt)**:
```python
import bcrypt

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)
```

### Encryption (Data at Rest)

**Use**: AES-256-GCM
**Don't use**: ECB mode, weak ciphers

**Python Example**:
```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def encrypt_data(plaintext: bytes, key: bytes) -> tuple:
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce, ciphertext

def decrypt_data(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
```

### TLS Configuration

**Minimum TLS Version**: 1.2
**Recommended TLS Version**: 1.3
**Cipher Suites**: Use Mozilla SSL Configuration Generator

---

## Dependency Security

### Automated Scanning

**JavaScript**:
```bash
npm audit --audit-level=moderate
npm audit fix
```

**Python**:
```bash
pip-audit
safety check
```

**Rust**:
```bash
cargo audit
```

**Go**:
```bash
go list -json -m all | nancy sleuth
```

### Lock Files

Always commit lock files:
- `package-lock.json` (npm)
- `yarn.lock` (yarn)
- `poetry.lock` (Poetry)
- `Cargo.lock` (Rust)
- `go.sum` (Go)

---

## Secret Management

### Never Commit Secrets

**Bad patterns to avoid**:
- Hardcoded API keys
- Database credentials in code
- Private keys in repository
- `.env` files in version control

**Good practices**:
- Use environment variables
- Use secret management services (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault)
- Use `.env.example` with dummy values
- Scan for secrets with gitleaks/trufflehog

### Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")
```

---

## Language-Specific Security

### Python

- Use `secrets` module for random tokens (not `random`)
- Avoid `eval()`, `exec()`, `pickle` with untrusted data
- Use `subprocess` instead of `os.system()`
- Enable virtual environments

### JavaScript/Node.js

- Don't use `eval()` or `new Function()` with user input
- Sanitize HTML with DOMPurify
- Use Content Security Policy (CSP)
- Validate environment variables on startup

### Rust

- Use `#![forbid(unsafe_code)]` where possible
- Avoid `unwrap()` in production code
- Use `cargo audit` regularly

### Go

- Use `crypto/rand` not `math/rand` for security
- Validate all inputs
- Use `sql.DB` with prepared statements

---

## API Security

### REST API Checklist

- [ ] HTTPS only (no HTTP)
- [ ] Authentication on all endpoints
- [ ] Rate limiting per user/IP
- [ ] Input validation on all parameters
- [ ] CORS properly configured
- [ ] Security headers set
- [ ] API versioning
- [ ] Request size limits

### Security Headers

```
Content-Security-Policy: default-src 'self'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Referrer-Policy: no-referrer
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### Rate Limiting

**Python/FastAPI Example**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/resource")
@limiter.limit("10/minute")
def get_resource(request: Request):
    return {"data": "resource"}
```

---

## Infrastructure Security

### Docker Security

**Checklist**:
- [ ] Use official base images
- [ ] Scan images for vulnerabilities (Trivy, Clair)
- [ ] Don't run as root
- [ ] Use multi-stage builds
- [ ] Minimize attack surface (Alpine Linux)
- [ ] No secrets in images

**Example Dockerfile**:
```dockerfile
FROM python:3.12-alpine AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.12-alpine
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
USER nobody
CMD ["python", "app.py"]
```

### Kubernetes Security

**Checklist**:
- [ ] RBAC enabled
- [ ] Network policies defined
- [ ] Pod security policies/standards
- [ ] Secrets management (sealed secrets, external secrets)
- [ ] Image scanning in CI/CD
- [ ] No privileged pods

---

## Automated Security Testing

### CI/CD Integration

```yaml
# GitHub Actions example
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1

      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'

      - name: Run npm audit
        run: npm audit --audit-level=moderate
```

---

## Security Testing Checklist

### Before Every Release

- [ ] Run automated security scans
- [ ] Review dependency updates
- [ ] Check for exposed secrets
- [ ] Validate authentication/authorization
- [ ] Test rate limiting
- [ ] Verify error handling doesn't leak info
- [ ] Check logs for sensitive data
- [ ] Review security headers
- [ ] Test input validation on all endpoints
- [ ] Verify TLS configuration

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Mozilla Web Security Guidelines](https://infosec.mozilla.org/guidelines/web_security)
