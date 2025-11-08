# Security Validation Guide

Comprehensive security validation covering OWASP Top 10, authentication, authorization, data protection, and compliance frameworks.

---

## OWASP Top 10 (2021) Deep Dive

### 1. Broken Access Control

**Description:** Users can access resources or perform actions they shouldn't be authorized for.

**Common Vulnerabilities:**
- Insecure Direct Object References (IDOR)
- Vertical privilege escalation (user → admin)
- Horizontal privilege escalation (user A → user B)
- Missing authorization checks
- Byp

assing access control via URL manipulation

**Prevention:**
```python
# BAD - No authorization check
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id: int):
    return db.get_user(user_id)  # Any user can access any profile

# GOOD - Authorization check
@app.get("/api/users/{user_id}/profile")
def get_profile(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(403, "Access denied")
    return db.get_user(user_id)
```

**Testing:**
```bash
# Test IDOR vulnerability
curl -H "Authorization: Bearer USER_A_TOKEN" \
  https://api.example.com/api/users/999/profile  # Try accessing user B's profile

# Test privilege escalation
curl -H "Authorization: Bearer USER_TOKEN" \
  https://api.example.com/api/admin/users  # Should return 403
```

**Checklist:**
- [ ] All protected resources have authorization checks
- [ ] Role-based access control (RBAC) implemented
- [ ] Attribute-based access control (ABAC) for fine-grained permissions
- [ ] JWT/session tokens properly validated
- [ ] Admin functions protected from non-admin access

---

### 2. Cryptographic Failures

**Description:** Sensitive data exposed due to weak encryption or lack thereof.

**Common Failures:**
- Passwords stored in plaintext
- Weak hashing algorithms (MD5, SHA1)
- HTTP used instead of HTTPS
- Sensitive data in logs
- Hardcoded encryption keys

**Prevention:**
```python
# BAD - Weak password hashing
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()

# GOOD - Strong password hashing with bcrypt
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))

# BAD - Hardcoded secret
SECRET_KEY = "my-secret-key-12345"

# GOOD - Environment variable
import os
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable must be set")
```

**Encryption Standards:**
- **At Rest:** AES-256-GCM
- **In Transit:** TLS 1.2+ (prefer TLS 1.3)
- **Passwords:** bcrypt (cost 12+), Argon2id, scrypt
- **Tokens:** HMAC-SHA256 or RS256 (for JWT)

**Checklist:**
- [ ] All sensitive data encrypted at rest
- [ ] TLS 1.2+ enforced for all connections
- [ ] Strong password hashing (bcrypt, Argon2)
- [ ] No secrets in code or version control
- [ ] Certificate validation enabled
- [ ] Perfect Forward Secrecy (PFS) enabled

---

### 3. Injection

**Description:** Untrusted data sent to an interpreter as part of a command or query.

**SQL Injection:**
```python
# BAD - String concatenation
query = f"SELECT * FROM users WHERE username = '{username}'"
cursor.execute(query)

# GOOD - Parameterized query
query = "SELECT * FROM users WHERE username = ?"
cursor.execute(query, (username,))

# ORM (safest)
user = db.query(User).filter(User.username == username).first()
```

**Command Injection:**
```python
# BAD - Shell command with user input
import os
os.system(f"ping {user_input}")

# GOOD - Avoid shell, use libraries
import subprocess
subprocess.run(["ping", "-c", "4", user_input], check=True)
```

**XSS Prevention:**
```javascript
// BAD - innerHTML with user data
element.innerHTML = userData;

// GOOD - textContent (auto-escapes)
element.textContent = userData;

// GOOD - React (auto-escapes)
return <div>{userData}</div>

// BAD - dangerouslySetInnerHTML
return <div dangerouslySetInnerHTML={{__html: userData}} />
```

**NoSQL Injection:**
```javascript
// BAD - Unvalidated object
db.users.find({ username: req.body.username })

// GOOD - String validation
const username = String(req.body.username)
db.users.find({ username: username })
```

**Checklist:**
- [ ] All SQL queries use parameterized statements or ORM
- [ ] No shell command execution with user input
- [ ] HTML output properly escaped
- [ ] JSON responses validated
- [ ] Input validation on all user data

---

### 4. Insecure Design

**Description:** Missing or ineffective security controls in the design phase.

**Threat Modeling:**
```
STRIDE Analysis:
- Spoofing: Can attackers impersonate users/services?
- Tampering: Can data be modified in transit or at rest?
- Repudiation: Can actions be denied without audit trail?
- Information Disclosure: Is sensitive data exposed?
- Denial of Service: Can service be overwhelmed?
- Elevation of Privilege: Can users gain unauthorized access?
```

**Secure Design Principles:**
1. **Defense in Depth:** Multiple layers of security
2. **Fail Secure:** System fails to secure state (deny access)
3. **Least Privilege:** Minimum permissions needed
4. **Separation of Duties:** No single user has complete control
5. **Complete Mediation:** Check every access
6. **Open Design:** Security not through obscurity

**Example - Secure Password Reset:**
```python
# Insecure Design - Password reset via GET with token
@app.get("/reset-password")
def reset_password(token: str, new_password: str):
    # Token in URL, leaked in logs/browser history
    # New password in URL, leaked in logs/browser history

# Secure Design - Multi-step process
@app.post("/request-password-reset")
def request_reset(email: str):
    # 1. Generate time-limited token
    token = generate_token(email, expiry=15*60)  # 15 min
    # 2. Send via email (not SMS - SIM swapping)
    send_email(email, reset_link_with_token)
    # 3. Rate limit (prevent brute force)
    rate_limiter.check(email)

@app.post("/reset-password")
def reset_password(token: str, new_password: str):
    # 1. Validate token (not expired, not already used)
    # 2. Enforce password policy
    # 3. Invalidate all existing sessions
    # 4. Log security event
    # 5. Send confirmation email
```

**Checklist:**
- [ ] Threat model created for critical features
- [ ] Security requirements documented
- [ ] Abuse cases considered
- [ ] Rate limiting implemented
- [ ] Security review before implementation

---

### 5. Security Misconfiguration

**Description:** Systems configured insecurely due to default settings, incomplete configs, or open cloud storage.

**Common Misconfigurations:**
- Default credentials
- Unnecessary features enabled
- Directory listing enabled
- Detailed error messages
- Missing security headers

**Security Headers:**
```nginx
# nginx configuration
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

**Cookie Security:**
```python
# Secure cookie settings
response.set_cookie(
    key="session_id",
    value=session_id,
    httponly=True,      # Prevents JavaScript access
    secure=True,        # Only sent over HTTPS
    samesite="Strict",  # CSRF protection
    max_age=3600,       # 1 hour expiry
)
```

**Environment Configuration:**
```python
# Production configuration check
class Config:
    DEBUG = False  # NEVER True in production
    SECRET_KEY = os.getenv("SECRET_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Validate configuration on startup
    def validate(self):
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY not set")
        if self.DEBUG:
            raise ValueError("DEBUG=True in production")
        if "localhost" in self.DATABASE_URL:
            raise ValueError("Using localhost database in production")
```

**Checklist:**
- [ ] All security headers configured
- [ ] Default credentials changed
- [ ] Unnecessary features/ports disabled
- [ ] Error messages don't leak system details
- [ ] CORS properly configured
- [ ] File upload restrictions enforced

---

### 6. Vulnerable and Outdated Components

**Description:** Using libraries/frameworks with known vulnerabilities.

**Dependency Scanning:**
```bash
# JavaScript
npm audit --audit-level=moderate
npm audit fix

# Python
pip-audit
pip install --upgrade pip pip-audit

# Rust
cargo audit
cargo audit fix --dry-run

# GitHub Dependabot (automated)
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

**Dependency Management Best Practices:**
```json
// package.json - Lock versions
{
  "dependencies": {
    "express": "4.18.2",  // Exact version, not ^4.18.2
    "jsonwebtoken": "~9.0.0"  // Patch updates only
  }
}
```

**Checklist:**
- [ ] All dependencies up to date
- [ ] Automated dependency scanning in CI/CD
- [ ] Security advisories monitored
- [ ] Unused dependencies removed
- [ ] Transitive dependencies reviewed
- [ ] Package lock files committed

---

### 7. Identification and Authentication Failures

**Description:** Authentication/session management implemented incorrectly.

**Strong Authentication:**
```python
# Password Policy
def validate_password(password: str):
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain lowercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain digit")
    if not re.search(r"[!@#$%^&*]", password):
        raise ValueError("Password must contain special character")

    # Check against common passwords
    if password in COMMON_PASSWORDS:
        raise ValueError("Password too common")

    # Check against breached passwords (Have I Been Pwned API)
    if is_password_breached(password):
        raise ValueError("Password found in breach database")
```

**Multi-Factor Authentication:**
```python
# TOTP-based MFA
import pyotp

# Setup MFA
secret = pyotp.random_base32()
totp = pyotp.TOTP(secret)
qr_code_url = totp.provisioning_uri(user.email, issuer_name="MyApp")

# Verify MFA
def verify_mfa(user, code):
    totp = pyotp.TOTP(user.mfa_secret)
    if not totp.verify(code, valid_window=1):
        raise ValueError("Invalid MFA code")
```

**Session Management:**
```python
# Secure session management
class SessionManager:
    def create_session(self, user_id):
        session_id = secrets.token_urlsafe(32)
        session_data = {
            "user_id": user_id,
            "created_at": time.time(),
            "ip_address": request.remote_addr,
            "user_agent": request.headers.get("User-Agent")
        }
        cache.setex(f"session:{session_id}", 3600, json.dumps(session_data))
        return session_id

    def validate_session(self, session_id):
        data = cache.get(f"session:{session_id}")
        if not data:
            raise ValueError("Session expired")

        session = json.loads(data)

        # Check session age
        if time.time() - session["created_at"] > 3600:
            raise ValueError("Session expired")

        # Check IP address (optional, may break mobile users)
        if session["ip_address"] != request.remote_addr:
            log.warning("Session IP mismatch")

        return session["user_id"]

    def invalidate_session(self, session_id):
        cache.delete(f"session:{session_id}")
```

**Account Enumeration Prevention:**
```python
# BAD - Different responses reveal valid usernames
@app.post("/login")
def login(username, password):
    user = db.get_user(username)
    if not user:
        return {"error": "User not found"}  # Reveals valid usernames
    if not verify_password(user, password):
        return {"error": "Invalid password"}

# GOOD - Same response for all failures
@app.post("/login")
def login(username, password):
    user = db.get_user(username)
    if not user or not verify_password(user, password):
        return {"error": "Invalid username or password"}  # Generic message

    # Add delay to prevent timing attacks
    time.sleep(random.uniform(0.1, 0.3))
```

**Checklist:**
- [ ] Password complexity enforced (12+ chars, mixed case, symbols)
- [ ] MFA available for all users
- [ ] Session timeout implemented (< 1 hour for sensitive apps)
- [ ] Session invalidation on logout/password change
- [ ] Account lockout after failed attempts
- [ ] No account enumeration vulnerability

---

### 8. Software and Data Integrity Failures

**Description:** Code/infrastructure that doesn't protect against integrity violations.

**CI/CD Pipeline Security:**
```yaml
# GitHub Actions - Secure pipeline
name: Secure CI/CD

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read  # Minimum permissions
      id-token: write  # For OIDC

    steps:
      - uses: actions/checkout@v4

      # Verify dependency integrity
      - name: Verify package-lock.json
        run: npm ci --audit --audit-level=moderate

      # Code signing
      - name: Sign artifacts
        run: |
          gpg --import ${{ secrets.GPG_PRIVATE_KEY }}
          gpg --detach-sign --armor dist/app.js

      # Attestation (Sigstore)
      - name: Generate attestation
        uses: actions/attest-build-provenance@v1
        with:
          subject-path: dist/app.js
```

**Dependency Integrity:**
```json
// package.json - Subresource Integrity
{
  "dependencies": {
    "lodash": "4.17.21"
  },
  "integrity": {
    "lodash": "sha512-v2kDEe57lecTulaDIuNTPy3Ry4gLGJ6Z1O3vE1krgXZNrsQ+LFTGHVxVjcXPs17LhbZVGedAJv8XZ1tvj5FvSg=="
  }
}
```

**Code Signing:**
```bash
# Sign release artifacts
gpg --detach-sign --armor myapp-v1.0.0.tar.gz

# Verify signature
gpg --verify myapp-v1.0.0.tar.gz.asc myapp-v1.0.0.tar.gz
```

**Checklist:**
- [ ] CI/CD pipeline uses minimum permissions
- [ ] Dependencies verified with checksums
- [ ] Release artifacts signed
- [ ] Auto-updates verify signatures
- [ ] No untrusted plugins/libraries

---

### 9. Security Logging and Monitoring Failures

**Description:** Insufficient logging and monitoring to detect/respond to breaches.

**What to Log:**
```python
# Security event logging
import logging
import json

security_logger = logging.getLogger("security")

def log_security_event(event_type, **kwargs):
    event = {
        "timestamp": time.time(),
        "event_type": event_type,
        "user_id": kwargs.get("user_id"),
        "ip_address": request.remote_addr,
        "user_agent": request.headers.get("User-Agent"),
        "details": kwargs
    }
    security_logger.warning(json.dumps(event))

# Log all authentication attempts
@app.post("/login")
def login(username, password):
    user = authenticate(username, password)
    if user:
        log_security_event("login_success", user_id=user.id, username=username)
    else:
        log_security_event("login_failure", username=username, reason="invalid_credentials")

# Log all authorization failures
@app.get("/admin/users")
def admin_users(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        log_security_event("authorization_failure", user_id=current_user.id, endpoint="/admin/users")
        raise HTTPException(403)
```

**Alerting Rules:**
```python
# Alert on suspicious activity
ALERT_RULES = {
    "brute_force": {
        "condition": "login_failure > 5 in 1 minute",
        "action": "lock_account_and_alert"
    },
    "privilege_escalation_attempt": {
        "condition": "authorization_failure > 3 in 5 minutes",
        "action": "alert_security_team"
    },
    "data_exfiltration": {
        "condition": "large_download > 100MB",
        "action": "alert_and_require_mfa"
    }
}
```

**Checklist:**
- [ ] All authentication events logged
- [ ] All authorization failures logged
- [ ] Sensitive actions logged (password reset, permission changes)
- [ ] Logs stored securely (write-once, encrypted)
- [ ] Log retention policy defined
- [ ] Real-time alerting configured
- [ ] Log monitoring dashboard

---

### 10. Server-Side Request Forgery (SSRF)

**Description:** Application fetches a remote resource without validating the user-supplied URL.

**Prevention:**
```python
# BAD - Unvalidated URL fetching
@app.post("/fetch-image")
def fetch_image(url: str):
    response = requests.get(url)  # Can access internal services!
    return response.content

# GOOD - URL validation with allowlist
ALLOWED_DOMAINS = ["cdn.example.com", "images.example.com"]

def validate_url(url: str):
    parsed = urlparse(url)

    # Only allow HTTP/HTTPS
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Invalid URL scheme")

    # Allowlist domains
    if parsed.netloc not in ALLOWED_DOMAINS:
        raise ValueError(f"Domain {parsed.netloc} not allowed")

    # Prevent IP addresses (especially private IPs)
    if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc):
        raise ValueError("IP addresses not allowed")

    # Prevent DNS rebinding
    try:
        ip = socket.gethostbyname(parsed.netloc)
        if ipaddress.ip_address(ip).is_private:
            raise ValueError("Private IP addresses not allowed")
    except socket.gaierror:
        raise ValueError("Invalid hostname")

    return url

@app.post("/fetch-image")
def fetch_image(url: str):
    validated_url = validate_url(url)
    response = requests.get(validated_url, timeout=5)
    return response.content
```

**Checklist:**
- [ ] All external URLs validated before fetching
- [ ] Allowlist of permitted domains
- [ ] No raw IP addresses accepted
- [ ] Private IP ranges blocked (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- [ ] Localhost/loopback blocked (127.0.0.1, ::1)
- [ ] Link-local addresses blocked (169.254.0.0/16)

---

## Compliance Frameworks

### GDPR (General Data Protection Regulation)

**Requirements:**
- [ ] **Data Minimization:** Collect only necessary data
- [ ] **Purpose Limitation:** Use data only for stated purposes
- [ ] **Right to Access:** Users can request their data
- [ ] **Right to Erasure:** Users can request deletion
- [ ] **Data Portability:** Export data in machine-readable format
- [ ] **Consent Management:** Clear opt-in/opt-out
- [ ] **Breach Notification:** Report breaches within 72 hours

```python
# GDPR Data Export
@app.get("/api/users/{user_id}/data-export")
def export_user_data(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.id != user_id:
        raise HTTPException(403)

    # Collect all user data
    user_data = {
        "profile": get_user_profile(user_id),
        "orders": get_user_orders(user_id),
        "preferences": get_user_preferences(user_id),
        "activity_log": get_user_activity(user_id)
    }

    return JSONResponse(content=user_data)

# GDPR Data Deletion
@app.delete("/api/users/{user_id}")
def delete_user_data(user_id: int, current_user: User = Depends(get_current_user)):
    if current_user.id != user_id:
        raise HTTPException(403)

    # Delete all user data
    db.delete_user(user_id)
    # Anonymize in logs
    anonymize_logs(user_id)
    # Remove from analytics
    remove_from_analytics(user_id)
```

### HIPAA (Health Insurance Portability and Accountability Act)

**Requirements:**
- [ ] **Encryption:** All ePHI encrypted at rest and in transit
- [ ] **Access Controls:** Role-based access to patient data
- [ ] **Audit Logs:** Track all access to ePHI
- [ ] **Business Associate Agreements:** With third-party vendors
- [ ] **Minimum Necessary:** Access limited to minimum needed

### SOC 2

**Trust Principles:**
- **Security:** Protection against unauthorized access
- **Availability:** System uptime and performance
- **Processing Integrity:** Complete, accurate processing
- **Confidentiality:** Sensitive data protection
- **Privacy:** PII handling and user consent

---

## Security Testing Tools

### Recommended Toolchain

```bash
# 1. Dependency Scanning
npm audit || pip-audit || cargo audit

# 2. SAST (Static Analysis)
semgrep --config=auto .
bandit -r . (Python)
brakeman . (Ruby on Rails)

# 3. Secret Detection
gitleaks detect --no-git
trufflehog filesystem .

# 4. Container Security
trivy image myapp:latest
docker scout cves myapp:latest

# 5. DAST (Dynamic Analysis)
zap-baseline.py -t https://myapp.example.com
nuclei -u https://myapp.example.com

# 6. Penetration Testing
burp suite (manual testing)
```

---

## Security Checklist Summary

**Authentication & Authorization:**
- [ ] Strong password policy enforced
- [ ] MFA available
- [ ] Session management secure
- [ ] Authorization checks on all resources

**Data Protection:**
- [ ] Encryption at rest (AES-256)
- [ ] Encryption in transit (TLS 1.2+)
- [ ] Sensitive data not logged
- [ ] PII handling compliant with regulations

**Input Validation:**
- [ ] All inputs validated and sanitized
- [ ] SQL injection prevented
- [ ] XSS prevented
- [ ] Command injection prevented

**Configuration:**
- [ ] Security headers configured
- [ ] CORS properly set
- [ ] Default credentials changed
- [ ] Error messages don't leak info

**Monitoring:**
- [ ] Security events logged
- [ ] Real-time alerting
- [ ] Incident response plan

**Dependencies:**
- [ ] All dependencies up to date
- [ ] Automated scanning enabled
- [ ] No known vulnerabilities
