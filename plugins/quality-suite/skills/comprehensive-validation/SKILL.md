---
name: comprehensive-validation
version: "2.2.1"
description: Multi-dimensional validation framework for code, APIs, and systems. Covers security scans, performance profiling, and production readiness checks.
---

# Comprehensive Validation

Systematic framework for ensuring software quality across multiple dimensions before deployment.

## 1. Validation Dimensions

- **Functional**: Verify happy paths, edge cases, and error handling.
- **Security**: Run SAST (Semgrep), SCA (dependency audit), and secret detection (Gitleaks). Check against OWASP Top 10.
- **Performance**: Detect N+1 queries, verify database indexes, and profile hot paths.
- **Accessibility**: Ensure WCAG 2.1 AA compliance using tools like axe-core or pa11y.
- **Operations**: Validate logging levels, metrics instrumentation, and health check endpoints.

## 2. Automated Validation Workflow

Use the validation orchestrator to run all checks:
```bash
# Run all automated validations (lint, test, security, build)
python scripts/run_all_validations.py
```

### Security Scanning
```bash
# Focused security scan
python scripts/security_scan.py --fast
```

## 3. Production Readiness Checklist

- [ ] **Secrets**: No secrets in code; retrieved from vault/env.
- [ ] **Resilience**: Timeouts, retries, and graceful shutdown implemented.
- [ ] **Observability**: RED metrics instrumented and dashboards available.
- [ ] **Documentation**: README, API specs, and runbooks are up to date.
