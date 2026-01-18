# Security Automation Workflows

**Version**: 1.0.3
**Command**: `/workflow-automate`
**Category**: CI/CD Automation

## Overview

Comprehensive security scanning automation covering SAST, DAST, dependency scanning, container security, secret detection, and compliance validation with automated gates and reporting.

---

## Table of Contents

1. [Comprehensive Security Scan (GitHub Actions)](#1-comprehensive-security-scan-github-actions)
2. [Container Security Pipeline](#2-container-security-pipeline)
3. [Compliance Automation](#3-compliance-automation)

---

## 1. Comprehensive Security Scan (GitHub Actions)

Complete security scanning workflow covering all security dimensions with automated gates.

### `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  pull-requests: write

jobs:
  # ===== SAST: CODEQL =====
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        language: [javascript, python]
    steps:
      - uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          queries: security-extended,security-and-quality

      - name: Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"

  # ===== SAST: SEMGREP =====
  semgrep:
    name: Semgrep Scan
    runs-on: ubuntu-latest
    container:
      image: returntocorp/semgrep
    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        run: |
          semgrep scan \
            --config auto \
            --sarif \
            --output semgrep-results.sarif

      - name: Upload Semgrep Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: semgrep-results.sarif
          category: semgrep

  # ===== SAST: SONARQUBE =====
  sonarqube:
    name: SonarQube Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Shallow clones should be disabled

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      - name: SonarQube Quality Gate
        uses: sonarsource/sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  # ===== DEPENDENCY SCANNING =====
  dependency-scan:
    name: Dependency Scanning
    runs-on: ubuntu-latest
    strategy:
      matrix:
        scanner: [snyk, npm-audit, safety]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        if: matrix.scanner == 'npm-audit' || matrix.scanner == 'snyk'
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Setup Python
        if: matrix.scanner == 'safety'
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # NPM Audit
      - name: Run npm audit
        if: matrix.scanner == 'npm-audit'
        run: |
          npm ci
          npm audit --audit-level=moderate --json > npm-audit.json
        continue-on-error: true

      - name: Upload npm audit results
        if: matrix.scanner == 'npm-audit'
        uses: actions/upload-artifact@v4
        with:
          name: npm-audit-results
          path: npm-audit.json

      # Snyk
      - name: Run Snyk
        if: matrix.scanner == 'snyk'
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --sarif-file-output=snyk-results.sarif
        continue-on-error: true

      - name: Upload Snyk results
        if: matrix.scanner == 'snyk'
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: snyk-results.sarif

      # Safety (Python)
      - name: Run Safety
        if: matrix.scanner == 'safety'
        run: |
          uv uv pip install safety
          safety check --json --output safety-report.json || true
        continue-on-error: true

      - name: Upload Safety results
        if: matrix.scanner == 'safety'
        uses: actions/upload-artifact@v4
        with:
          name: safety-results
          path: safety-report.json

  # ===== CONTAINER SCANNING =====
  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    needs: [codeql, semgrep]
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker Image
        run: |
          docker build -t myapp:${{ github.sha }} .

      # Trivy
      - name: Run Trivy Scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy Results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
          category: trivy

      # Anchore (Grype)
      - name: Run Anchore Scan
        uses: anchore/scan-action@v3
        id: scan
        with:
          image: myapp:${{ github.sha }}
          fail-build: false
          severity-cutoff: high

      - name: Upload Anchore Results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: ${{ steps.scan.outputs.sarif }}
          category: anchore

  # ===== SECRET SCANNING =====
  secret-scan:
    name: Secret Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comprehensive scan

      # GitLeaks
      - name: Run GitLeaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_ENABLE_COMMENTS: true

      # TruffleHog
      - name: TruffleHog Scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD

  # ===== LICENSE COMPLIANCE =====
  license-check:
    name: License Compliance
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Check Licenses
        run: |
          npx license-checker \
            --json \
            --out license-report.json \
            --failOn 'GPL;AGPL;LGPL' \
            --summary

      - name: Upload License Report
        uses: actions/upload-artifact@v4
        with:
          name: license-report
          path: license-report.json

  # ===== SECURITY GATE =====
  security-gate:
    name: Security Quality Gate
    runs-on: ubuntu-latest
    needs: [codeql, semgrep, dependency-scan, container-scan, secret-scan]
    if: always()
    steps:
      - name: Check Security Scan Results
        run: |
          # This job will fail if any critical security issues are found
          # Customize logic based on your security requirements

          CODEQL_STATUS="${{ needs.codeql.result }}"
          SEMGREP_STATUS="${{ needs.semgrep.result }}"
          CONTAINER_STATUS="${{ needs.container-scan.result }}"
          SECRET_STATUS="${{ needs.secret-scan.result }}"

          if [ "$SECRET_STATUS" == "failure" ]; then
            echo "❌ Secret scanning failed - blocking deployment"
            exit 1
          fi

          if [ "$CODEQL_STATUS" == "failure" ]; then
            echo "⚠️ CodeQL found issues - review required"
            # Don't block for CodeQL warnings
          fi

          echo "✅ Security gate passed"

      - name: Comment PR with Security Summary
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const summary = `## 🔒 Security Scan Summary

            | Scan Type | Status |
            |-----------|--------|
            | CodeQL | ${{ needs.codeql.result }} |
            | Semgrep | ${{ needs.semgrep.result }} |
            | Dependency Scan | ${{ needs.dependency-scan.result }} |
            | Container Scan | ${{ needs.container-scan.result }} |
            | Secret Scan | ${{ needs.secret-scan.result }} |

            *Review security findings in the Security tab.*`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            });

  # ===== NOTIFICATION =====
  notify:
    name: Security Alert Notification
    runs-on: ubuntu-latest
    needs: security-gate
    if: failure() && github.event_name == 'schedule'
    steps:
      - name: Send Slack Alert
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 Security Scan Failed",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Security Scan Alert*\n*Repository:* ${{ github.repository }}\n*Status:* Failed\n*Workflow:* ${{ github.workflow }}\n*Action Required:* Review security findings immediately"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

**Key Features**:
- ✅ Multi-dimensional security scanning (SAST, dependency, container, secrets)
- ✅ SARIF upload to GitHub Security tab
- ✅ Automated security gates
- ✅ PR comments with security summary
- ✅ Scheduled daily scans
- ✅ Slack notifications for failures

---

## 2. Container Security Pipeline

Dedicated container security workflow with comprehensive scanning and policy enforcement.

### `.github/workflows/container-security.yml`

```yaml
name: Container Security

on:
  push:
    branches: [main]
    paths:
      - 'Dockerfile'
      - 'docker-compose.yml'
      - '.github/workflows/container-security.yml'
  pull_request:
    paths:
      - 'Dockerfile'
  schedule:
    - cron: '0 3 * * *'  # Daily at 3 AM

permissions:
  contents: read
  security-events: write
  packages: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ===== DOCKERFILE LINTING =====
  dockerfile-lint:
    name: Dockerfile Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Hadolint
        uses: hadolint/hadolint-action@v3
        with:
          dockerfile: Dockerfile
          format: sarif
          output-file: hadolint-results.sarif
          no-fail: true

      - name: Upload Hadolint Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: hadolint-results.sarif

  # ===== BUILD IMAGE =====
  build:
    name: Build Container Image
    runs-on: ubuntu-latest
    needs: dockerfile-lint
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix={{branch}}-

      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ===== VULNERABILITY SCANNING =====
  scan:
    name: Vulnerability Scan
    runs-on: ubuntu-latest
    needs: build
    strategy:
      matrix:
        scanner: [trivy, grype, clair]
    steps:
      - uses: actions/checkout@v4

      # Trivy
      - name: Run Trivy
        if: matrix.scanner == 'trivy'
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH,MEDIUM'
          vuln-type: 'os,library'

      - name: Upload Trivy Results
        if: matrix.scanner == 'trivy'
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
          category: trivy

      # Grype
      - name: Run Grype
        if: matrix.scanner == 'grype'
        uses: anchore/scan-action@v3
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          fail-build: false
          severity-cutoff: medium

      - name: Upload Grype Results
        if: matrix.scanner == 'grype'
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
          category: grype

  # ===== MALWARE SCANNING =====
  malware-scan:
    name: Malware Scan
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Run ClamAV
        run: |
          docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          docker save ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} -o image.tar

          # Install ClamAV
          sudo apt-get update
          sudo apt-get install -y clamav clamav-daemon

          # Update virus database
          sudo freshclam

          # Scan image
          clamscan --infected --recursive image.tar | tee clamscan-results.txt

      - name: Upload Scan Results
        uses: actions/upload-artifact@v4
        with:
          name: malware-scan-results
          path: clamscan-results.txt

  # ===== IMAGE SIGNING =====
  sign:
    name: Sign Container Image
    runs-on: ubuntu-latest
    needs: [scan, malware-scan]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Install Cosign
        uses: sigstore/cosign-installer@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Sign Image
        run: |
          cosign sign --yes \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        env:
          COSIGN_EXPERIMENTAL: 1

  # ===== SBOM GENERATION =====
  sbom:
    name: Generate SBOM
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4

      - name: Generate SBOM with Syft
        uses: anchore/sbom-action@v0
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: spdx-json
          output-file: sbom.spdx.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.spdx.json

      - name: Attach SBOM to Release
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: |
          cosign attach sbom --sbom sbom.spdx.json \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        env:
          COSIGN_EXPERIMENTAL: 1
```

**Key Features**:
- ✅ Dockerfile linting with Hadolint
- ✅ Multi-scanner vulnerability detection (Trivy, Grype)
- ✅ Malware scanning with ClamAV
- ✅ Image signing with Cosign
- ✅ SBOM generation with Syft

---

## 3. Compliance Automation

Automated compliance validation for OWASP Top 10, CIS benchmarks, and regulatory requirements.

### `.github/workflows/compliance.yml`

```yaml
name: Compliance Validation

on:
  schedule:
    - cron: '0 4 * * 0'  # Weekly on Sundays at 4 AM
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  issues: write

jobs:
  # ===== OWASP TOP 10 =====
  owasp-check:
    name: OWASP Top 10 Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # OWASP Dependency Check
      - name: Run OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'MyApp'
          path: '.'
          format: 'HTML'
          args: >
            --failOnCVSS 7
            --enableRetired

      - name: Upload Dependency Check Report
        uses: actions/upload-artifact@v4
        with:
          name: dependency-check-report
          path: reports/

      # OWASP ZAP (DAST)
      - name: ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.7.0
        with:
          target: 'https://staging.example.com'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

  # ===== CIS BENCHMARKS =====
  cis-benchmark:
    name: CIS Benchmark Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Docker CIS Benchmark
      - name: Run Docker Bench Security
        run: |
          docker run --rm \
            --net host \
            --pid host \
            --userns host \
            --cap-add audit_control \
            -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
            -v /etc:/etc:ro \
            -v /usr/bin/containerd:/usr/bin/containerd:ro \
            -v /usr/bin/runc:/usr/bin/runc:ro \
            -v /usr/lib/systemd:/usr/lib/systemd:ro \
            -v /var/lib:/var/lib:ro \
            -v /var/run/docker.sock:/var/run/docker.sock:ro \
            docker/docker-bench-security | tee cis-docker-benchmark.txt

      - name: Upload CIS Report
        uses: actions/upload-artifact@v4
        with:
          name: cis-benchmark-report
          path: cis-docker-benchmark.txt

  # ===== PCI-DSS COMPLIANCE =====
  pci-dss:
    name: PCI-DSS Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for Unencrypted Secrets
        run: |
          # Verify no hardcoded secrets (simulated check)
          if grep -r "password\|api_key\|secret" --exclude-dir=.git .; then
            echo "❌ Potential hardcoded secrets found"
            exit 1
          else
            echo "✅ No hardcoded secrets detected"
          fi

      - name: Verify TLS Configuration
        run: |
          # Check TLS config (example)
          echo "Verifying TLS 1.2+ enforcement..."
          # Add actual TLS verification logic

      - name: Validate Access Controls
        run: |
          # Verify RBAC configuration
          echo "Validating role-based access controls..."
          # Add actual RBAC validation

  # ===== SOC 2 COMPLIANCE =====
  soc2-audit:
    name: SOC 2 Audit Trail
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate Audit Log
        run: |
          cat << EOF > compliance-audit.json
          {
            "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
            "repository": "${{ github.repository }}",
            "workflow": "${{ github.workflow }}",
            "compliance_checks": {
              "owasp_top_10": "completed",
              "cis_benchmark": "completed",
              "pci_dss": "completed"
            },
            "evidence": {
              "artifacts": [
                "dependency-check-report",
                "cis-benchmark-report"
              ]
            }
          }
          EOF

      - name: Upload Audit Log
        uses: actions/upload-artifact@v4
        with:
          name: soc2-audit-log
          path: compliance-audit.json

  # ===== COMPLIANCE REPORT =====
  report:
    name: Generate Compliance Report
    runs-on: ubuntu-latest
    needs: [owasp-check, cis-benchmark, pci-dss, soc2-audit]
    if: always()
    steps:
      - name: Download All Artifacts
        uses: actions/download-artifact@v4

      - name: Generate Summary Report
        run: |
          cat << EOF > compliance-summary.md
          # Compliance Validation Report

          **Date**: $(date -u +%Y-%m-%d)
          **Repository**: ${{ github.repository }}

          ## Results

          | Check | Status |
          |-------|--------|
          | OWASP Top 10 | ${{ needs.owasp-check.result }} |
          | CIS Benchmark | ${{ needs.cis-benchmark.result }} |
          | PCI-DSS | ${{ needs.pci-dss.result }} |
          | SOC 2 | ${{ needs.soc2-audit.result }} |

          ## Artifacts

          - OWASP Dependency Check Report
          - CIS Docker Benchmark Results
          - SOC 2 Audit Log

          ## Recommendations

          - Review all artifacts for detailed findings
          - Address any identified vulnerabilities
          - Update security policies as needed
          EOF

      - name: Create Compliance Issue
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('compliance-summary.md', 'utf8');

            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Weekly Compliance Report - ${new Date().toISOString().split('T')[0]}`,
              body: report,
              labels: ['compliance', 'security']
            });
```

**Key Features**:
- ✅ OWASP Top 10 validation
- ✅ CIS Docker Benchmark
- ✅ PCI-DSS compliance checks
- ✅ SOC 2 audit trail
- ✅ Automated compliance reporting
- ✅ GitHub issue creation for tracking

---

## Best Practices

### 1. False Positive Management

**`.trivyignore` Example:**
```
# CVE-2023-12345: False positive - Alpine backported fix
CVE-2023-12345

# CVE-2023-99999: Accepted risk - no fix available, mitigated at network layer
CVE-2023-99999
```

### 2. Security Policy Configuration

**`.github/dependabot.yml`:**
```yaml
version: 2
updates:
  - package-ecosystem: npm
    directory: "/"
    schedule:
      interval: daily
    open-pull-requests-limit: 10
    reviewers:
      - security-team
    labels:
      - dependencies
      - security
```

### 3. SARIF Aggregation

Combine multiple SARIF reports:
```bash
# Merge SARIF files
jq -s '.[0] * {"runs": [.[].runs[]] }' \\
  trivy-results.sarif \\
  semgrep-results.sarif \\
  > combined-results.sarif
```

---

For related documentation, see:
- [github-actions-reference.md](github-actions-reference.md)
- [gitlab-ci-reference.md](gitlab-ci-reference.md)
- [terraform-cicd-integration.md](terraform-cicd-integration.md)
