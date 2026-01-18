# Automated Validation Scripts

Comprehensive automation scripts for linting, testing, security scanning, performance profiling, and build verification across multiple tech stacks.

---

## Master Validation Script

**Complete automation pipeline executing all validation checks:**

```bash
#!/bin/bash
# File: run_all_validations.sh
# Purpose: Execute comprehensive validation across all dimensions

set -e  # Exit on first error

echo "========================================"
echo "COMPREHENSIVE VALIDATION PIPELINE"
echo "========================================"
echo ""

# Track validation results
FAILURES=0

# 1. Linting & Formatting
echo "📋 Step 1/7: Running linters and formatters..."
if ! ./scripts/lint_check.sh; then
    echo "❌ Linting failed"
    ((FAILURES++))
else
    echo "✅ Linting passed"
fi
echo ""

# 2. Type Checking
echo "📝 Step 2/7: Running type checking..."
if ! ./scripts/type_check.sh; then
    echo "❌ Type checking failed"
    ((FAILURES++))
else
    echo "✅ Type checking passed"
fi
echo ""

# 3. Tests with Coverage
echo "🧪 Step 3/7: Running tests with coverage..."
if ! ./scripts/test_runner.sh; then
    echo "❌ Tests failed"
    ((FAILURES++))
else
    echo "✅ Tests passed"
fi
echo ""

# 4. Security Scanning
echo "🔒 Step 4/7: Running security scans..."
if ! ./scripts/security_scan.sh; then
    echo "❌ Security scan found issues"
    ((FAILURES++))
else
    echo "✅ Security scan passed"
fi
echo ""

# 5. Performance Profiling (optional, warning only)
echo "⚡ Step 5/7: Performance profiling..."
./scripts/performance_profiler.sh || echo "⚠️  Performance profiling completed with warnings"
echo ""

# 6. Accessibility Testing (if web UI)
echo "♿ Step 6/7: Accessibility testing..."
if [ -f "./scripts/accessibility_check.sh" ]; then
    if ! ./scripts/accessibility_check.sh; then
        echo "❌ Accessibility check failed"
        ((FAILURES++))
    else
        echo "✅ Accessibility check passed"
    fi
else
    echo "ℹ️  No accessibility tests configured"
fi
echo ""

# 7. Build Verification
echo "🏗️  Step 7/7: Verifying build..."
if ! ./scripts/build_verify.sh; then
    echo "❌ Build verification failed"
    ((FAILURES++))
else
    echo "✅ Build verification passed"
fi
echo ""

# Final Report
echo "========================================"
echo "VALIDATION SUMMARY"
echo "========================================"
if [ $FAILURES -eq 0 ]; then
    echo "✅ ALL VALIDATIONS PASSED"
    exit 0
else
    echo "❌ $FAILURES VALIDATION(S) FAILED"
    exit 1
fi
```

---

## 1. Lint & Format Check

```bash
#!/bin/bash
# File: scripts/lint_check.sh
# Purpose: Run linters and formatters across all supported languages

set -e

echo "🔍 Running code linters and formatters..."

# Detect project type
HAS_JS=$(find . -name "package.json" -not -path "*/node_modules/*" | wc -l)
HAS_PY=$(find . -name "*.py" | wc -l)
HAS_RUST=$(find . -name "Cargo.toml" | wc -l)
HAS_GO=$(find . -name "go.mod" | wc -l)

# JavaScript/TypeScript
if [ $HAS_JS -gt 0 ]; then
    echo "  📦 JavaScript/TypeScript project detected"

    if [ -f "package.json" ]; then
        # ESLint
        if command -v eslint &> /dev/null; then
            echo "    Running ESLint..."
            npx eslint . --ext .js,.jsx,.ts,.tsx --max-warnings 0
        fi

        # Prettier
        if command -v prettier &> /dev/null; then
            echo "    Running Prettier check..."
            npx prettier --check .
        fi

        # TypeScript compiler check
        if [ -f "tsconfig.json" ]; then
            echo "    Running TypeScript compiler check..."
            npx tsc --noEmit
        fi
    fi
fi

# Python
if [ $HAS_PY -gt 0 ]; then
    echo "  🐍 Python project detected"

    # Ruff (fast linter + formatter)
    if command -v ruff &> /dev/null; then
        echo "    Running Ruff check..."
        ruff check .
        echo "    Running Ruff format check..."
        ruff format --check .
    else
        # Fallback: Black + Flake8
        if command -v black &> /dev/null; then
            echo "    Running Black check..."
            black --check .
        fi

        if command -v flake8 &> /dev/null; then
            echo "    Running Flake8..."
            flake8 .
        fi
    fi

    # isort (import sorting)
    if command -v isort &> /dev/null; then
        echo "    Running isort check..."
        isort --check-only .
    fi
fi

# Rust
if [ $HAS_RUST -gt 0 ]; then
    echo "  🦀 Rust project detected"

    # Clippy (linter)
    echo "    Running Clippy..."
    cargo clippy -- -D warnings

    # Rustfmt (formatter)
    echo "    Running Rustfmt check..."
    cargo fmt -- --check
fi

# Go
if [ $HAS_GO -gt 0 ]; then
    echo "  🐹 Go project detected"

    # golangci-lint (comprehensive linter)
    if command -v golangci-lint &> /dev/null; then
        echo "    Running golangci-lint..."
        golangci-lint run
    fi

    # gofmt (formatter)
    echo "    Running gofmt check..."
    gofmt -l . | (! grep .) || (echo "Files need formatting:" && gofmt -l . && exit 1)
fi

echo "✅ Linting and formatting checks passed"
```

---

## 2. Type Checking

```bash
#!/bin/bash
# File: scripts/type_check.sh
# Purpose: Run static type checkers

set -e

echo "📝 Running type checking..."

# TypeScript
if [ -f "tsconfig.json" ]; then
    echo "  TypeScript project detected"
    npx tsc --noEmit
fi

# Python with mypy
if command -v mypy &> /dev/null; then
    if find . -name "*.py" | grep -q .; then
        echo "  Python project detected, running mypy..."
        mypy .
    fi
fi

# Python with pyright (alternative to mypy)
if command -v pyright &> /dev/null; then
    if [ -f "pyrightconfig.json" ]; then
        echo "  Running pyright..."
        pyright
    fi
fi

echo "✅ Type checking passed"
```

---

## 3. Test Runner with Coverage

```bash
#!/bin/bash
# File: scripts/test_runner.sh
# Purpose: Run tests with coverage reporting

set -e

echo "🧪 Running tests with coverage..."

# Detect project type
HAS_JS=$(find . -name "package.json" -not -path "*/node_modules/*" | wc -l)
HAS_PY=$(find . -name "pytest.ini" -o -name "setup.py" | wc -l)
HAS_RUST=$(find . -name "Cargo.toml" | wc -l)
HAS_GO=$(find . -name "go.mod" | wc -l)

# JavaScript/TypeScript (Jest or Vitest)
if [ $HAS_JS -gt 0 ]; then
    echo "  Running JavaScript/TypeScript tests..."

    if grep -q "jest" package.json 2>/dev/null; then
        npm test -- --coverage --coverageThreshold='{"global":{"branches":80,"functions":80,"lines":80,"statements":80}}'
    elif grep -q "vitest" package.json 2>/dev/null; then
        npx vitest run --coverage --coverage.lines=80 --coverage.functions=80
    else
        npm test
    fi
fi

# Python (pytest)
if [ $HAS_PY -gt 0 ]; then
    echo "  Running Python tests..."

    if command -v pytest &> /dev/null; then
        pytest -v \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=html \
            --cov-fail-under=80
    fi
fi

# Rust (cargo test)
if [ $HAS_RUST -gt 0 ]; then
    echo "  Running Rust tests..."
    cargo test --all-features

    # Coverage with tarpaulin (if available)
    if command -v cargo-tarpaulin &> /dev/null; then
        cargo tarpaulin --out Html --output-dir coverage --fail-under 80
    fi
fi

# Go (go test)
if [ $HAS_GO -gt 0 ]; then
    echo "  Running Go tests..."
    go test ./... -v -race -coverprofile=coverage.out -covermode=atomic
    go tool cover -func=coverage.out

    # Check coverage threshold
    COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}' | sed 's/%//')
    if (( $(echo "$COVERAGE < 80" | bc -l) )); then
        echo "❌ Coverage $COVERAGE% is below 80% threshold"
        exit 1
    fi
fi

echo "✅ Tests passed with adequate coverage"
```

---

## 4. Security Scanning

```bash
#!/bin/bash
# File: scripts/security_scan.sh
# Purpose: Comprehensive security vulnerability scanning

set -e

echo "🔒 Running security scans..."

ISSUES_FOUND=0

# 1. Dependency Vulnerability Scanning
echo "  📦 Scanning dependencies for vulnerabilities..."

if [ -f "package.json" ]; then
    echo "    Auditing npm dependencies..."
    if ! npm audit --audit-level=moderate; then
        echo "⚠️  npm audit found vulnerabilities"
        ((ISSUES_FOUND++))
    fi
fi

if find . -name "requirements.txt" -o -name "pyproject.toml" | grep -q .; then
    if command -v pip-audit &> /dev/null; then
        echo "    Auditing Python dependencies..."
        if ! pip-audit; then
            echo "⚠️  pip-audit found vulnerabilities"
            ((ISSUES_FOUND++))
        fi
    fi
fi

if [ -f "Cargo.toml" ]; then
    if command -v cargo-audit &> /dev/null; then
        echo "    Auditing Rust dependencies..."
        if ! cargo audit; then
            echo "⚠️  cargo-audit found vulnerabilities"
            ((ISSUES_FOUND++))
        fi
    fi
fi

if [ -f "go.mod" ]; then
    echo "    Auditing Go dependencies..."
    if ! go list -json -m all | nancy sleuth; then
        echo "⚠️  nancy found vulnerabilities"
        ((ISSUES_FOUND++))
    fi
fi

# 2. Static Application Security Testing (SAST)
echo "  🔍 Running static analysis security testing..."

# Semgrep (multi-language)
if command -v semgrep &> /dev/null; then
    echo "    Running Semgrep..."
    if ! semgrep --config=auto . --error --quiet; then
        echo "⚠️  Semgrep found security issues"
        ((ISSUES_FOUND++))
    fi
fi

# Bandit (Python)
if find . -name "*.py" | grep -q .; then
    if command -v bandit &> /dev/null; then
        echo "    Running Bandit for Python..."
        if ! bandit -r . -ll; then
            echo "⚠️  Bandit found security issues"
            ((ISSUES_FOUND++))
        fi
    fi
fi

# 3. Secret Detection
echo "  🔑 Scanning for exposed secrets..."

# Gitleaks
if command -v gitleaks &> /dev/null; then
    echo "    Running Gitleaks..."
    if ! gitleaks detect --no-git --verbose; then
        echo "⚠️  Gitleaks found exposed secrets"
        ((ISSUES_FOUND++))
    fi
fi

# TruffleHog (alternative)
if command -v trufflehog &> /dev/null; then
    echo "    Running TruffleHog..."
    if ! trufflehog filesystem . --only-verified; then
        echo "⚠️  TruffleHog found exposed secrets"
        ((ISSUES_FOUND++))
    fi
fi

# 4. Container Security (if Dockerfile exists)
if [ -f "Dockerfile" ]; then
    echo "  🐳 Scanning Docker image..."

    # Trivy
    if command -v trivy &> /dev/null; then
        # Build image first
        IMAGE_NAME=$(basename $(pwd)):security-scan
        docker build -t $IMAGE_NAME .

        echo "    Running Trivy on Docker image..."
        if ! trivy image --severity HIGH,CRITICAL --exit-code 1 $IMAGE_NAME; then
            echo "⚠️  Trivy found vulnerabilities in Docker image"
            ((ISSUES_FOUND++))
        fi
    fi
fi

# Final Report
if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✅ No security issues found"
    exit 0
else
    echo "⚠️  Found $ISSUES_FOUND security issue(s)"
    echo "Review the output above and address critical/high severity issues"
    exit 1
fi
```

---

## 5. Performance Profiling

```bash
#!/bin/bash
# File: scripts/performance_profiler.sh
# Purpose: Profile application performance

set -e

echo "⚡ Running performance profiling..."

# Detect project type
HAS_PY=$(find . -name "*.py" | wc -l)
HAS_JS=$(find . -name "package.json" | wc -l)
HAS_RUST=$(find . -name "Cargo.toml" | wc -l)
HAS_GO=$(find . -name "go.mod" | wc -l)

# Python profiling
if [ $HAS_PY -gt 0 ]; then
    echo "  🐍 Profiling Python application..."

    # cProfile
    if [ -f "main.py" ]; then
        echo "    Running cProfile..."
        python -m cProfile -o profile.stats main.py

        # Pretty print top 20 slowest functions
        if command -v python &> /dev/null; then
            python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
        fi
    fi

    # Memory profiling with memory_profiler
    if command -v mprof &> /dev/null; then
        echo "    Running memory profiler..."
        mprof run main.py
        mprof plot -o memory_profile.png
    fi
fi

# Node.js profiling
if [ $HAS_JS -gt 0 ]; then
    echo "  📦 Profiling Node.js application..."

    # Find main entry point
    if [ -f "package.json" ]; then
        MAIN_FILE=$(node -e "console.log(require('./package.json').main || 'index.js')")

        if [ -f "$MAIN_FILE" ]; then
            echo "    Running node --prof..."
            node --prof $MAIN_FILE

            # Process profiling output
            PROF_FILE=$(ls isolate-*.log | head -1)
            if [ -f "$PROF_FILE" ]; then
                node --prof-process $PROF_FILE > cpu_profile.txt
                echo "    CPU profile saved to cpu_profile.txt"
            fi
        fi
    fi

    # Clinic.js (if available)
    if command -v clinic &> /dev/null; then
        echo "    Running Clinic.js doctor..."
        clinic doctor -- node $MAIN_FILE
    fi
fi

# Rust profiling
if [ $HAS_RUST -gt 0 ]; then
    echo "  🦀 Profiling Rust application..."

    # Benchmarks
    echo "    Running cargo bench..."
    cargo bench

    # Flamegraph (if available)
    if command -v cargo-flamegraph &> /dev/null; then
        echo "    Generating flamegraph..."
        cargo flamegraph
    fi
fi

# Go profiling
if [ $HAS_GO -gt 0 ]; then
    echo "  🐹 Profiling Go application..."

    # CPU profiling
    echo "    Running CPU profiling..."
    go test -bench . -cpuprofile cpu.prof
    go tool pprof -http=:8080 cpu.prof &

    # Memory profiling
    echo "    Running memory profiling..."
    go test -bench . -memprofile mem.prof
    go tool pprof -http=:8081 mem.prof &
fi

echo "✅ Performance profiling complete"
echo "Review the generated profile files for optimization opportunities"
```

---

## 6. Accessibility Testing

```bash
#!/bin/bash
# File: scripts/accessibility_check.sh
# Purpose: Automated accessibility testing for web applications

set -e

echo "♿ Running accessibility tests..."

# Check if web application
if [ ! -f "package.json" ]; then
    echo "ℹ️  No package.json found, skipping accessibility tests"
    exit 0
fi

# Start dev server in background
echo "  Starting development server..."
npm run dev &
DEV_PID=$!

# Wait for server to be ready
echo "  Waiting for server to be ready..."
sleep 5

# Base URL (customize as needed)
BASE_URL="http://localhost:3000"

# pa11y (automated accessibility testing)
if command -v pa11y &> /dev/null; then
    echo "  Running pa11y tests..."

    # Test home page
    if ! pa11y --runner axe --threshold 0 $BASE_URL; then
        echo "❌ Accessibility issues found on home page"
        kill $DEV_PID
        exit 1
    fi

    # Test additional pages if pa11y-ci config exists
    if [ -f ".pa11yci.json" ]; then
        if ! pa11y-ci; then
            echo "❌ Accessibility issues found"
            kill $DEV_PID
            exit 1
        fi
    fi
fi

# Lighthouse accessibility audit
if command -v lighthouse &> /dev/null; then
    echo "  Running Lighthouse accessibility audit..."

    lighthouse $BASE_URL \
        --only-categories=accessibility \
        --output=json \
        --output-path=./lighthouse-accessibility.json \
        --quiet

    # Check if score is above threshold (90+)
    SCORE=$(node -e "console.log(require('./lighthouse-accessibility.json').categories.accessibility.score * 100)")
    if (( $(echo "$SCORE < 90" | bc -l) )); then
        echo "❌ Accessibility score $SCORE is below 90 threshold"
        kill $DEV_PID
        exit 1
    fi

    echo "✅ Accessibility score: $SCORE/100"
fi

# Kill dev server
kill $DEV_PID

echo "✅ Accessibility tests passed"
```

---

## 7. Build Verification

```bash
#!/bin/bash
# File: scripts/build_verify.sh
# Purpose: Verify that the project builds successfully

set -e

echo "🏗️  Verifying build..."

# Detect project type
HAS_JS=$(find . -name "package.json" -not -path "*/node_modules/*" | wc -l)
HAS_PY=$(find . -name "setup.py" -o -name "pyproject.toml" | wc -l)
HAS_RUST=$(find . -name "Cargo.toml" | wc -l)
HAS_GO=$(find . -name "go.mod" | wc -l)

# JavaScript/TypeScript
if [ $HAS_JS -gt 0 ]; then
    echo "  📦 Building JavaScript/TypeScript project..."

    if grep -q "build" package.json 2>/dev/null; then
        npm run build
    elif grep -q "vite" package.json 2>/dev/null; then
        npx vite build
    elif [ -f "tsconfig.json" ]; then
        npx tsc
    fi

    # Check build output size
    if [ -d "dist" ] || [ -d "build" ]; then
        BUILD_DIR=$([ -d "dist" ] && echo "dist" || echo "build")
        BUILD_SIZE=$(du -sh $BUILD_DIR | awk '{print $1}')
        echo "  Build size: $BUILD_SIZE"

        # Warn if build is too large (>5MB for typical SPA)
        BUILD_SIZE_BYTES=$(du -sb $BUILD_DIR | awk '{print $1}')
        if [ $BUILD_SIZE_BYTES -gt 5242880 ]; then
            echo "⚠️  Build size $BUILD_SIZE exceeds 5MB"
            echo "Consider code splitting or optimization"
        fi
    fi
fi

# Python
if [ $HAS_PY -gt 0 ]; then
    echo "  🐍 Building Python package..."

    if command -v python &> /dev/null; then
        # Build with modern build backend
        python -m build

        # Verify wheel was created
        if [ ! -d "dist" ] || [ -z "$(ls -A dist/*.whl 2>/dev/null)" ]; then
            echo "❌ No wheel file created"
            exit 1
        fi

        echo "✅ Python package built successfully"
    fi
fi

# Rust
if [ $HAS_RUST -gt 0 ]; then
    echo "  🦀 Building Rust project..."

    # Build in release mode
    cargo build --release --all-features

    # Check binary size
    if [ -d "target/release" ]; then
        for binary in target/release/*; do
            if [ -x "$binary" ] && [ ! -d "$binary" ]; then
                BIN_SIZE=$(du -h "$binary" | awk '{print $1}')
                echo "  Binary size: $(basename $binary) - $BIN_SIZE"
            fi
        done
    fi
fi

# Go
if [ $HAS_GO -gt 0 ]; then
    echo "  🐹 Building Go project..."

    # Build all packages
    go build ./...

    # Build main binary if exists
    if [ -f "main.go" ]; then
        go build -o ./bin/app main.go

        BIN_SIZE=$(du -h ./bin/app | awk '{print $1}')
        echo "  Binary size: $BIN_SIZE"
    fi
fi

echo "✅ Build verification passed"
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/validation.yml
name: Comprehensive Validation

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up environment
        run: |
          # Install dependencies based on project type
          if [ -f "package.json" ]; then npm ci; fi
          if [ -f "requirements.txt" ]; then uv uv pip install -r requirements.txt; fi

      - name: Run comprehensive validation
        run: |
          chmod +x scripts/*.sh
          ./scripts/run_all_validations.sh

      - name: Upload coverage reports
        if: always()
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml,./coverage/lcov.info

      - name: Upload security scan results
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: security-scan-results
          path: |
            semgrep-report.json
            bandit-report.txt
```

---

## Pre-Commit Hook Integration

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit validation..."

# Quick validation (lint + type check only)
./scripts/lint_check.sh
./scripts/type_check.sh

# Optional: Run fast unit tests
# ./scripts/test_runner.sh --fast

if [ $? -eq 0 ]; then
    echo "✅ Pre-commit validation passed"
    exit 0
else
    echo "❌ Pre-commit validation failed"
    echo "Fix the issues above before committing"
    exit 1
fi
```

Install pre-commit hook:
```bash
chmod +x .git/hooks/pre-commit
```

Or use husky for JavaScript projects:
```bash
npx husky install
npx husky add .husky/pre-commit "./scripts/lint_check.sh && ./scripts/type_check.sh"
```

---

## Summary

**Available Scripts:**
- `run_all_validations.sh` - Master script running all validations
- `lint_check.sh` - Linting and formatting across all languages
- `type_check.sh` - Static type checking
- `test_runner.sh` - Tests with coverage reporting
- `security_scan.sh` - Comprehensive security scanning
- `performance_profiler.sh` - Performance profiling
- `accessibility_check.sh` - Web accessibility testing
- `build_verify.sh` - Build verification

**Usage:**
```bash
# Full validation
./scripts/run_all_validations.sh

# Individual checks
./scripts/lint_check.sh
./scripts/security_scan.sh
```

**Best Practices:**
- Run full validation before every pull request
- Integrate into CI/CD pipeline
- Use pre-commit hooks for quick checks
- Review and fix issues iteratively
- Keep scripts updated with new tools and standards
