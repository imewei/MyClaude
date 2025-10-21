---
description: Comprehensive multi-dimensional validation with automated testing, security scanning, and ultrathink reasoning
argument-hint: [work-to-validate] [--deep] [--security] [--performance]
color: orange
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Task, SlashCommand
agents:
  primary:
    - multi-agent-orchestrator
    - code-quality
    - code-reviewer
  conditional:
    - agent: research-intelligence
      trigger: pattern "research|paper|publication|methodology"
    - agent: systems-architect
      trigger: pattern "architecture|design.*pattern|system.*design|scalability"
    - agent: security-auditor
      trigger: pattern "security|auth|crypto|secrets|vulnerability"
    - agent: performance-engineer
      trigger: pattern "performance|optimization|bottleneck|latency"
    - agent: test-automator
      trigger: pattern "test|coverage|validation"
  orchestrated: true
---

# Comprehensive Double-Check & Validation

**Systematic multi-dimensional validation with automated verification, security analysis, and ultrathink reasoning**

## Your Task: $ARGUMENTS

## Validation Framework

Execute a comprehensive validation across **10 critical dimensions**:

---

### 1. Scope & Requirements Verification

**First, understand what was supposed to be done:**

1. **Extract Original Requirements**
   - Review conversation history for the original task
   - List all explicit requirements
   - Identify implicit expectations
   - Note any constraints or limitations mentioned

2. **Define "Complete" for This Task**
   - Functional requirements met?
   - Non-functional requirements (performance, security, UX)?
   - Documentation requirements?
   - Testing requirements?

3. **Requirement Traceability**
   - [ ] Every requirement has been addressed
   - [ ] No requirements were misinterpreted
   - [ ] No scope creep introduced unintended features
   - [ ] All acceptance criteria satisfied

---

### 2. Functional Correctness Analysis

**Does it actually work as intended?**

1. **Core Functionality Verification**
   - Test happy path scenarios
   - Verify expected outputs for typical inputs
   - Check integration with existing systems
   - Validate API contracts and interfaces

2. **Edge Case Coverage**
   - Empty inputs (null, undefined, empty strings, empty arrays)
   - Boundary values (min, max, zero, negative)
   - Invalid inputs (wrong types, malformed data)
   - Extreme scale (very large inputs, many concurrent requests)
   - Race conditions and timing issues

3. **Error Handling Robustness**
   - All error paths identified and handled
   - Error messages are clear and actionable
   - Graceful degradation when dependencies fail
   - No silent failures or swallowed exceptions
   - Proper logging at appropriate levels

4. **Automated Testing**
   ```bash
   # Run existing tests
   npm test || pytest || cargo test || go test ./...

   # Check test coverage
   npm run test:coverage || pytest --cov

   # Run integration tests if available
   npm run test:integration || pytest tests/integration
   ```

---

### 3. Code Quality & Maintainability

**Is the code clean, readable, and maintainable?**

1. **Code Review Checklist**
   - [ ] Follows project coding standards and conventions
   - [ ] Consistent naming (variables, functions, classes)
   - [ ] Functions are focused and single-purpose
   - [ ] No code duplication (DRY principle)
   - [ ] Appropriate abstraction levels
   - [ ] No magic numbers or hardcoded values
   - [ ] Proper use of language idioms and patterns

2. **Automated Linting & Formatting**
   ```bash
   # JavaScript/TypeScript
   npx eslint . && npx prettier --check .

   # Python
   ruff check . && black --check .

   # Rust
   cargo clippy && cargo fmt -- --check

   # Go
   golangci-lint run && go fmt ./...
   ```

3. **Complexity Analysis**
   - Cyclomatic complexity reasonable (<10 per function)
   - No deeply nested code (max 3-4 levels)
   - File sizes reasonable (<500 lines)
   - Function sizes reasonable (<50 lines)

4. **Documentation Quality**
   - [ ] Public APIs have docstrings/JSDoc
   - [ ] Complex logic has explanatory comments
   - [ ] README updated if needed
   - [ ] CHANGELOG updated for user-facing changes
   - [ ] Architecture decisions documented (ADRs)

---

### 4. Security Analysis

**Are there any security vulnerabilities?**

1. **Automated Security Scanning**
   ```bash
   # Dependency vulnerabilities
   npm audit || pip-audit || cargo audit

   # Static analysis security testing (SAST)
   semgrep --config=auto . || bandit -r .

   # Secret detection
   gitleaks detect --no-git || trufflehog filesystem .
   ```

2. **Security Checklist**
   - [ ] No secrets in code (API keys, passwords, tokens)
   - [ ] Input validation and sanitization
   - [ ] SQL injection prevention (parameterized queries)
   - [ ] XSS prevention (proper escaping/encoding)
   - [ ] CSRF protection for state-changing operations
   - [ ] Authentication and authorization properly enforced
   - [ ] Secure defaults (fail closed, not open)
   - [ ] No eval() or similar dangerous functions
   - [ ] Proper error handling (no sensitive info in errors)
   - [ ] Dependencies are up to date and vulnerability-free

3. **Data Protection**
   - [ ] Sensitive data encrypted at rest and in transit
   - [ ] PII handling complies with privacy regulations
   - [ ] Proper access control and least privilege
   - [ ] Audit logging for sensitive operations

---

### 5. Performance Analysis

**Is it fast enough? Are there bottlenecks?**

1. **Performance Profiling**
   ```bash
   # JavaScript
   node --prof app.js || clinic doctor -- node app.js

   # Python
   python -m cProfile -o profile.stats script.py

   # Rust
   cargo bench || cargo flamegraph
   ```

2. **Performance Checklist**
   - [ ] No N+1 query problems
   - [ ] Appropriate use of caching
   - [ ] Database indexes on queried fields
   - [ ] Efficient algorithms (no O(n²) where O(n log n) possible)
   - [ ] No unnecessary allocations or copies
   - [ ] Lazy loading where appropriate
   - [ ] Pagination for large datasets
   - [ ] Resource cleanup (connections, file handles, memory)

3. **Load Testing** (if applicable)
   ```bash
   # HTTP endpoints
   wrk -t12 -c400 -d30s http://localhost:3000

   # Or use k6
   k6 run load-test.js
   ```

4. **Bundle Size Analysis** (for frontend)
   ```bash
   # Webpack
   webpack-bundle-analyzer

   # Vite
   npx vite-bundle-visualizer
   ```

---

### 6. Accessibility & User Experience

**Is it usable and accessible?**

1. **Accessibility Audit** (for UI/web)
   ```bash
   # Run axe accessibility testing
   npm run test:a11y || pa11y http://localhost:3000

   # Lighthouse audit
   lighthouse http://localhost:3000 --view
   ```

2. **UX Checklist**
   - [ ] Clear and intuitive interface
   - [ ] Consistent with existing patterns
   - [ ] Responsive design (mobile, tablet, desktop)
   - [ ] Loading states and progress indicators
   - [ ] Error states with recovery options
   - [ ] Keyboard navigation support
   - [ ] Screen reader compatibility
   - [ ] Sufficient color contrast (WCAG AA/AAA)
   - [ ] No reliance on color alone for information

3. **API/CLI UX** (for non-UI)
   - [ ] Clear and consistent naming
   - [ ] Helpful error messages with suggestions
   - [ ] Sensible defaults
   - [ ] Progressive disclosure of complexity
   - [ ] Good documentation with examples

---

### 7. Testing Coverage & Strategy

**Are there adequate tests?**

1. **Test Coverage Analysis**
   ```bash
   # Generate coverage report
   npm run test:coverage || pytest --cov --cov-report=html

   # Check coverage thresholds
   # Aim for: >80% overall, >90% for critical paths
   ```

2. **Test Quality Checklist**
   - [ ] Unit tests for core logic
   - [ ] Integration tests for component interactions
   - [ ] End-to-end tests for critical user flows
   - [ ] Tests are fast and reliable (no flakiness)
   - [ ] Tests use meaningful assertions
   - [ ] Tests follow AAA pattern (Arrange, Act, Assert)
   - [ ] Tests are independent and can run in any order
   - [ ] Mock/stub external dependencies appropriately

3. **Test Coverage Gaps**
   - Identify uncovered code paths
   - Prioritize testing critical/risky code
   - Add missing tests before considering "done"

4. **Mutation Testing** (advanced)
   ```bash
   # JavaScript
   npx stryker run

   # Python
   mutmut run
   ```

---

### 8. Breaking Changes & Backward Compatibility

**Will this break existing functionality?**

1. **API Contract Analysis**
   - [ ] No breaking changes to public APIs
   - [ ] Deprecation warnings for old patterns
   - [ ] Migration guide provided if breaking
   - [ ] Semantic versioning followed

2. **Integration Testing**
   ```bash
   # Run full integration test suite
   npm run test:integration || pytest tests/integration

   # Run end-to-end tests
   npm run test:e2e
   ```

3. **Rollback Plan**
   - [ ] Can roll back safely if issues found
   - [ ] Database migrations are reversible
   - [ ] Feature flags for gradual rollout

---

### 9. Deployment & Operations Readiness

**Is it ready for production?**

1. **Configuration Management**
   - [ ] No hardcoded configuration
   - [ ] Environment-specific configs separated
   - [ ] Secrets managed via vault/env vars
   - [ ] Configuration validation on startup

2. **Observability**
   - [ ] Appropriate logging (not too verbose, not too sparse)
   - [ ] Structured logging format (JSON)
   - [ ] Metrics/telemetry instrumentation
   - [ ] Error tracking integration (Sentry, etc.)
   - [ ] Health check endpoints
   - [ ] Readiness/liveness probes (Kubernetes)

3. **Infrastructure as Code**
   ```bash
   # Validate Terraform/CloudFormation
   terraform validate && terraform plan

   # Validate Kubernetes manifests
   kubectl apply --dry-run=client -f k8s/
   ```

4. **CI/CD Pipeline**
   - [ ] All CI checks passing
   - [ ] Build artifacts created successfully
   - [ ] Deployment scripts tested
   - [ ] Smoke tests pass after deployment

---

### 10. Documentation & Knowledge Transfer

**Can others understand and maintain this?**

1. **Code Documentation**
   - [ ] README with setup instructions
   - [ ] API documentation (OpenAPI/Swagger)
   - [ ] Code comments for complex logic
   - [ ] Examples and usage guides
   - [ ] Troubleshooting section

2. **Architecture Documentation**
   - [ ] Architecture diagrams (if significant changes)
   - [ ] Decision records (ADRs) for key choices
   - [ ] Data models and schemas documented
   - [ ] Dependency relationships clear

3. **Runbooks & Operations**
   - [ ] Deployment guide
   - [ ] Monitoring and alerting guide
   - [ ] Incident response procedures
   - [ ] Common issues and solutions

---

## Comprehensive Validation Execution

### Phase 1: Automated Checks (Run All)

```bash
# 1. Run linters and formatters
echo "🔍 Running linters..."
npm run lint || ruff check . || cargo clippy

# 2. Run all tests with coverage
echo "🧪 Running tests..."
npm run test:coverage || pytest --cov

# 3. Security scanning
echo "🔒 Running security scans..."
npm audit --audit-level=moderate || pip-audit
semgrep --config=auto . || bandit -r .
gitleaks detect --no-git || echo "No secret scanner available"

# 4. Build verification
echo "🏗️ Verifying build..."
npm run build || python -m build || cargo build --release

# 5. Type checking (if applicable)
echo "📝 Type checking..."
npm run type-check || mypy . || cargo check

# 6. Accessibility testing (if web UI)
echo "♿ Accessibility testing..."
npm run test:a11y || pa11y-ci || echo "No a11y tests configured"
```

### Phase 2: Manual Review Checklist

Go through each dimension systematically:

1. **Requirements** ✓/✗
   - All requirements met?
   - Scope appropriate?

2. **Functionality** ✓/✗
   - Works correctly?
   - Edge cases handled?

3. **Code Quality** ✓/✗
   - Clean and maintainable?
   - Follows standards?

4. **Security** ✓/✗
   - No vulnerabilities?
   - Secure by default?

5. **Performance** ✓/✗
   - Fast enough?
   - No bottlenecks?

6. **UX/Accessibility** ✓/✗
   - Intuitive and accessible?
   - Good error handling?

7. **Testing** ✓/✗
   - Adequate coverage?
   - Tests pass reliably?

8. **Compatibility** ✓/✗
   - No breaking changes?
   - Migration path clear?

9. **Operations** ✓/✗
   - Production-ready?
   - Observable and debuggable?

10. **Documentation** ✓/✗
    - Complete and clear?
    - Maintainable by team?

### Phase 3: Gap Analysis & Prioritization

**Critical Gaps** (Block shipment):
- [ ] List any critical issues found
- [ ] Security vulnerabilities
- [ ] Breaking changes without migration
- [ ] Core functionality broken
- [ ] Data loss or corruption risks

**Important Gaps** (Address soon):
- [ ] Missing tests for edge cases
- [ ] Performance concerns
- [ ] Incomplete documentation
- [ ] Minor security issues

**Nice-to-Have Improvements**:
- [ ] Code refactoring opportunities
- [ ] Additional features
- [ ] Documentation enhancements

### Phase 4: Alternative Approaches Analysis

**Question the approach:**
1. **Is there a simpler way?**
   - Could this be done with less code?
   - Are we over-engineering?
   - Can we leverage existing solutions?

2. **Is there a more robust way?**
   - Better error handling patterns?
   - More resilient architecture?
   - Improved testability?

3. **Is there a more performant way?**
   - Different algorithm or data structure?
   - Caching opportunities?
   - Parallelization potential?

4. **Is this the right abstraction?**
   - Appropriate separation of concerns?
   - Good encapsulation?
   - Flexible for future changes?

---

## Final Validation Report

**Provide a structured report:**

### Summary
- Overall assessment: ✅ Ready / ⚠️ Needs work / ❌ Not ready
- Confidence level: High / Medium / Low

### Strengths
- What was done well
- Best practices followed
- Innovative solutions

### Issues Found

#### Critical (Must Fix)
1. [Issue description, location, impact]
2. [Recommended solution]

#### Important (Should Fix)
1. [Issue description, location, impact]
2. [Recommended solution]

#### Minor (Nice to Fix)
1. [Issue description, location, impact]
2. [Recommended solution]

### Recommendations
1. Immediate actions needed
2. Follow-up improvements
3. Long-term considerations

### Verification Evidence
- Tests run: [results]
- Coverage: [percentage]
- Security scans: [results]
- Performance benchmarks: [results]
- Build status: [success/failure]

---

## Advanced Validation Options

### --deep flag
Run extended analysis:
- Property-based testing (QuickCheck, Hypothesis)
- Fuzzing for input validation
- Dependency graph analysis
- Dead code detection
- Cyclomatic complexity metrics

### --security flag
Enhanced security focus:
- OWASP Top 10 validation
- Penetration testing checklist
- Security headers verification
- Certificate validation
- Cryptographic implementation review

### --performance flag
Deep performance analysis:
- CPU profiling
- Memory profiling
- Load testing
- Database query analysis
- Network latency analysis

---

**Execute comprehensive validation across all 10 dimensions, run automated checks, provide detailed findings with specific line numbers and actionable recommendations**
