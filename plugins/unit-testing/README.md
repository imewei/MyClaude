# Unit Testing

Production-grade test automation and debugging with 3 execution modes (quick/standard/enterprise), AI-driven RCA, comprehensive examples, and TDD excellence. Features iterative test fixing, automated test generation with scientific computing support, and 8 comprehensive external documentation files.

**Version:** 1.0.6 | **Category:** development | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/unit-testing.html)

## What's New in v1.0.6

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## ✨ What's New in v1.0.6

### Command Optimization & Execution Modes

**Both commands optimized** following the proven pattern from quality-engineering and systems-programming v1.0.3:

#### /run-all-tests Command
- **Optimized:** 1,469 → 492 lines (66.5% reduction)
- **3 Execution Modes:**
  - **Quick (30min-1h):** Single test file, max 3 iterations, basic error analysis
  - **Standard (2-4h):** Full suite, max 10 iterations, AI-assisted RCA, >80% coverage
  - **Enterprise (1-2d):** Entire codebase, unlimited iterations, >90% coverage, mutation testing

#### /test-generate Command
- **Optimized:** 643 → 552 lines (14.2% reduction)
- **3 Execution Modes:**
  - **Quick (30min-1h):** Single module, ~50-100 test cases, unit tests only
  - **Standard (2-4h):** Package/feature, ~200-500 test cases, unit + integration + property-based
  - **Enterprise (1-2d):** Entire project, ~1,000+ test cases, full suite + mutation + docs

### External Documentation (8 files - 5,858 lines)

**For /run-all-tests:**
- `framework-detection-guide.md` - Auto-detection for Jest, pytest, cargo, go test, Maven, Gradle
- `debugging-strategies.md` - AI-driven RCA, log correlation, flaky test detection
- `test-execution-workflows.md` - Iterative patterns, parallel strategies, CI/CD integration
- `multi-language-testing.md` - Cross-language patterns, monorepo strategies

**For /test-generate:**
- `test-generation-patterns.md` - AST parsing, test algorithms, mocking strategies
- `scientific-testing-guide.md` - JAX gradient verification, NumPy numerical validation, PyTorch autograd
- `property-based-testing.md` - Hypothesis patterns, QuickCheck, stateful testing
- `coverage-analysis-guide.md` - Coverage metrics, gap identification, pytest-cov/istanbul

### Key Improvements

- **Token Efficiency:** 50.6% command file reduction (2,112 → 1,044 lines)
- **Critical Fix:** Registered `e2e-testing-patterns` skill (was missing from plugin.json)
- **Scientific Computing:** JAX, NumPy, PyTorch test generation with gradient verification
- **Multi-Language:** Python, JavaScript/TypeScript, Rust, Go, Java, Julia support
- **Version Consistency:** All commands, agents, and skills now versioned 1.0.3

### Execution Mode Tables

#### /run-all-tests Modes

| Mode | Duration | Agents | Iterations | Coverage | Use Case |
|------|----------|--------|------------|----------|----------|
| Quick | 30min-1h | 1 | Max 3 | Basic | Rapid debugging |
| Standard | 2-4h | 2 | Max 10 | >80% | Pre-commit validation |
| Enterprise | 1-2d | 3 | Unlimited | >90% | Release validation |

#### /test-generate Modes

| Mode | Duration | Agents | Output | Test Types | Use Case |
|------|----------|--------|--------|------------|----------|
| Quick | 30min-1h | 1 | ~50-100 | Unit only | TDD workflow |
| Standard | 2-4h | 2 | ~200-500 | Unit + integration + property-based | Production suites |
| Enterprise | 1-2d | 3 | ~1,000+ | Full suite + mutation + docs | New projects |

[View Full Changelog →](./CHANGELOG.md)

## Agents (2)

### debugger

**Status:** active

AI-assisted debugging specialist with systematic 8-step process, LLM-driven root cause analysis, automated log correlation, distributed tracing integration, and performance profiling.

**Key Capabilities:**
- 8-step systematic debugging workflow
- AI-driven root cause analysis with LLM
- Automated log correlation across services
- Distributed tracing with OpenTelemetry/Jaeger
- Binary search and hypothesis-driven debugging
- Race condition and flaky test resolution
- Performance profiling with flame graphs
- Multi-language debugging (Python, JS/TS, Java, Go, Rust, C/C++)

**Example Usage:**
```python
# Debug intermittent test failure with race condition
# Agent systematically:
# 1. Captures context (test fails 20% of time)
# 2. Reproduces issue (non-atomic increment)
# 3. Forms hypothesis (race condition)
# 4. Tests with logging (confirms interleaving)
# 5. Isolates root cause (missing lock)
# 6. Implements minimal fix (add threading.Lock)
# 7. Verifies with stress test (1000 runs, 100% pass)
# 8. Prevents recurrence (adds documentation, stress test)
```

**Comprehensive Examples:**
- Intermittent test failure debugging (race condition with threading)
- Distributed system log correlation (API 500 → DB index, 417× speedup)
- Common debugging antipatterns and fixes

---

### test-automator

**Status:** active

Master test automation engineer with systematic 8-step process, TDD excellence, AI-powered testing, and comprehensive quality engineering.

**Key Capabilities:**
- 8-step systematic test automation workflow
- TDD red-green-refactor cycle mastery
- Page Object Model for maintainable UI tests
- Property-based testing with hypothesis
- Self-healing test automation
- CI/CD pipeline integration
- Test pyramid optimization (70/20/10)
- Comprehensive quality engineering

**Example Usage:**
```typescript
// E2E test with Page Object Pattern
const loginPage = new LoginPage(page);
const dashboardPage = new DashboardPage(page);

await loginPage.navigate();
await loginPage.login('test@example.com', 'password');
await expect(page).toHaveURL('/dashboard');
await dashboardPage.isLoaded();
```

```python
# TDD with property-based testing
@given(st.emails())
def test_all_emails_validated(email):
    assert validate_email(email) is True

# Automatically tests 200+ generated email addresses
# Finds edge cases: consecutive dots, boundary conditions
# Coverage: 100%, Bugs found: 2
```

**Comprehensive Examples:**
- E2E test with Page Object Pattern (Login + Dashboard, Playwright)
- TDD cycle with property-based testing (email validation, hypothesis)
- Common test automation antipatterns and fixes

## Commands (2)

### `/test-generate`

**Status:** active

Generate comprehensive test suites with scientific computing support, TDD patterns, and property-based testing

**Features:**
- Test generation for Python, Julia, JAX scientific computing
- Property-based testing with hypothesis/proptest
- TDD red-green-refactor patterns
- Numerical validation and benchmarks

### `/run-all-tests`

**Status:** active

Iteratively run and fix all tests until zero failures and 100% pass rate with AI-assisted debugging

**Features:**
- Automatic test execution and failure analysis
- AI-driven root cause analysis for failures
- Iterative fixing until 100% pass rate
- Flaky test detection and stabilization
- Coverage tracking and reporting

## Key Features

### Systematic Debugging Process

The debugger agent follows an 8-step workflow:
1. **Capture Context** - Errors, logs, traces, environment
2. **Reproduce Issue** - Minimal reproduction case
3. **Form Hypotheses** - Ranked list of potential causes
4. **Test Systematically** - Validate each hypothesis
5. **Isolate Root Cause** - Pinpoint exact failure
6. **Implement Fix** - Minimal, targeted solution
7. **Verify Resolution** - Comprehensive validation
8. **Prevent Recurrence** - Tests, monitoring, documentation

### Systematic Test Automation Process

The test-automator agent follows an 8-step workflow:
1. **Analyze Requirements** - Objectives, scope, risk areas
2. **Design Strategy** - Framework selection, test pyramid
3. **Implement Automation** - Scalable, maintainable tests
4. **Integrate CI/CD** - Automated execution, quality gates
5. **Comprehensive Coverage** - Happy paths, edge cases, negative tests
6. **Monitoring & Reporting** - Dashboards, flaky test detection
7. **Optimize Performance** - Parallel execution, sharding
8. **Plan Maintenance** - Documentation, contribution guidelines

### Quality Assurance Checkpoints

**debugger principles:**
- Root Cause Identified (not symptoms)
- Evidence-Based diagnosis
- Minimal Fix approach
- Test Coverage for prevention
- No Regressions
- Performance maintained
- Documentation complete
- Monitoring improved

**test-automator principles:**
- Test Coverage (critical paths)
- Test Reliability (>99% stability)
- Test Speed (<10 min)
- Test Maintainability (DRY)
- Test Clarity (descriptive names)
- Test Isolation (independent)
- Test Value (behavior, not implementation)
- CI/CD Integration (automated)

### Strategic Ambiguity Handling

**debugger questions (16 across 4 domains):**
- Error Context & Environment
- Reproduction & Frequency
- System State & Dependencies
- Impact & Urgency

**test-automator questions (16 across 4 domains):**
- Testing Scope & Requirements
- Test Framework & Tools
- Test Environment & Data
- Success Criteria & Constraints

### Comprehensive Examples

Each agent includes:
- **Good Example**: Production-ready implementation with best practices
- **Bad Example**: Common antipatterns to avoid with explanations
- **Annotated Example**: Step-by-step walkthrough with detailed reasoning

### Common Patterns Library

**debugger patterns:**
- Binary Search Debugging (git bisect, divide-and-conquer)
- Log Correlation Analysis (distributed tracing timeline)
- Performance Profiling Investigation (flame graphs, hotspots)

**test-automator patterns:**
- Page Object Model for UI Tests (encapsulation, maintainability)
- Test Data Factories and Fixtures (consistency, cleanup)
- TDD Red-Green-Refactor Cycle (test-first development)

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `unit-testing` plugin
3. Activate an agent (e.g., `@debugger` or `@test-automator`)
4. Try a command (e.g., `/test-generate` or `/run-all-tests`)

**Example debugging workflow:**
```bash
# Activate debugger agent
@debugger

# Ask for help with failing test
"This test fails intermittently in CI with 'expected 3, got 2'.
It involves concurrent threads incrementing a counter."

# The agent will:
# 1. Capture full context (test code, CI logs, frequency)
# 2. Form hypothesis (race condition in increment)
# 3. Design experiment (add logging to confirm)
# 4. Identify root cause (missing synchronization)
# 5. Implement minimal fix (add threading.Lock)
# 6. Verify with stress test (1000 runs)
# 7. Prevent recurrence (documentation + stress test)
```

**Example test automation workflow:**
```bash
# Activate test-automator agent
@test-automator

# Ask for comprehensive test suite
"Generate E2E tests for user authentication flow with page objects"

# The agent will:
# 1. Analyze requirements (login, error handling, navigation)
# 2. Design strategy (Playwright, Page Object pattern)
# 3. Create page objects (LoginPage, DashboardPage)
# 4. Implement tests (success, errors, edge cases)
# 5. Add CI/CD integration (parallel execution)
# 6. Set up monitoring (flaky test detection)
```

## Integration

See the full documentation for integration patterns and compatible plugins.

### Compatible Plugins
- **debugging-toolkit**: AI-assisted debugging and observability patterns
- **systems-programming**: Testing low-level systems code
- **cicd-automation**: CI/CD pipeline integration for automated testing
- **performance-engineering**: Performance testing and profiling

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/unit-testing.html)

To build documentation locally:

```bash
cd docs/
make html
```

## Advanced Features

### AI-Driven Root Cause Analysis
- LLM-powered error pattern recognition
- Automated hypothesis generation
- Cross-reference with historical bugs
- Probabilistic ranking of causes

### Distributed Tracing Integration
- OpenTelemetry, Jaeger, Zipkin support
- Span analysis for latency attribution
- Multi-service log correlation
- Cascading failure detection

### Test-Driven Development Excellence
- Red-Green-Refactor cycle automation
- Property-based testing with hypothesis
- TDD metrics tracking (cycle time, test growth)
- Chicago School and London School TDD

### Self-Healing Test Automation
- AI-powered test maintenance
- Dynamic element locators
- Intelligent test retries
- Flaky test detection and stabilization

### Performance Optimization
- Parallel test execution with sharding
- Smart test selection based on code changes
- Test result caching
- Containerized test environments
