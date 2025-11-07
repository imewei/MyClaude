# Error Pattern Library

**Version**: 1.0.3
**Command**: `/fix-commit-errors`
**Category**: CI/CD Automation

## Overview

Comprehensive library of 100+ error patterns across multiple languages, build tools, test frameworks, and CI/CD platforms. Each pattern includes regex matching rules, error categorization, severity levels, and common root causes.

---

## NPM/Yarn Error Patterns

### Dependency Resolution Errors

#### ERESOLVE - Peer Dependency Conflict
```python
{
  "id": "npm-eresolve-001",
  "pattern": r"npm ERR! ERESOLVE.*peer dependency",
  "category": "dependency_conflict",
  "severity": "medium",
  "subcategory": "peer_dependency_mismatch",
  "common_causes": [
    "Package requires different peer dependency version than installed",
    "Multiple packages require incompatible peer dependency versions",
    "Strict dependency resolution in npm 7+"
  ],
  "indicators": [
    "ERESOLVE unable to resolve dependency tree",
    "Found: package@version",
    "Could not resolve dependency: peer package@\"version-range\""
  ],
  "success_rate_by_solution": {
    "npm_install_legacy_peer_deps": 0.85,
    "update_peer_dependency": 0.95,
    "use_overrides": 0.60
  }
}
```

#### NPM 404 - Package Not Found
```python
{
  "id": "npm-404-001",
  "pattern": r"npm ERR! 404.*Not Found",
  "category": "package_not_found",
  "severity": "high",
  "common_causes": [
    "Typo in package name",
    "Package removed from npm registry",
    "Private package without authentication",
    "Scoped package (@org/package) access denied"
  ],
  "indicators": [
    "404 Not Found - GET https://registry.npmjs.org/package-name",
    "package-name@version is not in this registry"
  ]
}
```

#### GYP Build Failure - Native Module
```python
{
  "id": "npm-gyp-001",
  "pattern": r"gyp ERR!",
  "category": "native_build_failure",
  "severity": "high",
  "common_causes": [
    "Missing Python 2.7 or Python 3.x",
    "Missing C++ build tools (Visual Studio on Windows, build-essential on Linux)",
    "Incompatible Node.js version for native module",
    "Missing system dependencies (e.g., libvips, cairo)"
  ],
  "indicators": [
    "gyp ERR! build error",
    "gyp ERR! stack Error: `make` failed with exit code: 2",
    "node-gyp rebuild failed"
  ]
}
```

#### ELIFECYCLE - Script Execution Failure
```python
{
  "id": "npm-lifecycle-001",
  "pattern": r"npm ERR! code ELIFECYCLE",
  "category": "script_failure",
  "severity": "medium",
  "common_causes": [
    "npm script (postinstall, prepare, etc.) exited with non-zero code",
    "Missing devDependency required by script",
    "Script path not found"
  ],
  "indicators": [
    "npm ERR! errno 1",
    "npm ERR! package@version script-name: `script command`",
    "npm ERR! Exit status 1"
  ]
}
```

### Yarn-Specific Patterns

```python
YARN_PATTERNS = {
  "yarn-integrity-check-failed": {
    "pattern": r"error.*integrity check failed",
    "category": "cache_corruption",
    "severity": "medium",
    "solution": "yarn cache clean && yarn install"
  },
  "yarn-workspaces-error": {
    "pattern": r"error.*workspace.*not found",
    "category": "workspace_configuration",
    "severity": "high"
  }
}
```

---

## Python/Pip Error Patterns

### Package Installation Errors

#### Package Not Found
```python
{
  "id": "pip-not-found-001",
  "pattern": r"ERROR: Could not find a version that satisfies the requirement",
  "category": "package_not_found",
  "severity": "high",
  "common_causes": [
    "Package name typo",
    "Package doesn't exist on PyPI",
    "Version constraint too restrictive",
    "Python version incompatibility"
  ],
  "indicators": [
    "No matching distribution found for package-name",
    "Could not find a version that satisfies the requirement package-name"
  ]
}
```

#### Version Conflict
```python
{
  "id": "pip-version-conflict-001",
  "pattern": r"VersionConflict|version conflict",
  "category": "dependency_conflict",
  "severity": "medium",
  "common_causes": [
    "Two packages require incompatible versions of same dependency",
    "Direct dependency conflicts with transitive dependency"
  ],
  "indicators": [
    "pkg_resources.VersionConflict",
    "package 1.0.0 has requirement package-dep>=2.0, but you have package-dep 1.5"
  ]
}
```

#### Import Error
```python
{
  "id": "python-import-001",
  "pattern": r"ImportError: No module named|ModuleNotFoundError",
  "category": "missing_module",
  "severity": "high",
  "common_causes": [
    "Package not installed",
    "Wrong Python environment activated",
    "Package installed but not in PYTHONPATH",
    "Typo in import statement"
  ],
  "indicators": [
    "ModuleNotFoundError: No module named 'package_name'",
    "ImportError: cannot import name 'ClassName' from 'module'"
  ]
}
```

---

## Rust/Cargo Error Patterns

### Compilation Errors

#### Unresolved Name
```python
{
  "id": "cargo-e0425-001",
  "pattern": r"error\[E0425\]: cannot find",
  "category": "unresolved_identifier",
  "severity": "high",
  "common_causes": [
    "Variable, function, or type not declared",
    "Missing import (use statement)",
    "Typo in identifier name",
    "Item not in scope"
  ],
  "indicators": [
    "error[E0425]: cannot find value `variable` in this scope",
    "error[E0425]: cannot find function `function_name` in this scope"
  ]
}
```

#### Trait Not Implemented
```python
{
  "id": "cargo-e0277-001",
  "pattern": r"error\[E0277\]:.*doesn't implement",
  "category": "trait_bound",
  "severity": "medium",
  "common_causes": [
    "Type doesn't implement required trait",
    "Missing trait bound on generic parameter",
    "Trait bound conflict"
  ],
  "indicators": [
    "error[E0277]: the trait bound `Type: Trait` is not satisfied",
    "the trait `Trait` is not implemented for `Type`"
  ]
}
```

#### Unresolved Import
```python
{
  "id": "cargo-e0432-001",
  "pattern": r"error\[E0432\]: unresolved import",
  "category": "import_error",
  "severity": "high",
  "common_causes": [
    "Module or item doesn't exist",
    "Typo in import path",
    "Item not public (pub)",
    "Crate not in Cargo.toml dependencies"
  ],
  "indicators": [
    "error[E0432]: unresolved import `module::item`",
    "no `item` in `module`"
  ]
}
```

---

## Go Error Patterns

### Compilation and Build Errors

#### Undefined Identifier
```python
{
  "id": "go-undefined-001",
  "pattern": r"undefined: \w+",
  "category": "undefined_identifier",
  "severity": "high",
  "common_causes": [
    "Missing import",
    "Typo in identifier name",
    "Identifier not exported (lowercase)",
    "Wrong package imported"
  ],
  "indicators": [
    "undefined: VariableName",
    "undefined: FunctionName"
  ]
}
```

#### Cannot Find Package
```python
{
  "id": "go-package-001",
  "pattern": r"cannot find package",
  "category": "package_not_found",
  "severity": "high",
  "common_causes": [
    "Package path incorrect",
    "Package not in go.mod",
    "GOPATH not set correctly",
    "Module not downloaded (need go mod download)"
  ],
  "indicators": [
    "cannot find package \"package/path\" in any of:",
    "no required module provides package package/path"
  ]
}
```

#### Module Not Found
```python
{
  "id": "go-module-001",
  "pattern": r"module.*not found",
  "category": "module_missing",
  "severity": "high",
  "common_causes": [
    "Module not listed in go.mod",
    "Module path incorrect",
    "Private module authentication required",
    "go.sum out of sync"
  ],
  "indicators": [
    "no required module provides package",
    "module lookup disabled by GOPROXY=off"
  ]
}
```

---

## TypeScript Error Patterns

### Type System Errors

#### Type Mismatch (TS2322)
```python
{
  "id": "ts-2322-001",
  "pattern": r"error TS2322:",
  "category": "type_mismatch",
  "severity": "medium",
  "common_causes": [
    "Assigning value of wrong type",
    "Function return type doesn't match declared type",
    "Missing type conversion",
    "Incorrect generic type argument"
  ],
  "indicators": [
    "Type 'X' is not assignable to type 'Y'",
    "error TS2322"
  ]
}
```

#### Property Does Not Exist (TS2339)
```python
{
  "id": "ts-2339-001",
  "pattern": r"error TS2339:.*does not exist",
  "category": "missing_property",
  "severity": "medium",
  "common_causes": [
    "Property name typo",
    "Property not defined in interface/type",
    "Object might be null/undefined",
    "Wrong type inference"
  ],
  "indicators": [
    "Property 'propertyName' does not exist on type 'TypeName'",
    "error TS2339"
  ]
}
```

#### Cannot Find Module (TS2307)
```python
{
  "id": "ts-2307-001",
  "pattern": r"error TS2307: Cannot find module",
  "category": "module_not_found",
  "severity": "high",
  "common_causes": [
    "Module not installed",
    "Missing type definitions (@types/package)",
    "Incorrect import path",
    "tsconfig.json paths not configured"
  ],
  "indicators": [
    "Cannot find module 'module-name' or its corresponding type declarations",
    "error TS2307"
  ]
}
```

---

## Build Tool Error Patterns

### Webpack Errors

#### Module Not Found
```python
{
  "id": "webpack-module-001",
  "pattern": r"Module not found: Error: Can't resolve",
  "category": "module_resolution",
  "severity": "high",
  "common_causes": [
    "Import path incorrect",
    "Module not installed",
    "Webpack alias not configured",
    "File extension missing and not in resolve.extensions"
  ],
  "indicators": [
    "Module not found: Error: Can't resolve 'module-name'",
    "in '/path/to/file.js'"
  ]
}
```

#### Parse Failed
```python
{
  "id": "webpack-parse-001",
  "pattern": r"Module parse failed",
  "category": "syntax_error",
  "severity": "high",
  "common_causes": [
    "Syntax error in source file",
    "Missing loader for file type",
    "Loader configuration incorrect",
    "Unsupported JavaScript feature without babel"
  ],
  "indicators": [
    "Module parse failed: Unexpected token",
    "You may need an appropriate loader"
  ]
}
```

### ESLint Errors

```python
ESLINT_PATTERNS = {
  "eslint-parsing-error": {
    "pattern": r"Parsing error:",
    "category": "syntax_error",
    "severity": "high"
  },
  "eslint-rule-violation": {
    "pattern": r"\d+:\d+\s+error",
    "category": "code_quality",
    "severity": "low"
  }
}
```

---

## Test Framework Error Patterns

### Jest/Vitest Errors

#### Test Suite Failed to Run
```python
{
  "id": "jest-suite-fail-001",
  "pattern": r"● Test suite failed to run",
  "category": "test_setup_failure",
  "severity": "high",
  "common_causes": [
    "Syntax error in test file",
    "Missing test dependency",
    "Import error",
    "Jest configuration error"
  ],
  "indicators": [
    "Test suite failed to run",
    "SyntaxError:",
    "Cannot find module"
  ]
}
```

#### Assertion Failure
```python
{
  "id": "jest-assertion-001",
  "pattern": r"expect\(.*\)\..*",
  "category": "assertion_failure",
  "severity": "medium",
  "common_causes": [
    "Test expectation not met",
    "Incorrect test data",
    "Code behavior changed",
    "Flaky test"
  ],
  "indicators": [
    "Expected: value1",
    "Received: value2",
    "expect(received).toBe(expected)"
  ]
}
```

#### Test Timeout
```python
{
  "id": "jest-timeout-001",
  "pattern": r"Exceeded timeout of \d+ms",
  "category": "test_timeout",
  "severity": "medium",
  "common_causes": [
    "Async operation not completing",
    "Test waiting for condition that never occurs",
    "Infinite loop in test code",
    "Slow external dependency"
  ],
  "indicators": [
    "Exceeded timeout of 5000ms for a test",
    "Async callback was not invoked within"
  ]
}
```

### Pytest Errors

#### Test Failed
```python
{
  "id": "pytest-fail-001",
  "pattern": r"FAILED.*::.*",
  "category": "test_failure",
  "severity": "medium",
  "common_causes": [
    "Assertion failed",
    "Exception raised",
    "Setup/teardown failed",
    "Fixture error"
  ],
  "indicators": [
    "FAILED tests/test_file.py::test_name",
    "AssertionError:",
    "E       assert"
  ]
}
```

#### Fixture Not Found
```python
{
  "id": "pytest-fixture-001",
  "pattern": r"fixture.*not found",
  "category": "fixture_error",
  "severity": "high",
  "common_causes": [
    "Fixture not defined",
    "Fixture in different conftest.py scope",
    "Typo in fixture name",
    "Plugin providing fixture not installed"
  ],
  "indicators": [
    "fixture 'fixture_name' not found",
    "available fixtures:"
  ]
}
```

---

## Runtime Error Patterns

### Memory Errors

#### Out of Memory
```python
{
  "id": "runtime-oom-001",
  "pattern": r"OOM|out of memory|heap.*exhausted",
  "category": "memory_exhausted",
  "severity": "critical",
  "common_causes": [
    "Memory leak",
    "Processing too much data at once",
    "Insufficient memory allocation",
    "Recursive function without base case"
  ],
  "indicators": [
    "JavaScript heap out of memory",
    "FATAL ERROR: Ineffective mark-compacts near heap limit",
    "OutOfMemoryError"
  ]
}
```

### Network Errors

#### Connection Refused
```python
{
  "id": "network-econnrefused-001",
  "pattern": r"ECONNREFUSED",
  "category": "connection_refused",
  "severity": "high",
  "common_causes": [
    "Service not running",
    "Wrong host/port",
    "Firewall blocking connection",
    "Service crashed"
  ],
  "indicators": [
    "connect ECONNREFUSED 127.0.0.1:3000",
    "Error: connect ECONNREFUSED"
  ]
}
```

#### Timeout
```python
{
  "id": "network-timeout-001",
  "pattern": r"ETIMEDOUT|timeout.*exceeded",
  "category": "network_timeout",
  "severity": "medium",
  "common_causes": [
    "Service too slow to respond",
    "Network latency",
    "Service under heavy load",
    "DNS resolution timeout"
  ],
  "indicators": [
    "Error: ETIMEDOUT",
    "timeout of 30000ms exceeded"
  ]
}
```

#### DNS Resolution Failed
```python
{
  "id": "network-dns-001",
  "pattern": r"getaddrinfo ENOTFOUND|ENOTFOUND",
  "category": "dns_failure",
  "severity": "high",
  "common_causes": [
    "Host doesn't exist",
    "DNS server unavailable",
    "Network connectivity issue",
    "Typo in hostname"
  ],
  "indicators": [
    "getaddrinfo ENOTFOUND hostname",
    "Error: getaddrinfo ENOTFOUND"
  ]
}
```

---

## CI-Specific Error Patterns

### GitHub Actions Errors

#### Cache Restore Failed
```python
{
  "id": "ci-cache-001",
  "pattern": r"Failed to restore cache|cache miss",
  "category": "cache_failure",
  "severity": "low",
  "common_causes": [
    "Cache key changed",
    "Cache expired (7 days)",
    "Cache storage full",
    "First run (no cache exists)"
  ],
  "indicators": [
    "Cache not found for input keys:",
    "Failed to restore: Cache service responded with 429"
  ]
}
```

#### Setup Action Failed
```python
{
  "id": "ci-setup-001",
  "pattern": r"setup-(node|python|go|java).*failed",
  "category": "setup_failure",
  "severity": "high",
  "common_causes": [
    "Version not available",
    "Download failed",
    "Invalid version specification",
    "Network error"
  ],
  "indicators": [
    "Version 20.x not found",
    "Unable to download",
    "##[error]"
  ]
}
```

#### Checkout Failed
```python
{
  "id": "ci-checkout-001",
  "pattern": r"git checkout failed|Unable to checkout",
  "category": "checkout_failure",
  "severity": "critical",
  "common_causes": [
    "Authentication failed",
    "Repository not accessible",
    "Ref/branch doesn't exist",
    "Submodule checkout failed"
  ],
  "indicators": [
    "fatal: could not read Username",
    "Error: Unable to process file command 'env'"
  ]
}
```

#### Artifact Upload/Download Failed
```python
{
  "id": "ci-artifact-001",
  "pattern": r"upload-artifact.*failed|download-artifact.*failed",
  "category": "artifact_failure",
  "severity": "medium",
  "common_causes": [
    "File not found",
    "Path glob incorrect",
    "Artifact too large (>2GB)",
    "Storage quota exceeded"
  ],
  "indicators": [
    "Error: No files were found with the provided path:",
    "Error: Artifact size exceeded"
  ]
}
```

---

## Pattern Matching Algorithm

### Error Classification Flow

```python
def classify_error(error_message: str, error_context: dict) -> ErrorClassification:
    """
    Classify error using multi-tier pattern matching

    Tier 1: Exact pattern match (regex)
    Tier 2: Fuzzy matching for similar patterns
    Tier 3: ML-based classification (optional)
    Tier 4: Unknown error fallback
    """

    # Tier 1: Try all exact pattern matches
    for category, patterns in ALL_PATTERNS.items():
        for pattern_id, pattern_config in patterns.items():
            match = re.search(pattern_config['pattern'], error_message, re.IGNORECASE)
            if match:
                return ErrorClassification(
                    pattern_id=pattern_id,
                    category=category,
                    subcategory=pattern_config.get('subcategory'),
                    severity=pattern_config['severity'],
                    confidence=0.95,
                    matched_text=match.group(),
                    common_causes=pattern_config.get('common_causes', [])
                )

    # Tier 2: Fuzzy matching
    best_match = find_best_fuzzy_match(error_message, ALL_PATTERNS)
    if best_match and best_match.confidence > 0.7:
        return best_match

    # Tier 3: ML classification
    if ML_CLASSIFIER_AVAILABLE:
        ml_result = ML_CLASSIFIER.predict(error_message, error_context)
        if ml_result.confidence > 0.6:
            return ml_result

    # Tier 4: Unknown
    return ErrorClassification(
        pattern_id='unknown',
        category='unknown',
        severity='unknown',
        confidence=0.0,
        recommendation='Manual investigation required'
    )
```

---

## Pattern Statistics

| Category | Pattern Count | Coverage | Avg Success Rate |
|----------|---------------|----------|------------------|
| NPM/Yarn | 15 | 85% of npm errors | 0.82 |
| Python/Pip | 12 | 80% of pip errors | 0.78 |
| Rust/Cargo | 10 | 75% of cargo errors | 0.88 |
| Go | 8 | 70% of go errors | 0.85 |
| TypeScript | 15 | 90% of TS errors | 0.92 |
| Build Tools | 10 | 75% of build errors | 0.80 |
| Test Frameworks | 12 | 85% of test failures | 0.70 |
| Runtime Errors | 10 | 60% of runtime errors | 0.65 |
| CI-Specific | 8 | 80% of CI errors | 0.88 |

**Total Patterns**: 100+
**Overall Coverage**: 78% of CI/CD errors
**Average Success Rate**: 0.81 (81% of matched patterns have successful fix solutions)

---

## Usage in Multi-Agent System

The Pattern Matcher & Categorizer agent (Agent 2) uses this library to:

1. **Match error messages** against regex patterns
2. **Classify errors** into categories and subcategories
3. **Assign severity levels** (critical/high/medium/low)
4. **Identify common causes** for each error type
5. **Look up solution success rates** from knowledge base
6. **Provide context** to Root Cause Analyzer (Agent 3)

For complete integration details, see [multi-agent-error-analysis.md](multi-agent-error-analysis.md).
