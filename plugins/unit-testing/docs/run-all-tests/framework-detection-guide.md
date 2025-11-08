# Framework Detection Guide

Comprehensive guide for auto-detecting test frameworks, configuring test commands, and discovering test files across multiple languages and ecosystems.

## Overview

The /run-all-tests command automatically detects test frameworks by analyzing project structure, configuration files, and installed dependencies. This guide covers detection patterns, framework-specific commands, and configuration strategies.

## JavaScript/TypeScript Ecosystem

### Jest Detection

**Detection Patterns**:
```bash
# Configuration files
jest.config.js
jest.config.ts
jest.config.json
.jestrc
.jestrc.json
package.json (with "jest" key)

# Dependencies in package.json
"jest"
"@types/jest"
"ts-jest"
"babel-jest"
```

**Test File Patterns**:
```bash
**/*.test.js
**/*.test.ts
**/*.test.jsx
**/*.test.tsx
**/*.spec.js
**/*.spec.ts
__tests__/**/*.js
__tests__/**/*.ts
```

**Command Detection Priority**:
1. `npm test` (if "test" script exists in package.json)
2. `yarn test`
3. `pnpm test`
4. `npx jest`
5. `npx jest --config jest.config.js`

**Jest Configuration Examples**:

```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.ts', '**/*.test.ts'],
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts'
  ],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  transform: {
    '^.+\\.tsx?$': 'ts-jest'
  }
};
```

**Running Tests**:
```bash
# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- path/to/test.test.ts

# Run in watch mode
npm test -- --watch

# Run with verbose output
npm test -- --verbose

# Run tests matching pattern
npm test -- --testNamePattern="user login"
```

### Vitest Detection

**Detection Patterns**:
```bash
# Configuration files
vitest.config.js
vitest.config.ts
vite.config.js (with test section)
vite.config.ts (with test section)

# Dependencies
"vitest"
"@vitest/ui"
```

**Test File Patterns**:
```bash
**/*.test.js
**/*.test.ts
**/*.spec.js
**/*.spec.ts
```

**Command Detection**:
1. `npm test` (if script exists)
2. `npx vitest run`
3. `npx vitest`

**Vitest Configuration**:

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'dist/',
        '**/*.test.ts'
      ]
    },
    setupFiles: ['./vitest.setup.ts']
  }
});
```

**Running Tests**:
```bash
# Run all tests
npx vitest run

# Run with coverage
npx vitest run --coverage

# Watch mode
npx vitest

# UI mode
npx vitest --ui

# Run specific file
npx vitest run path/to/test.test.ts
```

### Mocha Detection

**Detection Patterns**:
```bash
# Configuration files
.mocharc.js
.mocharc.json
.mocharc.yaml
mocha.opts

# Dependencies
"mocha"
"@types/mocha"
```

**Command Detection**:
```bash
npx mocha
npx mocha --recursive
npx mocha 'test/**/*.test.js'
```

**Mocha Configuration**:

```javascript
// .mocharc.js
module.exports = {
  require: ['ts-node/register'],
  spec: 'test/**/*.test.ts',
  timeout: 5000,
  reporter: 'spec',
  ui: 'bdd'
};
```

### Ava Detection

**Detection Patterns**:
```bash
# package.json ava section
"ava": { ... }

# Dependencies
"ava"
```

**Configuration**:
```json
{
  "ava": {
    "files": [
      "test/**/*",
      "!test/helpers/**/*"
    ],
    "typescript": {
      "rewritePaths": {
        "src/": "dist/"
      },
      "compile": false
    }
  }
}
```

## Python Ecosystem

### Pytest Detection

**Detection Patterns**:
```bash
# Configuration files
pytest.ini
pyproject.toml (with [tool.pytest] section)
setup.cfg (with [tool:pytest] section)
tox.ini (with [pytest] section)

# Test file patterns
test_*.py
*_test.py
tests/test_*.py
tests/*_test.py

# Executable detection
which pytest
python -m pytest
```

**Configuration Examples**:

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-ra -q --strict-markers --cov=src --cov-report=html"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests",
    "unit: unit tests"
]
```

**Command Detection Priority**:
1. `pytest` (if in PATH)
2. `python -m pytest`
3. `python3 -m pytest`
4. `py.test`

**Running Tests**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run specific test function
pytest tests/test_api.py::test_user_creation

# Run tests matching keyword
pytest -k "user"

# Run with verbose output
pytest -v

# Run only failed tests from last run
pytest --lf

# Run in parallel
pytest -n auto

# Run with markers
pytest -m "not slow"

# Generate JUnit XML report
pytest --junitxml=results.xml
```

### Unittest Detection

**Detection Patterns**:
```bash
# Built-in Python module (always available)
# Test file patterns
test_*.py
*_test.py

# Test discovery
python -m unittest discover
```

**Running Tests**:
```bash
# Discover and run all tests
python -m unittest discover

# Run specific test module
python -m unittest tests.test_module

# Run specific test class
python -m unittest tests.test_module.TestClass

# Run specific test method
python -m unittest tests.test_module.TestClass.test_method

# Verbose output
python -m unittest discover -v

# Start directory
python -m unittest discover -s tests -p "test_*.py"
```

### Tox Detection

**Detection Patterns**:
```bash
# Configuration file
tox.ini

# Dependencies
"tox"
```

**Configuration**:
```ini
# tox.ini
[tox]
envlist = py38,py39,py310,py311,py312

[testenv]
deps =
    pytest
    pytest-cov
    pytest-xdist
commands =
    pytest --cov=src --cov-report=term --cov-report=html {posargs}
```

**Running Tests**:
```bash
# Run all environments
tox

# Run specific environment
tox -e py311

# Run with parallel execution
tox -p auto

# Recreate environments
tox -r
```

## Rust Ecosystem

### Cargo Test Detection

**Detection Patterns**:
```bash
# Configuration file
Cargo.toml

# Test file patterns
src/**/*_test.rs
tests/**/*.rs
src/lib.rs (with #[cfg(test)] modules)
src/main.rs (with #[cfg(test)] modules)

# Executable detection
which cargo
cargo --version
```

**Test Organization**:

```rust
// Unit tests (in src/*.rs files)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    #[should_panic]
    fn test_divide_by_zero() {
        divide(10, 0);
    }

    #[test]
    #[ignore]
    fn expensive_test() {
        // Expensive computation
    }
}

// Integration tests (in tests/*.rs)
use my_crate::*;

#[test]
fn integration_test() {
    assert!(external_function());
}
```

**Running Tests**:
```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_add

# Run tests in specific file
cargo test --test integration_test

# Run doc tests
cargo test --doc

# Run ignored tests
cargo test -- --ignored

# Run with multiple threads
cargo test -- --test-threads=4

# Show test output always
cargo test -- --show-output

# Run benchmarks
cargo bench
```

**Cargo.toml Test Configuration**:
```toml
[dev-dependencies]
proptest = "1.0"
criterion = "0.5"

[[test]]
name = "integration"
path = "tests/integration.rs"

[[bench]]
name = "performance"
harness = false
```

## Go Ecosystem

### Go Test Detection

**Detection Patterns**:
```bash
# Module file
go.mod

# Test file patterns
*_test.go

# Executable detection
which go
go version
```

**Test Organization**:

```go
// user_test.go
package user

import (
    "testing"
)

func TestCreateUser(t *testing.T) {
    user := CreateUser("John", "john@example.com")
    if user.Name != "John" {
        t.Errorf("Expected name John, got %s", user.Name)
    }
}

func TestValidateEmail(t *testing.T) {
    tests := []struct {
        name  string
        email string
        want  bool
    }{
        {"valid email", "user@example.com", true},
        {"invalid email", "invalid", false},
        {"empty email", "", false},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := ValidateEmail(tt.email)
            if got != tt.want {
                t.Errorf("ValidateEmail(%s) = %v, want %v",
                    tt.email, got, tt.want)
            }
        })
    }
}

func BenchmarkCreateUser(b *testing.B) {
    for i := 0; i < b.N; i++ {
        CreateUser("John", "john@example.com")
    }
}
```

**Running Tests**:
```bash
# Run all tests
go test ./...

# Run tests in current package
go test

# Run with verbose output
go test -v ./...

# Run specific test
go test -run TestCreateUser

# Run with coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run benchmarks
go test -bench=. ./...
go test -bench=BenchmarkCreateUser

# Run with race detector
go test -race ./...

# Run in parallel
go test -parallel 4 ./...

# Generate test binary
go test -c

# Run with timeout
go test -timeout 30s ./...
```

## Java Ecosystem

### Maven (JUnit) Detection

**Detection Patterns**:
```bash
# Configuration file
pom.xml

# Test file patterns
src/test/java/**/*Test.java
src/test/java/**/*Tests.java
src/test/java/**/Test*.java

# Dependencies in pom.xml
junit-jupiter
junit-platform
testng
```

**pom.xml Configuration**:

```xml
<project>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
                <configuration>
                    <includes>
                        <include>**/*Test.java</include>
                        <include>**/*Tests.java</include>
                    </includes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

**Running Tests**:
```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=UserServiceTest

# Run specific test method
mvn test -Dtest=UserServiceTest#testCreateUser

# Run with coverage (using JaCoCo)
mvn test jacoco:report

# Skip tests
mvn install -DskipTests

# Run integration tests
mvn verify

# Clean and test
mvn clean test
```

### Gradle Detection

**Detection Patterns**:
```bash
# Configuration files
build.gradle
build.gradle.kts
settings.gradle
settings.gradle.kts

# Test directories
src/test/java
src/test/kotlin
```

**build.gradle.kts Configuration**:

```kotlin
plugins {
    java
    jacoco
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
    testImplementation("org.mockito:mockito-core:5.5.0")
    testImplementation("org.assertj:assertj-core:3.24.2")
}

tasks.test {
    useJUnitPlatform()

    testLogging {
        events("passed", "skipped", "failed")
    }

    finalizedBy(tasks.jacocoTestReport)
}

tasks.jacocoTestReport {
    reports {
        xml.required.set(true)
        html.required.set(true)
    }
}
```

**Running Tests**:
```bash
# Run all tests
./gradlew test

# Run specific test class
./gradlew test --tests UserServiceTest

# Run specific test method
./gradlew test --tests UserServiceTest.testCreateUser

# Run with coverage
./gradlew test jacocoTestReport

# Run with info logging
./gradlew test --info

# Run integration tests
./gradlew integrationTest

# Clean and test
./gradlew clean test
```

## Other Frameworks

### Ruby (RSpec) Detection

**Detection Patterns**:
```bash
# Configuration files
.rspec
spec/spec_helper.rb

# Test file patterns
spec/**/*_spec.rb

# Dependencies
Gemfile (with "rspec")
```

**Running Tests**:
```bash
# Run all tests
bundle exec rspec

# Run specific file
bundle exec rspec spec/models/user_spec.rb

# Run with format
bundle exec rspec --format documentation

# Run with coverage
bundle exec rspec --require spec_helper
```

### PHP (PHPUnit) Detection

**Detection Patterns**:
```bash
# Configuration files
phpunit.xml
phpunit.xml.dist

# Test file patterns
tests/**/*Test.php
```

**Running Tests**:
```bash
# Run all tests
vendor/bin/phpunit

# Run with coverage
vendor/bin/phpunit --coverage-html coverage

# Run specific test
vendor/bin/phpunit tests/UserTest.php
```

### C/C++ (GoogleTest) Detection

**Detection Patterns**:
```bash
# Test file patterns
*_test.cpp
*_test.cc
test_*.cpp

# Build system
CMakeLists.txt (with gtest)
```

**Running Tests**:
```bash
# Build and run tests
cmake -B build
cmake --build build
cd build && ctest

# Run with verbose output
ctest -V

# Run specific test
./build/user_test
```

## Multi-Framework Detection Algorithm

```python
def detect_test_framework(project_path: str) -> Dict[str, Any]:
    """
    Detect all test frameworks in a project
    Returns dict with framework name, command, and test files
    """
    frameworks = []

    # Check for JavaScript/TypeScript frameworks
    if file_exists('package.json'):
        package_json = read_json('package.json')

        if 'jest' in package_json.get('dependencies', {}) or \
           'jest' in package_json.get('devDependencies', {}):
            frameworks.append({
                'name': 'jest',
                'command': detect_npm_command() + ' test',
                'config': find_file('jest.config.*'),
                'test_files': glob('**/*.test.{js,ts,jsx,tsx}')
            })

        if 'vitest' in package_json.get('devDependencies', {}):
            frameworks.append({
                'name': 'vitest',
                'command': 'npx vitest run',
                'config': find_file('vitest.config.*'),
                'test_files': glob('**/*.test.{js,ts}')
            })

    # Check for Python frameworks
    if command_exists('pytest'):
        frameworks.append({
            'name': 'pytest',
            'command': 'pytest',
            'config': find_file('pytest.ini') or find_file('pyproject.toml'),
            'test_files': glob('**/test_*.py') + glob('**/*_test.py')
        })

    # Check for Rust
    if file_exists('Cargo.toml'):
        frameworks.append({
            'name': 'cargo',
            'command': 'cargo test',
            'config': 'Cargo.toml',
            'test_files': glob('tests/**/*.rs')
        })

    # Check for Go
    if file_exists('go.mod'):
        frameworks.append({
            'name': 'go-test',
            'command': 'go test ./...',
            'config': 'go.mod',
            'test_files': glob('**/*_test.go')
        })

    # Check for Java
    if file_exists('pom.xml'):
        frameworks.append({
            'name': 'maven',
            'command': 'mvn test',
            'config': 'pom.xml',
            'test_files': glob('src/test/java/**/*Test.java')
        })

    if file_exists('build.gradle') or file_exists('build.gradle.kts'):
        frameworks.append({
            'name': 'gradle',
            'command': './gradlew test',
            'config': find_file('build.gradle*'),
            'test_files': glob('src/test/**/*.java')
        })

    return frameworks

def detect_npm_command() -> str:
    """Detect npm, yarn, or pnpm"""
    if file_exists('pnpm-lock.yaml'):
        return 'pnpm'
    elif file_exists('yarn.lock'):
        return 'yarn'
    else:
        return 'npm'
```

## Environment Setup

### Node.js Environment

```bash
# Install dependencies
npm install

# Or with specific package manager
yarn install
pnpm install

# Set environment variables
export NODE_ENV=test
export CI=true
```

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Or with specific test dependencies
pip install pytest pytest-cov pytest-xdist

# Set environment variables
export PYTHONPATH=src
export TESTING=true
```

### Rust Environment

```bash
# Update Rust
rustup update

# Install test tools
cargo install cargo-nextest  # Faster test runner
cargo install cargo-tarpaulin  # Coverage tool
```

### Go Environment

```bash
# Download dependencies
go mod download

# Install test tools
go install gotest.tools/gotestsum@latest
go install github.com/axw/gocov/gocov@latest
```

## Test Discovery Patterns

### Pattern Matching

```bash
# JavaScript/TypeScript
*.test.js
*.test.ts
*.spec.js
*.spec.ts
__tests__/**/*.js

# Python
test_*.py
*_test.py
tests/test_*.py

# Rust
*_test.rs
tests/*.rs
#[cfg(test)] modules

# Go
*_test.go

# Java
*Test.java
*Tests.java
Test*.java
```

### Directory Structures

```
project/
├── JavaScript/TypeScript
│   ├── src/
│   │   └── utils.ts
│   ├── __tests__/
│   │   └── utils.test.ts
│   └── tests/
│       └── integration/
│           └── api.test.ts
│
├── Python
│   ├── src/
│   │   └── utils.py
│   └── tests/
│       ├── unit/
│       │   └── test_utils.py
│       └── integration/
│           └── test_api.py
│
├── Rust
│   ├── src/
│   │   ├── lib.rs (with #[cfg(test)])
│   │   └── utils.rs
│   └── tests/
│       └── integration_test.rs
│
├── Go
│   ├── pkg/
│   │   ├── user.go
│   │   └── user_test.go
│   └── internal/
│       └── auth_test.go
│
└── Java
    ├── src/
    │   ├── main/
    │   │   └── java/
    │   └── test/
    │       └── java/
```

## Framework Comparison Matrix

| Feature | Jest | Vitest | Pytest | Cargo | Go Test | Maven | Gradle |
|---------|------|--------|--------|-------|---------|-------|--------|
| **Parallel** | ✅ | ✅ | ✅ (with -n) | ✅ | ✅ | ✅ | ✅ |
| **Watch Mode** | ✅ | ✅ | ✅ (with plugin) | ❌ | ❌ | ❌ | ❌ |
| **Coverage** | ✅ | ✅ | ✅ (with pytest-cov) | ✅ (with tarpaulin) | ✅ | ✅ (with JaCoCo) | ✅ (with JaCoCo) |
| **Snapshot Testing** | ✅ | ✅ | ✅ (with plugin) | ✅ (with insta) | ❌ | ❌ | ❌ |
| **Mocking** | ✅ | ✅ | ✅ (with pytest-mock) | ✅ (with mockall) | ✅ (manual) | ✅ (with Mockito) | ✅ (with Mockito) |
| **Speed** | Fast | Faster | Medium | Fast | Very Fast | Medium | Medium |

## Common Issues and Solutions

### Issue: Tests not discovered

**Solution**: Check test file naming patterns match framework expectations

```bash
# Jest/Vitest expects
*.test.js, *.spec.js

# Pytest expects
test_*.py, *_test.py

# Cargo expects
*_test.rs or #[cfg(test)] modules

# Go expects
*_test.go
```

### Issue: Wrong test command detected

**Solution**: Explicitly set test command in CI configuration

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: npm test  # Or: pytest, cargo test, go test ./..., etc.
```

### Issue: Environment variables not set

**Solution**: Create framework-specific environment file

```bash
# .env.test for Node.js
NODE_ENV=test
DATABASE_URL=sqlite::memory:

# pytest.ini for Python
[pytest]
env =
    TESTING=true
    DATABASE_URL=sqlite::memory:
```

## Best Practices

1. **Use framework detection in CI/CD pipelines**
2. **Document test commands in README**
3. **Maintain consistent test file naming**
4. **Configure coverage thresholds**
5. **Enable parallel execution**
6. **Use framework-specific reporters for CI integration**
7. **Set appropriate timeouts**
8. **Configure test retries for flaky tests**
9. **Use test tags/markers for selective execution**
10. **Keep test dependencies up to date**
