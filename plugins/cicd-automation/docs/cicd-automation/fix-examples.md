# Fix Examples

**Version**: 1.0.3
**Command**: `/fix-commit-errors`
**Category**: CI/CD Automation

## Overview

Real-world fix scenarios demonstrating the `/fix-commit-errors` command in action. Each example includes the complete error context, root cause analysis using the 3W1H framework (What/Why/When/How), solution implementation, and knowledge base impact.

---

## Example 1: NPM ERESOLVE Peer Dependency Conflict

### Before: Initial Error

```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR!
npm ERR! While resolving: my-app@1.0.0
npm ERR! Found: react@17.0.2
npm ERR! node_modules/react
npm ERR!   react@"^17.0.2" from the root project
npm ERR!
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^18.0.0" from react-dom@18.2.0
npm ERR! node_modules/react-dom
npm ERR!   react-dom@"^18.2.0" from the root project

Workflow: Node.js CI
Job: build
Exit Code: 1
```

### Root Cause Analysis (3W1H)

**What Failed**: npm install command during dependency resolution
**Why Failed**: React 17 specified in package.json conflicts with React DOM 18's peer dependency requirement for React 18
**When Started**: After updating react-dom from 17.x to 18.2.0 in commit a7f3c21
**How Propagates**: Blocks entire CI pipeline, prevents build and test stages from running

### Solution

**Before** (package.json):
```json
{
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^18.2.0"
  }
}
```

**After** (package.json):
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
```

**Workflow Update** (.github/workflows/ci.yml):
```yaml
# No changes needed - peer dependency now satisfied
```

**Alternative Quick Fix** (if React 18 upgrade not feasible):
```yaml
- name: Install dependencies
  run: npm ci --legacy-peer-deps
```

### Validation

```bash
# Locally validated
npm install
npm run build
npm test

# All passed ✓
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 2m 34s (previously failed at 0m 12s)
- ✅ Knowledge Base: Pattern `npm-eresolve-001` confidence increased to 0.95
- 🎯 Resolution Time: **3 minutes**

### Knowledge Base Impact

```json
{
  "pattern_id": "npm-eresolve-001",
  "successes": 47,
  "failures": 3,
  "confidence": 0.95,
  "preferred_solution": "update_peer_dependency",
  "alternative_solution": "npm_install_legacy_peer_deps"
}
```

---

## Example 2: TypeScript Type Error After Dependency Update

### Before: Initial Error

```
src/components/UserProfile.tsx:45:18 - error TS2339: Property 'username' does not exist on type 'User'.

45     const name = user.username;
                    ~~~~~~~~~~~~~~

src/utils/api.ts:23:7 - error TS2741: Property 'id' is missing in type '{ name: string; email: string; }' but required in type 'User'.

23       return { name: data.name, email: data.email };
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Found 2 errors in 2 files.

Workflow: TypeScript Build
Job: typecheck
Exit Code: 2
```

### Root Cause Analysis (3W1H)

**What Failed**: TypeScript compilation during type checking phase
**Why Failed**: Breaking change in @types/user-api v1.0.2 renamed `username` → `name` and added required `id` field
**When Started**: After automatic dependency update by Renovate bot (commit b4e9f12)
**How Propagates**: Blocks build stage, prevents deployment

### Solution

**Before** (src/components/UserProfile.tsx):
```typescript
interface User {
  username: string;
  email: string;
}

const UserProfile = ({ user }: { user: User }) => {
  const name = user.username;
  return <div>{name}</div>;
};
```

**After** (src/components/UserProfile.tsx):
```typescript
interface User {
  id: string;
  name: string;
  email: string;
}

const UserProfile = ({ user }: { user: User }) => {
  const name = user.name;
  return <div>{name}</div>;
};
```

**Before** (src/utils/api.ts):
```typescript
export const fetchUser = async (): Promise<User> => {
  const data = await fetch('/api/user').then(r => r.json());
  return { name: data.name, email: data.email };
};
```

**After** (src/utils/api.ts):
```typescript
export const fetchUser = async (): Promise<User> => {
  const data = await fetch('/api/user').then(r => r.json());
  return {
    id: data.id || crypto.randomUUID(), // Fallback for missing id
    name: data.name,
    email: data.email
  };
};
```

### Validation

```bash
# Type checking
npm run typecheck
# ✓ No errors

# Unit tests
npm test -- --findRelatedTests src/components/UserProfile.tsx src/utils/api.ts
# ✓ All tests passed

# Full build
npm run build
# ✓ Build succeeded
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 3m 12s
- ✅ Knowledge Base: Pattern `typescript-breaking-change-001` created with confidence 0.88
- 🎯 Resolution Time: **12 minutes**

### Knowledge Base Impact

```json
{
  "pattern_id": "typescript-breaking-change-001",
  "category": "type_error",
  "trigger": "@types/* package major version update",
  "solution_strategy": "review_changelog_and_update_types",
  "confidence": 0.88,
  "related_patterns": ["npm-major-update-001"]
}
```

---

## Example 3: Jest Test Timeout on CI (Flaky Test)

### Before: Initial Error

```
 FAIL  src/components/AsyncButton.test.tsx
  ● AsyncButton › should update text after async operation

    Exceeded timeout of 5000 ms for a test.
    Use jest.setTimeout(newTimeout) to increase the timeout value, if this is a long-running test.

      23 |   it('should update text after async operation', async () => {
      24 |     render(<AsyncButton />);
    > 25 |     fireEvent.click(screen.getByRole('button'));
         |     ^
      26 |     expect(screen.getByText('Loading...')).toBeInTheDocument();
      27 |     expect(await screen.findByText('Done')).toBeInTheDocument();
      28 |   });

    at Object.<anonymous> (src/components/AsyncButton.test.tsx:25:5)

Test Suites: 1 failed, 12 passed, 13 total
Tests:       1 failed, 87 passed, 88 total

Workflow: Test Suite
Job: test
Exit Code: 1
Pass Rate: 98.9% (locally: 100%)
```

### Root Cause Analysis (3W1H)

**What Failed**: Async test for button component with network request
**Why Failed**: Test doesn't properly await async state updates; CI environment is slower than local (network latency)
**When Started**: Intermittent failures starting 3 days ago, became consistent after infrastructure change
**How Propagates**: Flaky test blocks PR merges, reduces CI reliability, wastes developer time

### Solution

**Before** (src/components/AsyncButton.test.tsx):
```typescript
it('should update text after async operation', async () => {
  render(<AsyncButton />);
  fireEvent.click(screen.getByRole('button'));
  expect(screen.getByText('Loading...')).toBeInTheDocument();
  // ❌ Race condition: findByText might timeout before async operation completes
  expect(await screen.findByText('Done')).toBeInTheDocument();
});
```

**After** (src/components/AsyncButton.test.tsx):
```typescript
import { waitFor } from '@testing-library/react';

it('should update text after async operation', async () => {
  render(<AsyncButton />);

  const button = screen.getByRole('button');
  fireEvent.click(button);

  // Wait for loading state
  await waitFor(() => {
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  }, { timeout: 1000 });

  // Wait for completion with increased timeout for CI
  await waitFor(() => {
    expect(screen.getByText('Done')).toBeInTheDocument();
  }, { timeout: 10000 });
});
```

**Configuration Update** (jest.config.js):
```javascript
module.exports = {
  testTimeout: 10000, // Increased from default 5000
  // ... other config
};
```

### Validation

```bash
# Run test locally 10 times
for i in {1..10}; do npm test -- AsyncButton.test.tsx; done
# ✓ 10/10 passed

# Run on CI (via manual workflow trigger)
# ✓ Passed consistently across 5 runs
```

### After

- ✅ Workflow Status: Passing (5/5 consecutive runs)
- ✅ Test Time: 8.2s (was failing at 5.0s timeout)
- ✅ Knowledge Base: Pattern `jest-timeout-001` confidence increased to 0.92
- ✅ Flakiness Eliminated: 0% failure rate over 20 runs (was 35%)
- 🎯 Resolution Time: **8 minutes**

### Knowledge Base Impact

```json
{
  "pattern_id": "jest-timeout-001",
  "category": "flaky_test",
  "subcategory": "async_timing",
  "solution": "add_waitfor_with_increased_timeout",
  "confidence": 0.92,
  "prevention": "Use waitFor for all async assertions, increase CI timeout by 2x local"
}
```

---

## Example 4: Python Import Error After Requirements Update

### Before: Initial Error

```
Traceback (most recent call last):
  File "src/main.py", line 3, in <module>
    from fastapi import FastAPI, Request
ImportError: cannot import name 'Request' from 'fastapi' (/opt/hostedtoolcache/Python/3.11.8/x64/lib/python3.11/site-packages/fastapi/__init__.py)

Error: Process completed with exit code 1.

Workflow: Python CI
Job: test
Python Version: 3.11
```

### Root Cause Analysis (3W1H)

**What Failed**: Python module import during application startup
**Why Failed**: FastAPI 0.100.0 moved `Request` to `fastapi.requests` module (breaking change)
**When Started**: After Dependabot updated fastapi from 0.95.2 to 0.100.0
**How Propagates**: Application fails to start, all tests fail, deployment blocked

### Solution

**Before** (src/main.py):
```python
from fastapi import FastAPI, Request
from fastapi import HTTPException

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response
```

**After** (src/main.py):
```python
from fastapi import FastAPI, HTTPException
from fastapi.requests import Request  # Updated import path

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response
```

**Requirements Update** (requirements.txt):
```txt
# Updated with compatible versions
fastapi==0.100.0
starlette==0.27.0  # FastAPI 0.100 requires starlette>=0.27.0
```

### Validation

```bash
# Install updated dependencies
uv uv pip install -r requirements.txt

# Run application
python src/main.py
# ✓ Server started successfully

# Run tests
pytest tests/ -v
# ✓ 45 passed in 3.21s
```

### After

- ✅ Workflow Status: Passing
- ✅ Test Time: 1m 45s
- ✅ Knowledge Base: Pattern `python-import-breaking-change-001` created
- 🎯 Resolution Time: **6 minutes**

---

## Example 5: Go Module Not Found After go.mod Change

### Before: Initial Error

```
go: finding module for package github.com/user/mylib/v2
go: github.com/myapp/service imports
    github.com/user/mylib/v2: module github.com/user/mylib/v2: reading github.com/user/mylib/go.mod at revision v1.0.2: unknown revision v1.0.2

Error: Process completed with exit code 1.

Workflow: Go Build
Job: build
Go Version: 1.21
```

### Root Cause Analysis (3W1H)

**What Failed**: Go module resolution during dependency download
**Why Failed**: Import path changed from `/v2` to `/v3` but go.mod still references v2
**When Started**: After upgrading mylib to v3.0.0
**How Propagates**: Build fails, cannot compile application

### Solution

**Before** (go.mod):
```go
module github.com/myapp/service

go 1.21

require (
    github.com/user/mylib/v2 v2.1.0
)
```

**Before** (internal/handler/handler.go):
```go
import (
    "github.com/user/mylib/v2/client"
)
```

**After** (go.mod):
```go
module github.com/myapp/service

go 1.21

require (
    github.com/user/mylib/v3 v3.0.0
)
```

**After** (internal/handler/handler.go):
```go
import (
    "github.com/user/mylib/v3/client"
)
```

**Fix Command**:
```bash
# Update module path
go get github.com/user/mylib/v3@v3.0.0

# Update all imports
find . -name "*.go" -exec sed -i 's|github.com/user/mylib/v2|github.com/user/mylib/v3|g' {} \;

# Tidy dependencies
go mod tidy
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 2m 8s
- 🎯 Resolution Time: **5 minutes**

---

## Example 6: Rust Compilation Error - Unresolved Import

### Before: Initial Error

```
error[E0432]: unresolved import `serde::Deserialize`
 --> src/models/user.rs:1:5
  |
1 | use serde::{Serialize, Deserialize};
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^ no `Deserialize` in the root

error[E0433]: failed to resolve: use of undeclared type `Deserialize`
 --> src/models/user.rs:4:10
  |
4 | #[derive(Deserialize, Serialize, Debug)]
  |          ^^^^^^^^^^^ use of undeclared type `Deserialize`

error: aborting due to 2 previous errors

Workflow: Rust CI
Job: build
Rust Version: 1.75.0
Exit Code: 101
```

### Root Cause Analysis (3W1H)

**What Failed**: Rust compilation during derive macro expansion
**Why Failed**: `serde` feature `derive` not enabled in Cargo.toml
**When Started**: After adding new model with Deserialize trait
**How Propagates**: Compilation fails, cannot build binary

### Solution

**Before** (Cargo.toml):
```toml
[dependencies]
serde = "1.0"
serde_json = "1.0"
```

**After** (Cargo.toml):
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 3m 42s
- 🎯 Resolution Time: **4 minutes**

---

## Example 7: Webpack Module Resolution Failure

### Before: Initial Error

```
ERROR in ./src/components/Dashboard.tsx
Module not found: Error: Can't resolve '@/utils/api' in '/home/runner/work/app/app/src/components'

webpack compiled with 1 error

Workflow: Build Frontend
Job: build
Node Version: 18.x
Exit Code: 1
```

### Root Cause Analysis (3W1H)

**What Failed**: Webpack module resolution during compilation
**Why Failed**: Webpack alias `@` not configured in webpack.config.js
**When Started**: After refactoring imports to use absolute paths
**How Propagates**: Build fails, cannot create production bundle

### Solution

**Before** (webpack.config.js):
```javascript
module.exports = {
  // ... other config
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
};
```

**After** (webpack.config.js):
```javascript
const path = require('path');

module.exports = {
  // ... other config
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
};
```

**Also Update** (tsconfig.json for IDE support):
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 4m 15s
- 🎯 Resolution Time: **7 minutes**

---

## Example 8: Docker Build Failure - Missing System Dependency

### Before: Initial Error

```
#8 15.23 gcc: error: /usr/lib/x86_64-linux-gnu/libpthread.so: No such file or directory
#8 15.23 error: command 'gcc' failed with exit status 1
#8 15.23 ERROR: Failed building wheel for cryptography

ERROR: failed to solve: process "/bin/sh -c uv uv pip install -r requirements.txt" did not complete successfully: exit code: 1

Workflow: Build Docker Image
Job: build-and-push
Exit Code: 1
```

### Root Cause Analysis (3W1H)

**What Failed**: Python package compilation inside Docker build
**Why Failed**: System library `libffi-dev` missing, required by cryptography package
**When Started**: After updating cryptography from 38.x to 41.x
**How Propagates**: Docker build fails, cannot create container image

### Solution

**Before** (Dockerfile):
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN uv uv pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**After** (Dockerfile):
```dockerfile
FROM python:3.11-slim

# Install system dependencies required for cryptography
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN uv uv pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 5m 28s (increased due to apt install)
- ✅ Image Size: 245MB (was 180MB, acceptable trade-off)
- 🎯 Resolution Time: **10 minutes**

---

## Example 9: GitHub Actions Cache Corruption

### Before: Initial Error

```
Run actions/cache@v3
Cache restored from key: Linux-npm-a7f3c2... (partial match)

> npm ci
npm ERR! `npm ci` can only install packages when your package.json and package-lock.json are in sync.

npm ERR! A complete log of this run can be found in:
npm ERR!     /home/runner/.npm/_logs/2025-11-06T10_15_30_123Z-debug-0.log

Error: Process completed with exit code 1.

Workflow: Node.js CI
Job: build
Runs: Failed 3/3 times
```

### Root Cause Analysis (3W1H)

**What Failed**: npm ci command with cached node_modules
**Why Failed**: Cache contains node_modules from different package-lock.json version
**When Started**: After package-lock.json was updated but cache key didn't change
**How Propagates**: All npm ci runs fail with corrupted cache

### Solution

**Before** (.github/workflows/ci.yml):
```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-npm-${{ hashFiles('package.json') }}
    restore-keys: |
      ${{ runner.os }}-npm-
```

**After** (.github/workflows/ci.yml):
```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: ~/.npm
    # Fixed: Use package-lock.json hash, not package.json
    key: ${{ runner.os }}-npm-v2-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-npm-v2-
```

**Manual Cache Clear**:
```bash
# Delete corrupted cache via GitHub CLI
gh cache list
gh cache delete <cache-id>
```

### After

- ✅ Workflow Status: Passing
- ✅ Build Time: 1m 52s (first run without cache), 1m 15s (subsequent cached runs)
- ✅ Cache Hit Rate: 95% (after fix)
- 🎯 Resolution Time: **6 minutes**

---

## Example 10: GitLab CI Runner Out of Disk Space

### Before: Initial Error

```
ERROR: Job failed: prepare environment: exit status 1.
Check https://docs.gitlab.com/runner/shells/index.html#shell-profile-loading for more information

df: /home/gitlab-runner: No space left on device

Disk usage: 100% (50GB / 50GB)

Pipeline: Failed
Job: build-docker
Runner: gitlab-runner-01
```

### Root Cause Analysis (3W1H)

**What Failed**: GitLab CI job startup on runner
**Why Failed**: Runner disk full from accumulating Docker images and build artifacts
**When Started**: Gradual accumulation over 3 weeks
**How Propagates**: All jobs on this runner fail

### Solution

**Before** (.gitlab-ci.yml):
```yaml
build-docker:
  stage: build
  script:
    - docker build -t myapp:$CI_COMMIT_SHA .
    - docker push myapp:$CI_COMMIT_SHA
```

**After** (.gitlab-ci.yml):
```yaml
build-docker:
  stage: build
  before_script:
    # Clean up old Docker images
    - docker system prune -af --volumes --filter "until=24h"
  script:
    - docker build -t myapp:$CI_COMMIT_SHA .
    - docker push myapp:$CI_COMMIT_SHA
  after_script:
    # Clean up current build artifacts
    - docker rmi myapp:$CI_COMMIT_SHA || true
```

**Runner Maintenance Script** (scheduled job):
```yaml
cleanup-runner:
  stage: maintenance
  only:
    - schedules
  script:
    - docker system prune -af --volumes
    - rm -rf /tmp/*
    - gitlab-runner verify --delete
  tags:
    - maintenance
```

### After

- ✅ Workflow Status: Passing
- ✅ Disk Usage: 35% (freed 32.5GB)
- ✅ Scheduled Cleanup: Weekly via GitLab CI schedule
- 🎯 Resolution Time: **15 minutes**

---

## Example 11: Terraform State Lock Timeout

### Before: Initial Error

```
Error: Error acquiring the state lock

Error message: ConditionalCheckFailedException: The conditional request failed
Lock Info:
  ID:        a7f3c21e-1234-5678-90ab-cdef12345678
  Path:      terraform-state/production.tfstate
  Operation: OperationTypeApply
  Who:       CI-Pipeline-456@github-actions
  Version:   1.5.7
  Created:   2025-11-06 10:15:30 UTC
  Info:

Terraform acquires a state lock to protect the state from being written
by multiple users at the same time.

Pipeline: Failed
Job: terraform-apply
Duration: 5m 0s (timeout)
```

### Root Cause Analysis (3W1H)

**What Failed**: Terraform apply operation during state lock acquisition
**Why Failed**: Previous pipeline was cancelled mid-apply, left orphaned lock in DynamoDB
**When Started**: After force-cancelling previous pipeline
**How Propagates**: All subsequent Terraform runs timeout on lock

### Solution

**Manual Lock Release**:
```bash
# Force unlock (use with caution!)
terraform force-unlock a7f3c21e-1234-5678-90ab-cdef12345678

# Verify lock released
terraform plan
```

**Workflow Update** (.github/workflows/terraform.yml):
```yaml
terraform-apply:
  runs-on: ubuntu-latest
  timeout-minutes: 30  # Added timeout
  steps:
    - name: Terraform Apply
      id: apply
      run: terraform apply -auto-approve tfplan
      continue-on-error: true

    - name: Release Lock on Failure
      if: steps.apply.outcome == 'failure'
      run: |
        # Attempt graceful unlock
        terraform force-unlock -force $(terraform show -json | jq -r '.lock_id // empty')
      continue-on-error: true
```

### After

- ✅ Workflow Status: Passing
- ✅ Lock Released: Successfully
- ✅ Prevention: Added timeout and cleanup logic
- 🎯 Resolution Time: **10 minutes**

---

## Example 12: Kubernetes Deployment ImagePullBackOff

### Before: Initial Error

```
Events:
  Type     Reason     Age                From               Message
  ----     ------     ----               ----               -------
  Normal   Scheduled  2m                 default-scheduler  Successfully assigned default/myapp-6d8f7b9c-xyz to node-1
  Normal   Pulling    1m (x4 over 2m)    kubelet            Pulling image "myregistry.io/myapp:abc123"
  Warning  Failed     1m (x4 over 2m)    kubelet            Failed to pull image "myregistry.io/myapp:abc123": rpc error: code = Unknown desc = Error response from daemon: pull access denied for myregistry.io/myapp, repository does not exist or may require 'docker login'
  Warning  Failed     1m (x4 over 2m)    kubelet            Error: ImagePullBackOff

Pipeline: Deploy to Kubernetes
Job: deploy-production
Status: Deployed (but pods failing)
```

### Root Cause Analysis (3W1H)

**What Failed**: Kubernetes pod image pull from private registry
**Why Failed**: ImagePullSecret not created in target namespace
**When Started**: After deploying to new namespace
**How Propagates**: Deployment succeeds but pods never start

### Solution

**Before** (k8s/deployment.yaml):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      containers:
      - name: myapp
        image: myregistry.io/myapp:abc123
```

**After** (k8s/deployment.yaml):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      imagePullSecrets:
      - name: registry-credentials
      containers:
      - name: myapp
        image: myregistry.io/myapp:abc123
```

**Workflow Update** (.github/workflows/deploy.yml):
```yaml
- name: Create ImagePullSecret
  run: |
    kubectl create secret docker-registry registry-credentials \
      --docker-server=myregistry.io \
      --docker-username=${{ secrets.REGISTRY_USERNAME }} \
      --docker-password=${{ secrets.REGISTRY_PASSWORD }} \
      --namespace=production \
      --dry-run=client -o yaml | kubectl apply -f -

- name: Deploy Application
  run: kubectl apply -f k8s/ -n production
```

### After

- ✅ Deployment Status: Healthy (3/3 pods running)
- ✅ Image Pull: Successful
- ✅ Startup Time: 45s
- 🎯 Resolution Time: **8 minutes**

---

## Example 13: Database Migration Failure in CI

### Before: Initial Error

```
Running migrations...
Applying migration: 20251106_add_user_preferences.sql

ERROR: relation "users" does not exist
LINE 1: ALTER TABLE users ADD COLUMN preferences JSONB DEFAULT '{}'...
                    ^

Migration failed: exit code 1

Pipeline: Run Migrations
Job: migrate-database
Database: PostgreSQL 15
Environment: Staging
```

### Root Cause Analysis (3W1H)

**What Failed**: Database migration script execution
**Why Failed**: Migration assumes `users` table exists, but test database is empty
**When Started**: First run on fresh test database
**How Propagates**: Migrations fail, application cannot connect to database

### Solution

**Before** (migrations/20251106_add_user_preferences.sql):
```sql
ALTER TABLE users ADD COLUMN preferences JSONB DEFAULT '{}';
```

**After** (migrations/20251106_add_user_preferences.sql):
```sql
-- Check if users table exists before altering
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users') THEN
        ALTER TABLE users ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}';
    ELSE
        RAISE NOTICE 'Table users does not exist, skipping migration';
    END IF;
END $$;
```

**Better Solution** (ensure proper migration order):
```yaml
# .github/workflows/test.yml
- name: Run Migrations
  run: |
    # Reset database to known state
    npm run db:reset

    # Run all migrations in order
    npm run db:migrate

    # Verify migration
    npm run db:validate
```

### After

- ✅ Migration Status: Success
- ✅ Database Schema: Up to date
- ✅ Tests: All passing with updated schema
- 🎯 Resolution Time: **12 minutes**

---

## Example 14: Security Scan Blocking Deployment (False Positive)

### Before: Initial Error

```
Trivy Security Scan Results:

HIGH: CVE-2023-12345 in openssl (1.1.1k)
Severity: HIGH
CVSS Score: 7.5
Package: openssl@1.1.1k
Fixed Version: 1.1.1l

CRITICAL: CVE-2023-99999 in lodash (4.17.20)
Severity: CRITICAL
CVSS Score: 9.8
Package: lodash@4.17.20
Fixed Version: 4.17.21

Security scan failed: 2 vulnerabilities found (1 CRITICAL, 1 HIGH)

Pipeline: Security Scan
Job: trivy-scan
Policy: Block on CRITICAL
Status: FAILED
```

### Root Cause Analysis (3W1H)

**What Failed**: Security scan during Docker image analysis
**Why Failed**:
  - lodash vulnerability is real → needs update
  - openssl CVE-2023-12345 is false positive (Alpine Linux backported fix)
**When Started**: After weekly security scan
**How Propagates**: Blocks deployment despite false positive

### Solution

**Real Vulnerability Fix** (package.json):
```json
{
  "dependencies": {
    "lodash": "^4.17.21"  // Updated from 4.17.20
  }
}
```

**False Positive Suppression** (.trivyignore):
```
# CVE-2023-12345 in openssl
# Reason: Alpine Linux 3.18 backported the fix to 1.1.1k-r0
# Verified: https://security.alpinelinux.org/vuln/CVE-2023-12345
# Expiry: 2025-12-31
CVE-2023-12345
```

**Workflow Update** (.github/workflows/security.yml):
```yaml
- name: Run Trivy Scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: myapp:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'
    trivyignores: '.trivyignore'

- name: Upload to GitHub Security
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

### After

- ✅ Security Scan: Passing (0 CRITICAL, 0 HIGH after suppression)
- ✅ Deployment: Unblocked
- ✅ Real Vulnerabilities: Fixed (lodash updated)
- ✅ False Positives: Documented and suppressed with expiry
- 🎯 Resolution Time: **18 minutes**

---

## Example 15: Cross-Platform Build Failure (Windows vs Linux)

### Before: Initial Error

```
# Linux build: SUCCESS
# Windows build: FAILED

Run npm run build
> build
> webpack --mode production

ERROR in ./src/utils/paths.ts
Module build failed (from ./node_modules/ts-loader/index.js):
Error: ENOENT: no such file or directory, open 'C:\actions-runner\_work\myapp\myapp\src\config/settings.json'

Compilation failed.

Pipeline: Multi-Platform Build
Job: build-windows
OS: windows-latest
Exit Code: 1
```

### Root Cause Analysis (3W1H)

**What Failed**: Webpack compilation on Windows runner
**Why Failed**: Hardcoded forward slashes `/` in path don't work on Windows
**When Started**: After adding new config file import
**How Propagates**: Windows builds fail, cannot release Windows artifacts

### Solution

**Before** (src/utils/paths.ts):
```typescript
import settings from '../config/settings.json';

export const CONFIG_PATH = '../config/settings.json';
export const loadConfig = () => {
  return require(CONFIG_PATH);
};
```

**After** (src/utils/paths.ts):
```typescript
import path from 'path';
import settings from '../config/settings.json';

// Use path.join for cross-platform compatibility
export const CONFIG_PATH = path.join(__dirname, '..', 'config', 'settings.json');

export const loadConfig = () => {
  return require(CONFIG_PATH);
};
```

**Webpack Config Update** (webpack.config.js):
```javascript
const path = require('path');

module.exports = {
  resolve: {
    alias: {
      '@config': path.resolve(__dirname, 'src/config'),
    },
  },
};
```

**Better Approach** (using cross-platform imports):
```typescript
// Just use ES modules - webpack handles paths
import settings from '@config/settings.json';
```

### After

- ✅ Windows Build: Passing
- ✅ Linux Build: Passing
- ✅ macOS Build: Passing
- ✅ Build Time: 3m 45s (Windows), 2m 12s (Linux)
- 🎯 Resolution Time: **14 minutes**

---

## Summary Metrics

### Resolution Time Distribution

| Time Range | Examples | Percentage |
|------------|----------|------------|
| 0-5 min | 3 examples | 20% |
| 5-10 min | 7 examples | 47% |
| 10-15 min | 4 examples | 27% |
| 15+ min | 1 example | 6% |

**Average Resolution Time**: **9.2 minutes**

### Success Rate by Category

| Category | Examples | Success Rate | Avg Confidence |
|----------|----------|--------------|----------------|
| Dependency Conflicts | 5 | 100% | 0.92 |
| Type Errors | 2 | 100% | 0.88 |
| Test Issues | 1 | 100% | 0.92 |
| Build Failures | 3 | 100% | 0.85 |
| Infrastructure | 4 | 100% | 0.78 |

### Knowledge Base Impact

- **New Patterns Created**: 12
- **Existing Patterns Updated**: 3
- **Confidence Improvements**: +0.15 average increase
- **Cross-Repository Learning**: 8 patterns applicable to multiple projects

---

## Best Practices Demonstrated

### 1. Iterative Fix Approach
- Start with safest, most reversible fix
- Validate each step before proceeding
- Roll back on failure

### 2. Root Cause Analysis
- Always perform 3W1H analysis (What/Why/When/How)
- Identify cascading failures
- Trace regression points

### 3. Knowledge Base Integration
- Update confidence scores after each fix
- Extract reusable patterns
- Document false positives

### 4. Prevention Strategies
- Add automated detection
- Improve CI/CD configuration
- Document known issues

---

For complete error patterns, fix strategies, and multi-agent analysis, see:
- [multi-agent-error-analysis.md](multi-agent-error-analysis.md)
- [error-pattern-library.md](error-pattern-library.md)
- [fix-strategies.md](fix-strategies.md)
- [knowledge-base-system.md](knowledge-base-system.md)
