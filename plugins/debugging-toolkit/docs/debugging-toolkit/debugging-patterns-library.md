# Debugging Patterns Library

> Comprehensive collection of common error patterns, hypothesis generation frameworks, debugging decision trees, and code smell detection strategies

## Table of Contents

1. [Common Error Patterns](#common-error-patterns)
2. [Hypothesis Generation Frameworks](#hypothesis-generation-frameworks)
3. [Debugging Decision Trees](#debugging-decision-trees)
4. [Code Smell Detection](#code-smell-detection)
5. [Before/After Debugging Examples](#beforeafter-debugging-examples)

---

## Common Error Patterns

### 1. NullPointerException / Null Reference Errors

**Signature:**
```
NullPointerException at line X
TypeError: Cannot read property 'Y' of null/undefined
AttributeError: 'NoneType' object has no attribute 'X'
```

**Common Causes:**
- Uninitialized variables
- Missing null checks before dereferencing
- Failed API/database queries returning null
- Race conditions in asynchronous code

**Detection Strategy:**
1. Identify the exact line where null dereference occurs
2. Trace back to where the variable was assigned
3. Check for conditional branches that could skip initialization
4. Review error handling for external dependencies

**Quick Fix Pattern:**
```python
# Before
def process_user(user_id):
    user = db.get_user(user_id)  # May return None
    return user.email  # NullPointerException

# After
def process_user(user_id):
    user = db.get_user(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    return user.email
```

---

### 2. Connection Timeout Errors

**Signature:**
```
ConnectionTimeoutError: Connection to host:port timed out after N seconds
requests.exceptions.Timeout: HTTPSConnectionPool(host='api.example.com', port=443)
socket.timeout: timed out
```

**Common Causes:**
- Network connectivity issues
- Overloaded downstream services
- Firewall/security group misconfigurations
- DNS resolution failures
- Incorrect timeout configuration

**Detection Strategy:**
1. Check network connectivity: `ping`, `telnet`, `curl`
2. Verify DNS resolution: `nslookup`, `dig`
3. Review timeout configuration values
4. Check service health status and response times
5. Analyze connection pool exhaustion

**Diagnostic Commands:**
```bash
# Test connectivity
curl -v -m 5 https://api.example.com/health

# Check DNS resolution
nslookup api.example.com

# Test port connectivity
telnet api.example.com 443

# Trace network route
traceroute api.example.com
```

---

### 3. Memory Leak Patterns

**Signature:**
```
OutOfMemoryError: Java heap space
MemoryError: Unable to allocate X bytes
Process killed (OOM Killer)
Gradual memory growth over time in monitoring
```

**Common Causes:**
- Unclosed resources (files, connections, streams)
- Growing collections never cleared
- Event listeners not removed
- Circular references preventing garbage collection
- Large object retention in caches

**Detection Strategy:**
1. Monitor memory usage over time (heap dumps)
2. Analyze object retention with profilers
3. Check for resource cleanup in finally blocks
4. Review cache eviction policies
5. Inspect event listener registration/deregistration

**Example Pattern:**
```python
# Before - Memory Leak
class DataProcessor:
    def __init__(self):
        self.cache = {}  # Unbounded cache

    def process(self, key, data):
        self.cache[key] = data  # Never evicted
        return transform(data)

# After - Fixed
from cachetools import LRUCache

class DataProcessor:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)  # Bounded cache

    def process(self, key, data):
        self.cache[key] = data
        return transform(data)
```

---

### 4. Race Condition / Concurrency Errors

**Signature:**
```
ConcurrentModificationException
Race condition detected in variable X
Data race at memory address 0x...
Inconsistent state: expected X, got Y
```

**Common Causes:**
- Shared mutable state accessed by multiple threads
- Missing synchronization/locks
- Incorrect lock ordering (deadlocks)
- Atomic operation assumptions
- Non-thread-safe data structures

**Detection Strategy:**
1. Identify shared mutable state
2. Trace concurrent access patterns
3. Use thread sanitizers and race detectors
4. Add synchronization logging
5. Reproduce with stress testing

**Fix Pattern:**
```python
# Before - Race Condition
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1  # Not atomic!

# After - Thread-Safe
from threading import Lock

class Counter:
    def __init__(self):
        self.count = 0
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.count += 1
```

---

### 5. Database Deadlock Errors

**Signature:**
```
Deadlock detected when trying to get lock
Transaction deadlock: Resource X is locked by transaction Y
Database error: deadlock detected
```

**Common Causes:**
- Inconsistent lock acquisition order
- Long-running transactions holding locks
- Missing indexes causing table locks
- High contention on hot rows
- Circular dependencies in transaction logic

**Detection Strategy:**
1. Analyze database deadlock logs
2. Identify lock wait chains
3. Review transaction isolation levels
4. Check for missing indexes
5. Analyze transaction duration and frequency

**Prevention Pattern:**
```sql
-- Before - Potential Deadlock
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- After - Consistent Lock Ordering
BEGIN TRANSACTION;
-- Always lock in ascending ID order
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Or use explicit locking
BEGIN TRANSACTION;
SELECT * FROM accounts WHERE id IN (1, 2) ORDER BY id FOR UPDATE;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

---

### 6. Authentication / Authorization Failures

**Signature:**
```
401 Unauthorized
403 Forbidden: Access denied
AuthenticationError: Invalid credentials
TokenExpiredError: JWT token has expired
```

**Common Causes:**
- Expired tokens/sessions
- Missing or incorrect credentials
- Insufficient permissions
- Token/session not refreshed
- Clock skew issues with token validation

**Detection Strategy:**
1. Check token expiration timestamps
2. Verify credentials are being sent correctly
3. Review permission configuration
4. Check token refresh logic
5. Analyze authentication middleware

**Fix Pattern:**
```javascript
// Before - No Token Refresh
async function apiCall(url) {
  const token = localStorage.getItem('token');
  return fetch(url, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
}

// After - With Token Refresh
async function apiCall(url) {
  let token = localStorage.getItem('token');

  // Check if token is expired
  if (isTokenExpired(token)) {
    token = await refreshToken();
    localStorage.setItem('token', token);
  }

  return fetch(url, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
}
```

---

### 7. API Rate Limiting Errors

**Signature:**
```
429 Too Many Requests
RateLimitExceeded: API rate limit exceeded
Error: You have exceeded your request quota
```

**Common Causes:**
- Too many requests in time window
- Missing rate limit handling
- Inefficient API usage patterns
- Lack of request batching
- Missing exponential backoff

**Detection Strategy:**
1. Check API response headers (X-RateLimit-*)
2. Monitor request frequency
3. Identify burst patterns
4. Review retry logic
5. Analyze API usage efficiency

**Fix Pattern:**
```python
# Before - No Rate Limiting
def fetch_users(user_ids):
    users = []
    for uid in user_ids:
        users.append(api.get_user(uid))  # Individual calls
    return users

# After - Batched with Backoff
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60),
       stop=stop_after_attempt(5))
def fetch_users(user_ids):
    # Batch API call
    return api.get_users_batch(user_ids, batch_size=100)
```

---

### 8. JSON Parsing Errors

**Signature:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
SyntaxError: Unexpected token < in JSON at position 0
json.decoder.JSONDecodeError: Extra data
```

**Common Causes:**
- Receiving HTML error page instead of JSON
- Malformed JSON response
- Incorrect Content-Type header
- Empty response body
- Partial response due to truncation

**Detection Strategy:**
1. Log raw response body before parsing
2. Check Content-Type header
3. Verify response status code
4. Inspect for HTML error pages
5. Check for response truncation

**Fix Pattern:**
```javascript
// Before - Blind JSON Parsing
const response = await fetch('/api/data');
const data = response.json();  // May fail

// After - Safe JSON Parsing
const response = await fetch('/api/data');

if (!response.ok) {
  const text = await response.text();
  throw new Error(`API error (${response.status}): ${text}`);
}

const contentType = response.headers.get('content-type');
if (!contentType || !contentType.includes('application/json')) {
  throw new Error(`Expected JSON, got ${contentType}`);
}

const data = await response.json();
```

---

### 9. File I/O Errors

**Signature:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'file.txt'
PermissionError: [Errno 13] Permission denied: 'file.txt'
IOError: [Errno 28] No space left on device
```

**Common Causes:**
- Incorrect file paths
- Missing file permissions
- Disk space exhaustion
- Files locked by other processes
- Network file system issues

**Detection Strategy:**
1. Verify file path existence
2. Check file permissions
3. Monitor disk space
4. Review file locking mechanisms
5. Test with absolute paths

**Fix Pattern:**
```python
# Before - No Error Handling
def read_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# After - Robust Error Handling
import os
from pathlib import Path

def read_config():
    config_path = Path(__file__).parent / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not os.access(config_path, os.R_OK):
        raise PermissionError(f"Cannot read config file: {config_path}")

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config: {e}")
```

---

### 10. Infinite Loop / Hang Patterns

**Signature:**
```
Process hanging with 100% CPU usage
Thread stuck in infinite loop
Application unresponsive
Stack trace shows repeated function calls
```

**Common Causes:**
- Incorrect loop termination conditions
- Waiting for condition that never becomes true
- Deadlocks preventing progress
- Recursive function without base case
- Busy-wait loops

**Detection Strategy:**
1. Attach debugger and pause execution
2. Analyze stack traces
3. Check loop termination conditions
4. Review recursive function logic
5. Monitor thread states

**Fix Pattern:**
```python
# Before - Potential Infinite Loop
def wait_for_completion(task_id):
    while True:
        status = get_task_status(task_id)
        if status == 'completed':
            break
        time.sleep(1)

# After - Timeout Protection
import time

def wait_for_completion(task_id, timeout=300):
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

        status = get_task_status(task_id)
        if status == 'completed':
            return True
        elif status == 'failed':
            raise RuntimeError(f"Task {task_id} failed")

        time.sleep(1)
```

---

### 11. SQL Injection Vulnerabilities

**Signature:**
```
Unexpected SQL syntax errors
Data modification by unauthorized users
SQL error messages in application logs
Database warnings about query execution
```

**Common Causes:**
- String concatenation for SQL queries
- Unsanitized user input
- Dynamic query building without parameterization
- Lack of input validation

**Detection Strategy:**
1. Search for string concatenation in SQL queries
2. Test with SQL injection payloads
3. Use static analysis tools
4. Review database query logs
5. Check for parameterized queries

**Fix Pattern:**
```python
# Before - SQL Injection Vulnerability
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)
    # Vulnerable to: username = "' OR '1'='1"

# After - Parameterized Query
def get_user(username):
    query = "SELECT * FROM users WHERE username = ?"
    return db.execute(query, (username,))
```

---

### 12. Type Coercion Errors

**Signature:**
```
TypeError: unsupported operand type(s)
ValueError: invalid literal for int() with base 10
Type 'string' is not assignable to type 'number'
```

**Common Causes:**
- Implicit type conversions
- String vs number comparisons
- Missing type validation
- API response type assumptions

**Detection Strategy:**
1. Check variable types at error point
2. Review type conversion logic
3. Validate input data types
4. Use type hints/annotations
5. Enable strict type checking

**Fix Pattern:**
```typescript
// Before - Type Confusion
function calculateTotal(price, quantity) {
  return price * quantity;  // Fails if quantity is string "5"
}

// After - Type Safety
function calculateTotal(price: number, quantity: number): number {
  const validPrice = Number(price);
  const validQuantity = Number(quantity);

  if (isNaN(validPrice) || isNaN(validQuantity)) {
    throw new TypeError('Price and quantity must be valid numbers');
  }

  return validPrice * validQuantity;
}
```

---

### 13. Environment Configuration Errors

**Signature:**
```
Error: Missing required environment variable 'DATABASE_URL'
Configuration error: Invalid API endpoint
UnknownHostException: Cannot resolve hostname from config
```

**Common Causes:**
- Missing environment variables
- Incorrect configuration file paths
- Environment-specific configuration not loaded
- Hardcoded values instead of config
- Case sensitivity in variable names

**Detection Strategy:**
1. Verify all required environment variables
2. Check configuration loading order
3. Review default values
4. Validate configuration schema
5. Test across environments

**Fix Pattern:**
```python
# Before - No Validation
DATABASE_URL = os.environ['DATABASE_URL']
API_KEY = os.environ['API_KEY']

# After - With Validation and Defaults
import os
from typing import Optional

class Config:
    def __init__(self):
        self.database_url = self._get_required('DATABASE_URL')
        self.api_key = self._get_required('API_KEY')
        self.debug_mode = self._get_optional('DEBUG_MODE', 'false') == 'true'

    def _get_required(self, key: str) -> str:
        value = os.environ.get(key)
        if not value:
            raise EnvironmentError(f"Missing required environment variable: {key}")
        return value

    def _get_optional(self, key: str, default: str) -> str:
        return os.environ.get(key, default)

config = Config()
```

---

### 14. Asynchronous Operation Errors

**Signature:**
```
UnhandledPromiseRejectionWarning
RuntimeError: This event loop is already running
asyncio.TimeoutError: Task took too long to complete
```

**Common Causes:**
- Unhandled promise rejections
- Mixing sync and async code incorrectly
- Missing await keywords
- Event loop blocking operations
- Callback hell without proper error handling

**Detection Strategy:**
1. Search for unhandled promise rejections
2. Review async/await usage
3. Check for blocking operations in async functions
4. Verify error propagation
5. Use async debugging tools

**Fix Pattern:**
```javascript
// Before - Unhandled Rejection
async function fetchData() {
  fetch('/api/data').then(response => response.json());
  // No error handling
}

// After - Proper Async Error Handling
async function fetchData() {
  try {
    const response = await fetch('/api/data');

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to fetch data:', error);
    throw error;  // Re-throw for caller to handle
  }
}
```

---

### 15. Cross-Origin Resource Sharing (CORS) Errors

**Signature:**
```
Access to fetch at 'X' from origin 'Y' has been blocked by CORS policy
No 'Access-Control-Allow-Origin' header is present
CORS preflight request failed
```

**Common Causes:**
- Missing CORS headers on server
- Incorrect allowed origins configuration
- Preflight request not handled
- Credentials not properly configured
- HTTP vs HTTPS mismatch

**Detection Strategy:**
1. Check browser console for CORS errors
2. Inspect network tab for preflight OPTIONS requests
3. Verify Access-Control-* headers
4. Review server CORS configuration
5. Test with CORS proxy

**Fix Pattern:**
```javascript
// Backend (Express.js) - Proper CORS Configuration
const cors = require('cors');

// Before - No CORS Configuration
app.get('/api/data', (req, res) => {
  res.json({ data: 'value' });
});

// After - Proper CORS Setup
const corsOptions = {
  origin: ['https://app.example.com', 'https://staging.example.com'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

app.use(cors(corsOptions));

app.get('/api/data', (req, res) => {
  res.json({ data: 'value' });
});
```

---

## Hypothesis Generation Frameworks

### Framework 1: The 5 Whys Technique

**Purpose:** Drill down to root cause by repeatedly asking "why"

**Process:**
1. State the problem clearly
2. Ask "Why did this happen?"
3. For each answer, ask "Why?" again
4. Continue 5 times or until root cause found
5. Verify root cause addresses the problem

**Example:**
```
Problem: Website is slow for users
1. Why? → The database queries are taking too long
2. Why? → There are no indexes on frequently queried columns
3. Why? → The database schema was designed without performance testing
4. Why? → Performance requirements were not defined before development
5. Why? → The project lacked a performance testing phase in the SDLC

Root Cause: Missing performance requirements and testing process
```

---

### Framework 2: Fault Tree Analysis

**Purpose:** Visual decomposition of failure scenarios using boolean logic

**Process:**
1. Define the top-level failure event
2. Identify immediate causes (AND/OR gates)
3. Decompose each cause into sub-causes
4. Continue until basic events (root causes) reached
5. Calculate failure probabilities if quantitative

**Example Structure:**
```
[Top Event: Payment Processing Failure]
         |
     OR gate
    /    |    \
   /     |     \
[DB      [API   [Network
 Down]   Timeout] Error]
         |
     AND gate
      /    \
 [High    [No
 Load]   Cache]
```

---

### Framework 3: Timeline Reconstruction

**Purpose:** Chronologically map events leading to the error

**Process:**
1. Establish error timestamp
2. Collect logs from all relevant systems
3. Align timestamps (adjust for time zones)
4. Identify events before/during/after error
5. Look for correlations and causation

**Template:**
```
T-300s: Normal operation (100 req/s)
T-120s: Deploy new version to 20% of servers
T-90s:  Memory usage increases on new version
T-60s:  First timeout errors reported
T-30s:  Error rate reaches 5%
T-0s:   Circuit breaker trips, traffic rerouted
T+30s:  Errors stop, service stabilizes
T+300s: Rollback initiated

Hypothesis: New version has memory leak causing timeouts
```

---

### Framework 4: Differential Diagnosis

**Purpose:** Compare working vs failing scenarios to isolate variables

**Process:**
1. Identify a working scenario
2. List all variables (env, config, data, code, etc.)
3. Compare failing scenario variables
4. Identify differences
5. Test by changing one variable at a time

**Comparison Matrix:**
```
Variable          | Working | Failing | Impact
------------------|---------|---------|--------
Environment       | Staging | Prod    | HIGH
Python Version    | 3.9     | 3.9     | None
Database Version  | 5.7     | 8.0     | HIGH
Request Size      | <100KB  | >1MB    | HIGH
TLS Version       | 1.2     | 1.3     | Low
Load Balancer     | ALB     | ALB     | None

Hypotheses: Database version incompatibility or large request handling issue
```

---

### Framework 5: Ishikawa (Fishbone) Diagram

**Purpose:** Categorize potential causes by type

**Categories:**
- **People:** Human error, skill gaps, training
- **Process:** Workflows, procedures, standards
- **Technology:** Code, infrastructure, tools
- **Environment:** Configuration, dependencies, external services
- **Data:** Input quality, schema, volume

**Example:**
```
                    [Payment Failure]
                          |
      People ----/        |        \---- Technology
                          |
      Process ---\        |        /---- Environment
                          |
                      Data

People:
- Developer unfamiliar with payment API
- No code review performed

Process:
- Missing integration testing
- No rollback procedure

Technology:
- Library version incompatibility
- Missing error handling

Environment:
- API endpoint configuration incorrect
- Network firewall blocking requests

Data:
- Invalid currency codes in database
- Missing required fields
```

---

## Debugging Decision Trees

### Decision Tree 1: Error Origin Classification

```
Is the error reproducible?
├─ Yes → Is it environment-specific?
│   ├─ Yes → Configuration/Environment issue
│   │   └─ Check: env vars, configs, dependencies
│   └─ No → Code logic issue
│       └─ Check: code paths, business logic, algorithms
└─ No → Is it time-based or load-based?
    ├─ Time-based → Race condition or timing issue
    │   └─ Check: concurrency, async operations, caching
    └─ Load-based → Resource exhaustion or scaling issue
        └─ Check: memory, connections, rate limits
```

---

### Decision Tree 2: Performance Issue Classification

```
Is the issue latency or throughput?
├─ Latency → Where is time spent?
│   ├─ Database → Run query analysis
│   │   ├─ Missing indexes → Add indexes
│   │   ├─ N+1 queries → Implement eager loading
│   │   └─ Complex joins → Optimize query or denormalize
│   ├─ External API → Check network and API performance
│   │   ├─ Slow API → Implement caching or async calls
│   │   ├─ Network → Check connectivity and DNS
│   │   └─ Serialization → Optimize payload size
│   └─ Application Code → Profile code execution
│       ├─ Algorithm → Use more efficient algorithm
│       ├─ I/O → Implement buffering or streaming
│       └─ Computation → Cache results or use approximation
└─ Throughput → What is the bottleneck?
    ├─ CPU → Optimize algorithms or scale horizontally
    ├─ Memory → Reduce memory usage or increase capacity
    ├─ Network → Implement compression or CDN
    └─ Database Connections → Implement connection pooling
```

---

### Decision Tree 3: Data Integrity Issues

```
Is data missing, incorrect, or corrupted?
├─ Missing → Where did data loss occur?
│   ├─ Input validation → Add required field validation
│   ├─ Database constraint → Review DB constraints and migrations
│   └─ Application logic → Check for null handling
├─ Incorrect → When did data become incorrect?
│   ├─ At input → Strengthen validation rules
│   ├─ During processing → Review transformation logic
│   └─ At storage → Check serialization/deserialization
└─ Corrupted → How was data corrupted?
    ├─ Concurrent modification → Implement locking or versioning
    ├─ Failed transaction → Review transaction boundaries
    └─ Storage failure → Check data integrity and backups
```

---

## Code Smell Detection

### Smell 1: God Object/Class

**Indicators:**
- Class with >1000 lines or >20 methods
- Class name contains "Manager", "Handler", "Utility"
- High coupling (depends on many other classes)
- Low cohesion (methods unrelated to each other)

**Impact:** Hard to understand, test, and maintain

**Refactoring:**
```python
# Before - God Class
class UserManager:
    def create_user(self): ...
    def delete_user(self): ...
    def send_email(self): ...
    def process_payment(self): ...
    def generate_report(self): ...
    def log_activity(self): ...
    # 20+ more methods

# After - Single Responsibility
class UserService:
    def create_user(self): ...
    def delete_user(self): ...

class EmailService:
    def send_email(self): ...

class PaymentService:
    def process_payment(self): ...

class ReportGenerator:
    def generate_report(self): ...

class ActivityLogger:
    def log_activity(self): ...
```

---

### Smell 2: Deeply Nested Conditionals

**Indicators:**
- Indentation depth >4 levels
- Multiple nested if/else statements
- Hard to follow logic flow

**Impact:** Cognitive overload, error-prone modifications

**Refactoring:**
```python
# Before - Deep Nesting
def process_order(order):
    if order is not None:
        if order.status == 'pending':
            if order.items:
                if order.payment_method:
                    if order.payment_method.is_valid():
                        # Process order
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

# After - Guard Clauses
def process_order(order):
    if order is None:
        return False
    if order.status != 'pending':
        return False
    if not order.items:
        return False
    if not order.payment_method:
        return False
    if not order.payment_method.is_valid():
        return False

    # Process order
    return True
```

---

### Smell 3: Magic Numbers/Strings

**Indicators:**
- Hardcoded numeric or string literals
- Unclear meaning without context
- Same value repeated multiple times

**Impact:** Hard to maintain, prone to errors

**Refactoring:**
```javascript
// Before - Magic Numbers
function calculatePrice(quantity) {
  if (quantity > 100) {
    return quantity * 9.99 * 0.9;  // What are these numbers?
  }
  return quantity * 9.99;
}

// After - Named Constants
const UNIT_PRICE = 9.99;
const BULK_DISCOUNT_THRESHOLD = 100;
const BULK_DISCOUNT_RATE = 0.10;

function calculatePrice(quantity) {
  const basePrice = quantity * UNIT_PRICE;

  if (quantity >= BULK_DISCOUNT_THRESHOLD) {
    return basePrice * (1 - BULK_DISCOUNT_RATE);
  }

  return basePrice;
}
```

---

### Smell 4: Long Parameter Lists

**Indicators:**
- Functions with >4 parameters
- Boolean flags changing behavior
- Parameters that always passed together

**Impact:** Hard to call, easy to pass wrong arguments

**Refactoring:**
```typescript
// Before - Long Parameter List
function createUser(
  firstName: string,
  lastName: string,
  email: string,
  age: number,
  address: string,
  city: string,
  country: string,
  sendWelcomeEmail: boolean
) { ... }

// After - Parameter Object
interface UserProfile {
  firstName: string;
  lastName: string;
  email: string;
  age: number;
  address: Address;
}

interface Address {
  street: string;
  city: string;
  country: string;
}

interface UserCreationOptions {
  profile: UserProfile;
  sendWelcomeEmail?: boolean;
}

function createUser(options: UserCreationOptions) { ... }
```

---

### Smell 5: Duplicate Code

**Indicators:**
- Same code block in multiple places
- Copy-paste patterns
- Similar logic with slight variations

**Impact:** Multiple places to fix bugs, inconsistency

**Refactoring:**
```python
# Before - Duplication
def validate_user_email(email):
    if not email:
        return False
    if '@' not in email:
        return False
    if '.' not in email.split('@')[1]:
        return False
    return True

def validate_admin_email(email):
    if not email:
        return False
    if '@' not in email:
        return False
    if '.' not in email.split('@')[1]:
        return False
    if not email.endswith('@company.com'):
        return False
    return True

# After - Extract Common Logic
def is_valid_email_format(email):
    if not email:
        return False
    if '@' not in email:
        return False
    parts = email.split('@')
    if len(parts) != 2:
        return False
    if '.' not in parts[1]:
        return False
    return True

def validate_user_email(email):
    return is_valid_email_format(email)

def validate_admin_email(email):
    return is_valid_email_format(email) and email.endswith('@company.com')
```

---

## Before/After Debugging Examples

### Example 1: Fixing N+1 Query Problem

**Problem:** Application slows down with list of users

**Before:**
```python
# views.py
def get_users_with_orders(request):
    users = User.objects.all()  # 1 query
    result = []
    for user in users:
        orders = user.orders.all()  # N queries (one per user)
        result.append({
            'user': user.name,
            'order_count': len(orders)
        })
    return JsonResponse(result, safe=False)

# Performance: 1 + N queries, ~500ms for 100 users
```

**After:**
```python
# views.py
from django.db.models import Count

def get_users_with_orders(request):
    users = User.objects.annotate(
        order_count=Count('orders')
    ).all()  # 1 query with JOIN

    result = [
        {'user': user.name, 'order_count': user.order_count}
        for user in users
    ]
    return JsonResponse(result, safe=False)

# Performance: 1 query, ~50ms for 100 users (10x faster)
```

---

### Example 2: Fixing Memory Leak in React Component

**Problem:** Memory grows indefinitely on page with timer

**Before:**
```javascript
function Dashboard() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date());
    }, 1000);
    // Missing cleanup!
  }, []);

  return <div>Current time: {time.toLocaleTimeString()}</div>;
}

// Memory leak: interval keeps running even after component unmounts
```

**After:**
```javascript
function Dashboard() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date());
    }, 1000);

    // Cleanup function
    return () => {
      clearInterval(interval);
    };
  }, []);

  return <div>Current time: {time.toLocaleTimeString()}</div>;
}

// Fixed: interval is cleared when component unmounts
```

---

### Example 3: Fixing Race Condition in Order Processing

**Problem:** Order sometimes processed twice

**Before:**
```python
def process_order(order_id):
    order = Order.objects.get(id=order_id)

    if order.status == 'pending':
        # Race condition: two workers can both see 'pending'
        charge_payment(order)
        order.status = 'processed'
        order.save()

# Problem: Between checking status and updating, another worker
# can start processing the same order
```

**After:**
```python
from django.db import transaction
from django.db.models import F

def process_order(order_id):
    with transaction.atomic():
        # Select for update locks the row
        order = Order.objects.select_for_update().get(id=order_id)

        if order.status == 'pending':
            charge_payment(order)
            order.status = 'processed'
            order.save()

    # Alternative: Optimistic locking
    # updated = Order.objects.filter(
    #     id=order_id,
    #     status='pending'
    # ).update(status='processing')
    #
    # if updated == 1:
    #     order = Order.objects.get(id=order_id)
    #     charge_payment(order)
    #     order.status = 'processed'
    #     order.save()

# Fixed: Database lock prevents concurrent processing
```

---

### Example 4: Fixing Unhandled Promise Rejection

**Problem:** Application crashes on API error

**Before:**
```javascript
async function loadUserData(userId) {
  const response = await fetch(`/api/users/${userId}`);
  const user = await response.json();
  updateUI(user);
}

// Problem: If fetch fails or returns non-200, promise rejects
// and error is unhandled
```

**After:**
```javascript
async function loadUserData(userId) {
  try {
    const response = await fetch(`/api/users/${userId}`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const user = await response.json();
    updateUI(user);
  } catch (error) {
    console.error('Failed to load user data:', error);
    showErrorMessage('Unable to load user data. Please try again.');

    // Optionally report to error tracking
    reportError(error, { userId, context: 'loadUserData' });
  }
}

// Fixed: All errors are caught and handled appropriately
```

---

### Example 5: Fixing SQL Injection Vulnerability

**Problem:** User input concatenated into SQL query

**Before:**
```python
def search_products(search_term):
    query = f"SELECT * FROM products WHERE name LIKE '%{search_term}%'"
    return db.execute(query).fetchall()

# Vulnerability: search_term = "'; DROP TABLE products; --"
# Results in: SELECT * FROM products WHERE name LIKE '%'; DROP TABLE products; --%'
```

**After:**
```python
def search_products(search_term):
    query = "SELECT * FROM products WHERE name LIKE ?"
    safe_term = f"%{search_term}%"
    return db.execute(query, (safe_term,)).fetchall()

# Or using ORM:
def search_products(search_term):
    return Product.objects.filter(name__icontains=search_term)

# Fixed: Parameterized queries prevent SQL injection
```

---

## Quick Reference: Error Pattern Signatures

| Error Pattern | Key Signature | First Check |
|---------------|--------------|-------------|
| Null Reference | `NullPointerException`, `TypeError: ... null/undefined` | Variable initialization |
| Timeout | `TimeoutError`, `Connection timed out` | Network connectivity |
| Memory Leak | `OutOfMemoryError`, gradual memory growth | Resource cleanup |
| Race Condition | `ConcurrentModificationException`, inconsistent state | Shared mutable state |
| Deadlock | `Deadlock detected`, transaction hangs | Lock ordering |
| Auth Failure | `401 Unauthorized`, `403 Forbidden` | Token expiration |
| Rate Limit | `429 Too Many Requests` | Request frequency |
| JSON Parse | `JSONDecodeError`, `Unexpected token` | Response content-type |
| File I/O | `FileNotFoundError`, `PermissionError` | File path existence |
| Infinite Loop | 100% CPU, unresponsive | Loop termination |
| SQL Injection | Unexpected SQL errors | Query parameterization |
| Type Error | `TypeError`, `ValueError` | Type validation |
| Config Error | Missing env variable errors | Environment setup |
| Async Error | `UnhandledPromiseRejection` | Error handling |
| CORS Error | CORS policy blocked | Server headers |

---

## Usage Guidelines

**When to use this library:**
- During error triage to quickly identify error patterns
- When generating debugging hypotheses
- To select appropriate debugging strategies
- For learning common debugging patterns
- When training team members on debugging techniques

**How to reference:**
1. Identify error signature from logs
2. Find matching error pattern
3. Review detection strategy
4. Apply fix pattern or generate hypothesis
5. Use decision trees for complex scenarios
6. Check for related code smells

**Integration with smart-debug workflow:**
- **Step 1 (Triage):** Use error pattern signatures
- **Step 3 (Hypothesis):** Apply hypothesis generation frameworks
- **Step 4 (Strategy):** Use debugging decision trees
- **Step 8 (Fix):** Reference before/after examples and fix patterns
