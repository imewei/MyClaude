---
name: error-handling-patterns
version: "1.0.7"
description: Master error handling patterns including exception hierarchies, Result types, retry with exponential backoff, circuit breakers, graceful degradation, error aggregation, and structured error responses. Use when implementing fault-tolerant systems, designing API error responses, or building resilient distributed applications.
---

# Error Handling Patterns

Build resilient applications with robust error handling strategies.

## Error Categories

| Type | Examples | Strategy |
|------|----------|----------|
| Recoverable | Network timeout, rate limit, missing file | Retry, fallback |
| Validation | Invalid input, format errors | Return error, don't throw |
| Unrecoverable | OOM, stack overflow, bugs | Fail fast, log, alert |

## Python Patterns

### Custom Exception Hierarchy
```python
class ApplicationError(Exception):
    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

class ValidationError(ApplicationError): pass
class NotFoundError(ApplicationError): pass
class ExternalServiceError(ApplicationError):
    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(message, **kwargs)
        self.service = service
```

### Context Manager Cleanup
```python
@contextmanager
def database_transaction(session):
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

### Retry with Backoff
```python
def retry(max_attempts=3, backoff_factor=2.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt < max_attempts - 1:
                        time.sleep(backoff_factor ** attempt)
                    else:
                        raise
        return wrapper
    return decorator
```

## TypeScript Patterns

### Result Type
```typescript
type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };

function Ok<T>(value: T): Result<T, never> { return { ok: true, value }; }
function Err<E>(error: E): Result<never, E> { return { ok: false, error }; }

// Usage
const result = parseJSON<User>(json);
if (result.ok) console.log(result.value.name);
else console.error(result.error.message);
```

### Custom Error Classes
```typescript
class ApplicationError extends Error {
    constructor(message: string, public code: string, public statusCode = 500) {
        super(message);
        this.name = this.constructor.name;
    }
}

class NotFoundError extends ApplicationError {
    constructor(resource: string, id: string) {
        super(`${resource} not found`, 'NOT_FOUND', 404);
    }
}
```

## Go Pattern

```go
// Custom error type
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

// Sentinel errors
var ErrNotFound = errors.New("not found")

// Error wrapping
if err != nil {
    return fmt.Errorf("process failed: %w", err)
}
```

## Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60, success_threshold=2):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout)
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

    def call(self, func):
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        try:
            result = func()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

## Error Aggregation

```typescript
class ErrorCollector {
    private errors: Error[] = [];
    add(error: Error) { this.errors.push(error); }
    hasErrors() { return this.errors.length > 0; }
    throw() {
        if (this.errors.length === 1) throw this.errors[0];
        throw new AggregateError(this.errors);
    }
}

function validateUser(data: any) {
    const errors = new ErrorCollector();
    if (!data.email) errors.add(new ValidationError('Email required'));
    if (!data.name) errors.add(new ValidationError('Name required'));
    if (errors.hasErrors()) errors.throw();
}
```

## Graceful Degradation

```python
def with_fallback(primary, fallback, log_error=True):
    try:
        return primary()
    except Exception as e:
        if log_error: logger.error(f"Primary failed: {e}")
        return fallback()

# Multiple fallbacks
def get_rate(currency):
    return (try_fn(lambda: api1.get(currency))
         or try_fn(lambda: api2.get(currency))
         or try_fn(lambda: cache.get(currency))
         or DEFAULT_RATE)
```

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Fail fast | Validate input early |
| Preserve context | Include stack traces, metadata |
| Meaningful messages | Explain what happened and how to fix |
| Handle at right level | Catch where you can meaningfully handle |
| Clean up resources | Use try-finally, context managers |
| Don't swallow errors | Log or re-throw, never ignore |
| Type-safe errors | Use typed errors when possible |

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| `except Exception` | Hides bugs |
| Empty catch blocks | Silent failures |
| Logging AND re-throwing | Duplicate logs |
| Poor error messages | "Error occurred" not helpful |
| Ignoring async errors | Unhandled promise rejections |

## Checklist

- [ ] Custom exception hierarchy defined
- [ ] Context managers for resource cleanup
- [ ] Retry logic for transient failures
- [ ] Circuit breakers for external services
- [ ] Graceful degradation with fallbacks
- [ ] Structured error logging
- [ ] Error aggregation for validation
