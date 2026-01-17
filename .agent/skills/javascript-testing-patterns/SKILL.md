---
name: javascript-testing-patterns
version: "1.0.7"
description: Comprehensive testing with Jest/Vitest and Testing Library. Unit tests, integration tests, mocking, fixtures, and TDD workflows. Use when writing tests, setting up test infrastructure, or implementing testing best practices.
---

# JavaScript Testing Patterns

Robust testing strategies for JavaScript/TypeScript applications with modern frameworks.

## Framework Selection

| Framework | Use Case | Speed |
|-----------|----------|-------|
| Vitest | Vite projects | Fast |
| Jest | General purpose | Moderate |
| Testing Library | Component tests | - |
| Supertest | API integration | - |

## Vitest Configuration

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
      thresholds: { branches: 80, functions: 80, lines: 80 }
    },
    setupFiles: ['./src/test/setup.ts']
  }
});
```

## Unit Testing Patterns

### Pure Functions

```typescript
describe('Calculator', () => {
  it('should add two numbers', () => {
    expect(add(2, 3)).toBe(5);
  });

  it('should throw on division by zero', () => {
    expect(() => divide(10, 0)).toThrow('Division by zero');
  });
});
```

### Class Testing

```typescript
describe('UserService', () => {
  let service: UserService;

  beforeEach(() => {
    service = new UserService();
  });

  it('should create user', () => {
    const user = service.create({ id: '1', name: 'John' });
    expect(service.findById('1')).toEqual(user);
  });

  it('should throw if user exists', () => {
    service.create({ id: '1', name: 'John' });
    expect(() => service.create({ id: '1', name: 'Jane' })).toThrow('User already exists');
  });
});
```

### Async Testing

```typescript
describe('ApiService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should fetch user', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: '1', name: 'John' })
    });

    const user = await service.fetchUser('1');
    expect(user).toEqual({ id: '1', name: 'John' });
  });

  it('should throw on not found', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false });
    await expect(service.fetchUser('999')).rejects.toThrow('User not found');
  });
});
```

## Mocking Patterns

### Module Mocking

```typescript
vi.mock('nodemailer', () => ({
  default: {
    createTransport: vi.fn(() => ({
      sendMail: vi.fn().mockResolvedValue({ messageId: '123' })
    }))
  }
}));
```

### Dependency Injection

```typescript
describe('UserService', () => {
  let service: UserService;
  let mockRepository: IUserRepository;

  beforeEach(() => {
    mockRepository = { findById: vi.fn(), create: vi.fn() };
    service = new UserService(mockRepository);
  });

  it('should return user if found', async () => {
    vi.mocked(mockRepository.findById).mockResolvedValue({ id: '1', name: 'John' });
    const user = await service.getUser('1');
    expect(user.name).toBe('John');
  });
});
```

### Spying

```typescript
const loggerSpy = vi.spyOn(logger, 'info');

await service.processOrder('123');

expect(loggerSpy).toHaveBeenCalledWith('Processing order 123');
expect(loggerSpy).toHaveBeenCalledTimes(2);
```

## Integration Testing

### API Tests with Supertest

```typescript
describe('User API', () => {
  beforeEach(async () => {
    await pool.query('TRUNCATE TABLE users CASCADE');
  });

  it('should create user', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({ name: 'John', email: 'john@example.com', password: 'pass123' })
      .expect(201);

    expect(response.body).toHaveProperty('id');
    expect(response.body.email).toBe('john@example.com');
  });

  it('should return 409 if email exists', async () => {
    await request(app).post('/api/users').send({ name: 'John', email: 'john@example.com', password: 'pass' });
    await request(app).post('/api/users').send({ name: 'Jane', email: 'john@example.com', password: 'pass' }).expect(409);
  });

  it('should require auth for protected routes', async () => {
    await request(app).get('/api/users/me').expect(401);
  });
});
```

## React Component Testing

```typescript
import { render, screen, fireEvent } from '@testing-library/react';

describe('UserForm', () => {
  it('should render form inputs', () => {
    render(<UserForm onSubmit={vi.fn()} />);
    expect(screen.getByPlaceholderText('Name')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
  });

  it('should call onSubmit with form data', () => {
    const onSubmit = vi.fn();
    render(<UserForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByTestId('name-input'), { target: { value: 'John' } });
    fireEvent.change(screen.getByTestId('email-input'), { target: { value: 'john@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

    expect(onSubmit).toHaveBeenCalledWith({ name: 'John', email: 'john@example.com' });
  });
});
```

### Hook Testing

```typescript
import { renderHook, act } from '@testing-library/react';

describe('useCounter', () => {
  it('should increment', () => {
    const { result } = renderHook(() => useCounter(0));
    act(() => result.current.increment());
    expect(result.current.count).toBe(1);
  });
});
```

## Test Fixtures

```typescript
import { faker } from '@faker-js/faker';

export function createUserFixture(overrides?: Partial<User>): User {
  return {
    id: faker.string.uuid(),
    name: faker.person.fullName(),
    email: faker.internet.email(),
    ...overrides
  };
}
```

## Timer Testing

```typescript
it('should call after delay', () => {
  vi.useFakeTimers();
  const callback = vi.fn();

  setTimeout(callback, 1000);
  expect(callback).not.toHaveBeenCalled();

  vi.advanceTimersByTime(1000);
  expect(callback).toHaveBeenCalled();

  vi.useRealTimers();
});
```

## Best Practices

| Practice | Description |
|----------|-------------|
| AAA Pattern | Arrange, Act, Assert |
| Descriptive names | `should create user when valid data` |
| One assertion focus | Or logically related assertions |
| Mock externals | Keep tests isolated |
| Test edge cases | Not just happy paths |
| Test behavior | Not implementation details |
| Use fixtures | Consistent test data |
| 80%+ coverage | For critical paths |

## Test Organization

```typescript
describe('UserService', () => {
  describe('createUser', () => {
    it('should create user', () => {});
    it('should throw if email exists', () => {});
  });

  describe('updateUser', () => {
    it('should update user', () => {});
    it('should throw if not found', () => {});
  });
});
```

## Commands

```bash
vitest                    # Watch mode
vitest --coverage         # With coverage
vitest --ui               # UI mode
vitest run                # Single run
vitest run src/user.test.ts  # Specific file
```

## Testing Checklist

- [ ] Unit tests for business logic
- [ ] Integration tests for APIs
- [ ] Component tests for UI
- [ ] Mocks for external dependencies
- [ ] Edge cases covered
- [ ] Error handling tested
- [ ] 80%+ coverage on critical paths
- [ ] CI/CD integration
