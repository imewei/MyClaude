# Testing Strategies

> **Reference**: Comprehensive testing approaches for React & React Native components

---

## Component Test Generator

```typescript
class ComponentTestGenerator {
  generateTests(spec: ComponentSpec): string {
    return `
import { render, screen, fireEvent } from '@testing-library/react';
import { ${spec.name} } from './${spec.name}';

describe('${spec.name}', () => {
  const defaultProps = {
${spec.props.filter(p => p.required).map(p => `    ${p.name}: ${this.getMockValue(p.type)},`).join('\n')}
  };

  it('renders without crashing', () => {
    render(<${spec.name} {...defaultProps} />);
    expect(screen.getByRole('${this.inferAriaRole(spec.type)}')).toBeInTheDocument();
  });

  it('displays correct content', () => {
    render(<${spec.name} {...defaultProps} />);
    expect(screen.getByText(/content/i)).toBeVisible();
  });

${spec.props.filter(p => p.type.includes('()') || p.name.startsWith('on')).map(p => `
  it('calls ${p.name} when triggered', () => {
    const mock${this.capitalize(p.name)} = jest.fn();
    render(<${spec.name} {...defaultProps} ${p.name}={mock${this.capitalize(p.name)}} />);

    const trigger = screen.getByRole('button');
    fireEvent.click(trigger);

    expect(mock${this.capitalize(p.name)}).toHaveBeenCalledTimes(1);
  });`).join('\n')}

  it('meets accessibility standards', async () => {
    const { container } = render(<${spec.name} {...defaultProps} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
`;
  }

  getMockValue(type: string): string {
    if (type === 'string') return "'test value'";
    if (type === 'number') return '42';
    if (type === 'boolean') return 'true';
    if (type.includes('[]')) return '[]';
    if (type.includes('()')) return 'jest.fn()';
    return '{}';
  }

  private capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  private inferAriaRole(type: string): string {
    const roleMap: Record<string, string> = {
      'functional': 'region',
      'page': 'main',
      'layout': 'region',
      'form': 'form',
      'data-display': 'region'
    };
    return roleMap[type] || 'region';
  }
}
```

---

## Testing Pyramid

### Unit Tests (70%)

**Purpose**: Test individual component logic and rendering

**Tools**: Jest, Vitest, React Testing Library

**Examples**:
```typescript
// Component rendering
it('renders with required props', () => {
  render(<Button label="Click me" onClick={jest.fn()} />);
  expect(screen.getByRole('button')).toHaveTextContent('Click me');
});

// State changes
it('updates count when clicked', () => {
  render(<Counter />);
  const button = screen.getByRole('button');
  fireEvent.click(button);
  expect(screen.getByText('Count: 1')).toBeInTheDocument();
});

// Conditional rendering
it('shows loading state when isLoading is true', () => {
  render(<DataDisplay isLoading={true} />);
  expect(screen.getByRole('status')).toHaveTextContent('Loading...');
});
```

### Integration Tests (20%)

**Purpose**: Test component interactions and data flow

**Tools**: React Testing Library, MSW (Mock Service Worker)

**Examples**:
```typescript
// User flow
it('completes registration flow', async () => {
  render(<RegistrationForm />);

  await userEvent.type(screen.getByLabelText('Email'), 'user@example.com');
  await userEvent.type(screen.getByLabelText('Password'), 'password123');
  await userEvent.click(screen.getByRole('button', { name: 'Sign Up' }));

  await waitFor(() => {
    expect(screen.getByText('Registration successful')).toBeInTheDocument();
  });
});

// API integration
it('fetches and displays user data', async () => {
  const mockUser = { id: 1, name: 'John Doe' };
  server.use(
    rest.get('/api/user', (req, res, ctx) => {
      return res(ctx.json(mockUser));
    })
  );

  render(<UserProfile userId={1} />);

  await waitFor(() => {
    expect(screen.getByText('John Doe')).toBeInTheDocument();
  });
});
```

### E2E Tests (10%)

**Purpose**: Test complete user journeys across multiple pages

**Tools**: Playwright, Cypress, Detox (React Native)

**Examples**:
```typescript
// E2E user flow
test('user can complete checkout', async ({ page }) => {
  await page.goto('/products');
  await page.click('text=Add to Cart');
  await page.click('text=Checkout');
  await page.fill('input[name="email"]', 'user@example.com');
  await page.fill('input[name="card"]', '4242424242424242');
  await page.click('button:has-text("Place Order")');

  await expect(page.locator('text=Order Confirmed')).toBeVisible();
});
```

---

## Accessibility Testing

### axe-core Integration

```typescript
import { axe } from 'jest-axe';

it('has no accessibility violations', async () => {
  const { container } = render(<Component />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### Keyboard Navigation

```typescript
it('supports keyboard navigation', () => {
  render(<Navigation />);
  const firstLink = screen.getAllByRole('link')[0];

  firstLink.focus();
  expect(firstLink).toHaveFocus();

  fireEvent.keyDown(firstLink, { key: 'Tab' });
  expect(screen.getAllByRole('link')[1]).toHaveFocus();
});
```

### Screen Reader Testing

```typescript
it('provides descriptive labels for screen readers', () => {
  render(<ImageGallery images={mockImages} />);

  const images = screen.getAllByRole('img');
  images.forEach(img => {
    expect(img).toHaveAccessibleName();
  });
});
```

---

## React Native Testing

### Component Testing

```typescript
import { render, fireEvent } from '@testing-library/react-native';

describe('NativeButton', () => {
  it('handles press events', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <NativeButton label="Press me" onPress={onPress} />
    );

    fireEvent.press(getByText('Press me'));
    expect(onPress).toHaveBeenCalledTimes(1);
  });
});
```

### E2E with Detox

```typescript
describe('Login Flow', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  it('should login successfully', async () => {
    await element(by.id('email-input')).typeText('user@example.com');
    await element(by.id('password-input')).typeText('password123');
    await element(by.id('login-button')).tap();

    await expect(element(by.text('Welcome back!'))).toBeVisible();
  });
});
```

---

## Test Coverage Targets

| Component Type | Unit Tests | Integration Tests | E2E Tests | Total Coverage |
|----------------|-----------|-------------------|-----------|----------------|
| **Functional** | 80% | 10% | 0% | 90% |
| **Page** | 60% | 30% | 10% | 100% |
| **Layout** | 70% | 20% | 0% | 90% |
| **Form** | 70% | 25% | 5% | 100% |
| **Data Display** | 75% | 15% | 0% | 90% |

---

## Mock Value Patterns

```typescript
// String mocks
const mockString = 'test value';
const mockEmail = 'user@example.com';
const mockUrl = 'https://example.com';

// Number mocks
const mockNumber = 42;
const mockPrice = 19.99;
const mockId = 123;

// Boolean mocks
const mockBoolean = true;

// Array mocks
const mockArray: string[] = [];
const mockUsers: User[] = [
  { id: 1, name: 'John Doe', email: 'john@example.com' },
  { id: 2, name: 'Jane Smith', email: 'jane@example.com' }
];

// Function mocks
const mockFunction = jest.fn();
const mockCallback = jest.fn(() => Promise.resolve());

// Object mocks
const mockObject = {};
const mockUser = { id: 1, name: 'John Doe' };
```

---

## Testing Best Practices

1. **Test user behavior, not implementation**: Use `getByRole`, `getByLabelText` instead of `getByTestId`
2. **Write descriptive test names**: Use "it should..." or "it renders..." format
3. **Keep tests isolated**: Each test should be independent and not rely on others
4. **Mock external dependencies**: Use MSW for API mocking, jest.mock for modules
5. **Test accessibility**: Always include axe-core tests for critical components
6. **Achieve 90%+ coverage**: Target high coverage for business-critical components
7. **Use data-testid sparingly**: Only when role-based queries are not possible
8. **Test error states**: Verify error handling and fallback UI
9. **Test loading states**: Ensure loading indicators display correctly
10. **Run tests in CI/CD**: Automate testing with GitHub Actions or similar
