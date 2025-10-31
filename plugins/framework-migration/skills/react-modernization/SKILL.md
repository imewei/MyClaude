---
name: react-modernization
description: Upgrade React applications from legacy versions to React 18+, migrate class components to functional components with hooks (useState, useEffect, useContext, useMemo, useCallback), adopt concurrent features (Suspense, transitions, automatic batching), apply automated codemods for API transformations, optimize performance with React.memo and code splitting, and implement TypeScript for type safety. Use when upgrading React 16→17→18 with breaking change handling, migrating class components to functional components with hooks, converting lifecycle methods (componentDidMount, componentDidUpdate, componentWillUnmount) to useEffect, transforming this.setState to useState, replacing Higher-Order Components (HOCs) and render props with custom hooks, updating to new JSX transform (no React import needed), implementing React 18's createRoot API replacing ReactDOM.render, adopting automatic batching for all state updates, using transitions (useTransition, startTransition) for non-urgent updates, implementing Suspense for data fetching and code splitting with lazy(), optimizing render performance with useMemo and useCallback, applying React codemods for automated refactoring, converting PropTypes to TypeScript interfaces, migrating from React.createClass to functional components, implementing React.StrictMode for development warnings, adopting concurrent rendering features and error boundaries. Apply when working with React component files (.jsx, .tsx, .js with JSX), custom hooks files (useCustomHook.ts), context provider files (Context.tsx), HOC files (withAuth.jsx), test files using React Testing Library (@testing-library/react), and when planning systematic migration strategies for large React codebases with minimal disruption to production.
---

# React Modernization

Master React version upgrades, class to hooks migration, concurrent features adoption, and codemods for automated transformation.

## When to Use This Skill

- When upgrading React applications incrementally from React 16→17→18 with version-specific breaking changes
- When migrating class components to functional components using useState, useEffect, useContext, useReducer hooks
- When converting componentDidMount, componentDidUpdate, componentWillUnmount lifecycle methods to useEffect with dependency arrays
- When replacing this.setState with useState or useReducer for state management
- When transforming Higher-Order Components (HOCs) like withAuth, withRouter to custom hooks (useAuth, useRouter)
- When converting render props patterns to custom hooks for better code reusability
- When updating to React 18's createRoot API from ReactDOM.render for concurrent features
- When adopting automatic batching which batches all setState calls including in async functions and timeouts
- When implementing useTransition and startTransition for non-urgent UI updates (search results, filtering)
- When using Suspense for code splitting with React.lazy() and dynamic imports
- When applying React codemods using react-codeshift for automated transformations (rename-unsafe-lifecycles, new-jsx-transform)
- When optimizing component re-renders with React.memo, useMemo for expensive calculations, useCallback for stable function references
- When removing PropTypes in favor of TypeScript interfaces and type definitions
- When updating to new JSX transform that doesn't require "import React from 'react'" in every file
- When implementing error boundaries (componentDidCatch, getDerivedStateFromError) for graceful error handling
- When working with React component files (.jsx, .tsx), custom hooks (use*.ts), context providers, and HOCs
- When converting React.createClass to ES6 class components and then to functional components
- When enabling React.StrictMode to detect unsafe lifecycle methods and side effects
- When migrating from old context API (React.createContext with Consumer) to useContext hook
- When implementing lazy loading with Suspense and React.lazy for route-based code splitting
- When updating testing files to use React Testing Library instead of Enzyme
- When migrating forms from uncontrolled components to controlled components with useState
- When converting ref callbacks to useRef and forwardRef for ref forwarding
- When implementing custom hooks to extract and reuse stateful logic across components
- When adopting React 18 concurrent features while maintaining backward compatibility during gradual rollout
- When planning phased migration strategies starting with leaf components before migrating parent components

## Version Upgrade Path

### React 16 → 17 → 18

**Breaking Changes by Version:**

**React 17:**
- Event delegation changes
- No event pooling
- Effect cleanup timing
- JSX transform (no React import needed)

**React 18:**
- Automatic batching
- Concurrent rendering
- Strict Mode changes (double invocation)
- New root API
- Suspense on server

## Class to Hooks Migration

### State Management
```javascript
// Before: Class component
class Counter extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      count: 0,
      name: ''
    };
  }

  increment = () => {
    this.setState({ count: this.state.count + 1 });
  }

  render() {
    return (
      <div>
        <p>Count: {this.state.count}</p>
        <button onClick={this.increment}>Increment</button>
      </div>
    );
  }
}

// After: Functional component with hooks
function Counter() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');

  const increment = () => {
    setCount(count + 1);
  };

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>Increment</button>
    </div>
  );
}
```

### Lifecycle Methods to Hooks
```javascript
// Before: Lifecycle methods
class DataFetcher extends React.Component {
  state = { data: null, loading: true };

  componentDidMount() {
    this.fetchData();
  }

  componentDidUpdate(prevProps) {
    if (prevProps.id !== this.props.id) {
      this.fetchData();
    }
  }

  componentWillUnmount() {
    this.cancelRequest();
  }

  fetchData = async () => {
    const data = await fetch(`/api/${this.props.id}`);
    this.setState({ data, loading: false });
  };

  cancelRequest = () => {
    // Cleanup
  };

  render() {
    if (this.state.loading) return <div>Loading...</div>;
    return <div>{this.state.data}</div>;
  }
}

// After: useEffect hook
function DataFetcher({ id }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const fetchData = async () => {
      try {
        const response = await fetch(`/api/${id}`);
        const result = await response.json();

        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      } catch (error) {
        if (!cancelled) {
          console.error(error);
        }
      }
    };

    fetchData();

    // Cleanup function
    return () => {
      cancelled = true;
    };
  }, [id]); // Re-run when id changes

  if (loading) return <div>Loading...</div>;
  return <div>{data}</div>;
}
```

### Context and HOCs to Hooks
```javascript
// Before: Context consumer and HOC
const ThemeContext = React.createContext();

class ThemedButton extends React.Component {
  static contextType = ThemeContext;

  render() {
    return (
      <button style={{ background: this.context.theme }}>
        {this.props.children}
      </button>
    );
  }
}

// After: useContext hook
function ThemedButton({ children }) {
  const { theme } = useContext(ThemeContext);

  return (
    <button style={{ background: theme }}>
      {children}
    </button>
  );
}

// Before: HOC for data fetching
function withUser(Component) {
  return class extends React.Component {
    state = { user: null };

    componentDidMount() {
      fetchUser().then(user => this.setState({ user }));
    }

    render() {
      return <Component {...this.props} user={this.state.user} />;
    }
  };
}

// After: Custom hook
function useUser() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser().then(setUser);
  }, []);

  return user;
}

function UserProfile() {
  const user = useUser();
  if (!user) return <div>Loading...</div>;
  return <div>{user.name}</div>;
}
```

## React 18 Concurrent Features

### New Root API
```javascript
// Before: React 17
import ReactDOM from 'react-dom';

ReactDOM.render(<App />, document.getElementById('root'));

// After: React 18
import { createRoot } from 'react-dom/client';

const root = createRoot(document.getElementById('root'));
root.render(<App />);
```

### Automatic Batching
```javascript
// React 18: All updates are batched
function handleClick() {
  setCount(c => c + 1);
  setFlag(f => !f);
  // Only one re-render (batched)
}

// Even in async:
setTimeout(() => {
  setCount(c => c + 1);
  setFlag(f => !f);
  // Still batched in React 18!
}, 1000);

// Opt out if needed
import { flushSync } from 'react-dom';

flushSync(() => {
  setCount(c => c + 1);
});
// Re-render happens here
setFlag(f => !f);
// Another re-render
```

### Transitions
```javascript
import { useState, useTransition } from 'react';

function SearchResults() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isPending, startTransition] = useTransition();

  const handleChange = (e) => {
    // Urgent: Update input immediately
    setQuery(e.target.value);

    // Non-urgent: Update results (can be interrupted)
    startTransition(() => {
      setResults(searchResults(e.target.value));
    });
  };

  return (
    <>
      <input value={query} onChange={handleChange} />
      {isPending && <Spinner />}
      <Results data={results} />
    </>
  );
}
```

### Suspense for Data Fetching
```javascript
import { Suspense } from 'react';

// Resource-based data fetching (with React 18)
const resource = fetchProfileData();

function ProfilePage() {
  return (
    <Suspense fallback={<Loading />}>
      <ProfileDetails />
      <Suspense fallback={<Loading />}>
        <ProfileTimeline />
      </Suspense>
    </Suspense>
  );
}

function ProfileDetails() {
  // This will suspend if data not ready
  const user = resource.user.read();
  return <h1>{user.name}</h1>;
}

function ProfileTimeline() {
  const posts = resource.posts.read();
  return <Timeline posts={posts} />;
}
```

## Codemods for Automation

### Run React Codemods
```bash
# Install jscodeshift
npm install -g jscodeshift

# React 16.9 codemod (rename unsafe lifecycle methods)
npx react-codeshift <transform> <path>

# Example: Rename UNSAFE_ methods
npx react-codeshift --parser=tsx \
  --transform=react-codeshift/transforms/rename-unsafe-lifecycles.js \
  src/

# Update to new JSX Transform (React 17+)
npx react-codeshift --parser=tsx \
  --transform=react-codeshift/transforms/new-jsx-transform.js \
  src/

# Class to Hooks (third-party)
npx codemod react/hooks/convert-class-to-function src/
```

### Custom Codemod Example
```javascript
// custom-codemod.js
module.exports = function(file, api) {
  const j = api.jscodeshift;
  const root = j(file.source);

  // Find setState calls
  root.find(j.CallExpression, {
    callee: {
      type: 'MemberExpression',
      property: { name: 'setState' }
    }
  }).forEach(path => {
    // Transform to useState
    // ... transformation logic
  });

  return root.toSource();
};

// Run: jscodeshift -t custom-codemod.js src/
```

## Performance Optimization

### useMemo and useCallback
```javascript
function ExpensiveComponent({ items, filter }) {
  // Memoize expensive calculation
  const filteredItems = useMemo(() => {
    return items.filter(item => item.category === filter);
  }, [items, filter]);

  // Memoize callback to prevent child re-renders
  const handleClick = useCallback((id) => {
    console.log('Clicked:', id);
  }, []); // No dependencies, never changes

  return (
    <List items={filteredItems} onClick={handleClick} />
  );
}

// Child component with memo
const List = React.memo(({ items, onClick }) => {
  return items.map(item => (
    <Item key={item.id} item={item} onClick={onClick} />
  ));
});
```

### Code Splitting
```javascript
import { lazy, Suspense } from 'react';

// Lazy load components
const Dashboard = lazy(() => import('./Dashboard'));
const Settings = lazy(() => import('./Settings'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}
```

## TypeScript Migration

```typescript
// Before: JavaScript
function Button({ onClick, children }) {
  return <button onClick={onClick}>{children}</button>;
}

// After: TypeScript
interface ButtonProps {
  onClick: () => void;
  children: React.ReactNode;
}

function Button({ onClick, children }: ButtonProps) {
  return <button onClick={onClick}>{children}</button>;
}

// Generic components
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => React.ReactNode;
}

function List<T>({ items, renderItem }: ListProps<T>) {
  return <>{items.map(renderItem)}</>;
}
```

## Migration Checklist

```markdown
### Pre-Migration
- [ ] Update dependencies incrementally (not all at once)
- [ ] Review breaking changes in release notes
- [ ] Set up testing suite
- [ ] Create feature branch

### Class → Hooks Migration
- [ ] Identify class components to migrate
- [ ] Start with leaf components (no children)
- [ ] Convert state to useState
- [ ] Convert lifecycle to useEffect
- [ ] Convert context to useContext
- [ ] Extract custom hooks
- [ ] Test thoroughly

### React 18 Upgrade
- [ ] Update to React 17 first (if needed)
- [ ] Update react and react-dom to 18
- [ ] Update @types/react if using TypeScript
- [ ] Change to createRoot API
- [ ] Test with StrictMode (double invocation)
- [ ] Address concurrent rendering issues
- [ ] Adopt Suspense/Transitions where beneficial

### Performance
- [ ] Identify performance bottlenecks
- [ ] Add React.memo where appropriate
- [ ] Use useMemo/useCallback for expensive operations
- [ ] Implement code splitting
- [ ] Optimize re-renders

### Testing
- [ ] Update test utilities (React Testing Library)
- [ ] Test with React 18 features
- [ ] Check for warnings in console
- [ ] Performance testing
```

## Resources

- **references/breaking-changes.md**: Version-specific breaking changes
- **references/codemods.md**: Codemod usage guide
- **references/hooks-migration.md**: Comprehensive hooks patterns
- **references/concurrent-features.md**: React 18 concurrent features
- **assets/codemod-config.json**: Codemod configurations
- **assets/migration-checklist.md**: Step-by-step checklist
- **scripts/apply-codemods.sh**: Automated codemod script

## Best Practices

1. **Incremental Migration**: Don't migrate everything at once
2. **Test Thoroughly**: Comprehensive testing at each step
3. **Use Codemods**: Automate repetitive transformations
4. **Start Simple**: Begin with leaf components
5. **Leverage StrictMode**: Catch issues early
6. **Monitor Performance**: Measure before and after
7. **Document Changes**: Keep migration log

## Common Pitfalls

- Forgetting useEffect dependencies
- Over-using useMemo/useCallback
- Not handling cleanup in useEffect
- Mixing class and functional patterns
- Ignoring StrictMode warnings
- Breaking change assumptions
