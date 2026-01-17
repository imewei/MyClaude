---
name: react-modernization
version: "1.0.7"
maturity: "5-Expert"
specialization: React Migration & Hooks
description: Upgrade React 16→17→18, migrate class components to hooks (useState, useEffect, useContext), adopt concurrent features (Suspense, transitions), apply codemods, optimize with memo/useMemo/useCallback, and add TypeScript. Use for class-to-hooks migration, lifecycle conversion, or React 18 adoption.
---

# React Modernization

React upgrades, class-to-hooks migration, and concurrent features.

---

## Version Breaking Changes

| Version | Key Changes |
|---------|-------------|
| React 17 | Event delegation, no event pooling, new JSX transform |
| React 18 | Automatic batching, concurrent rendering, createRoot API, Strict Mode double-invoke |

---

## State Migration

```javascript
// Before: Class component
class Counter extends React.Component {
  state = { count: 0 };
  increment = () => this.setState({ count: this.state.count + 1 });
  render() {
    return <button onClick={this.increment}>{this.state.count}</button>;
  }
}

// After: Functional with hooks
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
```

---

## Lifecycle to useEffect

| Lifecycle | useEffect Equivalent |
|-----------|---------------------|
| componentDidMount | `useEffect(() => {}, [])` |
| componentDidUpdate | `useEffect(() => {}, [deps])` |
| componentWillUnmount | `useEffect(() => { return cleanup; }, [])` |

```javascript
// Before
componentDidMount() { this.fetchData(); }
componentDidUpdate(prevProps) {
  if (prevProps.id !== this.props.id) this.fetchData();
}
componentWillUnmount() { this.cancel(); }

// After
useEffect(() => {
  let cancelled = false;
  fetchData(id).then(data => { if (!cancelled) setData(data); });
  return () => { cancelled = true; };
}, [id]);
```

---

## HOCs to Custom Hooks

```javascript
// Before: HOC
function withUser(Component) {
  return class extends React.Component {
    state = { user: null };
    componentDidMount() { fetchUser().then(user => this.setState({ user })); }
    render() { return <Component {...this.props} user={this.state.user} />; }
  };
}

// After: Custom hook
function useUser() {
  const [user, setUser] = useState(null);
  useEffect(() => { fetchUser().then(setUser); }, []);
  return user;
}

function UserProfile() {
  const user = useUser();
  return user ? <div>{user.name}</div> : <div>Loading...</div>;
}
```

---

## React 18 Features

### New Root API
```javascript
// Before (React 17)
ReactDOM.render(<App />, document.getElementById('root'));

// After (React 18)
import { createRoot } from 'react-dom/client';
createRoot(document.getElementById('root')).render(<App />);
```

### Automatic Batching
```javascript
// React 18: All updates batched (even in setTimeout, promises)
setTimeout(() => {
  setCount(c => c + 1);
  setFlag(f => !f);
  // Single re-render!
}, 1000);
```

### Transitions
```javascript
const [isPending, startTransition] = useTransition();

const handleChange = (e) => {
  setQuery(e.target.value);  // Urgent: update input
  startTransition(() => {
    setResults(search(e.target.value));  // Non-urgent: can be interrupted
  });
};
```

### Suspense
```javascript
const Dashboard = lazy(() => import('./Dashboard'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Dashboard />
    </Suspense>
  );
}
```

---

## Performance Optimization

```javascript
// Memoize expensive calculation
const filtered = useMemo(() => items.filter(expensive), [items]);

// Memoize callback for child components
const handleClick = useCallback((id) => console.log(id), []);

// Prevent re-renders
const List = React.memo(({ items, onClick }) => {
  return items.map(item => <Item key={item.id} item={item} onClick={onClick} />);
});
```

---

## Codemods

```bash
# Rename unsafe lifecycle methods
npx react-codeshift --transform=rename-unsafe-lifecycles src/

# Update to new JSX transform
npx react-codeshift --transform=new-jsx-transform src/

# Class to hooks (third-party)
npx codemod react/hooks/convert-class-to-function src/
```

---

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
```

---

## Migration Checklist

### Pre-Migration
- [ ] Update dependencies incrementally
- [ ] Review breaking changes
- [ ] Set up testing

### Class → Hooks
- [ ] Start with leaf components
- [ ] Convert state to useState
- [ ] Convert lifecycle to useEffect
- [ ] Extract custom hooks
- [ ] Test thoroughly

### React 18
- [ ] Update to createRoot
- [ ] Test with StrictMode (double-invoke)
- [ ] Adopt Suspense/Transitions where beneficial

---

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| Missing useEffect deps | Stale closures, infinite loops |
| Over-using useMemo/useCallback | Premature optimization |
| No cleanup in useEffect | Memory leaks |
| Ignoring StrictMode warnings | Hidden bugs |
| Big-bang migration | Too risky |

---

**Version**: 1.0.5
