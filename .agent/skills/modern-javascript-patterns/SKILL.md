---
name: modern-javascript-patterns
version: "1.0.7"
description: Master modern JavaScript (ES6-ES2024) including async/await, destructuring, spread operators, arrow functions, modules, optional chaining, nullish coalescing, and functional programming patterns. Use when writing ES6+ code, refactoring legacy JS, implementing async patterns, or applying functional programming with map/filter/reduce.
---

# Modern JavaScript Patterns

## Feature

| Feature | Syntax | Use Case |
|---------|--------|----------|
| Arrow function | `(a, b) => a + b` | Callbacks, lexical `this` |
| Destructuring | `const { a, b } = obj` | Extract properties |
| Spread | `[...arr1, ...arr2]` | Merge arrays/objects |
| Template literal | `` `Hello ${name}` `` | String interpolation |
| Optional chaining | `obj?.prop?.nested` | Safe property access |
| Nullish coalescing | `val ?? 'default'` | Default for null/undefined |
| Async/await | `const x = await fn()` | Async operations |

## Arrow Functions

```javascript
const add = (a, b) => a + b;
const double = x => x * 2;
const getUser = () => ({ name: 'John', age: 30 });

// Lexical this (preserves context)
class Counter {
  count = 0;
  increment = () => this.count++;  // 'this' bound to instance

  delayed() {
    setTimeout(() => this.count++, 1000);  // 'this' preserved
  }
}
```

## Destructuring

```javascript
// Objects
const { name, email, age = 25 } = user;
const { name: userName, address: { city } } = user;
const { id, ...rest } = user;  // Rest

// Arrays
const [first, second] = arr;
const [head, ...tail] = arr;
let [a, b] = [b, a];  // Swap

// Function parameters
function greet({ name, age = 18 }) {
  return `Hello ${name}, ${age}`;
}
```

## Spread Operator

```javascript
// Arrays
const combined = [...arr1, ...arr2];
const copy = [...original];
const withNew = [...arr, newItem];

// Objects
const merged = { ...defaults, ...overrides };
const updated = { ...user, age: 31 };
const { removed, ...rest } = obj;  // Remove property
```

## Async/Await

```javascript
async function fetchUser(id) {
  try {
    const res = await fetch(`/api/users/${id}`);
    return await res.json();
  } catch (error) {
    console.error('Failed:', error);
    throw error;
  }
}

// Parallel execution
const [user, posts] = await Promise.all([
  fetchUser(id),
  fetchPosts(id)
]);

// Sequential when needed
for (const id of ids) {
  await processItem(id);  // One at a time
}
```

### Promise Combinators

| Method | Behavior | Use Case |
|--------|----------|----------|
| `Promise.all` | Fail if any fails | Parallel, all required |
| `Promise.allSettled` | Never fails | Get all results |
| `Promise.race` | First to complete | Timeout pattern |
| `Promise.any` | First to succeed | Fallback sources |

```javascript
// Retry with timeout
async function fetchWithRetry(url, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      return await Promise.race([
        fetch(url),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 5000))
      ]);
    } catch (e) {
      if (i === retries - 1) throw e;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
}
```

## Functional Array Methods

```javascript
const users = [
  { id: 1, name: 'John', age: 30, active: true },
  { id: 2, name: 'Jane', age: 25, active: false }
];

// Transform
const names = users.map(u => u.name);
const active = users.filter(u => u.active);
const totalAge = users.reduce((sum, u) => sum + u.age, 0);

// Search
const user = users.find(u => u.id === 1);
const idx = users.findIndex(u => u.name === 'Jane');
const hasActive = users.some(u => u.active);
const allAdults = users.every(u => u.age >= 18);

// Chain
const result = users
  .filter(u => u.active)
  .map(u => u.name.toUpperCase())
  .sort()
  .join(', ');

// Group by (reduce pattern)
const byStatus = users.reduce((groups, user) => ({
  ...groups,
  [user.active ? 'active' : 'inactive']: [
    ...(groups[user.active ? 'active' : 'inactive'] || []),
    user
  ]
}), {});
```

## Higher-Order Functions

```javascript
// Currying
const multiply = a => b => a * b;
const double = multiply(2);
const triple = multiply(3);

// Memoization
function memoize(fn) {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (!cache.has(key)) cache.set(key, fn(...args));
    return cache.get(key);
  };
}

// Composition
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);

const process = pipe(
  str => str.trim(),
  str => str.toLowerCase(),
  str => str.split(' ')
);
```

## Modern Operators

```javascript
// Optional chaining
const city = user?.address?.city;
const result = obj.method?.();
const item = arr?.[0];

// Nullish coalescing (only null/undefined)
const value = input ?? 'default';
const count = obj.count ?? 0;  // 0 preserved, unlike ||

// Logical assignment
a ??= 'default';  // a = a ?? 'default'
b ||= fallback;   // b = b || fallback
c &&= newValue;   // c = c && newValue
```

## ES6 Modules

```javascript
// Named exports
export const PI = 3.14159;
export function add(a, b) { return a + b; }

// Default export
export default class Calculator { }

// Imports
import Calculator, { PI, add } from './math.js';
import * as Math from './math.js';
import { add as sum } from './math.js';

// Dynamic import (code splitting)
const module = await import('./feature.js');
if (condition) {
  const { handler } = await import('./handler.js');
}
```

## Classes

```javascript
class User {
  #password;              // Private field
  static count = 0;       // Static field

  constructor(name, password) {
    this.name = name;
    this.#password = password;
    User.count++;
  }

  get displayName() { return this.name.toUpperCase(); }
  set password(val) { this.#password = this.#hash(val); }

  #hash(val) { return `hashed_${val}`; }  // Private method

  static create(name, pwd) { return new User(name, pwd); }
}

class Admin extends User {
  constructor(name, password, role) {
    super(name, password);
    this.role = role;
  }
}
```

## Generators & Iterators

```javascript
// Generator
function* range(start, end) {
  for (let i = start; i <= end; i++) yield i;
}

for (const n of range(1, 5)) console.log(n);

// Infinite sequence
function* fibonacci() {
  let [a, b] = [0, 1];
  while (true) {
    yield b;
    [a, b] = [b, a + b];
  }
}

// Async generator
async function* fetchPages(url) {
  let page = 1;
  while (true) {
    const data = await fetch(`${url}?page=${page++}`).then(r => r.json());
    if (!data.length) break;
    yield data;
  }
}

for await (const page of fetchPages('/api/items')) {
  console.log(page);
}
```

## Performance Utilities

```javascript
// Debounce (delay until pause)
function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
}

// Throttle (limit frequency)
function throttle(fn, limit) {
  let inThrottle;
  return (...args) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

const searchDebounced = debounce(search, 300);
const scrollThrottled = throttle(handleScroll, 100);
```

## Immutable Patterns

```javascript
// Arrays
const added = [...arr, newItem];
const removed = arr.filter(x => x !== item);
const updated = arr.map(x => x.id === id ? { ...x, ...changes } : x);

// Objects
const modified = { ...obj, key: newValue };
const { removed: _, ...rest } = obj;

// Deep clone
const clone = structuredClone(obj);  // Modern
const clone = JSON.parse(JSON.stringify(obj));  // Legacy
```

## Best Practices

| Practice | Guideline |
|----------|-----------|
| Variables | `const` by default, `let` when needed, avoid `var` |
| Functions | Arrow for callbacks, regular for methods |
| Async | async/await over .then() chains |
| Defaults | `??` for null/undefined, `||` for all falsy |
| Imports | Named exports for tree-shaking |
| Immutability | Spread operator for updates |
| Errors | try/catch with async/await |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| `this` in callbacks | Arrow functions or `.bind()` |
| Missing `await` | Async functions return Promises |
| `||` vs `??` | `??` preserves 0 and '' |
| Mutation | Use spread for immutable updates |
| Promise rejection | Always handle with catch/try |
| Blocking event loop | Use async for I/O operations |

## Checklist

- [ ] const/let used appropriately (no var)
- [ ] Destructuring for cleaner assignments
- [ ] async/await with proper error handling
- [ ] Optional chaining for safe property access
- [ ] Array methods instead of loops where clearer
- [ ] ES6 modules for code organization
- [ ] Performance utilities (debounce/throttle) where needed
