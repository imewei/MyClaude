---
name: typescript-advanced-types
version: "1.0.7"
description: Master TypeScript's advanced type system including generics, conditional types, mapped types, template literal types, utility types, branded types, and discriminated unions. Use when implementing complex type logic, creating reusable generic components, designing type-safe APIs, building form validation systems, or working with advanced TypeScript patterns.
---

# TypeScript Advanced Types

## Utility Types Quick Reference

| Type | Purpose | Example |
|------|---------|---------|
| `Partial<T>` | All optional | `Partial<User>` |
| `Required<T>` | All required | `Required<Config>` |
| `Readonly<T>` | All readonly | `Readonly<State>` |
| `Pick<T, K>` | Select props | `Pick<User, 'id' \| 'name'>` |
| `Omit<T, K>` | Remove props | `Omit<User, 'password'>` |
| `Record<K, T>` | Key-value map | `Record<string, number>` |
| `Extract<T, U>` | Filter union | `Extract<A \| B, A>` |
| `Exclude<T, U>` | Remove from union | `Exclude<A \| B, A>` |
| `NonNullable<T>` | Remove null/undefined | `NonNullable<string \| null>` |
| `ReturnType<T>` | Function return type | `ReturnType<typeof fn>` |
| `Parameters<T>` | Function params | `Parameters<typeof fn>` |

## Generics

```typescript
// Basic generic function
function identity<T>(value: T): T {
  return value;
}

// Generic constraints
interface HasLength { length: number; }

function logLength<T extends HasLength>(item: T): T {
  console.log(item.length);
  return item;
}

// Multiple type parameters
function merge<T, U>(a: T, b: U): T & U {
  return { ...a, ...b };
}

// Generic class
class Container<T> {
  constructor(private value: T) {}
  get(): T { return this.value; }
}
```

## Conditional Types

```typescript
// Basic conditional
type IsString<T> = T extends string ? true : false;

// Extract return type
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

// Extract array element
type ElementType<T> = T extends (infer U)[] ? U : never;

// Extract promise value
type Awaited<T> = T extends Promise<infer U> ? U : T;

// Distributive over unions
type ToArray<T> = T extends any ? T[] : never;
type Result = ToArray<string | number>;  // string[] | number[]
```

## Mapped Types

```typescript
// Make all properties optional
type Partial<T> = { [K in keyof T]?: T[K] };

// Make all properties readonly
type Readonly<T> = { readonly [K in keyof T]: T[K] };

// Key remapping (TypeScript 4.1+)
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K]
};

// Filter by value type
type PickByType<T, U> = {
  [K in keyof T as T[K] extends U ? K : never]: T[K]
};
```

## Template Literal Types

```typescript
// Event names
type EventName = 'click' | 'focus' | 'blur';
type EventHandler = `on${Capitalize<EventName>}`;
// 'onClick' | 'onFocus' | 'onBlur'

// String manipulation
type Upper = Uppercase<'hello'>;      // 'HELLO'
type Lower = Lowercase<'HELLO'>;      // 'hello'
type Cap = Capitalize<'hello'>;       // 'Hello'
type Uncap = Uncapitalize<'Hello'>;   // 'hello'
```

## Discriminated Unions

```typescript
// State machine pattern
type AsyncState<T> =
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: string };

function handle<T>(state: AsyncState<T>) {
  switch (state.status) {
    case 'loading': return 'Loading...';
    case 'success': return state.data;  // narrowed to { data: T }
    case 'error': return state.error;   // narrowed to { error: string }
  }
}

// Exhaustiveness checking
function assertNever(x: never): never {
  throw new Error(`Unexpected: ${x}`);
}
```

## Branded Types

```typescript
// Prevent primitive mixing
type UserId = string & { readonly __brand: 'UserId' };
type OrderId = string & { readonly __brand: 'OrderId' };

function createUserId(id: string): UserId {
  return id as UserId;
}

function getUser(id: UserId) { /* ... */ }

const userId = createUserId('123');
const orderId = '456' as OrderId;

getUser(userId);   // OK
// getUser(orderId);  // Error: not compatible
```

## Type-Safe Event Emitter

```typescript
type EventMap = {
  'user:created': { id: string; name: string };
  'user:deleted': { id: string };
};

class TypedEmitter<T extends Record<string, any>> {
  private listeners: { [K in keyof T]?: Array<(data: T[K]) => void> } = {};

  on<K extends keyof T>(event: K, cb: (data: T[K]) => void) {
    (this.listeners[event] ??= []).push(cb);
  }

  emit<K extends keyof T>(event: K, data: T[K]) {
    this.listeners[event]?.forEach(cb => cb(data));
  }
}

const emitter = new TypedEmitter<EventMap>();
emitter.on('user:created', ({ id, name }) => console.log(id, name));
```

## Deep Utility Types

```typescript
// Deep readonly
type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object
    ? T[K] extends Function ? T[K] : DeepReadonly<T[K]>
    : T[K];
};

// Deep partial
type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object
    ? DeepPartial<T[K]>
    : T[K];
};
```

## Type Guards

```typescript
// Type predicate
function isString(value: unknown): value is string {
  return typeof value === 'string';
}

// Assertion function
function assertString(value: unknown): asserts value is string {
  if (typeof value !== 'string') throw new Error('Not a string');
}

// Usage
const val: unknown = 'hello';
if (isString(val)) {
  val.toUpperCase();  // val is string
}

assertString(val);
val.toUpperCase();  // val is string
```

## Best Practices

| Practice | Guideline |
|----------|-----------|
| Use `unknown` | Prefer over `any` |
| `interface` for objects | Better error messages |
| `type` for unions | More flexible |
| Leverage inference | Avoid over-annotating |
| Use `as const` | Preserve literal types |
| Type guards over assertions | Safer narrowing |
| Enable strict mode | Catch more errors |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Overusing `any` | Use `unknown` + type guards |
| Ignoring strict null | Enable `strictNullChecks` |
| Complex deep types | Can slow compilation |
| Missing discriminants | Always include `type` or `kind` field |
| Circular references | Use lazy evaluation |

## Checklist

- [ ] Enable strict mode in tsconfig
- [ ] Use utility types instead of manual definitions
- [ ] Implement discriminated unions for state
- [ ] Create branded types for IDs
- [ ] Write type guards for runtime narrowing
- [ ] Test types with type assertions
- [ ] Document complex types with JSDoc
