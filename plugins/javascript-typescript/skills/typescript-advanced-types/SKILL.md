---
name: typescript-advanced-types
description: Master TypeScript's advanced type system including generics, conditional types, mapped types, template literal types, utility types (Partial, Required, Pick, Omit, Record), type inference with infer keyword, branded types, discriminated unions, and recursive types for building robust, type-safe applications with compile-time guarantees. Use when writing or editing TypeScript files (*.ts, *.tsx, *.d.ts), when implementing complex type logic with conditional types and type inference, when creating reusable generic components and functions with type constraints, when designing type-safe API clients with branded types and discriminated unions, when building form validation systems with type-safe schema definitions, when creating strongly-typed configuration objects with template literal types, when implementing type-safe state management with discriminated unions and exhaustiveness checking, when migrating JavaScript codebases to TypeScript with incremental type adoption, when using utility types to transform and manipulate existing types (Partial, Required, Pick, Omit, Readonly, Record, Extract, Exclude, NonNullable), when implementing mapped types for dynamic key generation and property transformation, when creating type guards and assertion functions for runtime type narrowing, when designing branded types to prevent primitive obsession and add semantic meaning, when building recursive types for tree structures and nested data, when implementing type-safe builders and fluent APIs with method chaining, when creating advanced generic constraints with extends and keyof, when using template literal types for string manipulation at type level, when implementing type inference with the infer keyword in conditional types, when designing type-safe event systems with discriminated unions, when creating type utilities for deep readonly, deep partial, or deep required transformations, when ensuring exhaustive checking in switch statements with never type, or when establishing TypeScript type patterns and advanced type usage standards for teams.
---

# TypeScript Advanced Types

Comprehensive guidance for mastering TypeScript's advanced type system including generics, conditional types, mapped types, template literal types, and utility types for building robust, type-safe applications.

## When to Use This Skill

### TypeScript File Creation and Editing
- Writing or editing TypeScript source files: `*.ts`, `*.tsx` (React)
- Creating TypeScript declaration files: `*.d.ts` for type definitions
- Working with type definition files in `@types/` packages
- Creating shared type files: `types.ts`, `models.ts`, `interfaces.ts`

### Generic Type Implementation
- Creating generic functions with type parameters (`<T>`, `<T, U>`)
- Implementing generic classes and interfaces for reusable components
- Using generic constraints with `extends` keyword for type safety
- Creating generic React components with typed props
- Implementing generic repository or service classes
- Building generic data structures (Stack, Queue, LinkedList)
- Creating type-safe HOCs (Higher-Order Components) in React
- Implementing generic utility functions with proper type inference

### Conditional Types and Type Inference
- Creating conditional types with `T extends U ? X : Y` syntax
- Using the `infer` keyword to extract types from complex structures
- Implementing custom `ReturnType`, `Parameters`, or `ConstructorParameters` utilities
- Creating distributive conditional types over union types
- Building type-level pattern matching with conditional types
- Extracting nested property types from complex objects
- Implementing custom type predicates with conditional logic

### Mapped Types and Type Transformations
- Creating mapped types with `{ [K in keyof T]: ... }` syntax
- Implementing custom `Partial`, `Required`, `Readonly` type utilities
- Using `Pick` to select specific properties from types
- Using `Omit` to exclude properties from types
- Creating mapped types with key remapping using `as` keyword
- Implementing deep transformations (DeepPartial, DeepReadonly)
- Building conditional property types based on value types

### Utility Type Usage
- Applying built-in utility types: `Partial<T>`, `Required<T>`, `Readonly<T>`
- Using `Pick<T, K>` to create subsets of types
- Using `Omit<T, K>` to remove properties from types
- Creating record types with `Record<K, V>` for key-value mappings
- Using `Extract<T, U>` and `Exclude<T, U>` for union type filtering
- Applying `NonNullable<T>` to remove null and undefined
- Using `ReturnType<T>`, `Parameters<T>`, `InstanceType<T>` for function/class types
- Implementing `Awaited<T>` for unwrapping Promise types

### Template Literal Types
- Creating template literal types for string pattern matching
- Implementing type-safe CSS-in-JS with template literals
- Building route path types with template literals
- Creating branded types with template literal strings
- Implementing string manipulation at type level (Uppercase, Lowercase, Capitalize, Uncapitalize)
- Building auto-completion for string literals with unions

### Discriminated Unions and Type Guards
- Creating discriminated unions with common discriminant properties
- Implementing exhaustive checking with switch statements and never type
- Writing type guard functions with `is` keyword for runtime narrowing
- Creating assertion functions with `asserts` keyword
- Implementing narrowing with typeof, instanceof, and custom guards
- Building type-safe state machines with discriminated unions
- Creating type-safe event systems with union types

### Branded Types and Nominal Typing
- Implementing branded types to prevent primitive obsession
- Creating nominal types for IDs, emails, URLs with type brands
- Using intersection types with unique symbols for branding
- Implementing validation with branded type constructors
- Creating type-safe newtypes for domain modeling
- Preventing accidental type mixing with branded primitives

### Advanced Generic Constraints
- Using `extends` keyword for generic constraints
- Implementing `keyof` constraint for object key access
- Creating generic constraints with multiple bounds
- Using `typeof` with generics for type capture
- Implementing recursive generic constraints
- Building complex generic hierarchies with variance

### Type Inference and Type Narrowing
- Leveraging automatic type inference in variable declarations
- Using `as const` for literal type inference
- Implementing control flow analysis for type narrowing
- Using discriminated unions for narrowing in switch/if statements
- Creating exhaustiveness checks with never type
- Implementing type narrowing with assertion functions

### Recursive and Self-Referential Types
- Creating recursive types for tree structures and nested data
- Implementing JSON types with recursive definitions
- Building recursive mapped types for deep transformations
- Creating recursive conditional types with depth limits
- Implementing linked list or tree node types with self-references

### Type-Safe APIs and Builders
- Designing type-safe REST API clients with branded types
- Implementing type-safe GraphQL clients with generated types
- Creating fluent API builders with method chaining
- Implementing type-safe configuration objects with template literals
- Building type-safe form builders with discriminated unions
- Creating type-safe query builders for databases

### Advanced Type Patterns
- Implementing variance in generics (covariance, contravariance)
- Creating phantom types for state machines
- Implementing type-level programming with conditional types
- Building type-safe dependency injection containers
- Creating opaque types for information hiding
- Implementing nominal typing patterns in TypeScript

### Type Definition Creation
- Writing `.d.ts` declaration files for JavaScript libraries
- Creating ambient module declarations with `declare module`
- Implementing module augmentation for third-party types
- Creating global type declarations with `declare global`
- Writing JSDoc type annotations for gradual TypeScript adoption
- Creating type-only imports and exports with `import type`

### Migration and Refactoring
- Migrating JavaScript to TypeScript incrementally
- Replacing `any` with proper generic types
- Refactoring callback functions to use generic constraints
- Converting JavaScript classes to TypeScript with proper typing
- Adding type annotations to existing codebases
- Implementing strict TypeScript compiler options gradually

### React-Specific Type Patterns
- Typing React component props with generics
- Creating discriminated unions for component variants
- Implementing type-safe context providers
- Typing React hooks with proper generic constraints
- Creating type-safe custom hooks
- Implementing render props with generic types
- Typing HOCs (Higher-Order Components) properly

### Form and Validation Type Safety
- Creating type-safe form schemas with Zod, Yup, or io-ts
- Implementing discriminated unions for form states
- Building type-safe validation functions
- Creating branded types for validated data
- Implementing type-safe form builders

### State Management Typing
- Typing Redux actions with discriminated unions
- Creating type-safe reducers with exhaustive checking
- Implementing type-safe selectors
- Typing Zustand stores with proper inference
- Creating type-safe state machines with XState
- Implementing type-safe context providers

### Performance and Optimization
- Using type-only imports to reduce bundle size
- Implementing proper type inference to avoid excessive annotations
- Creating reusable type utilities to reduce duplication
- Using `const` assertions for literal type optimization
- Optimizing type checking performance with proper constraints

## Core Concepts

### 1. Generics

**Purpose:** Create reusable, type-flexible components while maintaining type safety.

**Basic Generic Function:**
```typescript
function identity<T>(value: T): T {
  return value;
}

const num = identity<number>(42);        // Type: number
const str = identity<string>("hello");    // Type: string
const auto = identity(true);              // Type inferred: boolean
```

**Generic Constraints:**
```typescript
interface HasLength {
  length: number;
}

function logLength<T extends HasLength>(item: T): T {
  console.log(item.length);
  return item;
}

logLength("hello");           // OK: string has length
logLength([1, 2, 3]);         // OK: array has length
logLength({ length: 10 });    // OK: object has length
// logLength(42);             // Error: number has no length
```

**Multiple Type Parameters:**
```typescript
function merge<T, U>(obj1: T, obj2: U): T & U {
  return { ...obj1, ...obj2 };
}

const merged = merge(
  { name: "John" },
  { age: 30 }
);
// Type: { name: string } & { age: number }
```

### 2. Conditional Types

**Purpose:** Create types that depend on conditions, enabling sophisticated type logic.

**Basic Conditional Type:**
```typescript
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;    // true
type B = IsString<number>;    // false
```

**Extracting Return Types:**
```typescript
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

function getUser() {
  return { id: 1, name: "John" };
}

type User = ReturnType<typeof getUser>;
// Type: { id: number; name: string; }
```

**Distributive Conditional Types:**
```typescript
type ToArray<T> = T extends any ? T[] : never;

type StrOrNumArray = ToArray<string | number>;
// Type: string[] | number[]
```

**Nested Conditions:**
```typescript
type TypeName<T> =
  T extends string ? "string" :
  T extends number ? "number" :
  T extends boolean ? "boolean" :
  T extends undefined ? "undefined" :
  T extends Function ? "function" :
  "object";

type T1 = TypeName<string>;     // "string"
type T2 = TypeName<() => void>; // "function"
```

### 3. Mapped Types

**Purpose:** Transform existing types by iterating over their properties.

**Basic Mapped Type:**
```typescript
type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

interface User {
  id: number;
  name: string;
}

type ReadonlyUser = Readonly<User>;
// Type: { readonly id: number; readonly name: string; }
```

**Optional Properties:**
```typescript
type Partial<T> = {
  [P in keyof T]?: T[P];
};

type PartialUser = Partial<User>;
// Type: { id?: number; name?: string; }
```

**Key Remapping:**
```typescript
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K]
};

interface Person {
  name: string;
  age: number;
}

type PersonGetters = Getters<Person>;
// Type: { getName: () => string; getAge: () => number; }
```

**Filtering Properties:**
```typescript
type PickByType<T, U> = {
  [K in keyof T as T[K] extends U ? K : never]: T[K]
};

interface Mixed {
  id: number;
  name: string;
  age: number;
  active: boolean;
}

type OnlyNumbers = PickByType<Mixed, number>;
// Type: { id: number; age: number; }
```

### 4. Template Literal Types

**Purpose:** Create string-based types with pattern matching and transformation.

**Basic Template Literal:**
```typescript
type EventName = "click" | "focus" | "blur";
type EventHandler = `on${Capitalize<EventName>}`;
// Type: "onClick" | "onFocus" | "onBlur"
```

**String Manipulation:**
```typescript
type UppercaseGreeting = Uppercase<"hello">;  // "HELLO"
type LowercaseGreeting = Lowercase<"HELLO">;  // "hello"
type CapitalizedName = Capitalize<"john">;    // "John"
type UncapitalizedName = Uncapitalize<"John">; // "john"
```

**Path Building:**
```typescript
type Path<T> = T extends object
  ? { [K in keyof T]: K extends string
      ? `${K}` | `${K}.${Path<T[K]>}`
      : never
    }[keyof T]
  : never;

interface Config {
  server: {
    host: string;
    port: number;
  };
  database: {
    url: string;
  };
}

type ConfigPath = Path<Config>;
// Type: "server" | "database" | "server.host" | "server.port" | "database.url"
```

### 5. Utility Types

**Built-in Utility Types:**

```typescript
// Partial<T> - Make all properties optional
type PartialUser = Partial<User>;

// Required<T> - Make all properties required
type RequiredUser = Required<PartialUser>;

// Readonly<T> - Make all properties readonly
type ReadonlyUser = Readonly<User>;

// Pick<T, K> - Select specific properties
type UserName = Pick<User, "name" | "email">;

// Omit<T, K> - Remove specific properties
type UserWithoutPassword = Omit<User, "password">;

// Exclude<T, U> - Exclude types from union
type T1 = Exclude<"a" | "b" | "c", "a">;  // "b" | "c"

// Extract<T, U> - Extract types from union
type T2 = Extract<"a" | "b" | "c", "a" | "b">;  // "a" | "b"

// NonNullable<T> - Exclude null and undefined
type T3 = NonNullable<string | null | undefined>;  // string

// Record<K, T> - Create object type with keys K and values T
type PageInfo = Record<"home" | "about", { title: string }>;
```

## Advanced Patterns

### Pattern 1: Type-Safe Event Emitter

```typescript
type EventMap = {
  "user:created": { id: string; name: string };
  "user:updated": { id: string };
  "user:deleted": { id: string };
};

class TypedEventEmitter<T extends Record<string, any>> {
  private listeners: {
    [K in keyof T]?: Array<(data: T[K]) => void>;
  } = {};

  on<K extends keyof T>(event: K, callback: (data: T[K]) => void): void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event]!.push(callback);
  }

  emit<K extends keyof T>(event: K, data: T[K]): void {
    const callbacks = this.listeners[event];
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
}

const emitter = new TypedEventEmitter<EventMap>();

emitter.on("user:created", (data) => {
  console.log(data.id, data.name);  // Type-safe!
});

emitter.emit("user:created", { id: "1", name: "John" });
// emitter.emit("user:created", { id: "1" });  // Error: missing 'name'
```

### Pattern 2: Type-Safe API Client

```typescript
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";

type EndpointConfig = {
  "/users": {
    GET: { response: User[] };
    POST: { body: { name: string; email: string }; response: User };
  };
  "/users/:id": {
    GET: { params: { id: string }; response: User };
    PUT: { params: { id: string }; body: Partial<User>; response: User };
    DELETE: { params: { id: string }; response: void };
  };
};

type ExtractParams<T> = T extends { params: infer P } ? P : never;
type ExtractBody<T> = T extends { body: infer B } ? B : never;
type ExtractResponse<T> = T extends { response: infer R } ? R : never;

class APIClient<Config extends Record<string, Record<HTTPMethod, any>>> {
  async request<
    Path extends keyof Config,
    Method extends keyof Config[Path]
  >(
    path: Path,
    method: Method,
    ...[options]: ExtractParams<Config[Path][Method]> extends never
      ? ExtractBody<Config[Path][Method]> extends never
        ? []
        : [{ body: ExtractBody<Config[Path][Method]> }]
      : [{
          params: ExtractParams<Config[Path][Method]>;
          body?: ExtractBody<Config[Path][Method]>;
        }]
  ): Promise<ExtractResponse<Config[Path][Method]>> {
    // Implementation here
    return {} as any;
  }
}

const api = new APIClient<EndpointConfig>();

// Type-safe API calls
const users = await api.request("/users", "GET");
// Type: User[]

const newUser = await api.request("/users", "POST", {
  body: { name: "John", email: "john@example.com" }
});
// Type: User

const user = await api.request("/users/:id", "GET", {
  params: { id: "123" }
});
// Type: User
```

### Pattern 3: Builder Pattern with Type Safety

```typescript
type BuilderState<T> = {
  [K in keyof T]: T[K] | undefined;
};

type RequiredKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? never : K;
}[keyof T];

type OptionalKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? K : never;
}[keyof T];

type IsComplete<T, S> =
  RequiredKeys<T> extends keyof S
    ? S[RequiredKeys<T>] extends undefined
      ? false
      : true
    : false;

class Builder<T, S extends BuilderState<T> = {}> {
  private state: S = {} as S;

  set<K extends keyof T>(
    key: K,
    value: T[K]
  ): Builder<T, S & Record<K, T[K]>> {
    this.state[key] = value;
    return this as any;
  }

  build(
    this: IsComplete<T, S> extends true ? this : never
  ): T {
    return this.state as T;
  }
}

interface User {
  id: string;
  name: string;
  email: string;
  age?: number;
}

const builder = new Builder<User>();

const user = builder
  .set("id", "1")
  .set("name", "John")
  .set("email", "john@example.com")
  .build();  // OK: all required fields set

// const incomplete = builder
//   .set("id", "1")
//   .build();  // Error: missing required fields
```

### Pattern 4: Deep Readonly/Partial

```typescript
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object
    ? T[P] extends Function
      ? T[P]
      : DeepReadonly<T[P]>
    : T[P];
};

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object
    ? T[P] extends Array<infer U>
      ? Array<DeepPartial<U>>
      : DeepPartial<T[P]>
    : T[P];
};

interface Config {
  server: {
    host: string;
    port: number;
    ssl: {
      enabled: boolean;
      cert: string;
    };
  };
  database: {
    url: string;
    pool: {
      min: number;
      max: number;
    };
  };
}

type ReadonlyConfig = DeepReadonly<Config>;
// All nested properties are readonly

type PartialConfig = DeepPartial<Config>;
// All nested properties are optional
```

### Pattern 5: Type-Safe Form Validation

```typescript
type ValidationRule<T> = {
  validate: (value: T) => boolean;
  message: string;
};

type FieldValidation<T> = {
  [K in keyof T]?: ValidationRule<T[K]>[];
};

type ValidationErrors<T> = {
  [K in keyof T]?: string[];
};

class FormValidator<T extends Record<string, any>> {
  constructor(private rules: FieldValidation<T>) {}

  validate(data: T): ValidationErrors<T> | null {
    const errors: ValidationErrors<T> = {};
    let hasErrors = false;

    for (const key in this.rules) {
      const fieldRules = this.rules[key];
      const value = data[key];

      if (fieldRules) {
        const fieldErrors: string[] = [];

        for (const rule of fieldRules) {
          if (!rule.validate(value)) {
            fieldErrors.push(rule.message);
          }
        }

        if (fieldErrors.length > 0) {
          errors[key] = fieldErrors;
          hasErrors = true;
        }
      }
    }

    return hasErrors ? errors : null;
  }
}

interface LoginForm {
  email: string;
  password: string;
}

const validator = new FormValidator<LoginForm>({
  email: [
    {
      validate: (v) => v.includes("@"),
      message: "Email must contain @"
    },
    {
      validate: (v) => v.length > 0,
      message: "Email is required"
    }
  ],
  password: [
    {
      validate: (v) => v.length >= 8,
      message: "Password must be at least 8 characters"
    }
  ]
});

const errors = validator.validate({
  email: "invalid",
  password: "short"
});
// Type: { email?: string[]; password?: string[]; } | null
```

### Pattern 6: Discriminated Unions

```typescript
type Success<T> = {
  status: "success";
  data: T;
};

type Error = {
  status: "error";
  error: string;
};

type Loading = {
  status: "loading";
};

type AsyncState<T> = Success<T> | Error | Loading;

function handleState<T>(state: AsyncState<T>): void {
  switch (state.status) {
    case "success":
      console.log(state.data);  // Type: T
      break;
    case "error":
      console.log(state.error);  // Type: string
      break;
    case "loading":
      console.log("Loading...");
      break;
  }
}

// Type-safe state machine
type State =
  | { type: "idle" }
  | { type: "fetching"; requestId: string }
  | { type: "success"; data: any }
  | { type: "error"; error: Error };

type Event =
  | { type: "FETCH"; requestId: string }
  | { type: "SUCCESS"; data: any }
  | { type: "ERROR"; error: Error }
  | { type: "RESET" };

function reducer(state: State, event: Event): State {
  switch (state.type) {
    case "idle":
      return event.type === "FETCH"
        ? { type: "fetching", requestId: event.requestId }
        : state;
    case "fetching":
      if (event.type === "SUCCESS") {
        return { type: "success", data: event.data };
      }
      if (event.type === "ERROR") {
        return { type: "error", error: event.error };
      }
      return state;
    case "success":
    case "error":
      return event.type === "RESET" ? { type: "idle" } : state;
  }
}
```

## Type Inference Techniques

### 1. Infer Keyword

```typescript
// Extract array element type
type ElementType<T> = T extends (infer U)[] ? U : never;

type NumArray = number[];
type Num = ElementType<NumArray>;  // number

// Extract promise type
type PromiseType<T> = T extends Promise<infer U> ? U : never;

type AsyncNum = PromiseType<Promise<number>>;  // number

// Extract function parameters
type Parameters<T> = T extends (...args: infer P) => any ? P : never;

function foo(a: string, b: number) {}
type FooParams = Parameters<typeof foo>;  // [string, number]
```

### 2. Type Guards

```typescript
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isArrayOf<T>(
  value: unknown,
  guard: (item: unknown) => item is T
): value is T[] {
  return Array.isArray(value) && value.every(guard);
}

const data: unknown = ["a", "b", "c"];

if (isArrayOf(data, isString)) {
  data.forEach(s => s.toUpperCase());  // Type: string[]
}
```

### 3. Assertion Functions

```typescript
function assertIsString(value: unknown): asserts value is string {
  if (typeof value !== "string") {
    throw new Error("Not a string");
  }
}

function processValue(value: unknown) {
  assertIsString(value);
  // value is now typed as string
  console.log(value.toUpperCase());
}
```

## Best Practices

1. **Use `unknown` over `any`**: Enforce type checking
2. **Prefer `interface` for object shapes**: Better error messages
3. **Use `type` for unions and complex types**: More flexible
4. **Leverage type inference**: Let TypeScript infer when possible
5. **Create helper types**: Build reusable type utilities
6. **Use const assertions**: Preserve literal types
7. **Avoid type assertions**: Use type guards instead
8. **Document complex types**: Add JSDoc comments
9. **Use strict mode**: Enable all strict compiler options
10. **Test your types**: Use type tests to verify type behavior

## Type Testing

```typescript
// Type assertion tests
type AssertEqual<T, U> =
  [T] extends [U]
    ? [U] extends [T]
      ? true
      : false
    : false;

type Test1 = AssertEqual<string, string>;        // true
type Test2 = AssertEqual<string, number>;        // false
type Test3 = AssertEqual<string | number, string>; // false

// Expect error helper
type ExpectError<T extends never> = T;

// Example usage
type ShouldError = ExpectError<AssertEqual<string, number>>;
```

## Common Pitfalls

1. **Over-using `any`**: Defeats the purpose of TypeScript
2. **Ignoring strict null checks**: Can lead to runtime errors
3. **Too complex types**: Can slow down compilation
4. **Not using discriminated unions**: Misses type narrowing opportunities
5. **Forgetting readonly modifiers**: Allows unintended mutations
6. **Circular type references**: Can cause compiler errors
7. **Not handling edge cases**: Like empty arrays or null values

## Performance Considerations

- Avoid deeply nested conditional types
- Use simple types when possible
- Cache complex type computations
- Limit recursion depth in recursive types
- Use build tools to skip type checking in production

## Resources

- **TypeScript Handbook**: https://www.typescriptlang.org/docs/handbook/
- **Type Challenges**: https://github.com/type-challenges/type-challenges
- **TypeScript Deep Dive**: https://basarat.gitbook.io/typescript/
- **Effective TypeScript**: Book by Dan Vanderkam
