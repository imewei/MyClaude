# Migration Patterns Library

**Version:** 1.0.3 | **Category:** framework-migration | **Type:** Reference

Comprehensive catalog of code migration patterns, transformation strategies, and automated codemod examples for framework and language migrations.

---

## Table of Contents

1. [Framework Migration Patterns](#framework-migration-patterns)
2. [Language Migration Patterns](#language-migration-patterns)
3. [Database Migration Patterns](#database-migration-patterns)
4. [API Migration Patterns](#api-migration-patterns)
5. [Automated Transformation Tools](#automated-transformation-tools)
6. [Code Examples](#code-examples)

---

## Framework Migration Patterns

### React Class to Functional Components

**Pattern**: Convert class components with lifecycle methods to functional components with hooks.

**Transformation Rules**:
- `componentDidMount` → `useEffect(() => { ... }, [])`
- `componentDidUpdate` → `useEffect(() => { ... }, [dependencies])`
- `componentWillUnmount` → `useEffect(() => { return () => { ... } }, [])`
- `this.state` → `useState()`
- `this.props` → function parameters

**Codemod**: `npx react-codemod class-to-hooks`

### AngularJS to Angular Migration

**Pattern**: Hybrid mode migration using ngUpgrade.

**Key Steps**:
1. Bootstrap hybrid application
2. Downgrade Angular components for use in AngularJS
3. Upgrade AngularJS components for use in Angular
4. Migrate module by module
5. Remove AngularJS when complete

**Upgrade Adapter**: `@angular/upgrade/static`

### Vue 2 to Vue 3 Composition API

**Pattern**: Options API to Composition API transformation.

**Transformation**:
- `data()` → `ref()` or `reactive()`
- `computed` → `computed()`
- `methods` → regular functions
- `mounted()` → `onMounted()`
- `watch` → `watch()` or `watchEffect()`

**Migration Tool**: `npx @vue/compat`

---

## Language Migration Patterns

### Python 2 to Python 3

**Common Transformations**:
- `print statement` → `print()` function
- `unicode()` → `str()`
- `.iteritems()` → `.items()`
- `xrange()` → `range()`
- `raise Exception, msg` → `raise Exception(msg)`
- `except Exception, e` → `except Exception as e`

**Automated Tool**: `2to3` or `python-modernize`

### Java 8 to Java 17+

**Key Updates**:
- Lambda improvements (var in lambda parameters)
- Records for immutable data classes
- Pattern matching for `instanceof`
- Text blocks for multi-line strings
- Sealed classes for restricted inheritance

### JavaScript ES5 to ES6+

**Modern Patterns**:
- `var` → `const` / `let`
- `function` → arrow functions `=>`
- `callbacks` → Promises / async-await
- `prototype` → classes
- Module imports: `require()` → `import`

---

## Database Migration Patterns

### Zero-Downtime Schema Changes

**Safe Migration Sequence**:
1. **Phase 1**: Add new column/table (nullable)
2. **Phase 2**: Dual-write to old and new
3. **Phase 3**: Backfill historical data
4. **Phase 4**: Migrate reads to new schema
5. **Phase 5**: Remove old column/table

### SQL to NoSQL Data Model Transformation

**Relational → Document Mapping**:
- **One-to-One**: Embed subdocument
- **One-to-Many** (few): Embed array
- **One-to-Many** (many): Reference by ID
- **Many-to-Many**: Array of references

---

## API Migration Patterns

### REST to GraphQL

**Schema Generation Strategy**:
- HTTP GET → GraphQL Query
- HTTP POST/PUT/PATCH → GraphQL Mutation
- REST resource → GraphQL Type
- Query parameters → GraphQL arguments

### REST API Versioning

**Approaches**:
1. **URL versioning**: `/api/v1/users`
2. **Header versioning**: `Accept: application/vnd.api+json;version=2`
3. **Query parameter**: `/api/users?version=1`

---

## Automated Transformation Tools

### Codemods

**Available Codemods**:
- **React**: `react-codemod` (JSX transforms, prop types, deprecations)
- **JavaScript**: `jscodeshift` (custom transformations)
- **TypeScript**: `ts-migrate` (JS to TS migration)
- **Python**: `2to3`, `python-modernize`

### AST-Based Transformations

**Tool**: `jscodeshift`

**Example Custom Transform**:
```javascript
module.exports = function(fileInfo, api) {
  const j = api.jscodeshift;
  const root = j(fileInfo.source);

  // Find all class components
  return root
    .find(j.ClassDeclaration)
    .filter(path => {
      return path.value.superClass &&
             path.value.superClass.name === 'Component';
    })
    .forEach(path => {
      // Transform to functional component
      // (complex logic here)
    })
    .toSource();
};
```

---

## Code Examples

### Example 1: React Class to Functional Component

**Before** (Class Component):
```jsx
class UserProfile extends React.Component {
  constructor(props) {
    super(props);
    this.state = { data: null, loading: true };
  }

  componentDidMount() {
    fetchUser(this.props.userId)
      .then(data => this.setState({ data, loading: false }));
  }

  render() {
    const { data, loading } = this.state;
    if (loading) return <div>Loading...</div>;
    return <div>{data.name}</div>;
  }
}
```

**After** (Functional Component with Hooks):
```jsx
function UserProfile({ userId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUser(userId)
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  return <div>{data.name}</div>;
}
```

### Example 2: Python 2 to Python 3

**Before** (Python 2):
```python
print "Hello, World!"

for key, value in my_dict.iteritems():
    print key, value

result = 10 / 3  # Returns 3 (integer division)
```

**After** (Python 3):
```python
print("Hello, World!")

for key, value in my_dict.items():
    print(key, value)

result = 10 / 3  # Returns 3.333... (true division)
```

---

**For migration workflows using these patterns**, see:
- `/code-migrate` command for orchestrated migration
- `/legacy-modernize` command for comprehensive legacy system transformation
