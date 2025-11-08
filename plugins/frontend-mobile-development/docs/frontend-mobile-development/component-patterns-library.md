# Component Patterns Library

> **Reference**: React & React Native component implementation patterns with TypeScript

---

## Component Specification Interface

```typescript
interface ComponentSpec {
  name: string;
  type: 'functional' | 'page' | 'layout' | 'form' | 'data-display';
  props: PropDefinition[];
  state?: StateDefinition[];
  hooks?: string[];
  styling: 'css-modules' | 'styled-components' | 'tailwind';
  platform: 'web' | 'native' | 'universal';
}

interface PropDefinition {
  name: string;
  type: string;
  required: boolean;
  defaultValue?: any;
  description: string;
}

interface StateDefinition {
  name: string;
  type: string;
  initial: any;
}

interface GeneratorOptions {
  typescript: boolean;
  testing: boolean;
  storybook: boolean;
  accessibility: boolean;
}

interface ComponentFiles {
  component: string;
  types: string | null;
  styles: string;
  tests: string | null;
  stories: string | null;
  index: string;
}
```

---

## React Web Component Generator

### Class Implementation

```typescript
class ReactComponentGenerator {
  generate(spec: ComponentSpec, options: GeneratorOptions): ComponentFiles {
    return {
      component: this.generateComponent(spec, options),
      types: options.typescript ? this.generateTypes(spec) : null,
      styles: this.generateStyles(spec),
      tests: options.testing ? this.generateTests(spec) : null,
      stories: options.storybook ? this.generateStories(spec) : null,
      index: this.generateIndex(spec)
    };
  }

  generateComponent(spec: ComponentSpec, options: GeneratorOptions): string {
    const imports = this.generateImports(spec, options);
    const types = options.typescript ? this.generatePropTypes(spec) : '';
    const component = this.generateComponentBody(spec, options);
    const exports = this.generateExports(spec);

    return `${imports}\n\n${types}\n\n${component}\n\n${exports}`;
  }

  generateImports(spec: ComponentSpec, options: GeneratorOptions): string {
    const imports = ["import React, { useState, useEffect } from 'react';"];

    if (spec.styling === 'css-modules') {
      imports.push(`import styles from './${spec.name}.module.css';`);
    } else if (spec.styling === 'styled-components') {
      imports.push("import styled from 'styled-components';");
    }

    if (options.accessibility) {
      imports.push("import { useA11y } from '@/hooks/useA11y';");
    }

    return imports.join('\n');
  }

  generatePropTypes(spec: ComponentSpec): string {
    const props = spec.props.map(p => {
      const optional = p.required ? '' : '?';
      const comment = p.description ? `  /** ${p.description} */\n` : '';
      return `${comment}  ${p.name}${optional}: ${p.type};`;
    }).join('\n');

    return `export interface ${spec.name}Props {\n${props}\n}`;
  }

  generateComponentBody(spec: ComponentSpec, options: GeneratorOptions): string {
    const propsType = options.typescript ? `: React.FC<${spec.name}Props>` : '';
    const destructuredProps = spec.props.map(p => p.name).join(', ');

    let body = `export const ${spec.name}${propsType} = ({ ${destructuredProps} }) => {\n`;

    // Add state hooks
    if (spec.state) {
      body += spec.state.map(s =>
        `  const [${s.name}, set${this.capitalize(s.name)}] = useState${options.typescript ? `<${s.type}>` : ''}(${s.initial});\n`
      ).join('');
      body += '\n';
    }

    // Add effects
    if (spec.hooks?.includes('useEffect')) {
      body += `  useEffect(() => {\n`;
      body += `    // TODO: Add effect logic\n`;
      body += `  }, [${destructuredProps}]);\n\n`;
    }

    // Add accessibility
    if (options.accessibility) {
      body += `  const a11yProps = useA11y({\n`;
      body += `    role: '${this.inferAriaRole(spec.type)}',\n`;
      body += `    label: ${spec.props.find(p => p.name === 'label')?.name || `'${spec.name}'`}\n`;
      body += `  });\n\n`;
    }

    // JSX return
    body += `  return (\n`;
    body += this.generateJSX(spec, options);
    body += `  );\n`;
    body += `};`;

    return body;
  }

  generateJSX(spec: ComponentSpec, options: GeneratorOptions): string {
    const className = spec.styling === 'css-modules' ? `className={styles.${this.camelCase(spec.name)}}` : '';
    const a11y = options.accessibility ? '{...a11yProps}' : '';

    return `    <div ${className} ${a11y}>\n` +
           `      {/* TODO: Add component content */}\n` +
           `    </div>\n`;
  }

  private capitalize(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  private camelCase(str: string): string {
    return str.charAt(0).toLowerCase() + str.slice(1);
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

## React Native Component Generator

### Class Implementation

```typescript
class ReactNativeGenerator {
  generateComponent(spec: ComponentSpec): string {
    return `
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  AccessibilityInfo
} from 'react-native';

interface ${spec.name}Props {
${spec.props.map(p => `  ${p.name}${p.required ? '' : '?'}: ${this.mapNativeType(p.type)};`).join('\n')}
}

export const ${spec.name}: React.FC<${spec.name}Props> = ({
  ${spec.props.map(p => p.name).join(',\n  ')}
}) => {
  return (
    <View
      style={styles.container}
      accessible={true}
      accessibilityLabel="${spec.name} component"
    >
      <Text style={styles.text}>
        {/* Component content */}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  text: {
    fontSize: 16,
    color: '#333',
  },
});
`;
  }

  mapNativeType(webType: string): string {
    const typeMap: Record<string, string> = {
      'string': 'string',
      'number': 'number',
      'boolean': 'boolean',
      'React.ReactNode': 'React.ReactNode',
      'Function': '() => void'
    };
    return typeMap[webType] || webType;
  }
}
```

---

## Component Type Patterns

### Functional Component

**Use case**: Stateless UI components, presentational components

```typescript
// Basic functional component
export const Button: React.FC<ButtonProps> = ({ label, onClick, variant = 'primary' }) => {
  return (
    <button className={`btn btn-${variant}`} onClick={onClick}>
      {label}
    </button>
  );
};
```

### Page Component

**Use case**: Top-level route components, full-page layouts

```typescript
// Page component with data fetching
export const ProductPage: React.FC<ProductPageProps> = ({ productId }) => {
  const { data, isLoading } = useQuery(['product', productId], fetchProduct);

  if (isLoading) return <LoadingSpinner />;

  return (
    <main role="main">
      <ProductDetails product={data} />
      <RelatedProducts productId={productId} />
    </main>
  );
};
```

### Layout Component

**Use case**: Wrapper components, consistent page structure

```typescript
// Layout component with slots
export const AppLayout: React.FC<AppLayoutProps> = ({ header, sidebar, children, footer }) => {
  return (
    <div className="app-layout">
      <header>{header}</header>
      <div className="content-wrapper">
        <aside>{sidebar}</aside>
        <main>{children}</main>
      </div>
      <footer>{footer}</footer>
    </div>
  );
};
```

### Form Component

**Use case**: Input forms, user data collection

```typescript
// Form component with validation
export const ContactForm: React.FC<ContactFormProps> = ({ onSubmit }) => {
  const { register, handleSubmit, formState: { errors } } = useForm<FormData>();

  return (
    <form onSubmit={handleSubmit(onSubmit)} role="form">
      <input
        {...register('email', { required: 'Email is required' })}
        type="email"
        aria-invalid={!!errors.email}
      />
      {errors.email && <span role="alert">{errors.email.message}</span>}
      <button type="submit">Submit</button>
    </form>
  );
};
```

### Data Display Component

**Use case**: Tables, lists, cards displaying data

```typescript
// Data display component
export const UserList: React.FC<UserListProps> = ({ users, onUserSelect }) => {
  return (
    <ul role="list" aria-label="User list">
      {users.map(user => (
        <li key={user.id}>
          <UserCard user={user} onClick={() => onUserSelect(user)} />
        </li>
      ))}
    </ul>
  );
};
```

---

## Hook Patterns

### useState Pattern

```typescript
const [count, setCount] = useState<number>(0);
const [user, setUser] = useState<User | null>(null);
const [isLoading, setIsLoading] = useState<boolean>(false);
```

### useEffect Pattern

```typescript
useEffect(() => {
  // Fetch data on mount
  fetchData().then(setData);

  // Cleanup function
  return () => {
    cleanup();
  };
}, [dependency]);
```

### Custom Hook Pattern

```typescript
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}
```

---

## Platform Selection Guide

| Platform | Use Case | Technologies | Performance |
|----------|----------|--------------|-------------|
| **web** | Web applications, responsive design | React, Next.js, Vite | High (browser optimization) |
| **native** | Platform-specific features, optimal performance | React Native, Swift, Kotlin | Highest (native code) |
| **universal** | Cross-platform with shared codebase | React Native Web, Expo | Medium (compromise) |

---

## Component Naming Conventions

- **PascalCase** for component names: `UserProfile`, `ProductCard`
- **camelCase** for prop names: `onClick`, `isActive`, `userName`
- **camelCase** for state variables: `isLoading`, `userData`, `count`
- **UPPER_SNAKE_CASE** for constants: `API_URL`, `MAX_RETRIES`
- **kebab-case** for CSS classes: `user-profile`, `product-card`
