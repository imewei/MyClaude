# Styling Approaches

> **Reference**: CSS Modules, styled-components, and Tailwind CSS implementation patterns

---

## Style Generator Class

```typescript
class StyleGenerator {
  generateCSSModule(spec: ComponentSpec): string {
    const className = this.camelCase(spec.name);
    return `
.${className} {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background-color: var(--bg-primary);
}

.${className}Title {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}

.${className}Content {
  flex: 1;
  color: var(--text-secondary);
}
`;
  }

  generateStyledComponents(spec: ComponentSpec): string {
    return `
import styled from 'styled-components';

export const ${spec.name}Container = styled.div\`
  display: flex;
  flex-direction: column;
  padding: \${({ theme }) => theme.spacing.md};
  background-color: \${({ theme }) => theme.colors.background};
\`;

export const ${spec.name}Title = styled.h2\`
  font-size: \${({ theme }) => theme.fontSize.lg};
  font-weight: 600;
  color: \${({ theme }) => theme.colors.text.primary};
  margin-bottom: \${({ theme }) => theme.spacing.sm};
\`;
`;
  }

  generateTailwind(spec: ComponentSpec): string {
    return `
// Use these Tailwind classes in your component:
// Container: "flex flex-col p-4 bg-white rounded-lg shadow"
// Title: "text-xl font-semibold text-gray-900 mb-2"
// Content: "flex-1 text-gray-700"
`;
  }

  private camelCase(str: string): string {
    return str.charAt(0).toLowerCase() + str.slice(1);
  }
}
```

---

## CSS Modules

### Setup

```bash
# Install CSS Modules support
npm install --save-dev typescript-plugin-css-modules
```

### Usage Pattern

```typescript
// Component.module.css
.container {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background-color: var(--bg-primary);
}

.title {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}

.content {
  flex: 1;
  color: var(--text-secondary);
}

// Component.tsx
import styles from './Component.module.css';

export const Component: React.FC = () => {
  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Title</h2>
      <div className={styles.content}>Content</div>
    </div>
  );
};
```

### Composition

```css
/* Base styles */
.button {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
}

/* Compose with base */
.primaryButton {
  composes: button;
  background-color: var(--color-primary);
  color: white;
}

.secondaryButton {
  composes: button;
  background-color: var(--color-secondary);
  color: var(--text-primary);
}
```

### Pros & Cons

**Pros**:
- ✅ Local scope by default (no class name collisions)
- ✅ Type safety with TypeScript plugin
- ✅ Standard CSS syntax (easy learning curve)
- ✅ Great for component libraries

**Cons**:
- ❌ Requires build configuration
- ❌ No dynamic styling without inline styles
- ❌ Separate CSS file management

---

## styled-components

### Setup

```bash
# Install styled-components
npm install styled-components
npm install --save-dev @types/styled-components
```

### Usage Pattern

```typescript
import styled from 'styled-components';

// Basic styled component
export const Container = styled.div`
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing.md};
  background-color: ${({ theme }) => theme.colors.background};
`;

// With props
interface ButtonProps {
  variant?: 'primary' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
}

export const Button = styled.button<ButtonProps>`
  padding: ${({ size }) => {
    switch (size) {
      case 'sm': return '0.25rem 0.5rem';
      case 'lg': return '0.75rem 1.5rem';
      default: return '0.5rem 1rem';
    }
  }};
  background-color: ${({ variant, theme }) =>
    variant === 'primary' ? theme.colors.primary : theme.colors.secondary
  };
  color: white;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;

  &:hover {
    opacity: 0.9;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

// Usage
<Container>
  <Button variant="primary" size="lg">
    Click Me
  </Button>
</Container>
```

### Theme Provider

```typescript
// theme.ts
export const theme = {
  colors: {
    primary: '#3B82F6',
    secondary: '#10B981',
    background: '#FFFFFF',
    text: {
      primary: '#1F2937',
      secondary: '#6B7280'
    }
  },
  spacing: {
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem'
  },
  fontSize: {
    sm: '0.875rem',
    md: '1rem',
    lg: '1.25rem',
    xl: '1.5rem'
  },
  breakpoints: {
    mobile: '640px',
    tablet: '768px',
    desktop: '1024px'
  }
};

// App.tsx
import { ThemeProvider } from 'styled-components';
import { theme } from './theme';

export const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <YourComponents />
    </ThemeProvider>
  );
};
```

### Pros & Cons

**Pros**:
- ✅ Dynamic styling with props
- ✅ Theme support out of the box
- ✅ No class name collisions
- ✅ Component-scoped styles
- ✅ Server-side rendering support

**Cons**:
- ❌ Runtime overhead (styles generated in JS)
- ❌ Larger bundle size
- ❌ Requires learning new syntax

---

## Tailwind CSS

### Setup

```bash
# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Configuration

```javascript
// tailwind.config.js
module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        secondary: '#10B981',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      }
    },
  },
  plugins: [],
}
```

### Usage Pattern

```typescript
// Basic usage
export const Component: React.FC = () => {
  return (
    <div className="flex flex-col p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-semibold text-gray-900 mb-2">Title</h2>
      <div className="flex-1 text-gray-700">Content</div>
    </div>
  );
};

// Responsive design
<div className="
  grid
  grid-cols-1
  md:grid-cols-2
  lg:grid-cols-3
  gap-4
">
  {/* Cards */}
</div>

// Hover and focus states
<button className="
  bg-blue-500
  hover:bg-blue-600
  focus:ring-2
  focus:ring-blue-300
  text-white
  px-4
  py-2
  rounded
">
  Click Me
</button>

// Dark mode
<div className="bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
  Content adapts to theme
</div>
```

### Custom Components with Tailwind

```typescript
// Using clsx for conditional classes
import clsx from 'clsx';

interface ButtonProps {
  variant?: 'primary' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  children
}) => {
  return (
    <button
      className={clsx(
        'rounded font-medium transition',
        {
          'bg-blue-500 hover:bg-blue-600 text-white': variant === 'primary',
          'bg-gray-200 hover:bg-gray-300 text-gray-900': variant === 'secondary',
          'px-2 py-1 text-sm': size === 'sm',
          'px-4 py-2 text-base': size === 'md',
          'px-6 py-3 text-lg': size === 'lg',
        }
      )}
    >
      {children}
    </button>
  );
};
```

### Pros & Cons

**Pros**:
- ✅ Minimal bundle size (only used classes)
- ✅ No CSS file management
- ✅ Fast prototyping
- ✅ Responsive design utilities
- ✅ Built-in dark mode support

**Cons**:
- ❌ Long className strings
- ❌ Learning curve for utility classes
- ❌ Less semantic HTML

---

## React Native StyleSheet

### Usage Pattern

```typescript
import { StyleSheet, View, Text } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 8,
  },
  content: {
    flex: 1,
    color: '#6B7280',
  },
});

export const Component: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Title</Text>
      <Text style={styles.content}>Content</Text>
    </View>
  );
};
```

### Dynamic Styles

```typescript
// Array of styles
<View style={[styles.container, isActive && styles.active]} />

// Inline styles (avoid for performance)
<View style={{ backgroundColor: color, padding: 16 }} />

// Function to generate styles
const getButtonStyle = (variant: string) => {
  return [
    styles.button,
    variant === 'primary' && styles.primaryButton,
    variant === 'secondary' && styles.secondaryButton,
  ];
};

<TouchableOpacity style={getButtonStyle('primary')} />
```

---

## Styling Strategy Decision Tree

```
Choose styling approach based on:

1. Are you building for React Native?
   YES → Use StyleSheet.create
   NO → Continue

2. Do you need dynamic theming?
   YES → Use styled-components
   NO → Continue

3. Do you prefer utility-first approach?
   YES → Use Tailwind CSS
   NO → Use CSS Modules

4. Do you need component library consistency?
   YES → Use CSS Modules or styled-components
   NO → Use Tailwind CSS for rapid prototyping
```

---

## Performance Comparison

| Approach | Bundle Size | Runtime Overhead | Build Time | Type Safety |
|----------|-------------|------------------|------------|-------------|
| **CSS Modules** | Small | None | Fast | Good |
| **styled-components** | Medium | Yes (CSS-in-JS) | Medium | Excellent |
| **Tailwind CSS** | Very Small | None | Fast | Fair |
| **React Native StyleSheet** | N/A | Minimal | Fast | Good |

---

## Best Practices

### CSS Modules
- Use semantic class names: `.container`, `.title`, not `.div1`, `.text2`
- Leverage composition for reusable styles
- Use CSS variables for theming

### styled-components
- Extract styled components to separate files for large components
- Use `css` helper for shared style fragments
- Implement proper TypeScript types for props

### Tailwind CSS
- Use `@apply` directive for repeated patterns
- Create custom utilities in `tailwind.config.js`
- Use `clsx` or `classnames` for conditional classes
- Extract complex class combinations into components

### React Native
- Always use `StyleSheet.create` (performance optimization)
- Avoid inline styles in render method
- Use `Platform.select()` for platform-specific styles
- Leverage `useWindowDimensions` for responsive layouts
