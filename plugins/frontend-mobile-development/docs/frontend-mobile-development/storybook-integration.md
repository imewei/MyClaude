# Storybook Integration

> **Reference**: Storybook stories generation and documentation patterns

---

## Storybook Generator Class

```typescript
class StorybookGenerator {
  generateStories(spec: ComponentSpec): string {
    return `
import type { Meta, StoryObj } from '@storybook/react';
import { ${spec.name} } from './${spec.name}';

const meta: Meta<typeof ${spec.name}> = {
  title: 'Components/${spec.name}',
  component: ${spec.name},
  tags: ['autodocs'],
  argTypes: {
${spec.props.map(p => `    ${p.name}: { control: '${this.inferControl(p.type)}', description: '${p.description}' },`).join('\n')}
  },
};

export default meta;
type Story = StoryObj<typeof ${spec.name}>;

export const Default: Story = {
  args: {
${spec.props.map(p => `    ${p.name}: ${p.defaultValue || this.getMockValue(p.type)},`).join('\n')}
  },
};

export const Interactive: Story = {
  args: {
    ...Default.args,
  },
};
`;
  }

  inferControl(type: string): string {
    if (type === 'string') return 'text';
    if (type === 'number') return 'number';
    if (type === 'boolean') return 'boolean';
    if (type.includes('[]')) return 'object';
    return 'text';
  }

  private getMockValue(type: string): string {
    if (type === 'string') return "'Default text'";
    if (type === 'number') return '0';
    if (type === 'boolean') return 'false';
    if (type.includes('[]')) return '[]';
    if (type.includes('()')) return '() => {}';
    return '{}';
  }
}
```

---

## Setup Storybook

### Installation

```bash
# Initialize Storybook
npx storybook@latest init

# Install additional addons
npm install --save-dev @storybook/addon-a11y @storybook/addon-interactions
```

### Configuration

```typescript
// .storybook/main.ts
import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx)'],
  addons: [
    '@storybook/addon-links',
    '@storybook/addon-essentials',
    '@storybook/addon-interactions',
    '@storybook/addon-a11y',
  ],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  docs: {
    autodocs: 'tag',
  },
};

export default config;
```

---

## Story Patterns

### Basic Story

```typescript
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Primary: Story = {
  args: {
    variant: 'primary',
    label: 'Click Me',
    onClick: () => alert('Clicked!'),
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    label: 'Click Me',
  },
};

export const Disabled: Story = {
  args: {
    label: 'Disabled',
    disabled: true,
  },
};
```

### Story with Args

```typescript
const meta: Meta<typeof Card> = {
  title: 'Components/Card',
  component: Card,
  argTypes: {
    title: { control: 'text', description: 'Card title' },
    content: { control: 'text', description: 'Card content' },
    variant: {
      control: 'select',
      options: ['default', 'outlined', 'elevated'],
    },
    size: {
      control: 'radio',
      options: ['sm', 'md', 'lg'],
    },
    isLoading: { control: 'boolean' },
  },
};

export const Default: Story = {
  args: {
    title: 'Card Title',
    content: 'This is the card content',
    variant: 'default',
    size: 'md',
    isLoading: false,
  },
};
```

### Story with Decorators

```typescript
import { ThemeProvider } from 'styled-components';
import { theme } from '@/theme';

const meta: Meta<typeof ThemedButton> = {
  title: 'Components/ThemedButton',
  component: ThemedButton,
  decorators: [
    (Story) => (
      <ThemeProvider theme={theme}>
        <div style={{ padding: '2rem' }}>
          <Story />
        </div>
      </ThemeProvider>
    ),
  ],
};
```

### Story with Play Function

```typescript
import { userEvent, within } from '@storybook/testing-library';
import { expect } from '@storybook/jest';

export const FilledForm: Story = {
  args: {
    onSubmit: fn(),
  },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);

    await userEvent.type(canvas.getByLabelText('Email'), 'user@example.com');
    await userEvent.type(canvas.getByLabelText('Password'), 'password123');
    await userEvent.click(canvas.getByRole('button', { name: 'Submit' }));

    await expect(args.onSubmit).toHaveBeenCalled();
  },
};
```

---

## ArgTypes Controls

### Control Types

```typescript
argTypes: {
  // Text input
  title: { control: 'text' },

  // Number slider
  size: { control: { type: 'number', min: 0, max: 100, step: 5 } },

  // Range slider
  opacity: { control: { type: 'range', min: 0, max: 1, step: 0.1 } },

  // Boolean checkbox
  isActive: { control: 'boolean' },

  // Radio buttons
  variant: { control: 'radio', options: ['primary', 'secondary'] },

  // Select dropdown
  color: { control: 'select', options: ['red', 'green', 'blue'] },

  // Multi-select
  tags: { control: 'multi-select', options: ['tag1', 'tag2', 'tag3'] },

  // Color picker
  backgroundColor: { control: 'color' },

  // Date picker
  publishedAt: { control: 'date' },

  // Object editor
  config: { control: 'object' },

  // File upload
  avatar: { control: 'file', accept: '.jpg,.png' },
}
```

---

## Documentation

### JSDoc for Auto-generated Docs

```typescript
/**
 * Primary button component for user interactions
 *
 * @example
 * ```tsx
 * <Button variant="primary" onClick={handleClick}>
 *   Click Me
 * </Button>
 * ```
 */
export const Button: React.FC<ButtonProps> = ({ variant, label, onClick }) => {
  // Implementation
};

export interface ButtonProps {
  /**
   * Button variant style
   * @default 'primary'
   */
  variant?: 'primary' | 'secondary' | 'tertiary';

  /**
   * Button label text
   */
  label: string;

  /**
   * Click handler function
   */
  onClick?: () => void;

  /**
   * Disable the button
   * @default false
   */
  disabled?: boolean;
}
```

### MDX Documentation

```mdx
{/* Button.stories.mdx */}
import { Meta, Story, Canvas, ArgsTable } from '@storybook/blocks';
import { Button } from './Button';

<Meta title="Components/Button" component={Button} />

# Button

Buttons are used to trigger actions and events.

## Usage

```tsx
import { Button } from '@/components/Button';

<Button variant="primary" onClick={handleClick}>
  Click Me
</Button>
```

## Variants

<Canvas>
  <Story name="Primary">
    <Button variant="primary">Primary</Button>
  </Story>
  <Story name="Secondary">
    <Button variant="secondary">Secondary</Button>
  </Story>
</Canvas>

## Props

<ArgsTable of={Button} />
```

---

## Accessibility Testing in Storybook

### Configure A11y Addon

```typescript
// .storybook/preview.ts
import { withA11y } from '@storybook/addon-a11y';

export const decorators = [withA11y];

export const parameters = {
  a11y: {
    config: {
      rules: [
        {
          id: 'color-contrast',
          enabled: true,
        },
      ],
    },
  },
};
```

### Story-Level A11y Configuration

```typescript
export const AccessibleButton: Story = {
  args: {
    label: 'Accessible Button',
  },
  parameters: {
    a11y: {
      config: {
        rules: [
          {
            id: 'color-contrast',
            enabled: true,
          },
          {
            id: 'button-name',
            enabled: true,
          },
        ],
      },
    },
  },
};
```

---

## Component Variants Matrix

### Generate Multiple Variants

```typescript
const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
};

export default meta;
type Story = StoryObj<typeof Button>;

// Generate stories for all variant combinations
const variants = ['primary', 'secondary', 'tertiary'] as const;
const sizes = ['sm', 'md', 'lg'] as const;

variants.forEach(variant => {
  sizes.forEach(size => {
    const storyName = `${variant.charAt(0).toUpperCase() + variant.slice(1)}_${size.toUpperCase()}`;

    exports[storyName] = {
      args: {
        variant,
        size,
        label: `${variant} ${size}`,
      },
    } satisfies Story;
  });
});
```

---

## Responsive Stories

### Viewport Configuration

```typescript
export const ResponsiveCard: Story = {
  args: {
    title: 'Responsive Card',
  },
  parameters: {
    viewport: {
      viewports: {
        mobile: {
          name: 'Mobile',
          styles: { width: '375px', height: '667px' },
        },
        tablet: {
          name: 'Tablet',
          styles: { width: '768px', height: '1024px' },
        },
        desktop: {
          name: 'Desktop',
          styles: { width: '1440px', height: '900px' },
        },
      },
      defaultViewport: 'mobile',
    },
  },
};
```

---

## Storybook for React Native

### Setup React Native Storybook

```bash
npx storybook@latest init --type react_native
```

### React Native Story

```typescript
import { storiesOf } from '@storybook/react-native';
import { NativeButton } from './NativeButton';

storiesOf('Button', module)
  .add('Primary', () => (
    <NativeButton variant="primary" label="Primary Button" />
  ))
  .add('Secondary', () => (
    <NativeButton variant="secondary" label="Secondary Button" />
  ));
```

---

## Best Practices

1. **Organize Stories by Domain**: Group related components (e.g., `Forms/Input`, `Forms/Button`)
2. **Use Autodocs**: Enable `tags: ['autodocs']` for automatic documentation
3. **Provide All Variants**: Show all possible states (default, hover, disabled, loading, error)
4. **Add Play Functions**: Demonstrate interactions with `play` functions
5. **Test Accessibility**: Enable `@storybook/addon-a11y` for all components
6. **Document Props**: Use JSDoc comments for auto-generated prop tables
7. **Show Edge Cases**: Include stories for empty states, long text, overflow scenarios
8. **Use Decorators**: Wrap stories with necessary providers (theme, router, state)
9. **Responsive Testing**: Configure viewports for mobile, tablet, desktop
10. **Interaction Testing**: Add `@storybook/addon-interactions` for user flow validation

---

## Storybook Scripts

```json
// package.json
{
  "scripts": {
    "storybook": "storybook dev -p 6006",
    "build-storybook": "storybook build",
    "test-storybook": "test-storybook"
  }
}
```

---

## Export Stories for Testing

```typescript
// Button.stories.ts
export const Primary = { ... };
export const Secondary = { ... };

// Button.test.tsx
import { composeStories } from '@storybook/testing-react';
import * as stories from './Button.stories';

const { Primary, Secondary } = composeStories(stories);

describe('Button', () => {
  it('renders primary button', () => {
    render(<Primary />);
    expect(screen.getByRole('button')).toBeInTheDocument();
  });
});
```
