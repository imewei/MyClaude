# Library & CLI Tool Scaffolding Guide

> **Version:** 1.0.3 | **Category:** Packages & Tools | **Maturity:** 95%

## TypeScript Library Scaffolding

### Complete Library Structure

```
library-name/
├── package.json
├── tsconfig.json
├── tsconfig.build.json
├── .npmignore
├── README.md
├── LICENSE
├── src/
│   ├── index.ts                    # Main export
│   ├── core.ts                     # Core functionality
│   └── utils.ts                    # Utilities
├── tests/
│   ├── setup.ts
│   └── core.test.ts
└── dist/                           # Build output
    ├── index.js
    ├── index.d.ts
    └── package.json
```

### package.json for Library

```json
{
  "name": "@scope/library-name",
  "version": "0.1.0",
  "description": "Library description",
  "type": "module",
  "main": "./dist/index.js",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": {
        "types": "./dist/index.d.ts",
        "default": "./dist/index.js"
      }
    },
    "./utils": {
      "import": {
        "types": "./dist/utils.d.ts",
        "default": "./dist/utils.js"
      }
    }
  },
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "build": "tsc -p tsconfig.build.json",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "prepublishOnly": "pnpm build && pnpm test",
    "release": "changeset publish"
  },
  "keywords": ["typescript", "library"],
  "author": "Your Name",
  "license": "MIT",
  "devDependencies": {
    "typescript": "^5.3.0",
    "vitest": "^1.2.0",
    "@changesets/cli": "^2.27.0"
  },
  "peerDependencies": {},
  "dependencies": {}
}
```

### tsconfig.json (Development)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "bundler"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

### tsconfig.build.json (Production)

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "declaration": true,
    "declarationMap": false,
    "sourceMap": false
  },
  "exclude": ["node_modules", "dist", "tests", "**/*.test.ts"]
}
```

### Library Entry Point

```typescript
// src/index.ts
export { CoreClass, type CoreOptions } from './core.js'
export { utilityFunction } from './utils.js'
export type { UtilityOptions, UtilityResult } from './types.js'

// Re-export commonly used types
export type * from './types.js'
```

### Example Core Module

```typescript
// src/core.ts
export interface CoreOptions {
  debug?: boolean
  timeout?: number
}

export class CoreClass {
  private options: Required<CoreOptions>

  constructor(options: CoreOptions = {}) {
    this.options = {
      debug: options.debug ?? false,
      timeout: options.timeout ?? 5000,
    }
  }

  async process(input: string): Promise<string> {
    if (this.options.debug) {
      console.log(`Processing: ${input}`)
    }
    return input.toUpperCase()
  }
}
```

---

## CLI Tool Scaffolding

### Complete CLI Structure

```
cli-tool/
├── package.json
├── tsconfig.json
├── bin/
│   └── cli.ts                      # CLI entry point
├── src/
│   ├── index.ts                    # Programmatic API
│   ├── cli/
│   │   ├── commands/
│   │   │   ├── init.ts
│   │   │   ├── build.ts
│   │   │   └── deploy.ts
│   │   ├── ui/
│   │   │   ├── prompts.ts
│   │   │   ├── spinner.ts
│   │   │   └── logger.ts
│   │   └── utils/
│   │       └── config.ts
│   └── core/
│       └── processor.ts
└── tests/
    └── commands/
        └── init.test.ts
```

### package.json for CLI

```json
{
  "name": "cli-tool",
  "version": "0.1.0",
  "description": "CLI tool description",
  "type": "module",
  "bin": {
    "cli-tool": "./dist/bin/cli.js"
  },
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "files": [
    "dist",
    "README.md"
  ],
  "scripts": {
    "build": "tsc && chmod +x dist/bin/cli.js",
    "dev": "tsx watch bin/cli.ts",
    "test": "vitest",
    "cli": "tsx bin/cli.ts"
  },
  "dependencies": {
    "commander": "^11.1.0",
    "inquirer": "^9.2.12",
    "chalk": "^5.3.0",
    "ora": "^8.0.1"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@types/inquirer": "^9.0.7",
    "typescript": "^5.3.0",
    "tsx": "^4.7.0",
    "vitest": "^1.2.0"
  }
}
```

### CLI Entry Point with Commander.js

```typescript
#!/usr/bin/env node
// bin/cli.ts
import { Command } from 'commander'
import chalk from 'chalk'
import { initCommand } from '../src/cli/commands/init.js'
import { buildCommand } from '../src/cli/commands/build.js'

const program = new Command()

program
  .name('cli-tool')
  .description('A powerful CLI tool')
  .version('0.1.0')

program
  .command('init')
  .description('Initialize a new project')
  .option('-n, --name <name>', 'Project name')
  .option('-t, --template <template>', 'Template to use')
  .action(initCommand)

program
  .command('build')
  .description('Build the project')
  .option('-w, --watch', 'Watch mode')
  .option('-m, --minify', 'Minify output')
  .action(buildCommand)

program.parse()
```

### CLI Command with Inquirer

```typescript
// src/cli/commands/init.ts
import inquirer from 'inquirer'
import chalk from 'chalk'
import ora from 'ora'
import fs from 'fs/promises'

export async function initCommand(options: {
  name?: string
  template?: string
}) {
  console.log(chalk.bold.blue('🚀 Initializing project...'))

  // Prompt for missing options
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'name',
      message: 'Project name:',
      default: options.name || 'my-project',
      validate: (input) => {
        if (!input.trim()) return 'Project name is required'
        return true
      },
    },
    {
      type: 'list',
      name: 'template',
      message: 'Choose a template:',
      choices: ['basic', 'advanced', 'custom'],
      default: options.template || 'basic',
    },
    {
      type: 'confirm',
      name: 'install',
      message: 'Install dependencies?',
      default: true,
    },
  ])

  const spinner = ora('Creating project...').start()

  try {
    // Create project directory
    await fs.mkdir(answers.name, { recursive: true })

    // Create files based on template
    await createTemplate(answers.name, answers.template)

    spinner.succeed(chalk.green('Project created successfully!'))

    if (answers.install) {
      spinner.text = 'Installing dependencies...'
      spinner.start()
      // Install dependencies
      spinner.succeed(chalk.green('Dependencies installed!'))
    }

    console.log(chalk.bold.green('\n✨ Done! Next steps:'))
    console.log(chalk.gray(`  cd ${answers.name}`))
    console.log(chalk.gray(`  npm run dev`))
  } catch (error) {
    spinner.fail(chalk.red('Failed to create project'))
    console.error(error)
    process.exit(1)
  }
}
```

### Interactive Prompts with Validation

```typescript
// src/cli/ui/prompts.ts
import inquirer from 'inquirer'

export async function promptConfig() {
  return inquirer.prompt([
    {
      type: 'input',
      name: 'apiKey',
      message: 'Enter your API key:',
      validate: (input) => {
        if (input.length < 10) return 'API key must be at least 10 characters'
        return true
      },
      filter: (input) => input.trim(),
    },
    {
      type: 'list',
      name: 'environment',
      message: 'Select environment:',
      choices: ['development', 'staging', 'production'],
    },
    {
      type: 'checkbox',
      name: 'features',
      message: 'Select features to enable:',
      choices: [
        { name: 'Authentication', value: 'auth' },
        { name: 'Database', value: 'db' },
        { name: 'Caching', value: 'cache' },
      ],
    },
    {
      type: 'confirm',
      name: 'proceed',
      message: 'Proceed with configuration?',
      default: true,
    },
  ])
}
```

### Colorized Output

```typescript
// src/cli/ui/logger.ts
import chalk from 'chalk'

export const logger = {
  success: (message: string) => {
    console.log(chalk.green('✓ ' + message))
  },

  error: (message: string) => {
    console.error(chalk.red('✗ ' + message))
  },

  warning: (message: string) => {
    console.warn(chalk.yellow('⚠ ' + message))
  },

  info: (message: string) => {
    console.log(chalk.blue('ℹ ' + message))
  },

  debug: (message: string) => {
    if (process.env.DEBUG) {
      console.log(chalk.gray('  ' + message))
    }
  },
}
```

### Progress Indicators

```typescript
// src/cli/ui/spinner.ts
import ora from 'ora'

export async function withSpinner<T>(
  text: string,
  task: () => Promise<T>
): Promise<T> {
  const spinner = ora(text).start()

  try {
    const result = await task()
    spinner.succeed()
    return result
  } catch (error) {
    spinner.fail()
    throw error
  }
}

// Usage
await withSpinner('Downloading templates...', async () => {
  await downloadTemplates()
})
```

---

## Publishing Workflow

### npm Publishing

```bash
# 1. Build the library
pnpm build

# 2. Run tests
pnpm test

# 3. Update version
npm version patch # or minor, major

# 4. Publish to npm
npm publish --access public

# 5. Push tags
git push --tags
```

### Changesets Workflow

```bash
# 1. Add changeset
pnpm changeset

# 2. Version packages
pnpm changeset version

# 3. Publish
pnpm changeset publish
```

---

## Testing CLI Tools

```typescript
// tests/commands/init.test.ts
import { describe, it, expect, vi } from 'vitest'
import { initCommand } from '../../src/cli/commands/init.js'
import inquirer from 'inquirer'

describe('init command', () => {
  it('should create project with default options', async () => {
    // Mock inquirer
    vi.spyOn(inquirer, 'prompt').mockResolvedValue({
      name: 'test-project',
      template: 'basic',
      install: false,
    })

    await initCommand({})

    // Assert project was created
    expect(fs.existsSync('test-project')).toBe(true)
  })
})
```

---

## Distribution

### Creating Executables

```json
// package.json
{
  "scripts": {
    "build:bin": "pkg . --output bin/cli-tool"
  },
  "devDependencies": {
    "pkg": "^5.8.1"
  },
  "pkg": {
    "targets": ["node18-linux-x64", "node18-macos-x64", "node18-win-x64"]
  }
}
```

### GitHub Releases with Binaries

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - run: pnpm install
      - run: pnpm build
      - run: pnpm build:bin
      - uses: softprops/action-gh-release@v1
        with:
          files: bin/*
```

---

## Related Documentation

- [Project Scaffolding Guide](project-scaffolding-guide.md)
- [TypeScript Configuration](typescript-configuration.md)
- [Development Tooling](development-tooling.md)
