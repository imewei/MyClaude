# Import Resolution Strategies

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: Systematic approaches for resolving broken imports, path aliases, and module references

## Import Detection Patterns

### Language-Specific Import Scanning

**TypeScript/JavaScript**:
```bash
# ES6 imports
grep -r "^import.*from ['\"]" --include="*.ts" --include="*.tsx" --include="*.js" --include="*.jsx"

# CommonJS requires
grep -r "require(['\"]" --include="*.js" --include="*.ts"

# Dynamic imports
grep -r "import(['\"]" --include="*.ts" --include="*.js"
```

**Python**:
```bash
# Standard imports
grep -r "^import " --include="*.py"

# From imports
grep -r "^from .* import" --include="*.py"

# Relative imports
grep -r "^from \\..*import" --include="*.py"
```

**Rust**:
```bash
# Use statements
grep -r "^use " --include="*.rs"

# Crate imports
grep -r "extern crate" --include="*.rs"
```

## Path Resolution Algorithm

### Absolute vs Relative Path Detection

```typescript
class ImportPathResolver {
    constructor(
        private projectRoot: string,
        private aliases: Map<string, string>
    ) {}

    resolveImportPath(importStatement: string, sourceFilePath: string): Resolution {
        // Extract import path from statement
        const importPath = this.extractPath(importStatement);

        // Determine import type
        if (importPath.startsWith('./') || importPath.startsWith('../')) {
            return this.resolveRelativePath(importPath, sourceFilePath);
        } else if (this.isPathAlias(importPath)) {
            return this.resolveAlias(importPath);
        } else if (importPath.startsWith('/')) {
            return this.resolveAbsolutePath(importPath);
        } else {
            return this.resolveNodeModules(importPath);
        }
    }

    private resolveRelativePath(importPath: string, sourceFile: string): Resolution {
        const sourceDir = path.dirname(sourceFile);
        const absolutePath = path.resolve(sourceDir, importPath);

        // Try different extensions
        const extensions = ['.ts', '.tsx', '.js', '.jsx', '.d.ts', '/index.ts', '/index.js'];

        for (const ext of extensions) {
            const fullPath = absolutePath + ext;
            if (fs.existsSync(fullPath)) {
                return {
                    resolved: true,
                    path: fullPath,
                    type: 'relative'
                };
            }
        }

        return {
            resolved: false,
            path: importPath,
            type: 'relative',
            suggestions: this.findSimilarFiles(path.basename(importPath))
        };
    }

    private resolveAlias(importPath: string): Resolution {
        for (const [alias, target] of this.aliases.entries()) {
            if (importPath.startsWith(alias)) {
                const resolvedPath = importPath.replace(alias, target);
                return {
                    resolved: true,
                    path: resolvedPath,
                    type: 'alias',
                    alias: alias
                };
            }
        }

        return {
            resolved: false,
            path: importPath,
            type: 'alias',
            error: 'Alias not found in tsconfig.json'
        };
    }

    private findSimilarFiles(filename: string): string[] {
        // Use fuzzy search to find similar filenames
        const allFiles = this.getAllProjectFiles();
        const similarities = allFiles.map(file => ({
            file,
            score: this.calculateSimilarity(filename, path.basename(file))
        }));

        return similarities
            .filter(s => s.score > 0.7)
            .sort((a, b) => b.score - a.score)
            .slice(0, 5)
            .map(s => s.file);
    }

    private calculateSimilarity(str1: string, str2: string): number {
        const longer = str1.length > str2.length ? str1 : str2;
        const shorter = str1.length > str2.length ? str2 : str1;

        if (longer.length === 0) return 1.0;

        const editDistance = this.levenshteinDistance(longer, shorter);
        return (longer.length - editDistance) / longer.length;
    }
}
```

## Path Alias Configuration Detection

### TypeScript Configuration Parsing

```typescript
interface PathAliasConfig {
    baseUrl?: string;
    paths?: Record<string, string[]>;
}

function parseTsConfig(tsconfigPath: string): PathAliasConfig {
    const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf-8'));
    const compilerOptions = tsconfig.compilerOptions || {};

    return {
        baseUrl: compilerOptions.baseUrl,
        paths: compilerOptions.paths || {}
    };
}

function extractAliases(config: PathAliasConfig, projectRoot: string): Map<string, string> {
    const aliases = new Map<string, string>();
    const baseUrl = config.baseUrl || '.';

    for (const [alias, targets] of Object.entries(config.paths || {})) {
        // Remove wildcard
        const cleanAlias = alias.replace('/*', '');
        const cleanTarget = targets[0].replace('/*', '');

        const absoluteTarget = path.resolve(projectRoot, baseUrl, cleanTarget);
        aliases.set(cleanAlias, absoluteTarget);
    }

    return aliases;
}
```

### Common Path Alias Patterns

```json
{
  "compilerOptions": {
    "baseUrl": "./",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"],
      "@services/*": ["src/services/*"],
      "@models/*": ["src/models/*"],
      "@config/*": ["config/*"],
      "~/*": ["./"]
    }
  }
}
```

## Import Organization Strategies

### Grouping and Sorting Rules

```typescript
enum ImportGroup {
    EXTERNAL = 1,      // node_modules packages
    INTERNAL = 2,      // project imports with aliases
    RELATIVE_PARENT = 3,  // ../ imports
    RELATIVE_SIBLING = 4, // ./ imports
    SIDE_EFFECT = 5    // import './styles.css'
}

function organizeImports(imports: Import[]): Import[] {
    const grouped = new Map<ImportGroup, Import[]>();

    for (const imp of imports) {
        const group = categorizeImport(imp);
        if (!grouped.has(group)) {
            grouped.set(group, []);
        }
        grouped.get(group)!.push(imp);
    }

    // Sort within each group
    for (const [group, imps] of grouped.entries()) {
        imps.sort((a, b) => a.module.localeCompare(b.module));
    }

    // Combine all groups in order
    const organized: Import[] = [];
    for (let i = 1; i <= 5; i++) {
        if (grouped.has(i as ImportGroup)) {
            organized.push(...grouped.get(i as ImportGroup)!);
        }
    }

    return organized;
}

function categorizeImport(imp: Import): ImportGroup {
    if (imp.isSideEffect) {
        return ImportGroup.SIDE_EFFECT;
    }

    if (imp.module.startsWith('../')) {
        return ImportGroup.RELATIVE_PARENT;
    }

    if (imp.module.startsWith('./')) {
        return ImportGroup.RELATIVE_SIBLING;
    }

    if (imp.module.startsWith('@/') || imp.module.startsWith('~/')) {
        return ImportGroup.INTERNAL;
    }

    return ImportGroup.EXTERNAL;
}
```

### Import Formatting Preservation

```typescript
interface ImportStyle {
    quoteStyle: 'single' | 'double';
    semicolons: boolean;
    trailingComma: boolean;
    multilineThreshold: number;
}

function formatImport(imp: Import, style: ImportStyle): string {
    const quote = style.quoteStyle === 'single' ? "'" : '"';
    const semi = style.semicolons ? ';' : '';

    if (imp.isDefault) {
        return `import ${imp.defaultImport} from ${quote}${imp.module}${quote}${semi}`;
    }

    if (imp.isNamespace) {
        return `import * as ${imp.namespaceImport} from ${quote}${imp.module}${quote}${semi}`;
    }

    const namedImports = imp.namedImports || [];

    if (namedImports.length === 0) {
        // Side-effect import
        return `import ${quote}${imp.module}${quote}${semi}`;
    }

    if (namedImports.length === 1 || namedImports.join(', ').length < style.multilineThreshold) {
        // Single line
        return `import { ${namedImports.join(', ')} } from ${quote}${imp.module}${quote}${semi}`;
    }

    // Multi-line
    const comma = style.trailingComma ? ',' : '';
    const namedLines = namedImports.map(n => `  ${n}`).join(',\n');
    return `import {\n${namedLines}${comma}\n} from ${quote}${imp.module}${quote}${semi}`;
}
```

## Barrel Export Handling

### Barrel Export Detection

```typescript
function isBarrelExport(filePath: string): boolean {
    const basename = path.basename(filePath);
    return basename === 'index.ts' || basename === 'index.js';
}

function analyzeBarrelExport(barrelPath: string): BarrelAnalysis {
    const content = fs.readFileSync(barrelPath, 'utf-8');
    const exportStatements = content.match(/export .* from ['"].*['"]/g) || [];

    const reExports = exportStatements.map(stmt => {
        const match = stmt.match(/export (.*) from ['"](.*)['"]/);
        return {
            exports: match?.[1] || '',
            source: match?.[2] || ''
        };
    });

    return {
        isBarrel: true,
        reExportCount: reExports.length,
        reExports,
        shouldRefactor: reExports.length > 20  // Too many re-exports
    };
}
```

### Barrel Import Optimization

**Before** (imports from barrel):
```typescript
import { ComponentA, ComponentB, ComponentC } from './components';
```

**After** (direct imports):
```typescript
import { ComponentA } from './components/ComponentA';
import { ComponentB } from './components/ComponentB';
import { ComponentC } from './components/ComponentC';
```

**Why**: Improves tree-shaking and reduces bundle size

## Circular Dependency Detection

```typescript
class CircularDependencyDetector {
    private dependencyGraph: Map<string, Set<string>> = new Map();

    buildGraph(files: string[]): void {
        for (const file of files) {
            const imports = this.extractImports(file);
            this.dependencyGraph.set(file, new Set(imports));
        }
    }

    findCircularDependencies(): string[][] {
        const cycles: string[][] = [];
        const visited = new Set<string>();
        const recursionStack = new Set<string>();

        for (const node of this.dependencyGraph.keys()) {
            if (!visited.has(node)) {
                this.dfs(node, visited, recursionStack, [], cycles);
            }
        }

        return cycles;
    }

    private dfs(
        node: string,
        visited: Set<string>,
        recursionStack: Set<string>,
        path: string[],
        cycles: string[][]
    ): void {
        visited.add(node);
        recursionStack.add(node);
        path.push(node);

        const dependencies = this.dependencyGraph.get(node) || new Set();

        for (const dep of dependencies) {
            if (!visited.has(dep)) {
                this.dfs(dep, visited, recursionStack, path, cycles);
            } else if (recursionStack.has(dep)) {
                // Circular dependency found
                const cycleStart = path.indexOf(dep);
                const cycle = path.slice(cycleStart).concat([dep]);
                cycles.push(cycle);
            }
        }

        recursionStack.delete(node);
        path.pop();
    }
}
```

## Best Practices

### Import Statement Guidelines

1. **Prefer Absolute Imports** (with aliases):
   ```typescript
   // Good
   import { Button } from '@/components/Button';

   // Avoid (deep relative paths)
   import { Button } from '../../../components/Button';
   ```

2. **Group Imports Logically**:
   ```typescript
   // External packages
   import React from 'react';
   import { useQuery } from 'react-query';

   // Internal modules
   import { api } from '@/services/api';
   import { Button } from '@/components/Button';

   // Relative imports
   import { helper } from './utils';
   import './styles.css';
   ```

3. **Use Named Exports**:
   ```typescript
   // Good
   export { Button };
   import { Button } from './Button';

   // Avoid default exports for better refactoring
   export default Button;
   import Button from './Button';
   ```

4. **Avoid Circular Dependencies**:
   - Extract shared types to separate files
   - Use dependency injection
   - Refactor to eliminate coupling

5. **Tree-Shaking Friendly**:
   - Avoid barrel exports for large libraries
   - Use direct imports when possible
   - Mark side-effect-free packages in package.json
