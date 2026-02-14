---
version: "2.2.0"
description: Unified project and component scaffolding for TypeScript, Python, React, and Julia
argument-hint: <language> <name> [options]
category: engineering-suite
execution-modes:
  quick: "5-15 minutes"
  standard: "20-45 minutes"
  deep: "1-2 hours"
color: green
allowed-tools: [Write, Read, Task, Bash, Glob, Grep, Bash(uv:*)]
external-docs:
  - project-scaffolding-guide.md
  - component-patterns-library.md
  - testing-strategies.md
tags: [scaffolding, project-setup, boilerplate, templates]
---

# Project Scaffolding

$ARGUMENTS

## Languages

| Language | Description | Subtypes |
|----------|-------------|----------|
| `typescript` | TypeScript/Node.js projects | nextjs, react-vite, nodejs-api, library, cli |
| `python` | Python projects | fastapi, django, library, cli |
| `component` | React/React Native components | web, native, universal |
| `julia` | Julia packages | package, sciml |

**Examples:**
```bash
/scaffold typescript my-app --type nextjs
/scaffold python my-api --type fastapi
/scaffold component ProductCard --platform web --styling tailwind
/scaffold julia MyPackage
```

## Options

**All languages:**
- `--mode <depth>`: quick (minimal), standard (production-ready), deep (enterprise)

**TypeScript:**
- `--type <project>`: nextjs, react-vite, nodejs-api, library, cli

**Python:**
- `--type <project>`: fastapi, django, library, cli

**Component:**
- `--platform <target>`: web, native, universal
- `--styling <approach>`: css-modules, styled-components, tailwind
- `--tests`: Include test files
- `--storybook`: Generate Storybook stories
- `--accessibility`: Add ARIA and a11y features

---

## TypeScript Projects

### Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| Quick | 10-15 min | Basic config, essential deps |
| Standard | 20-30 min | Full config, tooling, testing |
| Deep | 40-60 min | + CI/CD, Docker, security |

### Project Types

| Type | Use Case | Key Features |
|------|----------|--------------|
| Next.js | Full-stack React with SSR | App Router, API routes, Server Components |
| React + Vite | Client-side SPAs | Fast HMR, optimized builds, Vitest |
| Node.js API | Backend services | Express/Fastify, middleware, auth |
| Library | Reusable packages | Tree-shakeable, dual format |
| CLI | Command-line tools | Commander.js, interactive prompts |

### Core Steps

1. **Initialize:** `mkdir project && cd project && pnpm init && git init`
2. **Project setup:** Based on type (Next.js, Vite, etc.)
3. **TypeScript config:** strict mode, paths, incremental
4. **Tooling:** ESLint, Prettier, Vitest, Husky
5. **Structure:** src/, tests/, config files

### Essential Files

| File | Purpose |
|------|---------|
| package.json | Scripts: dev, build, test, lint, type-check |
| tsconfig.json | TypeScript settings |
| .eslintrc | Linting rules |
| .prettierrc | Formatting |
| .env.example | Environment template |
| README.md | Setup instructions |

---

## Python Projects

### Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| Quick | 1-2h | Minimal structure (~15 files) |
| Standard | 3-6h | Complete FastAPI/Django (~50 files) |
| Deep | 1-2d | Multi-service + K8s (~100 files) |

### Project Types

| Type | Use Case |
|------|----------|
| FastAPI | REST APIs, microservices, async applications |
| Django | Full-stack web apps, admin panels, ORM-heavy |
| Library | Reusable packages, utilities |
| CLI | Command-line tools, automation scripts |

### Core Steps

1. **Initialize:**
   ```bash
   uv init <project-name>
   cd <project-name>
   git init
   uv venv && source .venv/bin/activate
   ```

2. **Configure pyproject.toml:**
   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py312"

   [tool.ruff.lint]
   select = ["E", "F", "I", "N", "W", "UP"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   ```

3. **Structure based on type:** FastAPI routes, Django apps, etc.

### Success Criteria

- Project initializes with uv
- Virtual environment activates
- Tests pass (`pytest`)
- Type checking passes (`mypy`)
- Linting passes (`ruff check`)

---

## React Components

### Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| Quick | 5-10 min | ComponentSpec interface and recommendations |
| Standard | 15-30 min | Complete component with TypeScript/styling |
| Deep | 30-60 min | Full scaffold with tests, Storybook, a11y |

### Component Classification

| Classification | Decision Rule |
|----------------|---------------|
| **Platform** | Web APIs → web; Native features → native; Shared → universal |
| **Type** | Complex state → page; Input validation → form; Wraps children → layout |
| **Styling** | React Native → StyleSheet; Theming → styled-components; Type safety → CSS Modules |

### Generated Files

| File | Content |
|------|---------|
| `{Name}.tsx` | Functional component with TypeScript props |
| `{Name}.types.ts` | Props interface with JSDoc |
| `index.ts` | Barrel exports |
| `{Name}.test.tsx` | React Testing Library tests (--tests) |
| `{Name}.stories.tsx` | Storybook stories (--storybook) |

### Styling Options

| Approach | Implementation |
|----------|----------------|
| CSS Modules | `.module.css` with CSS variables |
| styled-components | Theme props, dynamic styling |
| Tailwind CSS | Utility classes, responsive modifiers |
| React Native | StyleSheet.create, Platform.select |

### Validation

```bash
npx tsc --noEmit          # TypeScript check
npm test {Name}.test      # Run tests
npm run storybook         # Verify stories
npm run lint              # Lint check
```

---

## Julia Packages

### Validation

- PascalCase name ending with `.jl`
- Directory doesn't exist
- Author info available

### Generation

```julia
using PkgTemplates
t = Template(; user="username", julia=v"1.6", plugins=[...])
t("PackageName")
```

### Structure

```
PackageName/
├── .github/workflows/
├── docs/
├── src/PackageName.jl
├── test/runtests.jl
├── Project.toml
└── LICENSE
```

### Module Pattern

**Single file** (<500 lines):
```julia
module PackageName
export useful_function
useful_function(x) = x + 1
end
```

**Multi-file** (500-5000 lines):
```julia
module PackageName
export Type1, function1
include("types.jl")
include("functions.jl")
end
```

### Post-Creation

1. `cd PackageName`
2. `git remote add origin git@github.com:user/PackageName.jl.git`
3. `Pkg.add("Dependencies")`
4. Add `[compat]` entries
5. `git push -u origin main`

### CI/CD Setup (--ci flag)

Generate GitHub Actions workflows for Julia packages:

**Workflow Components:**
| Component | Purpose |
|-----------|---------|
| actions/checkout@v4 | Checkout code |
| julia-actions/setup-julia@v1 | Install Julia |
| julia-actions/cache@v1 | Cache packages |
| julia-actions/julia-buildpkg@v1 | Build |
| julia-actions/julia-runtest@v1 | Test |

**Platform Matrix:**
- Linux (ubuntu-latest): Always included
- macOS: For platform-specific code
- Windows: For Windows compatibility

**Julia Versions:**
- LTS (1.6): Long-term support
- Latest (1): Current stable
- Nightly: Future compatibility testing

**Optional Features:**
- `--coverage`: Codecov/Coveralls integration
- `--docs`: Documenter.jl workflow
- `--compat-helper`: Dependency update automation
- `--tagbot`: Release automation
- `--formatter`: JuliaFormatter checks
- `--quality`: Aqua.jl and JET.jl analysis

**Coverage Setup:**
1. Add julia-processcoverage action
2. Add codecov-action with lcov.info
3. Create .codecov.yml (target 80%)
4. Add badge to README

**Documentation Workflow:**
- Trigger on main, tags, PRs
- Install docs dependencies
- Deploy with DOCUMENTER_KEY secret

---

## Success Criteria

**All Projects:**
- Directory structure complete
- Configuration files in place
- Entry point functional
- README with setup instructions

**Standard+ Mode:**
- Testing framework configured
- Linting/formatting setup
- Git hooks configured

**Deep Mode:**
- CI/CD pipelines
- Docker support
- Security hardening
- Comprehensive documentation
