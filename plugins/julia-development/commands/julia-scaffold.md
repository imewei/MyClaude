---
version: "1.0.3"
category: "julia-development"
command: "/julia-scaffold"
description: Bootstrap new Julia package with proper structure following PkgTemplates.jl conventions, including testing infrastructure, documentation framework, and CI/CD
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<PackageName>"
color: cyan
execution_modes:
  quick: "5-10 minutes - Generate minimal package structure for rapid development"
  standard: "15-20 minutes - Generate complete package with CI, docs, and testing"
  comprehensive: "25-35 minutes - Generate production-ready package with full automation and quality tooling"
agents:
  primary:
    - julia-developer
  conditional:
    - agent: julia-pro
      trigger: pattern "performance|optimization"
  orchestrated: false
---

# Bootstrap New Julia Package

Create a well-structured Julia package following modern best practices with PkgTemplates.jl, including proper Project.toml, testing infrastructure, documentation framework, and CI/CD workflows.

## Quick Reference

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| **Complete Guide** | [package-scaffolding.md](../docs/package-scaffolding.md) | ~450 |
| **PkgTemplates** | [package-scaffolding.md#pkgtemplatesjl](../docs/package-scaffolding.md#pkgtemplatesjl) | ~150 |
| **Configuration** | [package-scaffolding.md#configuration-options](../docs/package-scaffolding.md#configuration-options) | ~100 |
| **Post-Creation** | [package-scaffolding.md#post-creation-setup](../docs/package-scaffolding.md#post-creation-setup) | ~80 |
| **Best Practices** | [package-scaffolding.md#best-practices](../docs/package-scaffolding.md#best-practices) | ~120 |

**Total External Documentation**: ~450 lines of templates and guidance

## Requirements

$ARGUMENTS

## Core Workflow

### Phase 1: Package Configuration

**Gather package details**:

1. **Package name**: From user argument (must end with `.jl`)
2. **Author information**: Name and email
3. **License**: MIT, Apache, GPL, BSD (default: MIT)
4. **Minimum Julia version**: 1.6 (LTS) or 1.0

**Validate**:
- Package name follows PascalCase convention
- Name ends with `.jl`
- Not already exists in current directory

### Phase 2: Template Selection

**Choose template tier** based on execution mode:

**Quick Mode** - Minimal Package:
```julia
Template(;
    user="username",
    plugins=[
        Git(; manifest=false),
        License(; name="MIT"),
    ]
)
```
**Use when**: Experimenting, internal packages

**Standard Mode** - Standard Package (DEFAULT):
```julia
Template(;
    user="username",
    julia=v"1.6",
    plugins=[
        Git(; manifest=false, ssh=true),
        GitHubActions(; coverage=true),
        Codecov(),
        License(; name="MIT"),
    ]
)
```
**Use when**: Public packages, collaborative projects

**Comprehensive Mode** - Production Package:
```julia
Template(;
    user="username",
    julia=v"1.6",
    plugins=[
        Git(; manifest=false, ssh=true),
        GitHubActions(; linux=true, osx=true, windows=true, coverage=true),
        Codecov(),
        Documenter{GitHubActions}(),
        CompatHelper(),
        TagBot(),
        License(; name="MIT"),
        Develop(),
    ]
)
```
**Use when**: Ecosystem packages, long-term projects

**Templates**: [package-scaffolding.md#configuration-options](../docs/package-scaffolding.md#configuration-options)

### Phase 3: Package Generation

**Generate package** using PkgTemplates:

```julia
using PkgTemplates

# Create configured template
t = Template(; [selected configuration])

# Generate package
t("PackageName")
```

**Generated structure**:
```
PackageName/
├── .git/                   # Git repository
├── .github/workflows/      # CI/CD (if included)
│   ├── CI.yml
│   ├── CompatHelper.yml
│   ├── TagBot.yml
│   └── Documentation.yml
├── docs/                   # Documentation (if included)
│   ├── make.jl
│   ├── Project.toml
│   └── src/index.md
├── src/
│   └── PackageName.jl      # Main module
├── test/
│   └── runtests.jl         # Test suite
├── .gitignore
├── LICENSE
├── Manifest.toml           # Locked dependencies (gitignored)
├── Project.toml            # Package metadata
└── README.md
```

**Details**: [package-scaffolding.md#package-structure](../docs/package-scaffolding.md#package-structure)

### Phase 4: Initial Module Setup

**Create initial module structure**:

**Simple package**:
```julia
# src/PackageName.jl
module PackageName

export useful_function

"""
    useful_function(x)

Brief description of what this function does.
"""
function useful_function(x)
    return x + 1
end

end # module
```

**Multi-file package**:
```julia
# src/PackageName.jl
module PackageName

export Type1, Type2, function1, function2

include("types.jl")
include("functions.jl")

end # module
```

**Guidance**: [package-scaffolding.md#module-structure](../docs/package-scaffolding.md#module-structure)

### Phase 5: Test Setup

**Create initial test structure**:

```julia
# test/runtests.jl
using PackageName
using Test

@testset "PackageName.jl" begin
    @testset "Feature 1" begin
        @test useful_function(1) == 2
    end

    # Add more test sets as needed
end
```

**Run tests**:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

**Patterns**: [package-scaffolding.md#testing-structure](../docs/package-scaffolding.md#testing-structure)

### Phase 6: Post-Creation Setup

**Guide user through setup steps**:

1. **Navigate to package**:
   ```bash
   cd PackageName
   ```

2. **Initialize Git** (if not auto-done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. **Create GitHub repository** and link:
   ```bash
   # Create repo on GitHub, then:
   git remote add origin git@github.com:username/PackageName.jl.git
   git push -u origin main
   ```

4. **Set up documentation keys** (if Documenter included):
   ```bash
   julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="username", repo="PackageName.jl")'
   ```

5. **Add dependencies**:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.add("DependencyName")
   ```

6. **Update [compat] entries** in Project.toml:
   ```toml
   [compat]
   julia = "1.6"
   DependencyName = "1.2"
   ```

**Checklist**: [package-scaffolding.md#post-creation-setup](../docs/package-scaffolding.md#post-creation-setup)

### Phase 7: Next Steps Guidance

**Provide development workflow**:

1. **Activate environment**:
   ```julia
   using Pkg; Pkg.activate(".")
   ```

2. **Use Revise.jl** for development:
   ```julia
   using Revise
   using PackageName
   # Code auto-reloads on changes!
   ```

3. **Run tests frequently**:
   ```julia
   Pkg.test()
   ```

4. **Build documentation locally** (if applicable):
   ```bash
   julia --project=docs docs/make.jl
   ```

5. **Push changes** and check CI

## Mode-Specific Execution

### Quick Mode (5-10 minutes)

**Generated**:
- Basic package structure
- Minimal Project.toml
- Simple src/ and test/ structure
- Git initialization
- MIT license

**Skip**: CI/CD, documentation, automation tools

**Output**: Functional package for rapid prototyping

### Standard Mode (15-20 minutes) - DEFAULT

**Generated**:
- Complete package structure
- GitHub Actions CI (Linux + latest Julia)
- Codecov integration
- Basic README with badges
- Proper .gitignore
- License file

**Include**: Post-creation checklist, development workflow

**Output**: Production-ready package for collaboration

### Comprehensive Mode (25-35 minutes)

**Generated**:
- All from standard mode
- Cross-platform CI (Linux/macOS/Windows)
- Documentation framework with Documenter.jl
- CompatHelper for dependency updates
- TagBot for releases
- Development best practices guide
- Registration instructions

**Include**: Complete setup guide, best practices, advanced patterns

**Output**: Enterprise-grade package with full automation

## Package Types & Recommendations

### Research/Academic Package
**Mode**: Standard
**Plugins**: Git, GitHubActions, Codecov, Documenter, License
**Why**: Documentation important for citations, moderate automation

### Open Source Library
**Mode**: Comprehensive
**Plugins**: All (Git, CI, coverage, docs, CompatHelper, TagBot)
**Why**: Professional quality, broad compatibility, automation

### Internal Tool
**Mode**: Quick
**Plugins**: Git, License only
**Why**: Fast setup, minimal overhead

### Performance-Critical Package
**Mode**: Comprehensive
**Plugins**: All + quality checks
**Why**: Need benchmarking, quality validation, optimization tracking

## Common Patterns

### Package Naming
✅ **Good**: `DataStructures.jl`, `HTTP.jl`, `MyPackage.jl`
❌ **Bad**: `my_package` (no .jl), `MP.jl` (unclear), `JuliaMyPackage.jl` (redundant)

**Guide**: [package-scaffolding.md#package-naming](../docs/package-scaffolding.md#package-naming)

### Module Organization
- **Single-file**: < 500 lines
- **Multi-file**: 500-5000 lines
- **Submodules**: > 5000 lines

**Patterns**: [package-scaffolding.md#module-structure](../docs/package-scaffolding.md#module-structure)

### API Design
- Export only public API
- Use `!` for mutating functions
- Use `?` for predicates
- Avoid abbreviations

**Guide**: [package-scaffolding.md#api-design](../docs/package-scaffolding.md#api-design)

## Success Criteria

✅ Package generated with proper structure
✅ Project.toml has correct metadata
✅ Initial module and tests created
✅ Git repository initialized
✅ CI/CD workflows configured (if included)
✅ Documentation framework set up (if included)
✅ README.md with badges
✅ Post-creation checklist provided
✅ Development workflow guidance given
✅ External documentation referenced

## Agent Integration

- **julia-developer**: Primary agent for package structure and best practices
- **julia-pro**: Triggered for performance-oriented package configuration

## Post-Generation

After package is generated:

1. **Follow setup checklist** from Phase 6
2. **Add initial functionality** to src/
3. **Write tests** for new functions
4. **Run tests locally** before pushing
5. **Push to GitHub** and verify CI
6. **Iterate**: Develop → Test → Commit → Push

**Development Guide**: [package-scaffolding.md#package-development-workflow](../docs/package-scaffolding.md#package-development-workflow)

**See Also**:
- `/julia-package-ci` - Add/update CI workflows
- `/sciml-setup` - Add SciML functionality
- [package-scaffolding.md](../docs/package-scaffolding.md) - Complete package guide

---

Focus on **proper structure from the start**, **appropriate automation for package type**, and **clear development workflow**.
