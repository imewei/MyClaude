# Julia Package Scaffolding

Comprehensive guide to creating well-structured Julia packages using PkgTemplates.jl and modern best practices.

## Table of Contents

- [PkgTemplates.jl](#pkgtemplatesjl)
- [Package Structure](#package-structure)
- [Configuration Options](#configuration-options)
- [Post-Creation Setup](#post-creation-setup)
- [Best Practices](#best-practices)

---

## PkgTemplates.jl

### Installation

```julia
using Pkg
Pkg.add("PkgTemplates")
```

### Basic Usage

```julia
using PkgTemplates

# Create template with defaults
t = Template()

# Generate package
t("MyPackage")
```

### Recommended Template

```julia
using PkgTemplates

t = Template(;
    user="yourusername",
    authors=["Your Name <your.email@example.com>"],
    julia=v"1.6",  # Minimum Julia version
    plugins=[
        Git(; manifest=false, ssh=true),
        GitHubActions(;
            linux=true,
            osx=true,
            windows=true,
            x64=true,
            coverage=true
        ),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
        License(; name="MIT"),
        ProjectFile(; version=v"0.1.0"),
    ],
)

t("MyAwesomePackage")
```

---

## Package Structure

### Generated Directory Layout

```
MyPackage/
├── .git/                   # Git repository
├── .github/
│   └── workflows/
│       ├── CI.yml          # Continuous integration
│       ├── CompatHelper.yml # Dependency updates
│       ├── TagBot.yml      # Release automation
│       └── Documentation.yml # Docs deployment
├── docs/
│   ├── make.jl             # Documentation builder
│   ├── Project.toml        # Docs dependencies
│   └── src/
│       └── index.md        # Documentation content
├── src/
│   └── MyPackage.jl        # Main module file
├── test/
│   └── runtests.jl         # Test suite
├── .gitignore              # Git ignore patterns
├── LICENSE                 # License file
├── Manifest.toml           # Exact dependency versions
├── Project.toml            # Package metadata
└── README.md               # Package documentation
```

### Core Files

#### `Project.toml`

```toml
name = "MyPackage"
uuid = "12345678-1234-1234-1234-123456789012"
authors = ["Your Name <email@example.com>"]
version = "0.1.0"

[deps]
# List dependencies here

[compat]
julia = "1.6"
# Add version constraints for dependencies
```

#### `src/MyPackage.jl`

```julia
module MyPackage

# Export public API
export my_function, MyType

# Include source files
include("types.jl")
include("functions.jl")

end # module
```

#### `test/runtests.jl`

```julia
using MyPackage
using Test

@testset "MyPackage.jl" begin
    @testset "Feature 1" begin
        @test my_function(1) == 2
    end

    @testset "Feature 2" begin
        # More tests
    end
end
```

---

## Configuration Options

### Plugin Options

#### Git Plugin

```julia
Git(;
    manifest=false,  # Don't track Manifest.toml
    ssh=true,        # Use SSH URLs
    jl=true,         # Add .jl extension
    ignore=[".DS_Store", "*.swp"]  # Additional ignores
)
```

#### GitHubActions Plugin

```julia
GitHubActions(;
    linux=true,       # Test on Linux
    osx=true,         # Test on macOS
    windows=true,     # Test on Windows
    x64=true,         # Test x64 architecture
    x86=false,        # Skip x86
    coverage=true,    # Enable coverage
    extra_versions=["nightly"]  # Test nightly Julia
)
```

#### Codecov Plugin

```julia
Codecov(;
    file=".codecov.yml",  # Config file path
    config_file="""
    coverage:
      status:
        project:
          default:
            target: 80%
    """
)
```

#### Documenter Plugin

```julia
Documenter{GitHubActions}(;
    make_jl="docs/make.jl",
    assets=[],
    logo="",
    canonical_url=nothing
)
```

#### License Plugin

```julia
License(;
    name="MIT",  # MIT, Apache, GPL, BSD, etc.
    path="LICENSE",
    destination="LICENSE"
)
```

### Template Customization

#### Minimal Package (Quick Start)

```julia
t = Template(;
    user="yourusername",
    plugins=[
        Git(; manifest=false),
        License(; name="MIT"),
    ]
)
```

**Use when**: Experimenting, internal packages

#### Standard Package (Recommended)

```julia
t = Template(;
    user="yourusername",
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

#### Production Package (Full-Featured)

```julia
t = Template(;
    user="yourusername",
    authors=["Name <email>"],
    julia=v"1.6",
    plugins=[
        Git(; manifest=false, ssh=true),
        GitHubActions(;
            linux=true,
            osx=true,
            windows=true,
            coverage=true,
            extra_versions=["nightly"]
        ),
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

---

## Post-Creation Setup

### Step 1: Initialize Git Repository

```bash
cd MyPackage
git add .
git commit -m "Initial commit"
```

### Step 2: Create GitHub Repository

```bash
# Create repo on GitHub, then:
git remote add origin git@github.com:username/MyPackage.jl.git
git push -u origin main
```

### Step 3: Set Up Documentation Keys

```julia
using DocumenterTools
DocumenterTools.genkeys(user="username", repo="MyPackage.jl")
```

Follow instructions to add keys to GitHub.

### Step 4: Add Dependencies

```julia
using Pkg
Pkg.activate(".")
Pkg.add("SomePackage")
```

### Step 5: Set Up [compat] Entries

Edit `Project.toml`:

```toml
[compat]
julia = "1.6"
SomePackage = "1.2"
```

**Use**: `CompatHelper.jl` will auto-update these.

### Step 6: Write First Test

```julia
# test/runtests.jl
using MyPackage
using Test

@testset "MyPackage.jl" begin
    @test 1 + 1 == 2  # Replace with real tests
end
```

### Step 7: Run Tests Locally

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Step 8: Push and Verify CI

```bash
git add .
git commit -m "Add initial functionality"
git push
```

Check GitHub Actions tab for CI status.

---

## Best Practices

### Package Naming

**Do** ✅:
- `MyPackage.jl` (PascalCase + .jl extension)
- `DataStructures.jl` (descriptive, clear purpose)
- `HTTP.jl` (acronyms allowed)

**Don't** ❌:
- `my_package` (snake_case)
- `MyPackage` (missing .jl)
- `MP.jl` (unclear abbreviation)
- `JuliaMyPackage.jl` (redundant "Julia")

### Module Structure

#### Single-File Module (Small packages)

```julia
# src/MyPackage.jl
module MyPackage

export useful_function

function useful_function(x)
    return x + 1
end

end
```

#### Multi-File Module (Larger packages)

```julia
# src/MyPackage.jl
module MyPackage

export Type1, Type2, function1, function2

include("types.jl")
include("functions.jl")
include("algorithms.jl")

end

# src/types.jl
struct Type1
    field1::Int
    field2::Float64
end

# src/functions.jl
function1(x) = x + 1
```

#### Submodules (Complex packages)

```julia
# src/MyPackage.jl
module MyPackage

include("Core.jl")
include("Algorithms.jl")
include("Utils.jl")

using .Core
using .Algorithms
using .Utils

end

# src/Core.jl
module Core
export Type1, Type2
# ...
end
```

### API Design

#### Exports

```julia
# Export only public API
export MyType, my_function

# Keep internal functions unexported
function _internal_helper(x)
    # ...
end
```

#### Type Hierarchy

```julia
# Abstract types for extensibility
abstract type AbstractWidget end

struct ConcreteWidget <: AbstractWidget
    data::Vector{Float64}
end
```

#### Function Names

- Lowercase with underscores: `my_function`
- Predicates end with `?`: `is_valid?`
- Mutating functions end with `!`: `update!`
- Avoid abbreviations: `calculate` not `calc`

### Testing Structure

#### Organized Test Sets

```julia
@testset "MyPackage.jl" begin
    @testset "Type Construction" begin
        @test MyType(1, 2.0) isa MyType
    end

    @testset "Basic Operations" begin
        x = MyType(1, 2.0)
        @test operation(x) == expected_result
    end

    @testset "Edge Cases" begin
        @test_throws ArgumentError invalid_input()
    end
end
```

#### Test Files

For large packages, split tests:

```
test/
├── runtests.jl
├── types_test.jl
├── functions_test.jl
└── algorithms_test.jl
```

```julia
# test/runtests.jl
using Test

@testset "MyPackage.jl" begin
    include("types_test.jl")
    include("functions_test.jl")
    include("algorithms_test.jl")
end
```

### Documentation

#### Docstrings

```julia
"""
    my_function(x::Int, y::Float64) -> Float64

Compute the magic number from `x` and `y`.

# Arguments
- `x::Int`: The integer input.
- `y::Float64`: The floating-point multiplier.

# Returns
- `Float64`: The computed result.

# Examples
```jldoctest
julia> my_function(2, 3.5)
7.5
```

# See Also
- [`related_function`](@ref)
"""
function my_function(x::Int, y::Float64)
    return x * y + 0.5
end
```

#### README.md Template

```markdown
# MyPackage.jl

[![CI](https://github.com/username/MyPackage.jl/workflows/CI/badge.svg)](https://github.com/username/MyPackage.jl/actions)
[![Coverage](https://codecov.io/gh/username/MyPackage.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/username/MyPackage.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://username.github.io/MyPackage.jl/stable)

Brief description of what your package does.

## Installation

```julia
using Pkg
Pkg.add("MyPackage")
```

## Quick Start

```julia
using MyPackage

result = my_function(42)
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Documentation

See the [documentation](https://username.github.io/MyPackage.jl/stable) for detailed usage.

## Contributing

Contributions are welcome! Please open an issue or PR.

## License

MIT License
```

---

## Common Pitfalls

### Pitfall 1: Not Using `const` for Globals

❌ **Bad**:
```julia
MODULE_VERSION = "1.0.0"  # Type-unstable
```

✅ **Good**:
```julia
const MODULE_VERSION = "1.0.0"  # Type-stable
```

### Pitfall 2: Exporting Too Much

❌ **Bad**:
```julia
export everything, including_helpers, _internal_stuff
```

✅ **Good**:
```julia
export only_public_api

# Internal functions stay unexported
_helper_function(x) = ...
```

### Pitfall 3: Missing [compat] Entries

❌ **Bad**:
```toml
[deps]
SomePackage = "uuid"
# No compat entry!
```

✅ **Good**:
```toml
[deps]
SomePackage = "uuid"

[compat]
julia = "1.6"
SomePackage = "1.2, 2"
```

### Pitfall 4: Not Testing Edge Cases

❌ **Bad**:
```julia
@test my_function(5) == 10
```

✅ **Good**:
```julia
@testset "my_function" begin
    @test my_function(5) == 10
    @test my_function(0) == 0
    @test my_function(-5) == -10
    @test_throws ArgumentError my_function(NaN)
end
```

---

## Package Development Workflow

### Development Cycle

1. **Activate package environment**:
   ```julia
   using Pkg
   Pkg.activate(".")
   ```

2. **Make changes** to `src/` files

3. **Load changes**:
   ```julia
   using Revise  # Auto-reloads changes
   using MyPackage
   ```

4. **Test manually**:
   ```julia
   my_function(test_input)
   ```

5. **Run tests**:
   ```julia
   Pkg.test()
   ```

6. **Commit** and **push**

### Using Revise.jl

```julia
# In startup.jl (~/.julia/config/startup.jl)
try
    using Revise
catch e
    @warn "Error initializing Revise" exception=(e, catch_backtrace())
end
```

**Benefit**: Automatic code reloading during development.

---

## Registration

### Registering Your Package

1. **Prepare package**:
   - All tests passing
   - Documentation complete
   - [compat] entries set
   - Version tagged

2. **Use Registrator.jl**:
   - Install: https://github.com/JuliaRegistries/Registrator.jl
   - Comment `@JuliaRegistrator register` on a commit or release

3. **Wait for approval**:
   - AutoMerge (if tests pass)
   - Manual review (first-time packages)

4. **Available in registry**:
   ```julia
   Pkg.add("YourPackage")
   ```

---

## Checklist

### Package Creation

- [ ] Template configured with desired plugins
- [ ] Package generated with `t("PackageName")`
- [ ] Git repository initialized
- [ ] GitHub repository created and linked
- [ ] Documentation keys generated and added

### Package Structure

- [ ] `Project.toml` has correct metadata
- [ ] [compat] entries added for all dependencies
- [ ] Module structure organized (single/multi-file)
- [ ] Public API clearly defined with exports
- [ ] Tests organized in `@testset` blocks

### Quality

- [ ] All functions have docstrings
- [ ] README.md describes package purpose
- [ ] CI workflows passing
- [ ] Code coverage > 80%
- [ ] No type instabilities in hot paths

### Release

- [ ] Version bumped in `Project.toml`
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] Documentation deployed
- [ ] Ready for registration

---

**Version**: 1.0.3
**Last Updated**: 2025-11-07
**Plugin**: julia-development
