---
name: package-development-workflow
version: "2.1.0"
maturity: "5-Expert"
specialization: Julia Packages
description: Create Julia packages following community standards with proper structure, exports, and PkgTemplates.jl. Use when creating new packages or organizing source code.
---

# Julia Package Development

Create packages following ecosystem standards.

---

## Package Structure

```
MyPackage/
├── Project.toml
├── src/MyPackage.jl
├── test/runtests.jl
├── docs/make.jl
├── README.md
└── LICENSE
```

---

## Module Pattern

```julia
module MyPackage

export public_function, PublicType

function public_function(x)
    internal_helper(x)
end

function internal_helper(x)
    # Not exported
end

end # module
```

---

## PkgTemplates.jl

```julia
using PkgTemplates

tpl = Template(;
    user="username",
    plugins=[
        Git(),
        License(name="MIT"),
        GitHubActions(),
        Documenter{GitHubActions}()
    ]
)

tpl("MyPackage")
```

---

## Checklist

- [ ] Project.toml with deps and compat
- [ ] Module exports defined
- [ ] Test suite in test/
- [ ] Documentation setup
- [ ] LICENSE and README

## Parallelization

- Use `TestEnv.activate(); include("test/runtests.jl")` with `julia -t auto`
- Configure CI with `julia-version: [...]` matrix for parallel testing
- Use `Distributed` for parallel test execution where appropriate

---

**Version**: 1.0.5
