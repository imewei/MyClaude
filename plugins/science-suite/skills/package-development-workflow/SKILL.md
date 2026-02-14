---
name: package-development-workflow
version: "2.2.0"
maturity: "5-Expert"
specialization: Julia Packages
description: Create Julia packages following community standards with proper structure, exports, and PkgTemplates.jl. Use when creating new packages or organizing source code.
---

# Julia Package Development

Create packages following ecosystem standards.

## Expert Agent

For package structure, CI/CD configuration, and documentation workflows, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia package development and testing.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: PkgTemplates.jl configuration, GitHub Actions setup, and comprehensive test suite generation.

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

---

**Version**: 1.0.6
