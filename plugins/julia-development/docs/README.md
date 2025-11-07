# Julia Development Plugin - Documentation Hub

Comprehensive external documentation for the julia-development plugin commands.

## Quick Navigation

| Documentation | Description | Lines | Related Command |
|--------------|-------------|-------|-----------------|
| **[SciML Templates](sciml-templates.md)** | Complete code templates for ODE, PDE, SDE, and optimization problems | ~550 | `/sciml-setup` |
| **[Optimization Patterns](optimization-patterns.md)** | Type stability, allocation reduction, parallelization strategies | ~400 | `/julia-optimize` |
| **[Profiling Guide](profiling-guide.md)** | BenchmarkTools, Profile.jl, type stability analysis | ~350 | `/julia-optimize` |
| **[CI/CD Workflows](ci-cd-workflows.md)** | GitHub Actions, test matrices, coverage, automation | ~400 | `/julia-package-ci` |
| **[Package Scaffolding](package-scaffolding.md)** | PkgTemplates.jl, structure, configuration, best practices | ~450 | `/julia-scaffold` |

**Total Documentation**: ~2,150 lines of detailed guides, examples, and best practices

---

## Documentation by Command

### /sciml-setup Command

**External Docs**: [sciml-templates.md](sciml-templates.md)

**Contents**:
- Complete ODE templates (direct API & symbolic)
- PDE templates (Method of Lines)
- SDE templates (stochastic dynamics)
- Optimization templates (parameter estimation)
- Problem type detection keywords
- Solver selection guide
- Best practices & common pitfalls

**Use when**: Need detailed template code, solver guidance, or SciML examples

---

### /julia-optimize Command

**External Docs**:
- [optimization-patterns.md](optimization-patterns.md)
- [profiling-guide.md](profiling-guide.md)

**Contents**:

#### Optimization Patterns
- Type stability patterns & fixes
- Allocation reduction techniques
- Parallelization strategies (threads, distributed, GPU)
- Memory optimization (StaticArrays, layout, SoA vs AoS)
- Algorithm improvements

#### Profiling Guide
- BenchmarkTools.jl complete guide
- Profile.jl & ProfileView.jl (flame graphs)
- @code_warntype interpretation
- Memory profiling techniques
- Benchmark interpretation guide

**Use when**: Optimizing Julia code, profiling performance, fixing type instabilities

---

### /julia-package-ci Command

**External Docs**: [ci-cd-workflows.md](ci-cd-workflows.md)

**Contents**:
- Complete GitHub Actions workflow templates
- Test matrix configurations (platform, version, arch)
- Coverage reporting (Codecov, Coveralls)
- Documentation deployment with Documenter.jl
- Automation tools (CompatHelper, TagBot, JuliaFormatter)
- Security scanning
- Performance benchmarking
- Workflow optimization techniques

**Use when**: Setting up CI/CD, configuring test matrices, automating workflows

---

### /julia-scaffold Command

**External Docs**: [package-scaffolding.md](package-scaffolding.md)

**Contents**:
- PkgTemplates.jl complete guide
- Package structure best practices
- Configuration options (all plugins)
- Post-creation setup (step-by-step)
- Module organization patterns
- API design guidelines
- Testing structure
- Documentation templates
- Registration process

**Use when**: Creating new packages, configuring templates, organizing package structure

---

## Quick Start Guides

### Performance Optimization

1. **Profile first**: `@benchmark my_function($args)`
2. **Check types**: `@code_warntype my_function(args)`
3. **Find bottlenecks**: `@profile` + `ProfileView.view()`
4. **Apply patterns**: See [optimization-patterns.md](optimization-patterns.md)
5. **Verify improvement**: Re-benchmark

**Docs**: [optimization-patterns.md](optimization-patterns.md), [profiling-guide.md](profiling-guide.md)

### SciML Project Setup

1. **Identify problem type**: ODE, PDE, SDE, or optimization
2. **Choose template**: See [sciml-templates.md](sciml-templates.md)
3. **Customize**: Fill in TODOs, adjust parameters
4. **Select solver**: Use solver selection guide
5. **Add features**: Callbacks, ensemble, sensitivity

**Docs**: [sciml-templates.md](sciml-templates.md)

### Package Creation

1. **Configure template**: Choose plugins based on needs
2. **Generate package**: `t("PackageName")`
3. **Set up Git**: Initialize and push to GitHub
4. **Add documentation keys**: For automatic docs deployment
5. **Develop**: Write code, tests, docs
6. **Register**: Use Registrator.jl when ready

**Docs**: [package-scaffolding.md](package-scaffolding.md)

### CI/CD Setup

1. **Choose workflow**: Minimal, standard, or full-featured
2. **Configure matrix**: Platform + Julia versions
3. **Enable coverage**: Codecov or Coveralls
4. **Set up docs**: Documenter.jl with GitHubActions
5. **Add automation**: CompatHelper, TagBot
6. **Optimize**: Caching, conditional execution

**Docs**: [ci-cd-workflows.md](ci-cd-workflows.md)

---

## Common Workflows

### Debug Performance Issues

```julia
# 1. Measure baseline
@benchmark slow_function($data)

# 2. Profile
using Profile, ProfileView
@profile slow_function(data)
ProfileView.view()

# 3. Check type stability
@code_warntype slow_function(data)

# 4. Identify issue from docs
# - Red in @code_warntype → optimization-patterns.md#type-stability
# - High allocations → optimization-patterns.md#allocation-reduction
# - Single hot spot → optimization-patterns.md#algorithm-improvements

# 5. Apply fix and verify
@benchmark fast_function($data)
```

**Related Docs**: [optimization-patterns.md](optimization-patterns.md), [profiling-guide.md](profiling-guide.md)

### Set Up SciML Project

```julia
# 1. Identify problem (e.g., "stochastic population dynamics")
# → SDE problem

# 2. Get template from docs/sciml-templates.md#sde-templates

# 3. Customize
function drift!(du, u, p, t)
    # Your drift dynamics
end

function diffusion!(du, u, p, t)
    # Your noise model
end

# 4. Select solver (docs: solver selection guide)
sol = solve(prob, SOSRI())  # Recommended for SDEs

# 5. Add features if needed (docs: ensemble, callbacks)
```

**Related Docs**: [sciml-templates.md](sciml-templates.md)

### Create Production Package

```julia
# 1. Configure comprehensive template
using PkgTemplates

t = Template(;
    user="yourusername",
    julia=v"1.6",
    plugins=[
        Git(; manifest=false, ssh=true),
        GitHubActions(; linux=true, osx=true, windows=true, coverage=true),
        Codecov(),
        Documenter{GitHubActions}(),
        CompatHelper(),
        TagBot(),
        License(; name="MIT"),
    ]
)

# 2. Generate
t("MyProductionPackage")

# 3. Follow post-creation steps in docs/package-scaffolding.md#post-creation-setup

# 4. Set up CI in docs/ci-cd-workflows.md
```

**Related Docs**: [package-scaffolding.md](package-scaffolding.md), [ci-cd-workflows.md](ci-cd-workflows.md)

---

## By Topic

### Type Stability
- **Main Guide**: [optimization-patterns.md#type-stability-patterns](optimization-patterns.md#type-stability-patterns)
- **Analysis**: [profiling-guide.md#type-stability-analysis](profiling-guide.md#type-stability-analysis)
- **Topics**: Detection, common patterns, fixes, function barriers

### Memory & Allocations
- **Main Guide**: [optimization-patterns.md#allocation-reduction](optimization-patterns.md#allocation-reduction)
- **Profiling**: [profiling-guide.md#memory-profiling](profiling-guide.md#memory-profiling)
- **Topics**: Pre-allocation, in-place operations, views, static arrays

### Parallelization
- **Main Guide**: [optimization-patterns.md#parallelization-strategies](optimization-patterns.md#parallelization-strategies)
- **Topics**: Multi-threading, distributed computing, GPU, ensemble simulations

### Testing & CI
- **CI Setup**: [ci-cd-workflows.md#github-actions-workflows](ci-cd-workflows.md#github-actions-workflows)
- **Test Structure**: [package-scaffolding.md#testing-structure](package-scaffolding.md#testing-structure)
- **Topics**: Test matrices, coverage, quality checks, automation

### SciML Ecosystem
- **Templates**: [sciml-templates.md](sciml-templates.md)
- **Topics**: ODE/PDE/SDE templates, optimization, solver selection, callbacks, ensembles

### Package Development
- **Scaffolding**: [package-scaffolding.md](package-scaffolding.md)
- **CI/CD**: [ci-cd-workflows.md](ci-cd-workflows.md)
- **Topics**: PkgTemplates, structure, configuration, registration

---

## Checklists

### Performance Optimization Checklist

From [optimization-patterns.md](optimization-patterns.md#optimization-checklist):

- [ ] Type stability verified with `@code_warntype`
- [ ] Allocations minimized (check with `@benchmark`)
- [ ] Pre-allocated arrays where possible
- [ ] In-place operations used (`!` functions)
- [ ] Views used instead of copies (`@view`)
- [ ] Correct loop order (column-major)
- [ ] SIMD used for independent loops
- [ ] Parallelization considered
- [ ] Algorithm complexity optimal
- [ ] Profiling done to find bottlenecks

### Package Creation Checklist

From [package-scaffolding.md](package-scaffolding.md#checklist):

- [ ] Template configured with desired plugins
- [ ] Package generated
- [ ] Git repository initialized
- [ ] GitHub repository created
- [ ] Documentation keys added
- [ ] `Project.toml` has correct metadata
- [ ] [compat] entries added
- [ ] Module structure organized
- [ ] Public API clearly defined
- [ ] Tests organized in `@testset` blocks
- [ ] All functions have docstrings
- [ ] CI workflows passing
- [ ] Code coverage > 80%

### CI Configuration Checklist

From [ci-cd-workflows.md](ci-cd-workflows.md#ci-configuration-checklist):

- [ ] `.github/workflows/CI.yml` created
- [ ] Test matrix configured
- [ ] Cross-platform testing set up
- [ ] Caching enabled
- [ ] Coverage reporting configured
- [ ] Documentation workflow created
- [ ] CompatHelper.yml added
- [ ] TagBot.yml added
- [ ] Quality checks included
- [ ] Badges added to README

---

## Version Information

**Plugin Version**: 1.0.3
**Documentation Version**: 1.0.3
**Last Updated**: 2025-11-07

---

## Related Plugin Documentation

- **Agent Documentation**: `agents/` directory for julia-pro, sciml-pro, turing-pro, julia-developer
- **Command Documentation**: `commands/` directory for command-specific instructions
- **Skill Documentation**: `skills/` directory for reusable skill patterns

---

**Maintained by**: Wei Chen
**Project**: [MyClaude](https://myclaude.readthedocs.io/)
**Plugin**: julia-development
