---
name: auditing-julia-package-practices
disable-model-invocation: true
description: Use when you review an existing or in-progress Julia package against the Julia package practices in science-suite and want prioritized, evidence-based findings and concrete remediation suggestions
---

# Auditing Julia Package Practices

This is an audit: inspect, cite evidence, suggest changes. Do not edit the package unless the user separately asks for fixes.

## Establish the package and its Julia target

```sh
rg --files --hidden -g '!.git/**' -g '!Manifest.toml' -g '!**/.DS_Store'
```

Read `AGENTS.md`/`CLAUDE.md` and other project instructions first. Read `Project.toml` for name, dependencies, `[compat]`, `[workspace]`, and the declared `julia` bound. Judge the package against its **declared** compatibility range, not only current Julia. Do not recommend syntax above its minimum version (`public`, `[sources]`: 1.11; `[workspace]`, `[apps]`: 1.12) without flagging the compat impact.

## Audit areas

Check only what applies. Use the matching skill as the detailed standard and report each root cause once.

### Structure and dependencies ([generating-julia-package](../generating-julia-package/SKILL.md))

- Root `Project.toml` and `src/<Name>.jl` form a loadable package with a clear entry point.
- Runtime, test, docs, and tooling dependencies live in the right projects, not all in `[deps]`.
- Version-feature gate: flag `[workspace]` (needs 1.12), `[sources]`/`public` (need 1.11), or `[apps]` (needs 1.12) when `[compat] julia` is lower, as a `REGISTRY BLOCKER` if CI on the lowest version fails and `USER-FACING BUG` otherwise. Conversely, do not demand a workspace when compat is below 1.12.
- Where a `[workspace]` is valid, it covers `test`, `docs`, `benchmark`; each project declares what it imports.
- CI matrix includes the lowest supported Julia version, `1` (latest stable), and ideally `pre`; action versions are current (check tags, do not trust memory).
- `[compat]` present for every dependency and `julia`.
- Heavy optional integrations use extensions/weakdeps rather than hard deps.

### API and implementation ([developing-julia-package](../developing-julia-package/SKILL.md), [fixing-julia-exports](../fixing-julia-exports/SKILL.md))

- Exports and `public` match the documented API; tests exercise public behavior first.
- Idiomatic dispatch, separate configuration and result types, actionable exceptions, no type piracy.
- Performance claims backed by profiles ([profiling-julia-performance](../profiling-julia-performance/SKILL.md)); steady-state cost separated from first-call latency ([precompiling-julia-package](../precompiling-julia-package/SKILL.md)).

### Tests, docs, tooling ([testing-julia-package](../testing-julia-package/SKILL.md), [documenting-julia-package](../documenting-julia-package/SKILL.md), [using-juliacheck](../using-juliacheck/SKILL.md))

- `test/runtests.jl` and its environment are reproducible; a full `Pkg.test()` path exists. Parallel runners and focused selection are optional extras.
- User-facing APIs documented; Documenter builds and doctests wired into CI.
- Aqua, JET, JuliaFormatter, JuliaCheck only where they answer a relevant question. Separate style/structure findings from correctness and API-design findings.
- Numerical code: tolerances explicit, RNG seeded, results type-stable.

### Apps and release readiness ([creating-julia-app](../creating-julia-app/SKILL.md))

- CLI packages: `@main`, `[apps]`, argument handling, stdout/stderr discipline. Note that Pkg app support is experimental.
- Registration candidates: version, compat, tests, docs, TagBot/CompatHelper ([ci-cd-patterns](../ci-cd-patterns/SKILL.md)). Do not recommend release as a fix for unrelated findings.

## Report format

For each finding:

1. **Category**: `REGISTRY BLOCKER` (blocks registration or breaks CI/users), `USER-FACING BUG`, `RECOMMENDED BEFORE ANNOUNCEMENT`, or `POLISH`.
2. **Priority**: high, medium, low, by user impact and release risk.
3. **Evidence**: `file:line`, command output, or a precise statement of what is missing. Do not infer a defect from an absent convention alone.
4. **Why it matters**: concrete consequence for users, maintainers, compatibility, or reproducibility.
5. **Suggested change**: specific and reviewable.

Merge related observations. Separate confirmed issues from questions needing maintainer intent or runtime evidence. End with strengths, top-priority fixes, and which checks were or were not run.

If asked to apply fixes, re-audit afterward: fixes can unmask hidden failures (a missing test environment may have hidden a failing test). Re-run the relevant commands before the final verdict.

Do not label optional preferences as violations; a package does not need every practice here. For release audits, end with a go/no-go that separates registry blockers from announcement polish, e.g. "GO for registration once the two REGISTRY BLOCKER items are fixed; add a README before announcing."

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
