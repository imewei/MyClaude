---
name: using-juliacheck
disable-model-invocation: true
description: Use when you want to run JuliaCheck.jl, a JuliaSyntax-based rule checker, against Julia source files, choose rules or output formats, or add static code checks to a workflow
---

# Checking Julia Code with JuliaCheck.jl

JuliaCheck.jl is a rule-based static checker. It parses files with JuliaSyntax.jl and reports style and structural violations. It does not execute code or prove correctness; use it alongside tests ([testing-julia-package](../testing-julia-package/SKILL.md)) and JET ([profiling-julia-performance](../profiling-julia-performance/SKILL.md)).

The setup step downloads third-party code from GitHub. If the environment cannot run networked commands, say so immediately and give the user the commands or workflow files to run themselves.

## Obtain a pinned checkout

Pick a tagged release compatible with the Julia version in use (check the repository's releases). Pin a tag or commit; do not track a moving branch in CI. Keep the checkout outside the project:

```sh
export JULIACHECK_DIR="${JULIACHECK_DIR:-$HOME/.cache/JuliaCheck.jl}"
git clone --branch <release-tag> --depth 1 https://github.com/tiobe/JuliaCheck.jl.git "$JULIACHECK_DIR"
julia --project="$JULIACHECK_DIR" -e 'using Pkg; Pkg.instantiate()'
```

If the checkout exists, only run the `instantiate` line. Replace `<release-tag>` with a real release before running.

## Run it

All rules are on by default:

```sh
julia --project="$JULIACHECK_DIR" "$JULIACHECK_DIR/src/JuliaCheck.jl" -- src/MyPkg.jl src/*.jl
```

Select rules after `--enable`, then separate the file list with `--`:

```sh
julia --project="$JULIACHECK_DIR" "$JULIACHECK_DIR/src/JuliaCheck.jl" \
  --enable module-name-casing single-module-file -- src/MyPkg.jl
```

Ask the tool for its options and installed rules before building automation on the CLI; they can change between releases:

```sh
julia --project="$JULIACHECK_DIR" "$JULIACHECK_DIR/src/JuliaCheck.jl" --help
```

## Output

Terminal-highlighted text is the default for local work. Use JSON when another tool consumes it:

```sh
julia --project="$JULIACHECK_DIR" "$JULIACHECK_DIR/src/JuliaCheck.jl" \
  --output json --outputfile juliacheck.json -- src/*.jl
```

## Policy

- Review enabled rules against the project's conventions. Defaults can be broader than a given style or API policy; enable a focused set.
- Keep JuliaCheck in a tooling or CI environment, never the package's runtime dependencies.
- After a Julia or parser upgrade, update the pinned release and review output before making the check mandatory.
- Triage findings: style and structure findings are not correctness bugs; report them separately ([auditing-julia-package-practices](../auditing-julia-package-practices/SKILL.md)).

When generating a script or CI workflow, replace every `<release-tag>` placeholder, say whether you validated the syntax, and ask the user to run it locally before committing.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
