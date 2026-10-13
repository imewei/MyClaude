---
name: creating-julia-app
disable-model-invocation: true
description: Use when you create a command-line application or MCP server in Julia — a package with a `@main` entry point and an `[apps]` table in `Project.toml`, installed with `Pkg.Apps` and run by name from the shell
---

# Creating a Julia App

An app is a Julia package run as a standalone program (`mytool args...`) rather than loaded as a library. It is a normal package plus:

- a `@main` entry point in the package module,
- an `[apps]` table in `Project.toml` naming the executables.

Requires Julia 1.12+. Pkg app support is experimental; expect changes.

```julia
# src/MyReverseApp.jl
module MyReverseApp

function (@main)(ARGS)
    for arg in ARGS
        print(stdout, reverse(arg), " ")
    end
    return 0
end

end # module
```

```toml
# Project.toml (standard package fields above)
[apps]
reverse = {}
```

The empty table `{}` holds optional metadata. One package can define several apps, point one at a submodule, and set per-app Julia flags:

```toml
[apps]
main-app = {}
cli-app = { submodule = "CLI" }
fast-app = { julia_flags = ["--threads=auto", "--optimize=2"] }
```

## Entry-point conventions

- Return an `Integer` exit code from `@main` (`0` success, non-zero failure); a `return` of `nothing` exits 0.
- Print results to `stdout`, diagnostics and errors to `stderr`.
- Validate `ARGS` up front and print usage on bad input. Use ArgParse.jl or Comonicon.jl for non-trivial option parsing rather than hand-rolled loops.
- Keep `@main` thin: parse arguments, call a library function, return a code. Test the library function with [testing-julia-package](../testing-julia-package/SKILL.md), not the process.
- Start-up latency matters for CLIs; see [precompiling-julia-package](../precompiling-julia-package/SKILL.md). For a fully standalone binary, see [compiler-patterns](../compiler-patterns/SKILL.md) (PackageCompiler.jl).

## Installing and running

```julia-repl
pkg> app add MyReverseApp
pkg> app add https://github.com/example/MyReverseApp.jl
pkg> app develop path/to/MyReverseApp   # local path must be a git repository
```

`develop` runs live from a local checkout so edits take effect immediately. Installation creates a shim in `~/.julia/bin`, which you must add to `PATH`. Then:

```sh
reverse some input string
```

Manage with `pkg> app status`, `app update [name]`, `app rm name`. An app uses the Julia executable that installed it; override with `JULIA_APPS_JULIA_CMD`.

**Security:** installing an app installs third-party code that runs when invoked. Prefer a registry entry or an immutable tag/commit SHA over a mutable branch, and review source you did not write.

## MCP servers

A Julia MCP server is an app speaking JSON-RPC over stdio: requests on stdin, responses on stdout. Use the same `@main`/`[apps]` structure so it installs and launches by name. Keep stdout reserved for protocol messages; send all logs to stderr, since a stray `println` corrupts the stream. See the [Model Context Protocol](https://modelcontextprotocol.io/) spec for the wire format.

References: [Apps](https://pkgdocs.julialang.org/v1/apps/), [Multiple Apps per Package](https://pkgdocs.julialang.org/v1/apps/#Multiple-Apps-per-Package).

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
