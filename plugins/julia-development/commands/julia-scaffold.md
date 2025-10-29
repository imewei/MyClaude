# /julia-scaffold - Bootstrap New Julia Package

**Priority**: 3
**Agent**: julia-developer
**Description**: Bootstrap new Julia package with proper structure following PkgTemplates.jl conventions.

## Usage
```
/julia-scaffold "<PackageName>"
```

## Generated Structure
```
PackageName/
├── Project.toml
├── src/PackageName.jl
├── test/runtests.jl
├── docs/make.jl
├── README.md
├── LICENSE
└── .gitignore
```

## Example
```
/julia-scaffold "MyAwesomePackage"
```

Generates complete package structure ready for development.
