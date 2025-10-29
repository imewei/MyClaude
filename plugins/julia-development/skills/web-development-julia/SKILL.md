---
name: web-development-julia
description: Genie.jl MVC framework, HTTP.jl server development, API patterns, JSON3.jl, and Oxygen.jl lightweight APIs. Use for building web applications and REST APIs in Julia.
---

# Web Development in Julia

Build web applications with Genie.jl and HTTP.jl.

## Genie.jl (MVC Framework)
```julia
using Genie, Genie.Router, Genie.Renderer.Json

route("/") do
    "Welcome"
end

route("/api/:id::Int") do
    id = payload(:id)
    json(:result => compute(id))
end

up(8000)
```

## HTTP.jl (Lightweight)
```julia
using HTTP, JSON3

HTTP.serve("0.0.0.0", 8000) do req
    if req.target == "/"
        return HTTP.Response(200, "Welcome")
    else
        return HTTP.Response(404, "Not Found")
    end
end
```

## Resources
- **Genie.jl**: https://genieframework.com/
- **HTTP.jl**: https://github.com/JuliaWeb/HTTP.jl
