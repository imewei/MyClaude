---
name: web-development-julia
description: Master web application development with Genie.jl MVC framework, HTTP.jl server development, REST API patterns, JSON3.jl, and Oxygen.jl lightweight APIs for building web services in Julia. Use when creating web applications (.jl files with Genie routes), building REST APIs with HTTP.jl server, implementing MVC patterns with Genie.jl, handling HTTP requests and responses, working with JSON data (JSON3.jl for serialization/deserialization), creating lightweight APIs with Oxygen.jl, building web services endpoints, serving static files, implementing authentication/authorization, or deploying Julia-based web servers. Essential for building web applications, REST APIs, and web services in Julia.
---

# Web Development in Julia

Build web applications with Genie.jl and HTTP.jl.

## When to use this skill

- Creating web applications with Genie.jl MVC framework
- Building REST APIs with HTTP.jl server
- Implementing routes and controllers in Genie
- Handling HTTP requests and responses
- Working with JSON data (JSON3.jl for ser ialization/deserialization)
- Creating lightweight APIs with Oxygen.jl
- Building web service endpoints for Julia computations
- Serving static files and assets
- Implementing authentication and authorization
- Deploying Julia-based web servers
- Creating microservices in Julia
- Building API servers for scientific computing workflows

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
