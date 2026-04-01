---
name: web-development-julia
maturity: "5-Expert"
specialization: Julia Web Development
description: Build web applications with Genie.jl MVC framework and HTTP.jl. Use when creating REST APIs, handling HTTP requests, or building web services in Julia.
---

# Julia Web Development

REST APIs and web services with Genie.jl and HTTP.jl.

## Genie.jl REST API

```julia
using Genie, Genie.Router, Genie.Renderer.Json

route("/") do
    "Welcome"
end

route("/api/items/:id::Int") do
    id = payload(:id)
    json(:result => compute(id))
end

route("/api/items", method = POST) do
    data = jsonpayload()
    json(:created => process(data))
end

up(8000)
```

## Middleware

```julia
function auth_middleware(handler)
    req = handler
    function(req)
        haskey(req.headers, "Authorization") || return HTTP.Response(401, "Unauthorized")
        handler(req)
    end
end
```

## Framework Selection

| Framework | Use Case |
|-----------|----------|
| Genie.jl | Full MVC, authentication, ORM |
| HTTP.jl | Lightweight, custom control |
| Oxygen.jl | Minimal API framework |
