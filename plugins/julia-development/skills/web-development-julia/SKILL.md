---
name: web-development-julia
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Web Development
description: Build web applications with Genie.jl MVC framework and HTTP.jl. Use when creating REST APIs, handling HTTP requests, or building web services in Julia.
---

# Julia Web Development

Web applications and REST APIs with Genie.jl and HTTP.jl.

---

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

---

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

---

## Framework Selection

| Framework | Use Case |
|-----------|----------|
| Genie.jl | Full MVC, authentication, ORM |
| HTTP.jl | Lightweight, custom control |
| Oxygen.jl | Minimal API framework |

---

## Checklist

- [ ] Framework selected for use case
- [ ] Routes defined
- [ ] JSON serialization configured
- [ ] Error handling implemented

---

**Version**: 1.0.5
