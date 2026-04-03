---
name: web-development-julia
description: Build web applications with Genie.jl MVC framework and HTTP.jl. Use when creating REST APIs, handling HTTP requests, or building web services in Julia. Also use when adding JSON endpoints, setting up middleware or authentication, deploying Julia web servers with Docker or systemd, or choosing between Genie.jl, HTTP.jl, and Oxygen.jl. Use proactively when the user mentions serving a Julia model over HTTP or building a dashboard backend in Julia.
---

# Julia Web Development

## Expert Agent

For Julia web development with Genie.jl and HTTP.jl, delegate to:

- **`julia-pro`**: Julia web frameworks, REST APIs, and web services.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

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

## HTTP.jl Server

```julia
using HTTP, JSON3

function router(req::HTTP.Request)
    if req.method == "GET" && req.target == "/health"
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             JSON3.write(Dict(:status => "ok")))
    elseif req.method == "POST" && startswith(req.target, "/api/compute")
        body = JSON3.read(String(req.body))
        result = process(body)
        return HTTP.Response(200, JSON3.write(result))
    end
    return HTTP.Response(404, "Not Found")
end

HTTP.serve(router, "0.0.0.0", 8080)
```

## Genie.jl MVC Pattern

```julia
# routes.jl
using Genie.Router, Genie.Renderer.Json

route("/api/v1/models", method = GET) do
    models = SearchLight.all(SimulationModel)
    json(:models => models)
end

route("/api/v1/models", method = POST) do
    payload = jsonpayload()
    model = SimulationModel(name=payload["name"], params=payload["params"])
    SearchLight.save!(model)
    json(:id => model.id, :status => "created")
end

route("/api/v1/models/:id/run", method = POST) do
    id = payload(:id)
    model = SearchLight.findone(SimulationModel, id=id)
    result = run_simulation(model)
    json(:result => result)
end
```

## JSON Handling

```julia
using JSON3, StructTypes

struct SimulationRequest
    model::String
    parameters::Dict{String, Float64}
    steps::Int
end

StructTypes.StructType(::Type{SimulationRequest}) = StructTypes.Struct()

# Parse request body
req = JSON3.read(body, SimulationRequest)
```

## Deployment

| Strategy | Configuration |
|----------|---------------|
| Systemd service | `ExecStart=julia --sysimage=app.so server.jl` |
| Docker | Multi-stage build with PackageCompiler sysimage |
| Reverse proxy | Nginx/Caddy in front of Julia HTTP server |
| Environment config | `ENV["GENIE_ENV"] = "prod"` |

## Checklist

- [ ] Routes organized by resource with RESTful verbs
- [ ] JSON serialization via StructTypes for type safety
- [ ] Error responses return appropriate HTTP status codes
- [ ] Middleware handles authentication and CORS
- [ ] System image compiled for production startup time
- [ ] Health check endpoint exposed for load balancer probes
