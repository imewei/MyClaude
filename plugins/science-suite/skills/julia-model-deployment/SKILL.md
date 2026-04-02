---
name: julia-model-deployment
description: Deploy trained Julia models to production. Covers ONNX.jl model export, Genie.jl/Oxygen.jl REST API serving, PackageCompiler.jl system images for startup elimination, Docker containerization, and interop with Python serving stacks via PythonCall.jl. Use when deploying Julia ML models or reducing startup latency.
---

# Julia Model Deployment

## Expert Agent

For deploying Julia models to production, delegate to:

- **`julia-ml-hpc`** at `plugins/science-suite/agents/julia-ml-hpc.md`

## Model Serialization with JLD2

Save and load trained model parameters:

```julia
using JLD2, Lux

model = Chain(Dense(784, 256, relu), Dense(256, 10))
ps, st = Lux.setup(Random.default_rng(), model)

# Train model...

# Save
jldsave("model.jld2"; ps=ps, st=st)

# Load
saved = load("model.jld2")
ps_loaded = saved["ps"]
st_loaded = saved["st"]
```

## ONNX Export with ONNXRunTime

Export models for cross-platform inference:

```julia
using ONNXRunTime

# Load an ONNX model for inference
model = ONNXRunTime.load("model.onnx")

# Run inference
input = Dict("input" => randn(Float32, 784, 1))
output = model(input)
```

To export Julia models to ONNX, trace through the model and serialize:

```julia
using Flux, ONNXExporter

flux_model = Chain(Dense(784, 256, relu), Dense(256, 10))
dummy_input = randn(Float32, 784, 1)

ONNXExporter.export_onnx("model.onnx", flux_model, dummy_input)
```

## Genie.jl REST API

Serve predictions via HTTP:

```julia
using Genie, Genie.Router, Genie.Renderer.Json
using JLD2, Lux

# Load model at startup
const MODEL = Chain(Dense(784, 256, relu), Dense(256, 10))
const PARAMS = load("model.jld2")

route("/predict", method=POST) do
    payload = jsonpayload()
    x = Float32.(payload["input"])
    y, _ = MODEL(x, PARAMS["ps"], PARAMS["st"])
    json(Dict("prediction" => Array(y)))
end

route("/health") do
    json(Dict("status" => "healthy", "version" => "1.0.0"))
end

up(8080; async=false)
```

## Oxygen.jl Lightweight Alternative

Minimal API server for simple deployments:

```julia
using Oxygen, HTTP

@post "/predict" function(req::HTTP.Request)
    data = json(req)
    x = Float32.(data["input"])
    y, _ = MODEL(x, PARAMS["ps"], PARAMS["st"])
    return Dict("prediction" => Array(y))
end

@get "/health" function()
    return Dict("status" => "healthy")
end

serve(; host="0.0.0.0", port=8080)
```

## PackageCompiler.jl System Image

Eliminate startup latency with a custom system image:

```julia
using PackageCompiler

# Create sysimage with precompiled packages
create_sysimage(
    [:Lux, :Genie, :JLD2];
    sysimage_path="serving.so",
    precompile_execution_file="precompile_script.jl"
)
```

The precompile script exercises key code paths:

```julia
# precompile_script.jl
using Lux, JLD2, Genie
model = Chain(Dense(10, 5, relu), Dense(5, 2))
ps, st = Lux.setup(Random.default_rng(), model)
x = randn(Float32, 10, 1)
model(x, ps, st)
```

Run with the system image:

```bash
julia --sysimage=serving.so serve.jl
```

### Standalone Application

Build a fully standalone executable:

```julia
create_app(
    ".",                        # Source directory
    "build/serve_app";          # Output directory
    precompile_execution_file="precompile_script.jl",
    force=true
)
```

## Docker Multi-Stage Build

```dockerfile
# Stage 1: Build sysimage
FROM julia:1.11 AS builder

WORKDIR /app
COPY Project.toml Manifest.toml ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

COPY . .
RUN julia --project=. -e '
    using PackageCompiler
    create_sysimage(
        [:Lux, :Genie, :JLD2];
        sysimage_path="serving.so",
        precompile_execution_file="precompile_script.jl"
    )'

# Stage 2: Runtime
FROM julia:1.11-slim

WORKDIR /app
COPY --from=builder /app/serving.so .
COPY --from=builder /app/serve.jl .
COPY --from=builder /app/model.jld2 .

EXPOSE 8080
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

CMD ["julia", "--sysimage=serving.so", "serve.jl"]
```

Build and run:

```bash
docker build -t julia-model-server .
docker run -p 8080:8080 julia-model-server
```

## Python Interop via PythonCall.jl

Call Julia models from Python serving stacks:

```python
# Python side -- using juliacall
from juliacall import Main as jl

jl.include("load_model.jl")      # Loads model, ps, st
predictions = jl.predict(input_data)
```

Or call Python from Julia:

```julia
using PythonCall

np = pyimport("numpy")
flask = pyimport("flask")

# Bridge Julia inference into a Python web framework
function julia_predict(input_array)
    x = pyconvert(Matrix{Float32}, input_array)
    y, _ = MODEL(x, PARAMS["ps"], PARAMS["st"])
    return Py(Array(y))
end
```

## Latency Optimization

| Strategy | Startup Time | Binary Size | Use Case |
|----------|-------------|-------------|----------|
| Default Julia | ~30s (TTFX) | N/A | Development |
| Sysimage | ~2s | ~500MB | Production servers |
| Standalone app | ~0.5s | ~1GB | Edge / containerized |
| PrecompileTools.jl | ~5s (cached) | N/A | Package development |

## Production Checklist

- [ ] Serialize models with JLD2 (include both `ps` and `st`)
- [ ] Add `/health` endpoint for load balancer probes
- [ ] Build a sysimage to eliminate first-call latency
- [ ] Use multi-stage Docker builds to minimize image size
- [ ] Pin Julia version and `Manifest.toml` for reproducibility
- [ ] Add request validation and error handling middleware
- [ ] Set up structured logging (JSON format for log aggregators)
- [ ] Load test with realistic payloads before deploying
