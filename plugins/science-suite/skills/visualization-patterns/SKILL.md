---
name: visualization-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Visualization
description: Master Plots.jl and Makie.jl for data visualization in Julia. Use when creating plots, selecting backends, building statistical visualizations, or making publication-quality figures.
---

# Julia Visualization Patterns

Plots.jl (unified interface) and Makie.jl (high-performance) visualization.

---

## Backend Selection

| Backend | Use Case |
|---------|----------|
| GR (Plots.jl) | Fast, default |
| Plotly | Interactive HTML |
| CairoMakie | Publication-quality |
| GLMakie | 3D, interactive |

---

## Plots.jl

```julia
using Plots

gr()  # Set backend
x = range(0, 2π, length=100)

plot(x, sin.(x), label="sin", xlabel="x", ylabel="y")
plot!(x, cos.(x), label="cos")  # Add to existing
scatter!(x[1:10:end], sin.(x[1:10:end]))

# Subplots
plot(plot(x, sin.(x)), plot(x, cos.(x)), layout=(1, 2))

# Heatmap
heatmap(rand(20, 20), c=:viridis)
```

---

## Makie.jl

```julia
using CairoMakie  # or GLMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
lines!(ax, x, sin.(x))
scatter!(ax, x[1:10:end], sin.(x[1:10:end]))
save("figure.pdf", fig)

# Animation
record(fig, "anim.mp4", 1:100; framerate=30) do i
    # Update plot each frame
end
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Quick prototyping | Plots.jl with GR |
| Publication figures | CairoMakie |
| Interactive 3D | GLMakie |
| Statistical plots | StatsPlots.jl |

---

## Checklist

- [ ] Appropriate backend selected
- [ ] Labels and legends added
- [ ] Color scheme accessible
- [ ] Figure saved at proper resolution

---

**Version**: 1.0.5
