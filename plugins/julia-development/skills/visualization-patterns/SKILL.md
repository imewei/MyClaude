---
name: visualization-patterns
description: Master Plots.jl, Makie.jl, and StatsPlots.jl for data visualization in Julia. Use for creating line plots, scatter plots, heatmaps, 3D visualizations, statistical plots, and interactive graphics.
---

# Visualization Patterns

Master Julia's visualization ecosystem with Plots.jl (unified interface), Makie.jl (high-performance), and StatsPlots.jl (statistical recipes).

## Plots.jl (Unified Interface)

```julia
using Plots

# Multiple backends
gr()         # Default, fast
plotly()     # Interactive HTML
pyplot()     # Matplotlib backend

# Basic plotting
x = range(0, 2π, length=100)
plot(x, sin.(x), label="sin(x)", xlabel="x", ylabel="y", title="Trigonometric")
plot!(x, cos.(x), label="cos(x)")  # Add to existing plot
scatter!(x[1:10:end], sin.(x[1:10:end]), label="samples")

# Subplots
p1 = plot(x, sin.(x))
p2 = plot(x, cos.(x))
plot(p1, p2, layout=(1, 2))

# Heatmaps
data = rand(20, 20)
heatmap(data, c=:viridis)
```

## Makie.jl (High Performance)

```julia
using GLMakie  # OpenGL backend, also CairoMakie, WGLMakie

# Basic figure
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
lines!(ax, x, sin.(x))
scatter!(ax, x[1:10:end], sin.(x[1:10:end]))
fig

# Interactive 3D
surface(rand(50, 50))

# Animations
record(fig, "animation.mp4", 1:100; framerate=30) do i
    # Update plot each frame
end
```

## Best Practices

- Use Plots.jl for quick prototyping and standard plots
- Use Makie.jl for publication-quality graphics and complex visualizations
- Choose backend based on needs: GR (fast), Plotly (interactive), Cairo (publication)
- Use StatsPlots for statistical visualizations (marginal plots, density plots)

## Resources

- **Plots.jl**: https://docs.juliaplots.org/
- **Makie.jl**: https://docs.makie.org/
