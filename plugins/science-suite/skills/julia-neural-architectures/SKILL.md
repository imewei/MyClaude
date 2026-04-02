---
name: julia-neural-architectures
description: Design neural network architectures in Julia using Lux.jl. Covers Transformers (multi-head attention, positional encoding), CNNs, RNNs/LSTMs/GRUs (RecurrentLayers.jl), autoencoders, and custom layer design with explicit parameterization. Use when designing or implementing neural architectures in Julia.
---

# Julia Neural Architectures

## Expert Agents

For architecture design and implementation in Julia, delegate to:

- **`julia-ml-hpc`**: Julia ML/HPC specialist for Lux.jl implementation.
  - *Location*: `plugins/science-suite/agents/julia-ml-hpc.md`
  - *Capabilities*: Layer composition, custom layers, GPU optimization.
- **`neural-network-master`**: For architecture theory and design decisions.
  - *Location*: `plugins/science-suite/agents/neural-network-master.md`
  - *Capabilities*: Architecture selection, capacity planning, design trade-offs.

## CNN Patterns

### Basic CNN

```julia
using Lux, Random

cnn = Chain(
    Conv((3, 3), 1 => 32, relu; pad=SamePad()),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu; pad=SamePad()),
    MaxPool((2, 2)),
    FlattenLayer(),
    Dense(64 * 7 * 7, 256, relu),
    Dense(256, 10)
)
```

### Residual Block

```julia
function ResidualBlock(channels::Int)
    return Parallel(
        +,                          # Element-wise addition
        Chain(
            Conv((3, 3), channels => channels, relu; pad=SamePad()),
            Conv((3, 3), channels => channels; pad=SamePad())
        ),
        NoOpLayer()                 # Skip connection (identity)
    )
end

resnet = Chain(
    Conv((7, 7), 3 => 64, relu; stride=2, pad=3),
    MaxPool((3, 3); stride=2, pad=1),
    ResidualBlock(64),
    ResidualBlock(64),
    GlobalMeanPool(),
    FlattenLayer(),
    Dense(64, 10)
)
```

## Transformer

### Multi-Head Attention

```julia
function MultiHeadAttention(embed_dim::Int, num_heads::Int; dropout=0.0f0)
    head_dim = embed_dim div num_heads
    return Chain(
        Parallel(
            nothing,
            Dense(embed_dim, embed_dim),   # Q
            Dense(embed_dim, embed_dim),   # K
            Dense(embed_dim, embed_dim)    # V
        ),
        WrappedFunction(qkv -> scaled_dot_product_attention(qkv..., num_heads)),
        Dense(embed_dim, embed_dim),
        Dropout(dropout)
    )
end

function scaled_dot_product_attention(Q, K, V, num_heads)
    d_k = size(Q, 1) / num_heads
    scores = (Q' * K) ./ Float32(sqrt(d_k))
    weights = softmax(scores; dims=2)
    return V * weights'
end
```

### Transformer Block

```julia
function TransformerBlock(embed_dim::Int, num_heads::Int, ff_dim::Int; dropout=0.0f0)
    return Chain(
        SkipConnection(
            Chain(
                LayerNorm((embed_dim,)),
                MultiHeadAttention(embed_dim, num_heads; dropout)
            ),
            +
        ),
        SkipConnection(
            Chain(
                LayerNorm((embed_dim,)),
                Dense(embed_dim, ff_dim, gelu),
                Dropout(dropout),
                Dense(ff_dim, embed_dim),
                Dropout(dropout)
            ),
            +
        )
    )
end
```

### Positional Encoding

```julia
function positional_encoding(embed_dim::Int, max_len::Int)
    pe = zeros(Float32, embed_dim, max_len)
    for pos in 1:max_len, i in 1:2:embed_dim
        pe[i, pos]   = sin(pos / 10000.0f0^((i-1) / embed_dim))
        pe[i+1, pos] = cos(pos / 10000.0f0^((i-1) / embed_dim))
    end
    return pe
end
```

## RNN / LSTM / GRU via RecurrentLayers.jl

```julia
using RecurrentLayers, Lux

# LSTM for sequence modeling
lstm_model = Chain(
    Recurrence(LSTMCell(128, 256)),
    Dense(256, 10)
)

# GRU (lighter weight)
gru_model = Chain(
    Recurrence(GRUCell(128, 256)),
    Dense(256, 10)
)

# Bidirectional
bilstm = Chain(
    BidirectionalRNN(LSTMCell(128, 256)),
    Dense(512, 10)   # 2x hidden for bidirectional
)
```

## VAE Autoencoder

```julia
encoder = Chain(
    Dense(784, 512, relu),
    Dense(512, 256, relu),
    Parallel(nothing,
        Dense(256, 20),    # mu
        Dense(256, 20)     # log_var
    )
)

decoder = Chain(
    Dense(20, 256, relu),
    Dense(256, 512, relu),
    Dense(512, 784, sigmoid)
)

function vae_loss(encoder, decoder, ps_enc, ps_dec, st_enc, st_dec, x)
    (mu, log_var), st_enc = encoder(x, ps_enc, st_enc)

    # Reparameterization trick
    std = exp.(0.5f0 .* log_var)
    eps = randn(Float32, size(std))
    z = mu .+ std .* eps

    x_recon, st_dec = decoder(z, ps_dec, st_dec)

    # Reconstruction + KL divergence
    recon_loss = Flux.Losses.mse(x_recon, x)
    kl_loss = -0.5f0 * mean(1 .+ log_var .- mu .^ 2 .- exp.(log_var))

    return recon_loss + kl_loss, st_enc, st_dec
end
```

## Custom Layer

```julia
using Lux, Random

struct ScaledDense{F} <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::F
end

function Lux.initialparameters(rng::AbstractRNG, l::ScaledDense)
    return (
        weight = randn(rng, Float32, l.out_dims, l.in_dims) .* sqrt(2.0f0 / l.in_dims),
        bias = zeros(Float32, l.out_dims),
        scale = ones(Float32, 1)
    )
end

function Lux.initialstates(::AbstractRNG, ::ScaledDense)
    return (training = Val(true),)
end

function (l::ScaledDense)(x, ps, st)
    y = ps.scale .* (ps.weight * x .+ ps.bias)
    return l.activation.(y), st
end
```

## Composition Patterns

| Pattern | Lux Construct | Use Case |
|---------|--------------|----------|
| Sequential | `Chain(l1, l2, ...)` | Feedforward networks |
| Residual | `Parallel(+, Chain(...), NoOpLayer())` | Skip connections |
| Branching | `Parallel(cat_fn, branch1, branch2)` | Multi-input fusion |
| Conditional | `BranchLayer(gate, path_a, path_b)` | Mixture of experts |
| Repeated | `Chain([Block(i) for i in 1:N]...)` | Stacked layers |

## Weight Initialization with WeightInitializers.jl

```julia
using WeightInitializers

# Kaiming (He) initialization for ReLU networks
Dense(256, 128, relu; init_weight=kaiming_normal)

# Xavier/Glorot for tanh/sigmoid
Dense(256, 128, tanh; init_weight=glorot_uniform)

# Orthogonal for RNNs
Dense(256, 128; init_weight=orthogonal)
```

## Architecture Selection

| Task | Architecture | Key Layers |
|------|-------------|------------|
| Image classification | ResNet / CNN | Conv, BatchNorm, Residual |
| Sequence modeling | Transformer | MultiHeadAttention, LayerNorm |
| Time series | LSTM / GRU | RecurrentLayers.jl |
| Generation | VAE / GAN | Encoder-Decoder, reparameterization |
| Graph data | GNN | GNNChain (GraphNeuralNetworks.jl) |
| Point clouds | PointNet | Shared MLP, global pooling |
| Tabular | MLP | Dense, Dropout, BatchNorm |

## Checklist

- [ ] Select architecture based on data modality and task requirements
- [ ] Use `Parallel(+, ...)` for residual/skip connections
- [ ] Apply appropriate weight initialization (Kaiming for ReLU, Glorot for tanh)
- [ ] Add regularization (Dropout, BatchNorm) to prevent overfitting
- [ ] Use RecurrentLayers.jl for sequence models (not raw cell implementations)
- [ ] Test custom layers with `Lux.setup` and verify parameter shapes
- [ ] Profile memory usage for large architectures before full training
