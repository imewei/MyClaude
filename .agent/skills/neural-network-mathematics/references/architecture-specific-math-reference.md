# allow-torch
# Architecture-Specific Mathematics - Complete Reference

Mathematical foundations specific to neural network architectures: CNNs, RNNs, Transformers, and Generative Models.

---

## Table of Contents

1. [Convolutional Neural Networks](#convolutional-neural-networks)
2. [Recurrent Neural Networks](#recurrent-neural-networks)
3. [Transformers and Attention](#transformers-and-attention)
4. [Generative Models](#generative-models)

---

## Convolutional Neural Networks

### Convolution Operation

**1D Discrete Convolution:**
```
(f * g)[n] = Σₘ f[m] · g[n - m]

For signals f and kernel g
```

**2D Discrete Convolution (Images):**
```
(I * K)[i, j] = ΣₘΣₙ I[i+m, j+n] · K[m, n]

where:
- I: input image
- K: convolution kernel (filter)
```

**Cross-Correlation (Actually Used):**
```
(I ⋆ K)[i, j] = ΣₘΣₙ I[i-m, j-n] · K[m, n]

Deep learning frameworks use cross-correlation, not true convolution
```

**Multiple Channels:**
```
Output[d_out, i, j] = Σ_{d_in} Σₘ Σₙ Input[d_in, i-m, j-n] · Kernel[d_out, d_in, m, n]

where:
- Input: (d_in, H, W)
- Kernel: (d_out, d_in, k_h, k_w)
- Output: (d_out, H_out, W_out)
```

**Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_manual(input: torch.Tensor,
                 weight: torch.Tensor,
                 bias: torch.Tensor = None,
                 stride: int = 1,
                 padding: int = 0) -> torch.Tensor:
    """
    Manual 2D convolution implementation.

    Args:
        input: (N, C_in, H, W)
        weight: (C_out, C_in, K_h, K_w)
        bias: (C_out,)
        stride: Stride
        padding: Padding

    Returns:
        output: (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = weight.shape

    # Apply padding
    if padding > 0:
        input = F.pad(input, [padding] * 4, mode='constant', value=0)
        H += 2 * padding
        W += 2 * padding

    # Compute output dimensions
    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1

    output = torch.zeros(N, C_out, H_out, W_out)

    # Convolution
    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    # Extract patch
                    h_start = i * stride
                    w_start = j * stride
                    patch = input[n, :, h_start:h_start+K_h, w_start:w_start+K_w]

                    # Compute dot product
                    output[n, c_out, i, j] = torch.sum(patch * weight[c_out])

            # Add bias
            if bias is not None:
                output[n, c_out] += bias[c_out]

    return output


# Use PyTorch built-in (much faster)
conv = nn.Conv2d(in_channels=3, out_channels=64,
                kernel_size=3, stride=1, padding=1)
output = conv(input)
```

### Output Dimension Calculation

**Formula:**
```
H_out = ⌊(H_in + 2P - K) / S⌋ + 1
W_out = ⌊(W_in + 2P - K) / S⌋ + 1

where:
- H_in, W_in: input height/width
- P: padding
- K: kernel size
- S: stride
```

**Receptive Field:**
```
Receptive field of neuron at layer l:
RF_l = RF_{l-1} + (K_l - 1) · Π_{i=1}^{l-1} S_i

where:
- K_l: kernel size at layer l
- S_i: stride at layer i
```

**Implementation:**

```python
def compute_receptive_field(layers):
    """
    Compute receptive field of CNN.

    Args:
        layers: List of (kernel_size, stride) tuples

    Returns:
        Receptive field size
    """
    rf = 1
    stride_product = 1

    for kernel_size, stride in layers:
        rf += (kernel_size - 1) * stride_product
        stride_product *= stride

    return rf


# Example: VGG-like network
layers = [
    (3, 1), (3, 1), (2, 2),  # Conv-Conv-Pool
    (3, 1), (3, 1), (2, 2),  # Conv-Conv-Pool
    (3, 1), (3, 1), (2, 2),  # Conv-Conv-Pool
]
rf = compute_receptive_field(layers)
print(f"Receptive field: {rf}×{rf}")
```

### Fourier Transform Perspective

**Convolution Theorem:**
```
f * g = F⁻¹(F(f) · F(g))

where F is Fourier transform

Convolution in spatial domain = multiplication in frequency domain
```

**Frequency Analysis:**
```
Low-pass filter (blur): Attenuates high frequencies
High-pass filter (edge): Attenuates low frequencies
```

**Implementation:**

```python
def fft_conv2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolution using FFT (Fast Fourier Transform).

    Faster for large kernels.

    Args:
        input: (H, W) image
        kernel: (K, K) filter

    Returns:
        Convolved output
    """
    # Pad kernel to input size
    H, W = input.shape
    K = kernel.shape[0]

    kernel_padded = torch.zeros(H, W)
    kernel_padded[:K, :K] = kernel

    # FFT
    input_fft = torch.fft.fft2(input)
    kernel_fft = torch.fft.fft2(kernel_padded)

    # Multiply in frequency domain
    output_fft = input_fft * kernel_fft

    # Inverse FFT
    output = torch.fft.ifft2(output_fft).real

    return output
```

### Pooling Operations

**Max Pooling:**
```
Output[i, j] = max_{m,n ∈ Window} Input[i*s+m, j*s+n]

where s is stride
```

**Average Pooling:**
```
Output[i, j] = (1/K²) Σ_{m,n ∈ Window} Input[i*s+m, j*s+n]
```

**Global Average Pooling:**
```
Output[c] = (1/HW) Σᵢ Σⱼ Input[c, i, j]

Reduces spatial dimensions to 1×1
```

---

## Recurrent Neural Networks

### Vanilla RNN

**Forward Pass:**
```
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
yₜ = Wₕᵧhₜ + bᵧ

where:
- xₜ: input at time t
- hₜ: hidden state at time t
- yₜ: output at time t
- Wₓₕ, Wₕₕ, Wₕᵧ: weight matrices
```

**Backpropagation Through Time (BPTT):**
```
∂L/∂hₜ = ∂L/∂yₜ · ∂yₜ/∂hₜ + ∂L/∂hₜ₊₁ · ∂hₜ₊₁/∂hₜ

Chain rule across time steps
```

**Gradient Flow:**
```
∂hₜ/∂hₜ₋ₖ = Πᵢ₌₁ᵏ ∂hₜ₋ᵢ₊₁/∂hₜ₋ᵢ
          = Πᵢ₌₁ᵏ Wₕₕ · diag(tanh'(hₜ₋ᵢ))

If ||Wₕₕ · diag(tanh')||< 1: vanishing gradients
If ||Wₕₕ · diag(tanh')|| > 1: exploding gradients
```

**Implementation:**

```python
class VanillaRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h_prev: (batch, hidden_size) or None

        Returns:
            outputs: (batch, seq_len, output_size)
            h: (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_prev

        outputs = []
        for t in range(seq_len):
            h = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h))
            y = self.W_hy(h)
            outputs.append(y)

        outputs = torch.stack(outputs, dim=1)
        return outputs, h
```

### Long Short-Term Memory (LSTM)

**Gates:**
```
Forget gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
Input gate:  iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
Output gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)

Cell candidate: C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)

Cell state: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
Hidden state: hₜ = oₜ ⊙ tanh(Cₜ)

where σ is sigmoid, ⊙ is element-wise product
```

**Why LSTM Helps:**
```
Gradient flow through cell state:
∂Cₜ/∂Cₜ₋₁ = fₜ

If fₜ ≈ 1: Gradient flows unimpeded (no vanishing)
Cell state acts as "highway" for gradients
```

**Implementation:**

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined weight matrix for efficiency
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, h_prev: tuple = None):
        """
        Args:
            x: (batch, input_size)
            h_prev: (h, c) where h, c are (batch, hidden_size)

        Returns:
            h, c: Updated hidden and cell states
        """
        batch_size = x.size(0)

        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = h_prev

        # Concatenate input and hidden
        combined = torch.cat([x, h], dim=1)

        # Compute all gates at once
        gates = self.W(combined)
        i, f, c_tilde, o = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        c_tilde = torch.tanh(c_tilde)  # Cell candidate
        o = torch.sigmoid(o)  # Output gate

        # Update cell and hidden states
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)

        return h_new, c_new
```

### Gated Recurrent Unit (GRU)

**Simplified Gates:**
```
Reset gate: rₜ = σ(Wr·[hₜ₋₁, xₜ])
Update gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])

Candidate: h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])

Hidden state: hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

**Fewer parameters than LSTM, similar performance**

---

## Transformers and Attention

### Scaled Dot-Product Attention

**Formula:**
```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) V

where:
- Q: Query matrix (n × dₖ)
- K: Key matrix (m × dₖ)
- V: Value matrix (m × dᵥ)
- dₖ: Key dimension
```

**Why Scale by √dₖ?**
```
Dot product QKᵀ has variance dₖ for random Q, K
Without scaling: softmax saturates for large dₖ
Scaling by √dₖ: variance = 1, prevents saturation
```

**Implementation:**

```python
def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Args:
        Q: Queries (batch, n, d_k)
        K: Keys (batch, m, d_k)
        V: Values (batch, m, d_v)
        mask: Attention mask (batch, n, m)

    Returns:
        output: (batch, n, d_v)
        attention_weights: (batch, n, m)
    """
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask (set masked positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attention_weights = torch.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### Multi-Head Attention

**Formula:**
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wₒ

where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

Parameters:
- WᵢQ, WᵢK: (d_model, d_k)
- WᵢV: (d_model, d_v)
- Wₒ: (h·d_v, d_model)
```

**Why Multiple Heads?**
```
Different heads learn different attention patterns:
- Positional relationships
- Semantic relationships
- Syntactic relationships
```

**Implementation:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
               mask: torch.Tensor = None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_O(output)

        return output
```

### Positional Encoding

**Sinusoidal Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
- pos: position in sequence
- i: dimension index
```

**Why Sinusoidal?**
```
- Smooth interpolation to unseen positions
- Relative positions: PE(pos+k) is linear function of PE(pos)
- Fixed (no parameters to learn)
```

**Implementation:**

```python
def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        Positional encodings (seq_len, d_model)
    """
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

---

## Generative Models

### Variational Autoencoders (VAE)

**Objective (ELBO):**
```
log p(x) ≥ E_q[log p(x|z)] - D_KL(q(z|x)||p(z))

where:
- p(x|z): decoder (likelihood)
- q(z|x): encoder (variational posterior)
- p(z): prior N(0, I)
```

**Reparameterization Trick:**
```
z ~ q(z|x) = N(μ(x), σ²(x))

Reparameterize: z = μ(x) + σ(x) ⊙ ε, ε ~ N(0, I)

Enables backpropagation through sampling
```

**KL Divergence (Gaussian):**
```
D_KL(N(μ, σ²)||N(0, 1)) = 0.5 * (σ² + μ² - 1 - log σ²)
```

### Generative Adversarial Networks (GANs)

**Minimax Game:**
```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

where:
- G: Generator
- D: Discriminator
- x: real data
- z: random noise
```

**Optimal Discriminator:**
```
D*(x) = p_data(x) / (p_data(x) + p_model(x))
```

**Nash Equilibrium:**
```
At optimum:
- D(x) = 0.5 for all x
- p_model = p_data
```

**Training Objective:**
```
Discriminator: max E_x[log D(x)] + E_z[log(1 - D(G(z)))]
Generator: min E_z[log(1 - D(G(z)))]

In practice, use: max E_z[log D(G(z))] (non-saturating loss)
```

### Normalizing Flows

**Change of Variables:**
```
x = f(z), z ~ p_Z(z)

p_X(x) = p_Z(f⁻¹(x)) |det(∂f⁻¹/∂x)|
       = p_Z(z) |det(∂f/∂z)|⁻¹

Log-likelihood:
log p_X(x) = log p_Z(z) - log|det(∂f/∂z)|
```

**Jacobian Determinant:**
```
For invertible f: z → x
Need to compute det(∂f/∂z) efficiently

Affine coupling layers: Triangular Jacobian → O(n) determinant
```

**Implementation:**

```python
class AffineCouplingLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.dim1 = dim // 2
        self.dim2 = dim - self.dim1

        # Networks for scale and translation
        self.scale_net = nn.Sequential(
            nn.Linear(self.dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.dim2)
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.dim2)
        )

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Forward: z → x

        Args:
            z: Latent variable (batch, dim)

        Returns:
            x: Data variable (batch, dim)
            log_det: Log determinant of Jacobian
        """
        z1, z2 = z[:, :self.dim1], z[:, self.dim1:]

        # Affine transformation
        scale = self.scale_net(z1)
        translate = self.translate_net(z1)

        x2 = z2 * torch.exp(scale) + translate
        x = torch.cat([z1, x2], dim=1)

        # Log determinant (sum of log scales)
        log_det = scale.sum(dim=1)

        return x, log_det

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse: x → z
        """
        x1, x2 = x[:, :self.dim1], x[:, self.dim1:]

        scale = self.scale_net(x1)
        translate = self.translate_net(x1)

        z2 = (x2 - translate) * torch.exp(-scale)
        z = torch.cat([x1, z2], dim=1)

        return z
```

---

## Quick Reference

### Architecture Comparison

| Architecture | Best For | Key Math Concept |
|-------------|---------|------------------|
| CNN | Images, spatial data | Convolution, translation invariance |
| RNN/LSTM | Sequences, time series | Recurrence, temporal dependencies |
| Transformer | Long sequences, parallelization | Self-attention, positional encoding |
| VAE | Generation with latent structure | Variational inference, ELBO |
| GAN | High-quality generation | Minimax game, adversarial training |
| Flow | Exact likelihood | Change of variables, invertibility |

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Convolution (naive) | O(C_in C_out K² HW) | O(C_out HW) |
| Convolution (FFT) | O(C_in C_out HW log(HW)) | O(HW) |
| Self-Attention | O(n² d) | O(n²) |
| RNN forward | O(n h²) | O(nh) |
| LSTM forward | O(4n h²) | O(nh) |

---

## References

1. **CNNs:**
   - "ImageNet Classification with Deep CNNs" - Krizhevsky et al., 2012 (AlexNet)
   - "Very Deep CNNs for Large-Scale Recognition" - Simonyan & Zisserman, 2014 (VGG)
   - "Deep Residual Learning" - He et al., 2015 (ResNet)

2. **RNNs:**
   - "Long Short-Term Memory" - Hochreiter & Schmidhuber, 1997
   - "Learning to Forget" - Gers et al., 2000
   - "Gated Recurrent Units" - Cho et al., 2014

3. **Transformers:**
   - "Attention is All You Need" - Vaswani et al., 2017
   - "BERT" - Devlin et al., 2018
   - "GPT-3" - Brown et al., 2020

4. **Generative Models:**
   - "Auto-Encoding Variational Bayes" - Kingma & Welling, 2013 (VAE)
   - "Generative Adversarial Nets" - Goodfellow et al., 2014 (GAN)
   - "Normalizing Flows" - Rezende & Mohamed, 2015
   - "Denoising Diffusion Probabilistic Models" - Ho et al., 2020

---

*Architecture-specific mathematics enables deep understanding and effective implementation of modern neural network architectures.*
