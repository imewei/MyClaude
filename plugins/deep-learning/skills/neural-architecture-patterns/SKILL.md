---
name: neural-architecture-patterns
description: Design patterns and best practices for neural network architectures covering CNNs, RNNs, transformers, and modern architectures. Use when designing new architectures or understanding architectural principles and inductive biases.
---

# Neural Architecture Patterns

Design patterns, best practices, and architectural principles for building effective neural networks.

## When to Use

- Designing new neural network architectures
- Understanding why certain architectures work
- Selecting appropriate architecture for task
- Adapting architectures to new domains
- Learning architectural design principles

## Core Architectural Patterns

### 1. Skip Connections / Residual Learning

**Pattern:**
```python
def residual_block(x):
    residual = x
    x = conv(x)
    x = activation(x)
    x = conv(x)
    return x + residual  # Skip connection
```

**Why it works:**
- Addresses vanishing gradients (gradient highways)
- Easier optimization (identity mapping as default)
- Enables very deep networks (ResNet-1000+)

**When to use:**
- Deep networks (>10 layers)
- Training convergence issues
- Want to enable deeper architectures

### 2. Attention Mechanisms

**Self-Attention:**
```python
# Attention(Q, K, V) = softmax(QK^T/√d_k)V
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
```

**Why it works:**
- Content-based addressing (vs position-based)
- Captures long-range dependencies
- Parallelizable (unlike RNNs)

**Types:**
- Self-attention: Within sequence
- Cross-attention: Between sequences
- Multi-head: Parallel attention subspaces

**When to use:**
- Long-range dependencies important
- Sequence modeling
- Need interpretability (attention weights)

### 3. Normalization Layers

**Batch Normalization:**
```python
# Normalize across batch dimension
mean = x.mean(dim=0)
var = x.var(dim=0)
x_norm = (x - mean) / sqrt(var + eps)
output = gamma * x_norm + beta  # Learnable scale/shift
```

**Layer Normalization:**
- Normalize per sample (better for RNNs/transformers)
- Batch size independent

**When to use each:**
- Batch Norm: CNNs, large batches
- Layer Norm: Transformers, small batches, RNNs
- Group Norm: Computer vision, small batches

### 4. Encoder-Decoder

**Pattern:**
```python
# Encoder: Compress input to representation
z = encoder(x)  # x → z (bottleneck)

# Decoder: Reconstruct from representation
x_recon = decoder(z)  # z → x_recon
```

**Applications:**
- Autoencoders: Unsupervised learning
- Seq2Seq: Machine translation
- VAE: Generative modeling

### 5. Multi-Scale Processing

**Feature Pyramid Networks:**
```python
# Process at multiple resolutions
features_high = conv(x)
features_mid = downsample_and_conv(x)
features_low = downsample_and_conv(features_mid)

# Combine multi-scale features
output = combine([features_high, features_mid, features_low])
```

**Why it works:**
- Captures both local details and global context
- Better for detection, segmentation

**When to use:**
- Object detection
- Semantic segmentation
- Need multi-scale understanding

## Architecture Design Principles

### Inductive Biases

**Convolution (CNNs):**
- Translation equivariance
- Local connectivity
- Parameter sharing

**Recurrence (RNNs):**
- Temporal processing
- Sequential dependencies
- Hidden state memory

**Attention (Transformers):**
- Permutation equivariance (with positional encoding)
- Content-based routing
- Minimal inductive bias

### Capacity vs Regularization

**Increase Capacity:**
- More layers (depth)
- More neurons per layer (width)
- More complex architectures

**Add Regularization:**
- Dropout
- Weight decay
- Data augmentation
- Early stopping

**Balance:**
- Start simple, add capacity if underfitting
- Add regularization if overfitting

### Depth vs Width

**Depth (more layers):**
- More expressive (hierarchical features)
- Can be harder to optimize
- Better for complex patterns

**Width (more neurons):**
- Easier to optimize
- More parameters per layer
- Better for simple patterns

**Rule of thumb:** Depth preferred for most modern architectures

## Task-Specific Architectures

### Computer Vision

**Image Classification:**
- CNNs: ResNet, EfficientNet
- Vision Transformers: ViT, Swin

**Object Detection:**
- Two-stage: Faster R-CNN
- Single-stage: YOLO, RetinaNet

**Segmentation:**
- U-Net (encoder-decoder with skip connections)
- DeepLab (atrous convolution)

### Natural Language Processing

**Text Classification:**
- BERT-based: Fine-tune pretrained
- CNN: Fast, works well

**Sequence Generation:**
- GPT-style: Autoregressive transformers
- Seq2Seq: Encoder-decoder

**Question Answering:**
- BERT with QA head
- Retrieval-augmented

### Multimodal

**Vision-Language:**
- CLIP: Contrastive learning
- Vision encoder + Language encoder

**Audio-Visual:**
- Separate encoders + fusion
- Cross-modal attention

## Modern Architecture Innovations

### Transformers
- Self-attention replaces recurrence
- Scaled to billions of parameters
- Dominant in NLP, expanding to vision

### Diffusion Models
- Iterative denoising process
- State-of-the-art image generation
- Text-to-image (Stable Diffusion)

### Neural ODEs
- Continuous depth
- Memory-efficient backprop
- Irregular time series

### Graph Neural Networks
- Message passing on graphs
- Non-Euclidean data
- Molecular property prediction

## Architecture Selection Guide

| Task | Recommended Architecture | Why |
|------|-------------------------|-----|
| Image Classification | ResNet/EfficientNet/ViT | Proven, scalable |
| Object Detection | YOLO/Faster R-CNN | Speed vs accuracy trade-off |
| Semantic Segmentation | U-Net/DeepLab | Multi-scale, skip connections |
| Text Classification | BERT/RoBERTa | Pretrained, transfer learning |
| Text Generation | GPT/T5 | Autoregressive, large scale |
| Speech Recognition | Conformer/Wav2Vec | Audio-specific |
| Multimodal | CLIP/Flamingo | Cross-modal understanding |

---

*Comprehensive design patterns and principles for building effective neural network architectures across domains.*
