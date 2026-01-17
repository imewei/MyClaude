---
name: neural-architecture-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Neural Network Architecture
description: Design neural architectures with skip connections, attention, normalization, and encoder-decoders. Use when designing CNNs, transformers, U-Nets, or selecting architectures for vision, NLP, and multimodal tasks.
---

# Neural Architecture Patterns

Design patterns and principles for building effective neural networks.

---

## Core Patterns

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Skip/Residual | Enable deep networks | >10 layers, vanishing gradients |
| Attention | Long-range dependencies | Sequences, global context |
| Normalization | Stable training | All deep networks |
| Encoder-Decoder | Compress/reconstruct | Translation, segmentation |
| Multi-Scale | Capture detail + context | Detection, segmentation |

---

## Skip Connections (ResNet)

```python
def residual_block(x):
    residual = x
    x = conv(x)
    x = activation(x)
    x = conv(x)
    return x + residual  # Skip connection
```

**Why**: Gradient highways, easier optimization, enables 100+ layers

---

## Attention Mechanism

```python
# Self-attention: Attention(Q, K, V) = softmax(QK^T/√d_k)V
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
```

| Type | Use Case |
|------|----------|
| Self-attention | Within sequence |
| Cross-attention | Between sequences |
| Multi-head | Parallel subspaces |

---

## Normalization

| Type | Use Case | Normalize Over |
|------|----------|----------------|
| BatchNorm | CNNs, large batches | Batch dimension |
| LayerNorm | Transformers, small batches | Feature dimension |
| GroupNorm | Vision, small batches | Channel groups |

---

## Inductive Biases

| Architecture | Bias | Best For |
|--------------|------|----------|
| CNN | Translation equivariance | Spatial data (images) |
| RNN | Temporal processing | Sequential data |
| Transformer | Content-based routing | Long-range, large scale |

---

## Task-Specific Selection

| Task | Architectures |
|------|---------------|
| Image Classification | ResNet, EfficientNet, ViT |
| Object Detection | YOLO, Faster R-CNN, DETR |
| Segmentation | U-Net, DeepLab, SegFormer |
| Text Classification | BERT, RoBERTa |
| Text Generation | GPT, T5 |
| Multimodal | CLIP, Flamingo |

---

## Design Principles

| Principle | Guidance |
|-----------|----------|
| Depth vs Width | Depth for complex patterns, width easier to optimize |
| Capacity | Start simple, add if underfitting |
| Regularization | Add dropout/decay if overfitting |

---

## Modern Innovations

| Innovation | Key Feature |
|------------|-------------|
| Transformers | Self-attention replaces recurrence |
| Diffusion Models | Iterative denoising for generation |
| Neural ODEs | Continuous depth, memory-efficient |
| Graph Neural Networks | Message passing on non-Euclidean data |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Match inductive bias | CNN for images, Transformer for sequences |
| Use pretrained | Fine-tune for efficiency |
| Skip connections | For deep networks |
| Appropriate normalization | LayerNorm for transformers, BatchNorm for CNNs |
| Balance capacity/regularization | Monitor train/val gap |

---

## Checklist

- [ ] Architecture matches data type (spatial, sequential, graph)
- [ ] Skip connections for deep networks
- [ ] Appropriate normalization layer
- [ ] Attention if long-range dependencies needed
- [ ] Considered pretrained models
- [ ] Balanced depth/width for task complexity

---

**Version**: 1.0.5
