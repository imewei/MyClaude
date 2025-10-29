---
name: model-optimization-deployment
description: Comprehensive guide for optimizing and deploying neural networks to production environments. Covers quantization, pruning, knowledge distillation, ONNX export, and deployment strategies. Use when preparing models for edge devices, cloud deployment, or production serving.
---

# Model Optimization & Deployment

Systematic approaches for optimizing neural networks and deploying to production.

## When to Use

- Deploying models to production environments
- Optimizing for edge devices (mobile, embedded)
- Reducing model size or inference latency
- Converting models between frameworks
- Setting up model serving infrastructure
- Optimizing for specific hardware (GPU, TPU, CPU)

## Model Optimization Techniques

### 1. Quantization

**Post-Training Quantization:**
```python
# PyTorch example
import torch
model_fp32 = torch.load('model.pt')
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
# 4x smaller, 2-4x faster
```

**Quantization-Aware Training:**
- Insert fake quantization ops during training
- Model learns to be robust to quantization
- Better accuracy than post-training quantization

**When to use:**
- Need 4x size reduction
- Deploying to mobile/edge
- CPU inference

### 2. Pruning

**Unstructured Pruning:**
- Remove individual weights (sparse matrices)
- Requires sparse inference support

**Structured Pruning:**
```python
# Remove entire channels/neurons
# Maintain dense operations (GPU friendly)
```

**Magnitude-based:** Remove smallest weights
**Lottery Ticket:** Train, prune, retrain from initialization

**When to use:**
- Large models with redundancy
- Need inference speedup
- Limited memory

### 3. Knowledge Distillation

```python
# Train small "student" model to mimic large "teacher"
loss = alpha * student_loss + (1-alpha) * distillation_loss
distillation_loss = KL_divergence(student_logits/T, teacher_logits/T)
```

**Benefits:**
- Student often better than training from scratch
- Transfers "dark knowledge"
- Flexible student architecture

**When to use:**
- Have large accurate model
- Need smaller deployed model
- Can't afford retraining large model

### 4. Architecture Optimization

**Efficient Architectures:**
- MobileNet, EfficientNet for mobile
- Dist

ilBERT, TinyBERT for NLP
- Depth-separable convolutions
- Inverted residuals

**Neural Architecture Search:**
- AutoML for efficient architectures
- Hardware-aware NAS

## Deployment Strategies

### 1. ONNX Export (Framework Agnostic)

```python
# PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Load in any framework supporting ONNX
import onnxruntime
session = onnxruntime.InferenceSession("model.onnx")
```

**Benefits:** Run on any hardware, optimize with ONNX Runtime

### 2. TensorRT (NVIDIA GPUs)

```python
# Optimize for NVIDIA GPUs
# INT8 quantization, kernel fusion, layer optimization
# Can achieve 2-10x speedup
```

### 3. TensorFlow Lite (Mobile)

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 4. Model Serving Frameworks

**TorchServe:**
- PyTorch native
- RESTful API
- Batch inference
- Model versioning

**TensorFlow Serving:**
- High-performance
- gRPC and REST
- Model management

**Triton Inference Server:**
- Multi-framework
- Dynamic batching
- GPU optimization

## Deployment Checklist

### Pre-Deployment
- [ ] Optimize model (quantization/pruning/distillation)
- [ ] Export to deployment format (ONNX/TFLite)
- [ ] Benchmark inference latency and throughput
- [ ] Test on target hardware
- [ ] Validate accuracy after optimization

### Deployment
- [ ] Set up model serving infrastructure
- [ ] Configure autoscaling
- [ ] Add monitoring (latency, throughput, errors)
- [ ] Implement A/B testing capability
- [ ] Set up model versioning
- [ ] Document inference API

### Post-Deployment
- [ ] Monitor production metrics
- [ ] Set up alerting for issues
- [ ] Plan for model updates
- [ ] Collect feedback for improvements
- [ ] Track model drift

## Performance Optimization

### Batch Inference
```python
# Process multiple inputs together
# Better GPU utilization
# Latency vs throughput trade-off
```

### Model Compilation
- TorchScript (PyTorch)
- XLA (TensorFlow, JAX)
- Ahead-of-time compilation for faster inference

### Hardware-Specific Optimization
- **GPU:** Batch operations, mixed precision, TensorRT
- **CPU:** ONNX Runtime, quantization, threading
- **TPU:** XLA compilation, large batches
- **Mobile:** TFLite, CoreML, quantization

## Quick Reference: Optimization Techniques

| Technique | Size Reduction | Speed Increase | Accuracy Impact |
|-----------|---------------|----------------|-----------------|
| INT8 Quantization | 4x | 2-4x | Small (<1%) |
| Pruning (90% sparse) | 10x | Varies | Small-Medium |
| Knowledge Distillation | Custom | Custom | Small |
| Architecture Change | Varies | Varies | Varies |

---

*Comprehensive guide for optimizing neural networks and deploying to production with quantization, pruning, knowledge distillation, and serving infrastructure.*
