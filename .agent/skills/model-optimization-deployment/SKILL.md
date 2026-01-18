---
name: model-optimization-deployment
version: "1.0.7"
maturity: "5-Expert"
specialization: Neural Network Optimization
description: Optimize and deploy neural networks with quantization, pruning, knowledge distillation, and production serving. Use when compressing models, converting between frameworks (ONNX, TFLite), or setting up TorchServe/Triton serving infrastructure.
---

# Model Optimization & Deployment

Optimize neural networks and deploy to production environments.

---

## Optimization Techniques

| Technique | Size Reduction | Speed Increase | Accuracy Impact |
|-----------|----------------|----------------|-----------------|
| INT8 Quantization | 4x | 2-4x | Small (<1%) |
| Pruning (90% sparse) | 10x | Varies | Small-Medium |
| Knowledge Distillation | Custom | Custom | Small |
| FP16 Mixed Precision | 2x | 1.5-2x | Minimal |

---

## Quantization

```python
# allow-torch
import torch

# Post-Training Dynamic Quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
# Result: 4x smaller, 2-4x faster on CPU
```

| Type | When to Use |
|------|-------------|
| Post-training dynamic | Quick deployment, NLP models |
| Post-training static | Vision models, calibration dataset available |
| Quantization-aware training | Maximum accuracy preservation |

---

## Framework Conversion

| Source | Target | Command/Method |
|--------|--------|----------------|
| PyTorch | ONNX | `torch.onnx.export(model, input, "model.onnx")` |
| TensorFlow | TFLite | `tf.lite.TFLiteConverter.from_saved_model()` |
| ONNX | TensorRT | `trtexec --onnx=model.onnx` |
| PyTorch | TorchScript | `torch.jit.trace(model, input)` |

---

## Serving Frameworks

| Framework | Best For | Key Features |
|-----------|----------|--------------|
| TorchServe | PyTorch models | REST API, batching, versioning |
| TF Serving | TensorFlow models | gRPC/REST, high performance |
| Triton | Multi-framework | Dynamic batching, GPU optimization |
| ONNX Runtime | Cross-platform | CPU/GPU, edge deployment |

---

## Hardware Optimization

| Target | Optimization |
|--------|--------------|
| NVIDIA GPU | TensorRT, mixed precision, batch ops |
| CPU | ONNX Runtime, INT8 quantization |
| Mobile | TFLite, Core ML, quantization |
| TPU | XLA compilation, large batches |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Benchmark first | Profile latency before optimizing |
| Validate accuracy | Test after each optimization step |
| Start simple | Try quantization before pruning |
| Match hardware | Optimize for deployment target |
| Version models | Track optimized model lineage |
| Monitor production | Track latency, throughput, errors |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Over-optimization | Stop when accuracy drops unacceptably |
| Wrong target | Profile actual deployment hardware |
| Ignoring batch size | Tune batch size for throughput vs latency |
| No rollback plan | Keep original model, version everything |
| Missing monitoring | Add latency/error rate tracking |

---

## Deployment Checklist

**Pre-Deployment:**
- [ ] Model optimized (quantization/pruning)
- [ ] Exported to deployment format
- [ ] Benchmarked on target hardware
- [ ] Accuracy validated after optimization

**Deployment:**
- [ ] Serving infrastructure configured
- [ ] Autoscaling enabled
- [ ] Monitoring active (latency, throughput)
- [ ] A/B testing capability ready

**Post-Deployment:**
- [ ] Production metrics monitored
- [ ] Alerting configured
- [ ] Rollback mechanism tested
- [ ] Model update pipeline ready

---

**Version**: 1.0.5
