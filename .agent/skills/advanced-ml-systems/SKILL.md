---
name: advanced-ml-systems
version: "1.0.7"
maturity: "5-Expert"
specialization: Deep Learning Systems
description: Build advanced deep learning with PyTorch 2.x, TensorFlow, JAX including architectures (CNNs, Transformers, GANs), distributed training (DDP, FSDP, DeepSpeed), hyperparameter optimization (Optuna, Ray Tune), and model optimization (quantization, pruning, distillation). Use when training scripts, custom architectures, or scaling to multi-GPU/TPU.
---

# Advanced ML Systems

Deep learning frameworks, distributed training, and production-ready optimization.

---

## Framework Selection

| Framework | Ease | Performance | Production | Research |
|-----------|------|-------------|------------|----------|
| PyTorch 2.x | ★★★★★ | ★★★★ | ★★★ | ★★★★★ |
| TensorFlow | ★★★ | ★★★★ | ★★★★★ | ★★★ |
| JAX | ★★ | ★★★★★ | ★★★ | ★★★★★ |

---

## PyTorch 2.x Patterns

### torch.compile and Mixed Precision
```python
# allow-torch
import torch
from torch.cuda.amp import autocast, GradScaler

# Compile for 2-10x speedup
model = YourModel()
compiled_model = torch.compile(model)

# Mixed precision training
scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Distributed Training

| Method | Use Case | Memory | Complexity |
|--------|----------|--------|------------|
| DDP | Multi-GPU, same node | Low | Easy |
| DeepSpeed ZeRO-2 | Large models, multi-node | High | Medium |
| FSDP | Very large models (>1B) | Very High | Medium |
| Model Parallel | Model > single GPU | High | Hard |

### DDP Pattern
```python
# allow-torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = DDP(YourModel().to(rank), device_ids=[rank])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch in DataLoader(dataset, sampler=sampler):
            # Training loop
            pass

# Launch: torchrun --nproc_per_node=4 train.py
```

### DeepSpeed Config
```python
ds_config = {
    "train_batch_size": 64,
    "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}},
    "fp16": {"enabled": True, "initial_scale_power": 16}
}
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)
```

---

## Hyperparameter Optimization

### Optuna
```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = create_model(dropout=dropout)
    for epoch in range(10):
        val_loss = train_and_evaluate(model, lr, batch_size)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_loss

study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
```

---

## Model Optimization

| Technique | Speed Up | Size Reduction | Accuracy Impact |
|-----------|----------|----------------|-----------------|
| Quantization (INT8) | 2-4x | 4x | <1% |
| Pruning (50%) | 1.5-2x | 2x | <2% |
| Distillation | Varies | Varies | 1-5% |
| Mixed Precision | 2-3x | None | <0.1% |

### Quantization
```python
# Dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Knowledge Distillation
```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    soft_targets = torch.softmax(teacher_logits / T, dim=-1)
    soft_student = torch.log_softmax(student_logits / T, dim=-1)
    kd_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (T**2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss
```

---

## LoRA Fine-Tuning

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(AutoModelForCausalLM.from_pretrained("gpt2"), lora_config)
model.print_trainable_parameters()  # Only 0.2% trainable!
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| torch.compile | Use on hot paths for 2-10x speedup |
| Mixed precision | FP16/BF16 for 2-3x faster training |
| Gradient accumulation | Simulate larger batches on small GPUs |
| DDP before FSDP | Start simple, scale complexity as needed |
| Profile first | Use torch.profiler before optimizing |

---

## Common Pitfalls

| Pitfall | Problem |
|---------|---------|
| No gradient checkpointing | OOM on large models |
| Wrong ZeRO stage | ZeRO-3 overhead for small models |
| Ignoring data loading | CPU bottleneck, GPU idle |
| No warmup | Unstable early training |

---

## Checklist

- [ ] torch.compile on model for inference
- [ ] Mixed precision enabled
- [ ] DDP/FSDP for multi-GPU
- [ ] Gradient checkpointing for large models
- [ ] Learning rate warmup configured
- [ ] Quantization for deployment
- [ ] Hyperparameter search completed

---

**Version**: 1.0.5
