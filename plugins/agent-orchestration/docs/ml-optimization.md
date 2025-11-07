# ML Optimization Patterns

## PyTorch/TensorFlow Training Optimization

### Pattern 1: Mixed Precision Training (2x Speedup)
```python
# PyTorch with AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Result: 2x faster training, minimal accuracy loss
```

### Pattern 2: Gradient Accumulation (Large Batches on Small GPU)
```python
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Effective batch size: batch_size × accumulation_steps
```

### Pattern 3: DataLoader Optimization (3-5x Speedup)
```python
# Optimized DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,          # Parallel data loading
    pin_memory=True,        # Faster CPU→GPU transfer
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True # Reuse workers
)
```

### Pattern 4: Model Quantization (4x Inference Speedup)
```python
# PyTorch dynamic quantization
import torch.quantization

model_fp32 = MyModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

# Result: 4x faster inference, 75% memory reduction
```

### Pattern 5: torch.compile() (PyTorch 2.0+, 20-40% Speedup)
```python
import torch

model = torch.compile(MyModel())  # One-line optimization!

# Automatically applies:
# - Operator fusion
# - Memory optimization
# - GPU kernel optimization
```

---

**See also**: [Scientific Patterns](scientific-patterns.md), [Examples](examples/jax-training-optimization.md)
