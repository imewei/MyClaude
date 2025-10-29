---
name: advanced-ml-systems
description: Advanced machine learning systems including deep learning architectures (CNNs, RNNs, Transformers), distributed training (Horovod, DeepSpeed, FSDP), hyperparameter optimization, and model optimization techniques (pruning, quantization, distillation). Use when building complex neural networks, scaling training to multiple GPUs/nodes, or optimizing models for production deployment.
---

# Advanced ML Systems

Build and optimize sophisticated machine learning systems with deep learning frameworks, distributed training, and production-ready model optimization.

---

## When to Use

- Implementing deep learning architectures (CNNs, RNNs, Transformers)
- Distributed training across multiple GPUs or nodes
- Hyperparameter tuning for complex models
- Model optimization (pruning, quantization, knowledge distillation)
- Transfer learning and fine-tuning pre-trained models
- Handling large-scale datasets and models
- Production model optimization for inference

---

## Deep Learning Frameworks

### PyTorch 2.x Modern Features

**torch.compile for Performance:**
```python
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

# Compile for 2-10x speedup
model = ResNetBlock(64, 128)
compiled_model = torch.compile(model)  # Automatic optimization

# Use like normal model
output = compiled_model(input_tensor)
```

**Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Automatic mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Scaled backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### TensorFlow 2.x/Keras Best Practices

**tf.function for Graph Optimization:**
```python
import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    @tf.function  # Convert to graph for 10-50x speedup
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = tf.nn.dropout(x, rate=0.5)
        return self.dense2(x)

# Mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model = CustomModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

---

## Distributed Training

### PyTorch DDP (DistributedDataParallel)

**Multi-GPU Training:**
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = YourModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Distributed sampler for data loading
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently each epoch

        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    cleanup()

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

### DeepSpeed for Large Models

**ZeRO Optimization:**
```python
import deepspeed

# DeepSpeed config
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO-2: partition optimizer states + gradients
        "offload_optimizer": {
            "device": "cpu"  # Offload to CPU for memory savings
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}

# Initialize model and engine
model = YourLargeModel()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop
for batch in dataloader:
    inputs, labels = batch
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)

    model_engine.backward(loss)
    model_engine.step()
```

### Fully Sharded Data Parallel (FSDP)

**For Models Too Large for Single GPU:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# Auto-wrap policy for transformers
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={GPT2Block}
)

model = GPT2Model(config)
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    device_id=torch.cuda.current_device()
)
```

---

## Hyperparameter Optimization

### Optuna for Automated Tuning

**Bayesian Optimization:**
```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

def objective(trial):
    """Define hyperparameter search space and training."""

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Build model with suggested hyperparameters
    model = create_model(n_layers=n_layers, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train for a few epochs
    for epoch in range(10):
        train_loss = train_epoch(model, optimizer, dataloader, batch_size)
        val_loss = evaluate(model, val_dataloader)

        # Report intermediate value for pruning
        trial.report(val_loss, epoch)

        # Prune unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss

# Create study and optimize
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(),
    pruner=MedianPruner()
)
study.optimize(objective, n_trials=100, timeout=3600)

# Best hyperparameters
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### Ray Tune for Distributed HPO

**Scalable Hyperparameter Search:**
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def train_model(config):
    """Training function with config from Ray Tune."""
    model = create_model(
        lr=config["lr"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    )

    for epoch in range(10):
        train_loss = train_epoch(model)
        val_loss = evaluate(model)

        # Report metrics to Ray Tune
        tune.report(loss=val_loss, accuracy=val_acc)

# Define search space
config = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "hidden_dim": tune.choice([64, 128, 256, 512]),
    "dropout": tune.uniform(0.1, 0.5),
}

# Use ASHA scheduler for early stopping
scheduler = ASHAScheduler(
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

# Bayesian optimization with Optuna
search_alg = OptunaSearch(metric="loss", mode="min")

# Run distributed hyperparameter search
analysis = tune.run(
    train_model,
    config=config,
    num_samples=100,
    scheduler=scheduler,
    search_alg=search_alg,
    resources_per_trial={"cpu": 2, "gpu": 0.25}
)

# Best config
best_config = analysis.get_best_config(metric="loss", mode="min")
```

---

## Model Optimization

### Quantization for Inference Speed

**PyTorch Quantization:**
```python
import torch.quantization

# Dynamic quantization (easiest, for LSTM/Linear layers)
model_fp32 = YourModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# Static quantization (better performance, requires calibration)
model_fp32 = YourModel()
model_fp32.eval()

# Fuse modules for better quantization
model_fp32 = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn', 'relu']])

# Specify quantization configuration
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_fp32, inplace=True)

# Calibrate with representative data
with torch.no_grad():
    for batch in calibration_dataloader:
        model_fp32(batch)

# Convert to quantized model
model_int8 = torch.quantization.convert(model_fp32, inplace=False)

# 4x smaller model, 2-4x faster inference
torch.save(model_int8.state_dict(), 'model_int8.pth')
```

### Model Pruning

**Magnitude-Based Pruning:**
```python
import torch.nn.utils.prune as prune

model = YourModel()

# Prune 30% of connections in all Conv2d layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Make pruning permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Structured pruning (entire filters)
prune.ln_structured(
    model.conv1, name="weight",
    amount=0.5, n=2, dim=0  # Prune 50% of filters
)
```

### Knowledge Distillation

**Teacher-Student Training:**
```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Combined loss: KL divergence from teacher + cross-entropy with labels.
    """
    # Soft targets from teacher
    soft_targets = torch.softmax(teacher_logits / temperature, dim=-1)
    soft_student = torch.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss (scaled by temperature^2)
    distillation_loss = torch.nn.functional.kl_div(
        soft_student, soft_targets, reduction='batchmean'
    ) * (temperature ** 2)

    # Standard cross-entropy loss
    student_loss = torch.nn.functional.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * distillation_loss + (1 - alpha) * student_loss

# Training loop
teacher_model.eval()
student_model.train()

for batch in dataloader:
    inputs, labels = batch

    # Get teacher predictions (no gradients)
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)

    # Train student
    student_logits = student_model(inputs)
    loss = distillation_loss(student_logits, teacher_logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Transfer Learning & Fine-Tuning

### Hugging Face Transformers

**Fine-Tuning Pre-trained Models:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Freeze early layers, only fine-tune classifier
for param in model.bert.encoder.layer[:8].parameters():
    param.requires_grad = False

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Mixed precision
)

# Trainer handles training loop
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

### LoRA for Efficient Fine-Tuning

**Low-Rank Adaptation:**
```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of update matrices
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none"
)

# Wrap model with LoRA
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(model, lora_config)

# Only 0.1% of parameters are trainable!
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.237
```

---

## Quick Reference

### Framework Comparison

| Feature | PyTorch | TensorFlow | JAX |
|---------|---------|------------|-----|
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Production | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Research | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Ecosystem | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Distributed Training Options

| Method | Use Case | Memory Efficiency | Complexity |
|--------|----------|-------------------|------------|
| DDP | Multi-GPU, same node | Low | Easy |
| DeepSpeed ZeRO-2 | Large models, multi-node | High | Medium |
| FSDP | Very large models | Very High | Medium |
| Model Parallel | Model doesn't fit single GPU | High | Hard |

### Model Optimization Techniques

| Technique | Speed Up | Size Reduction | Accuracy Impact |
|-----------|----------|----------------|-----------------|
| Quantization (INT8) | 2-4x | 4x | <1% |
| Pruning (50%) | 1.5-2x | 2x | <2% |
| Distillation | Varies | Varies | 1-5% |
| Mixed Precision | 2-3x | None | <0.1% |

---

*Build sophisticated ML systems with modern frameworks, distributed training, and production-ready optimizations.*
