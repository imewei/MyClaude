---
name: computer-vision
description: "Implement computer vision pipelines with OpenCV, torchvision, and Hugging Face Vision Transformers including object detection (YOLO, DETR), image segmentation, feature extraction, and video analysis. Use when building CV models, processing images, or implementing detection/segmentation pipelines."
---

# Computer Vision

Build end-to-end computer vision pipelines for detection, segmentation, and classification.

## Expert Agent

For designing and training computer vision models, delegate to the expert agent:

- **`neural-network-master`**: Deep learning specialist for CNN architectures, vision transformers, and training pipelines.
  - *Location*: `plugins/science-suite/agents/neural-network-master.md`
  - *Capabilities*: Architecture design, transfer learning, distributed training, model optimization.

## Image Preprocessing Pipeline

```python
import cv2
import numpy as np
from torchvision import transforms

def build_preprocessing(target_size: tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Standard preprocessing for vision models."""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def load_and_preprocess(path: str) -> np.ndarray:
    """Load image with OpenCV, convert BGR->RGB, apply CLAHE."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

## CNN Architecture Patterns

| Architecture | Use Case | Params | Top-1 Acc |
|-------------|----------|--------|-----------|
| ResNet-50 | Baseline classification | 25M | 76.1% |
| EfficientNet-B4 | Balanced accuracy/speed | 19M | 82.9% |
| ConvNeXt-T | Modern CNN baseline | 29M | 82.1% |
| ViT-B/16 | Large-scale classification | 86M | 84.5% |

## Object Detection with YOLO

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train on custom dataset
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    augment=True,
)

# Inference
detections = model.predict("image.jpg", conf=0.25, iou=0.45)
for det in detections[0].boxes:
    cls_id = int(det.cls)
    conf = float(det.conf)
    x1, y1, x2, y2 = det.xyxy[0].tolist()
```

## DETR: End-to-End Object Detection

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("scene.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process: threshold=0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]
```

## Semantic Segmentation

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # (B, num_classes, H/4, W/4)
pred = logits.argmax(dim=1)  # (B, H/4, W/4)
```

## Data Augmentation Strategy

```python
import albumentations as A

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# For detection: use BboxParams
det_transform = A.Compose([
    A.RandomResizedCrop(640, 640),
    A.HorizontalFlip(p=0.5),
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
```

## Transfer Learning Checklist

- [ ] Start with pretrained backbone (ImageNet or COCO)
- [ ] Freeze backbone, train head for 5-10 epochs
- [ ] Unfreeze with discriminative learning rates (backbone: 1e-5, head: 1e-3)
- [ ] Use cosine annealing or OneCycleLR scheduler
- [ ] Monitor validation mAP/IoU, not just loss
- [ ] Apply test-time augmentation (TTA) for final evaluation

## Evaluation Metrics

| Task | Primary Metric | Formula |
|------|---------------|---------|
| Classification | Top-1 Accuracy | correct / total |
| Detection | mAP@[0.5:0.95] | mean AP across IoU thresholds |
| Segmentation | mIoU | mean(TP / (TP + FP + FN)) per class |
| Instance Seg | Mask AP | AP using mask IoU |

## Video Analysis Pattern

```python
def process_video(path: str, model, frame_skip: int = 3):
    """Process video with detection model, skipping frames for speed."""
    cap = cv2.VideoCapture(path)
    frame_idx = 0
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            detections = model.predict(frame, verbose=False)
            results.append((frame_idx, detections))
        frame_idx += 1
    cap.release()
    return results
```
