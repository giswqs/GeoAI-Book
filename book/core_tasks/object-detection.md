---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: geo
  language: python
  name: python3
---
# Object Detection

## Introduction

## Learning Objectives

## Understanding Object Detection

### Classification vs. Detection

### Key Concepts

## Detection Architectures

### Two-Stage Detectors

### Single-Stage Detectors

### Transformer-Based Detectors

### Zero-Shot Detection

### Choosing an Architecture

## Preparing Detection Datasets

### Annotation Formats

### The NWPU-VHR-10 Dataset

## Evaluating Detection Results

### Mean Average Precision (mAP)

### Precision-Recall Curves

### IoU Thresholds

## Import Libraries

```{code-cell} ipython3
import os
import json

import geoai
```

## Download the NWPU-VHR-10 Dataset

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/NWPU-VHR-10.zip"
data_dir = geoai.download_file(url)
```

```{code-cell} ipython3
print(f"Dataset directory: {data_dir}")
print(f"Contents: {os.listdir(data_dir)}")
```

## Explore the Dataset

```{code-cell} ipython3
print(f"\nNWPU-VHR-10 Classes:")
for i, name in enumerate(geoai.NWPU_VHR10_CLASSES):
    print(f"  {i}: {name}")
```

## Prepare the Dataset

```{code-cell} ipython3
splits = geoai.prepare_nwpu_vhr10(data_dir, val_split=0.2, seed=42)
```

```{code-cell} ipython3
print(f"Images directory: {splits['images_dir']}")
print(f"Number of classes: {splits['num_classes']}")
print(f"Class names: {splits['class_names']}")
print(f"Training images: {len(splits['train_image_ids'])}")
print(f"Validation images: {len(splits['val_image_ids'])}")
```

## Visualize Sample Annotations

```{code-cell} ipython3
geoai.visualize_coco_annotations(
    annotations_path=splits["annotations_path"],
    images_dir=splits["images_dir"],
    num_samples=6,
    random=True,
    seed=1,
    cols=3,
    figsize=(12, 6),
)
```

## Train a Multi-Class Detection Model

```{code-cell} ipython3
output_dir = "nwpu_output"

model_path = geoai.train_multiclass_detector(
    images_dir=splits["images_dir"],
    annotations_path=splits["train_annotations"],
    output_dir=output_dir,
    model_name="fasterrcnn_resnet50_fpn_v2",
    class_names=splits["class_names"],
    num_channels=3,
    batch_size=4,
    num_epochs=10,
    learning_rate=0.005,
    val_split=0.1,
    seed=42,
    pretrained=True,
    verbose=True,
)
```

## Plot Training Metrics

```{code-cell} ipython3
geoai.plot_detection_training_history(
    history_path=os.path.join(output_dir, "training_history.pth"),
)
```

## Evaluate with COCO Metrics

```{code-cell} ipython3
metrics = geoai.evaluate_multiclass_detector(
    model_path=model_path,
    images_dir=splits["images_dir"],
    annotations_path=splits["val_annotations"],
    num_classes=splits["num_classes"],
    class_names=splits["class_names"][1:],  # Exclude background
    batch_size=4,
)
```

## Run Inference on Sample Images

```{code-cell} ipython3
# Load validation data to pick a test image
with open(splits["val_annotations"], "r") as f:
    val_data = json.load(f)

test_img_info = val_data["images"][0]
test_img_path = os.path.join(splits["images_dir"], test_img_info["file_name"])
print(f"Test image: {test_img_path}")
```

```{code-cell} ipython3
output_raster = "nwpu_detection_output.tif"

result_path, inference_time, detections = geoai.multiclass_detection(
    input_path=test_img_path,
    output_path=output_raster,
    model_path=model_path,
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
    window_size=512,
    overlap=256,
    confidence_threshold=0.5,
    batch_size=4,
    num_channels=3,
)

print(f"\nInference time: {inference_time:.2f}s")
print(f"Total detections: {len(detections)}")
```

## Visualize Detections

```{code-cell} ipython3
geoai.visualize_multiclass_detections(
    image_path=test_img_path,
    detections=detections,
    class_names=splits["class_names"],
    confidence_threshold=0.5,
    figsize=(12, 10),
)
```

## Batch Inference on Multiple Images

```{code-cell} ipython3
val_image_paths = [
    os.path.join(splits["images_dir"], img["file_name"])
    for img in val_data["images"][:4]
]

results = geoai.batch_multiclass_detection(
    image_paths=val_image_paths,
    output_dir="nwpu_batch_output",
    model_path=model_path,
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
    confidence_threshold=0.5,
    num_channels=3,
    figsize=(16, 12),
)
```

## Publish and Reuse Models

### Push to Hugging Face Hub

```{code-cell} ipython3
from huggingface_hub import notebook_login

notebook_login()
```

```{code-cell} ipython3
url = geoai.push_detector_to_hub(
    model_path=model_path,
    repo_id="your-username/nwpu-vhr10-fasterrcnn",
    model_name="fasterrcnn_resnet50_fpn_v2",
    num_classes=splits["num_classes"],
    class_names=splits["class_names"],
)
```

### Run Inference from Hub

```{code-cell} ipython3
sample_img_path = os.path.join(splits["images_dir"], "608.jpg")

result_path, inference_time, detections = geoai.predict_detector_from_hub(
    input_path=sample_img_path,
    output_path="hub_detection.tif",
    repo_id="giswqs/nwpu-vhr10-fasterrcnn",
    confidence_threshold=0.5,
)

print(f"Inference time: {inference_time:.2f}s")
print(f"Total detections: {len(detections)}")

# Clean up
if os.path.exists("hub_detection.tif"):
    os.remove("hub_detection.tif")
```

```{code-cell} ipython3
geoai.visualize_multiclass_detections(
    image_path=sample_img_path,
    detections=detections,
    class_names=geoai.NWPU_VHR10_CLASSES,
    confidence_threshold=0.5,
    figsize=(12, 10),
)
```

## Key Takeaways

## Exercises

### Exercise 1: Training with a Different Architecture

```{code-cell} ipython3

```

### Exercise 2: Confidence Threshold Analysis

```{code-cell} ipython3

```

### Exercise 3: Hyperparameter Sensitivity

```{code-cell} ipython3

```
