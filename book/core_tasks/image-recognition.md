---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Image Recognition

## Introduction

## Learning Objectives

## Understanding Image Classification

### From Pixels to Labels

### How Convolutional Neural Networks Work

### Transfer Learning

## Classification Architectures

### ResNet

### EfficientNet

### Vision Transformers

### ConvNeXt

### Choosing an Architecture

## Preparing Data for Classification

### ImageFolder Structure

### The EuroSAT Dataset

## Install Packages

```{code-cell} ipython3
# %pip install geoai-py
```

## Import Libraries

```{code-cell} ipython3
import os
from geoai.utils import download_file
from geoai.recognize import (
    load_image_dataset,
    train_image_classifier,
    predict_images,
    evaluate_classifier,
    plot_training_history,
    plot_confusion_matrix,
    plot_predictions,
)
```

## Download the EuroSAT RGB Dataset

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/EuroSAT-RGB.zip"
download_dir = download_file(url)
```

```{code-cell} ipython3
data_dir = os.path.join(download_dir, "EuroSAT_RGB")
print(f"Dataset directory: {data_dir}")
print(f"Classes: {sorted(os.listdir(data_dir))}")
```

## Explore the Dataset

```{code-cell} ipython3
dataset_info = load_image_dataset(data_dir)
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
from PIL import Image

class_names = dataset_info["class_names"]
image_paths = dataset_info["image_paths"]
labels = dataset_info["labels"]
class_to_idx = dataset_info["class_to_idx"]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, class_name in enumerate(class_names):
    ax = axes[idx // 5, idx % 5]
    # Find first image of this class
    img_idx = labels.index(class_to_idx[class_name])
    img = Image.open(image_paths[img_idx])
    ax.imshow(img)
    ax.set_title(class_name, fontsize=12)
    ax.axis("off")

plt.suptitle("Sample Image from Each Class", fontsize=14)
plt.tight_layout()
plt.show()
```

## Train a ResNet50 Classifier

```{code-cell} ipython3
result = train_image_classifier(
    data_dir=data_dir,
    model_name="resnet50",
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    image_size=64,
    in_channels=3,
    pretrained=True,
    output_dir="image_recognition_output/resnet50",
    num_workers=4,
    seed=42,
)
```

## Plot Training History

```{code-cell} ipython3
fig = plot_training_history("image_recognition_output/resnet50/models")
plt.show()
```

## Evaluate on Test Set

```{code-cell} ipython3
eval_result = evaluate_classifier(
    model=result["model"],
    dataset=result["test_dataset"],
    class_names=result["class_names"],
)
```

## Plot Confusion Matrix

```{code-cell} ipython3
fig = plot_confusion_matrix(
    eval_result["confusion_matrix"],
    result["class_names"],
)
plt.show()
```

```{code-cell} ipython3
fig = plot_confusion_matrix(
    eval_result["confusion_matrix"],
    result["class_names"],
    normalize=True,
)
plt.show()
```

## Visualize Predictions

```{code-cell} ipython3
test_dataset = result["test_dataset"]
test_paths = test_dataset.image_paths
test_labels = test_dataset.labels

pred_result = predict_images(
    model=result["model"],
    image_paths=test_paths[:20],
    class_names=result["class_names"],
    image_size=64,
    in_channels=3,
)

fig = plot_predictions(
    image_paths=test_paths[:20],
    predictions=pred_result["predictions"],
    true_labels=test_labels[:20],
    class_names=result["class_names"],
    probabilities=pred_result["probabilities"],
)
plt.show()
```

## Train an EfficientNet-B0 Classifier

```{code-cell} ipython3
result_effnet = train_image_classifier(
    data_dir=data_dir,
    model_name="efficientnet_b0",
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    image_size=64,
    in_channels=3,
    pretrained=True,
    output_dir="image_recognition_output/efficientnet_b0",
    num_workers=4,
    seed=42,
)
```

```{code-cell} ipython3
eval_effnet = evaluate_classifier(
    model=result_effnet["model"],
    dataset=result_effnet["test_dataset"],
    class_names=result_effnet["class_names"],
)
```

```{code-cell} ipython3
fig = plot_confusion_matrix(
    eval_effnet["confusion_matrix"],
    result_effnet["class_names"],
    normalize=True,
)
plt.show()
```

## Compare Results

```{code-cell} ipython3
print(f"ResNet50 accuracy:       {eval_result['accuracy']:.4f}")
print(f"EfficientNet-B0 accuracy: {eval_effnet['accuracy']:.4f}")
```

## Key Takeaways

## Exercises

### Exercise 1: Train a Multi-Architecture Comparison

### Exercise 2: Experiment with Transfer Learning Strategies

### Exercise 3: Evaluate on a Custom Dataset

### Exercise 4: Hyperparameter Sensitivity Analysis

### Exercise 5: Multi-Band Classification with GeoTIFF
