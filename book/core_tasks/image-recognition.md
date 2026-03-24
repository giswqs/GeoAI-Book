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
    push_classifier_to_hub,
    predict_images_from_hub,
)
```

## Download the EuroSAT RGB Dataset

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/EuroSAT-RGB.zip"
data_dir = download_file(url)
```

```{code-cell} ipython3
print(f"Dataset directory: {data_dir}")
print(f"Files: {sorted(os.listdir(data_dir))}")
```

## Explore the Dataset

```{code-cell} ipython3
dataset_info = load_image_dataset(data_dir)
print(f"Classes ({len(dataset_info['class_names'])}): {dataset_info['class_names']}")
print(f"Total images: {len(dataset_info['image_paths'])}")
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

## Train a ResNet-50 Classifier

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
import random

test_dataset = result["test_dataset"]
test_paths = test_dataset.image_paths
test_labels = test_dataset.labels
n_samples = min(10, len(test_paths))

rng = random.Random(42)
sample_indices = rng.sample(range(len(test_paths)), k=n_samples)
sample_paths = [test_paths[i] for i in sample_indices]
sample_labels = [test_labels[i] for i in sample_indices]

pred_result = predict_images(
    model=result["model"],
    image_paths=sample_paths,
    class_names=result["class_names"],
    image_size=64,
    in_channels=3,
)

fig = plot_predictions(
    image_paths=sample_paths,
    predictions=pred_result["predictions"],
    true_labels=sample_labels,
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
fig = plot_training_history("image_recognition_output/efficientnet_b0/models")
plt.show()
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
print(f"ResNet50 accuracy:        {eval_result['accuracy']:.4f}")
print(f"EfficientNet-B0 accuracy: {eval_effnet['accuracy']:.4f}")
```

## Publish and Reuse Models

### Authenticate with Hugging Face

```{code-cell} ipython3
from huggingface_hub import notebook_login

notebook_login()
```

### Push the Trained Model to the Hub

```{code-cell} ipython3
repo_url = push_classifier_to_hub(
    model_path=result["checkpoint_path"],
    repo_id="your-username/eurosat-resnet50",
    model_name="resnet50",
    num_classes=len(result["class_names"]),
    in_channels=3,
    class_names=result["class_names"],
    commit_message="EuroSAT ResNet-50 classifier trained for 5 epochs",
)
print(repo_url)
```

### Run Inference from the Hub

```{code-cell} ipython3
n_samples = 10
hub_result = predict_images_from_hub(
    image_paths=test_paths[:n_samples],
    repo_id="your-username/eurosat-resnet50",
    image_size=64,
)

fig = plot_predictions(
    image_paths=test_paths[:n_samples],
    predictions=hub_result["predictions"],
    true_labels=test_labels[:n_samples],
    class_names=hub_result["class_names"],
    probabilities=hub_result["probabilities"],
)
plt.show()
```

## Key Takeaways

## Exercises

### Exercise 1: Train a Multi-Architecture Comparison

```{code-cell} ipython3

```

### Exercise 2: Experiment with Transfer Learning Strategies

```{code-cell} ipython3

```

### Exercise 3: Evaluate on a Custom Dataset

```{code-cell} ipython3

```

### Exercise 4: Hyperparameter Sensitivity Analysis

```{code-cell} ipython3

```
