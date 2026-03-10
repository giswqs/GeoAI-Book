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
# Semantic Segmentation

## Introduction

## Learning Objectives

## Understanding Semantic Segmentation

### From Classification to Segmentation

### How Segmentation Models Work

## Segmentation Architectures

### U-Net

### DeepLabV3+

### SegFormer

### Feature Pyramid Network (FPN)

### Choosing an Architecture

## Preparing Data for Segmentation

### Dataset Structure

### The SegmentationDataset Class

```{code-cell} python
import geoai

# The SegmentationDataset loads GeoTIFF image-mask pairs
# image_paths: list of paths to image tiles
# mask_paths: list of paths to corresponding mask tiles
# num_channels: number of input bands to use (e.g., 3 for RGB)
```

## Training a Segmentation Model

### Configuring the Model

### Training Loop

### Monitoring Training

### Using geoai for Training

```{code-cell} python
import geoai

# Directories containing image tiles and corresponding label masks
images_dir = "building_tiles/images"
labels_dir = "building_tiles/labels"
output_dir = "building_model"

# Train a U-Net model with ResNet-34 encoder
geoai.train_segmentation_model(
    images_dir=images_dir,
    labels_dir=labels_dir,
    output_dir=output_dir,
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=3,
    num_classes=2,
    batch_size=8,
    num_epochs=2,         # Use more epochs (e.g., 50) for real training
    learning_rate=0.001,
    val_split=0.2,
    loss_function="crossentropy",
    verbose=True,
)
```

```{code-cell} python
# Train a DeepLabV3+ model with ResNet-50 encoder
geoai.train_segmentation_model(
    images_dir=images_dir,
    labels_dir=labels_dir,
    output_dir="building_model_deeplabv3",
    architecture="deeplabv3plus",
    encoder_name="resnet50",
    num_classes=2,
    num_epochs=2,
    loss_function="focal",  # Use focal loss for imbalanced data
)
```

## Loss Functions for Segmentation

### Cross-Entropy Loss

### Dice Loss

### Focal Loss

```{code-cell} python
# Focal loss is especially useful when one class dominates the image
geoai.train_segmentation_model(
    images_dir=images_dir,
    labels_dir=labels_dir,
    output_dir="building_model_focal",
    architecture="unet",
    encoder_name="resnet34",
    num_classes=2,
    num_epochs=2,
    loss_function="focal",
    focal_gamma=2.0,     # Higher gamma = more focus on hard pixels
)
```

### Combined Losses

## Evaluating Segmentation Results

### Metrics

### Confusion Matrix

```{code-cell} python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming y_true and y_pred are flattened arrays of pixel labels
# y_true = ground_truth_mask.flatten()
# y_pred = predicted_mask.flatten()

# cm = confusion_matrix(y_true, y_pred)
# print(classification_report(y_true, y_pred, target_names=["background", "building"]))
```

### Visual Inspection

```{code-cell} python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Create a side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0]: Original image
# axes[1]: Ground truth mask
# axes[2]: Predicted mask

# Example with placeholder data
img = np.random.rand(256, 256, 3)
mask_true = np.random.randint(0, 2, (256, 256))
mask_pred = np.random.randint(0, 2, (256, 256))

axes[0].imshow(img)
axes[0].set_title("NAIP Image")
axes[0].axis("off")

axes[1].imshow(mask_true, cmap="gray")
axes[1].set_title("Ground Truth")
axes[1].axis("off")

axes[2].imshow(mask_pred, cmap="gray")
axes[2].set_title("Prediction")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

## Running Inference

### Predicting on New Images

```{code-cell} python
import geoai

# Run inference on a new image
geoai.semantic_segmentation(
    input_path="naip_test.tif",
    output_path="prediction.tif",
    model_path="building_model/best_model.pth",
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
)
```

### Handling Large Images with Sliding Window

### Generating Prediction Maps

```{code-cell} python
import leafmap

# Visualize the prediction map on an interactive map
m = leafmap.Map()
m.add_basemap("Esri.WorldImagery")
# m.add_raster("prediction.tif", layer_name="Building Predictions")
m
```

## Land Cover Classification

### Understanding Land Cover Classification Schemes

### Spectral Indices

```{code-cell} ipython3
import geoai
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/s2-minnesota-2025-08-31-subset.tif"
s2_path = geoai.download_file(url)
```

```{code-cell} ipython3
ds = rioxarray.open_rasterio(s2_path)
nir = ds[3].astype(float)
red = ds[0].astype(float)
green = ds[1].astype(float)
swir = ds[4].astype(float) if ds.shape[0] > 4 else None

ndvi = (nir - red) / (nir + red + 1e-10)
ndvi.plot(cmap="RdYlGn", vmin=-1, vmax=1)
plt.title("NDVI")
plt.show()
```

```{code-cell} ipython3
ndwi = (green - nir) / (green + nir + 1e-10)
ndwi.plot(cmap="RdYlBu", vmin=-1, vmax=1)
plt.title("NDWI")
plt.show()
```

```{code-cell} ipython3
if swir is not None:
    ndbi = (swir - nir) / (swir + nir + 1e-10)
    ndbi.plot(cmap="OrRd", vmin=-1, vmax=1)
    plt.title("NDBI")
    plt.show()
```

### Preparing Multi-Class Training Data

```{code-cell} ipython3
image_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/m_3807511_ne_18_060_20181104.tif"
mask_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/m_3807511_ne_18_060_20181104_landcover.tif"

image_path = geoai.download_file(image_url)
mask_path = geoai.download_file(mask_url)
```

```{code-cell} ipython3
geoai.view_raster(image_path)
```

```{code-cell} ipython3
geoai.view_raster(mask_path)
```

```{code-cell} ipython3
geoai.export_landcover_tiles(
    image=image_path,
    mask=mask_path,
    output_dir="landcover_tiles",
    tile_size=256,
)
```

```{code-cell} ipython3
class_weights = geoai.compute_class_weights(mask_dir="landcover_tiles/masks")
print("Class weights:", class_weights)
```

### Training a Land Cover Model

```{code-cell} ipython3
loss_fn = geoai.get_landcover_loss_function(
    loss_type="focal",
    class_weights=class_weights,
)
print("Loss function:", loss_fn)
```

```{code-cell} ipython3
geoai.train_segmentation_landcover(
    image_path=image_path,
    mask_path=mask_path,
    output_dir="landcover_model",
    encoder_name="resnet50",
    num_classes=len(class_weights),
    loss_function=loss_fn,
    epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    tile_size=256,
    val_split=0.2,
    seed=42,
)
```

```{code-cell} ipython3
geoai.plot_training_history(log_dir="landcover_model")
```

### Evaluating Classification Results

```{code-cell} ipython3
predictions = geoai.predict_segmentation(
    model="landcover_model",
    raster=image_path,
)
```

```{code-cell} ipython3
geoai.plot_prediction_comparison(pred=predictions, true=mask_path)
```

```{code-cell} ipython3
from sklearn.metrics import classification_report, confusion_matrix
import rioxarray

true_mask = rioxarray.open_rasterio(mask_path).values.flatten()
pred_flat = predictions.flatten()

print(classification_report(true_mask, pred_flat))
```

```{code-cell} ipython3
cm = confusion_matrix(true_mask, pred_flat)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
plt.colorbar(im)
plt.tight_layout()
plt.show()
```

### Post-Processing and Map Generation

```{code-cell} ipython3
cleaned = geoai.clean_segmentation_mask(
    mask=predictions,
    min_area=100,
)
```

```{code-cell} ipython3
geoai.plot_prediction_comparison(pred=cleaned, true=mask_path)
```

### Multi-Temporal Classification

```{code-cell} ipython3
zip_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/landcover-sample-data.zip"
sample_dir = geoai.download_file(zip_url)
```

## Key Takeaways

## Exercises

### Exercise 1: Train a Building Segmentation Model

### Exercise 2: Compare Segmentation Architectures

### Exercise 3: Experiment with Loss Functions

### Exercise 4: Evaluate and Visualize Predictions

### Exercise 5: Sliding Window Inference on a Large Image

### Exercise 6: Land Cover Loss Function Comparison

### Exercise 7: Post-Processing Parameter Tuning

### Exercise 8: Multi-Temporal Feature Engineering
