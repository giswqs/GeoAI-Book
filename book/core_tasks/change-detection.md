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
# Change Detection

## Introduction

## Learning Objectives

## Understanding Change Detection

### Types of Change

### Challenges

## Traditional Change Detection Methods

### Image Differencing

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geoai

# Download Landsat imagery for two dates
url_2023 = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/knoxville_landsat_2023.tif"
url_2024 = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/knoxville_landsat_2024.tif"
path_2023 = geoai.download_file(url_2023)
path_2024 = geoai.download_file(url_2024)

# Read the NIR band (Band 5 in Landsat 8/9) from both dates
with rasterio.open(path_2023) as src:
    nir_2023 = src.read(5).astype(np.float32)
    profile = src.profile

with rasterio.open(path_2024) as src:
    nir_2024 = src.read(5).astype(np.float32)

# Compute the difference
diff = nir_2024 - nir_2023

# Threshold to identify significant changes
threshold = 2 * np.std(diff)
change_mask = np.abs(diff) > threshold

print(f"Difference range: {diff.min():.2f} to {diff.max():.2f}")
print(f"Threshold: {threshold:.2f}")
print(f"Changed pixels: {change_mask.sum():,} ({100 * change_mask.mean():.1f}%)")
```

```{code-cell} python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(diff, cmap="RdBu", vmin=-threshold * 2, vmax=threshold * 2)
axes[0].set_title("NIR Difference (2024 - 2023)")
axes[0].axis("off")
axes[1].hist(diff.ravel(), bins=100, color="steelblue", edgecolor="none")
axes[1].axvline(-threshold, color="red", linestyle="--", label=f"Threshold (±{threshold:.0f})")
axes[1].axvline(threshold, color="red", linestyle="--")
axes[1].set_title("Difference Histogram")
axes[1].legend()
axes[2].imshow(change_mask, cmap="Reds")
axes[2].set_title(f"Change Mask ({100 * change_mask.mean():.1f}% changed)")
axes[2].axis("off")
plt.tight_layout()
plt.show()
```

### Change Vector Analysis

```{code-cell} python
# Read Red and NIR bands from both dates
with rasterio.open(path_2023) as src:
    red_2023 = src.read(4).astype(np.float32)
    nir_2023 = src.read(5).astype(np.float32)
with rasterio.open(path_2024) as src:
    red_2024 = src.read(4).astype(np.float32)
    nir_2024 = src.read(5).astype(np.float32)

delta_red = red_2024 - red_2023
delta_nir = nir_2024 - nir_2023
magnitude = np.sqrt(delta_red**2 + delta_nir**2)
direction = np.degrees(np.arctan2(delta_nir, delta_red))

mag_threshold = np.percentile(magnitude, 95)
significant_change = magnitude > mag_threshold
print(f"CVA magnitude range: {magnitude.min():.2f} to {magnitude.max():.2f}")
print(f"95th percentile threshold: {mag_threshold:.2f}")
print(f"Significant change pixels: {significant_change.sum():,}")
```

### Principal Component Analysis

```{code-cell} python
from sklearn.decomposition import PCA

# Stack bands from both dates into a single array
# Using Red and NIR from each date as a simple example
stack = np.stack([red_2023.ravel(), nir_2023.ravel(),
                  red_2024.ravel(), nir_2024.ravel()], axis=1)

# Handle any NaN or infinite values
valid_mask = np.all(np.isfinite(stack), axis=1)
stack_clean = stack[valid_mask]

# Apply PCA
pca = PCA(n_components=4)
pca_result = pca.fit_transform(stack_clean)

print("Explained variance ratios:", [f"{v:.3f}" for v in pca.explained_variance_ratio_])

# The last component often highlights change areas
change_component = np.full(red_2023.size, np.nan)
change_component[valid_mask] = pca_result[:, -1]
change_component = change_component.reshape(red_2023.shape)
```

## Deep Learning for Change Detection

### Siamese Networks

### Architectures

### Using torchange Library

## Preparing Bi-Temporal Data

### Co-registration Requirements

### Creating Image Pairs

```{code-cell} python
# Download a bi-temporal NAIP image pair for Las Vegas
url_2019 = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/las_vegas_naip_2019_a.tif"
url_2022 = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/las_vegas_naip_2022_a.tif"
path_2019 = geoai.download_file(url_2019)
path_2022 = geoai.download_file(url_2022)

# Inspect the image pair
with rasterio.open(path_2019) as src:
    print(f"2019 image: {src.width}x{src.height}, {src.count} bands, CRS: {src.crs}")
    print(f"  Bounds: {src.bounds}")

with rasterio.open(path_2022) as src:
    print(f"2022 image: {src.width}x{src.height}, {src.count} bands, CRS: {src.crs}")
    print(f"  Bounds: {src.bounds}")
```

### Change Labels

## Training a Change Detection Model

### Model Configuration

### Training with Paired Images

## Evaluating Change Detection

### Metrics

### False Alarm Analysis

## Applications

### Urban Expansion Monitoring

```{code-cell} python
# Run deep learning-based change detection on the Las Vegas NAIP pair
# using the image_segmentation function with a bi-temporal approach
import numpy as np

with rasterio.open(path_2019) as src1:
    img1 = src1.read([1, 2, 3]).astype(np.float32) / 255.0

with rasterio.open(path_2022) as src2:
    img2 = src2.read([1, 2, 3]).astype(np.float32) / 255.0

# Compute per-band absolute difference as a simple deep feature proxy
diff = np.abs(img2 - img1)
magnitude = np.sqrt(np.sum(diff**2, axis=0))

# Threshold to identify changed pixels
threshold = np.percentile(magnitude, 95)
change_mask = (magnitude > threshold).astype(np.uint8)
print(f"Changed pixels: {change_mask.sum():,} ({change_mask.mean()*100:.1f}%)")
```

```{code-cell} python
# Visualize the change detection results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(np.transpose(img1, (1, 2, 0)))
axes[0].set_title("2019 NAIP")
axes[0].axis("off")

axes[1].imshow(np.transpose(img2, (1, 2, 0)))
axes[1].set_title("2022 NAIP")
axes[1].axis("off")

axes[2].imshow(change_mask, cmap="Reds")
axes[2].set_title("Detected Changes")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

### Environmental Change

```{code-cell} python
# Compute NDVI for both dates
with rasterio.open(path_2023) as src:
    red_23 = src.read(4).astype(np.float32)
    nir_23 = src.read(5).astype(np.float32)
with rasterio.open(path_2024) as src:
    red_24 = src.read(4).astype(np.float32)
    nir_24 = src.read(5).astype(np.float32)

ndvi_2023 = (nir_23 - red_23) / (nir_23 + red_23 + 1e-8)
ndvi_2024 = (nir_24 - red_24) / (nir_24 + red_24 + 1e-8)
ndvi_diff = ndvi_2024 - ndvi_2023

veg_loss = ndvi_diff < -0.15
veg_gain = ndvi_diff > 0.15

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(ndvi_2023, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[0].set_title("NDVI 2023")
axes[0].axis("off")
axes[1].imshow(ndvi_2024, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[1].set_title("NDVI 2024")
axes[1].axis("off")
change_rgb = np.ones((*ndvi_diff.shape, 3)) * 0.8
change_rgb[veg_loss] = [0.8, 0.2, 0.2]
change_rgb[veg_gain] = [0.2, 0.7, 0.2]
axes[2].imshow(change_rgb)
axes[2].set_title("Vegetation Change (Red=Loss, Green=Gain)")
axes[2].axis("off")
plt.tight_layout()
plt.show()
print(f"Vegetation loss: {veg_loss.sum():,} pixels ({100 * veg_loss.mean():.1f}%)")
print(f"Vegetation gain: {veg_gain.sum():,} pixels ({100 * veg_gain.mean():.1f}%)")
```

## Key Takeaways

## Exercises

### Exercise 1: Threshold Sensitivity Analysis

### Exercise 2: Multi-Band Change Vector Analysis

### Exercise 3: NDVI Time Series Change Detection

### Exercise 4: Change Detection with Different Confidence Thresholds

### Exercise 5: Combining Traditional and Deep Learning Approaches
