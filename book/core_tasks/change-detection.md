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
# Change Detection

## Introduction

## Learning Objectives

## Understanding Change Detection

### Types of Change

### Challenges

## Traditional Change Detection Methods

### Image Differencing

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geoai

# Download Landsat imagery for two dates
url_2023 = "https://data.source.coop/opengeos/geoai/knoxville_landsat_2023.tif"
url_2024 = "https://data.source.coop/opengeos/geoai/knoxville_landsat_2024.tif"
path_2023 = geoai.download_file(url_2023)
path_2024 = geoai.download_file(url_2024)

# Read the NIR band (Band 5 in Landsat 8/9) from both dates
with rasterio.open(path_2023) as src:
    nir_2023 = src.read(5).astype(np.float32)

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

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(diff, cmap="RdBu", vmin=-threshold * 2, vmax=threshold * 2)
axes[0].set_title("NIR Difference (2024 - 2023)")
axes[0].axis("off")
axes[1].hist(diff.ravel(), bins=100, color="steelblue", edgecolor="none")
axes[1].axvline(-threshold, color="red", linestyle="--", label=f"Threshold (±{threshold:.2f})")
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

```{code-cell} ipython3
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

## Deep Learning for Change Detection

### Siamese Networks

### ChangeStar

## Import Libraries

```{code-cell} ipython3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import geoai
from geoai.change_detection import (
    ChangeStarDetection,
    changestar_detect,
    list_changestar_models,
)
```

## List Available Models

```{code-cell} ipython3
models = list_changestar_models()
for short_name, full_name in models.items():
    print(f"{short_name:30s} -> {full_name}")
```

## Setup

```{code-cell} ipython3
device = geoai.get_device()
print(f"Using device: {device}")

out_folder = "changestar_results"
Path(out_folder).mkdir(exist_ok=True)
print(f"Output directory: {out_folder}")
```

## Download Sample Data

```{code-cell} ipython3
naip_2019_url = "https://data.source.coop/opengeos/geoai/las_vegas_naip_2019_a.tif"
naip_2022_url = "https://data.source.coop/opengeos/geoai/las_vegas_naip_2022_a.tif"

naip_2019_path = geoai.download_file(naip_2019_url)
naip_2022_path = geoai.download_file(naip_2022_url)

print(f"Downloaded 2019 NAIP: {naip_2019_path}")
print(f"Downloaded 2022 NAIP: {naip_2022_path}")
```

## Visualize Input Imagery

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=naip_2019_path,
    right_layer=naip_2022_path,
    left_label="NAIP 2019",
    right_label="NAIP 2022",
)
```

## Initialize the ChangeStar Model

```{code-cell} ipython3
detector = ChangeStarDetection(model_name="s1_s1c1_vitb")
```

## Run Change Detection

```{code-cell} ipython3
result = detector.predict(
    naip_2019_path,
    naip_2022_path,
    output_change=os.path.join(out_folder, "change_map.tif"),
    output_t1_semantic=os.path.join(out_folder, "t1_buildings.tif"),
    output_t2_semantic=os.path.join(out_folder, "t2_buildings.tif"),
    output_vector=os.path.join(out_folder, "changes.gpkg"),
)
```

```{code-cell} ipython3
print("Result keys:", list(result.keys()))
for key, value in result.items():
    if hasattr(value, "shape"):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")
```

## Visualize Results

```{code-cell} ipython3
fig = detector.visualize(
    naip_2019_path,
    naip_2022_path,
    result=result,
    figsize=(25, 10),
    title1="NAIP 2019",
    title2="NAIP 2022",
)
plt.show()
```

### Overlay Visualization

```{code-cell} ipython3
fig = detector.visualize_overlay(
    naip_2019_path,
    naip_2022_path,
    result=result,
    figsize=(20, 6),
    title1="NAIP 2019",
    title2="NAIP 2022",
)
plt.show()
```

## Using the Convenience Function

```{code-cell} ipython3
result2 = changestar_detect(
    naip_2019_path,
    naip_2022_path,
    model_name="s1_s1c1_vitb",
    output_change=os.path.join(out_folder, "change_map_v2.tif"),
)

print(f"Change pixels: {result2['change_map'].sum():,}")
print(
    f"Changed area: {result2['change_map'].sum() / result2['change_map'].size * 100:.2f}%"
)
```

## Comparing Model Variants

```{code-cell} ipython3
result_s1 = result

detector_s9 = ChangeStarDetection(model_name="s9_s9c1_vitb")
result_s9 = detector_s9.predict(naip_2019_path, naip_2022_path)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(result_s1["change_map"], cmap="gray")
axes[0].set_title(f"S1 Model\n(Changed pixels: {result_s1['change_map'].sum():,})")
axes[0].axis("off")

axes[1].imshow(result_s9["change_map"], cmap="gray")
axes[1].set_title(f"S9 Model\n(Changed pixels: {result_s9['change_map'].sum():,})")
axes[1].axis("off")

combined = np.zeros((*result_s1["change_map"].shape, 3), dtype=np.uint8)
combined[result_s1["change_map"] == 1, 0] = 255  # S1 in red
combined[result_s9["change_map"] == 1, 2] = 255  # S9 in blue
both = (result_s1["change_map"] == 1) & (result_s9["change_map"] == 1)
combined[both] = [255, 0, 255]

axes[2].imshow(combined)
axes[2].set_title("Comparison\n(Red=S1 only, Blue=S9 only, Magenta=Both)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

## Adjusting the Threshold

```{code-cell} ipython3
thresholds = [0.3, 0.5, 0.7]
fig, axes = plt.subplots(1, len(thresholds), figsize=(18, 6))

for ax, thresh in zip(axes, thresholds):
    result_t = detector.predict(naip_2019_path, naip_2022_path, threshold=thresh)
    ax.imshow(result_t["change_map"], cmap="gray")
    ax.set_title(
        f"Threshold = {thresh}\n"
        f"(Changed pixels: {result_t['change_map'].sum():,})"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
```

## View Saved Outputs

```{code-cell} ipython3
for f in sorted(os.listdir(out_folder)):
    fpath = os.path.join(out_folder, f)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"{f:40s} {size_mb:.2f} MB")
```

```{code-cell} ipython3
change_map_path = os.path.join(out_folder, "change_map.tif")
geoai.view_raster(
    change_map_path,
    nodata=0,
    cmap="Reds",
    opacity=0.8,
    basemap=naip_2019_path,
    backend="ipyleaflet",
)
```

## Key Takeaways

## Exercises

### Exercise 1: Threshold Sensitivity Analysis

```{code-cell} ipython3

```

### Exercise 2: Comparing All Model Variants

```{code-cell} ipython3

```

### Exercise 3: Traditional vs. Deep Learning Comparison

```{code-cell} ipython3

```

### Exercise 4: Change Detection on a Different Image Pair

```{code-cell} ipython3

```

### Exercise 5: Vector Output Analysis

```{code-cell} ipython3

```
