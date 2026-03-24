---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Image Translation

## Introduction

## Learning Objectives

## Foundations of Image Translation

### What Is Image-to-Image Translation?

### Super-Resolution for Remote Sensing

### Latent Diffusion Models for Super-Resolution

## Import Libraries

```{code-cell} ipython3
import geoai
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
```

## Download Sample Data

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/S2C-MSIL2A-20250920T162001-Knoxville.tif"
s2_path = geoai.download_file(url)
```

### Inspecting the Input Data

```{code-cell} ipython3
with rio.open(s2_path) as src:
    print(f"Bands: {src.count}")
    print(f"Size: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res[0]:.2f} m")
    print(f"Dtype: {src.dtypes[0]}")
```

## Visualize the Input RGB Composite

```{code-cell} ipython3
with rio.open(s2_path) as src:
    rgb = src.read([1, 2, 3]).astype(np.float32)

for i in range(3):
    band = rgb[i]
    p2, p98 = np.percentile(band, (2, 98))
    rgb[i] = (band - p2) / (p98 - p2)
rgb = np.clip(rgb, 0, 1)

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb.transpose(1, 2, 0))
ax.set_title("Sentinel-2 RGB Composite (10 m)")
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

## Single-Patch Super-Resolution

```{code-cell} ipython3
sr_output = "sr_output.tif"
sr_image, _ = geoai.super_resolution(
    input_lr_path=s2_path,
    output_sr_path=sr_output,
    rgb_nir_bands=[1, 2, 3, 4],
    window=(700, 1300, 128, 128),
    sampling_steps=100,
)
```

```{code-cell} ipython3
print(f"Input shape:  (4, 128, 128) at 10 m")
print(f"Output shape: {sr_image.shape} at 2.5 m")
```

### Comparing Low-Resolution and Super-Resolution

```{code-cell} ipython3
geoai.plot_sr_comparison(s2_path, sr_output, bands=[1, 2, 3])
plt.show()
```

```{code-cell} ipython3
with rio.open(sr_output) as src:
    print(f"SR Bands: {src.count}")
    print(f"SR Size: {src.width} x {src.height}")
    print(f"SR CRS: {src.crs}")
    print(f"SR Resolution: {src.res[0]:.2f} m")
```

## Uncertainty Estimation

```{code-cell} ipython3
sr_unc_output = "sr_with_uncertainty.tif"
unc_output = "uncertainty.tif"
sr_image2, uncertainty = geoai.super_resolution(
    input_lr_path=s2_path,
    output_sr_path=sr_unc_output,
    output_uncertainty_path=unc_output,
    rgb_nir_bands=[1, 2, 3, 4],
    window=(700, 1300, 128, 128),
    compute_uncertainty=True,
    n_variations=5,
    sampling_steps=100,
)
```

### Visualizing the Uncertainty Map

```{code-cell} ipython3
geoai.plot_sr_uncertainty(unc_output)
plt.show()
```

## Tiled Inference for Larger Regions

```{code-cell} ipython3
sr_large = "sr_large.tif"
sr_large_img, _ = geoai.super_resolution(
    input_lr_path=s2_path,
    output_sr_path=sr_large,
    rgb_nir_bands=[1, 2, 3, 4],
    window=(700, 1300, 256, 256),
    patch_size=128,
    overlap=16,
    sampling_steps=100,
)
```

```{code-cell} ipython3
print(f"Input shape:  (4, 256, 256) at 10 m")
print(f"Output shape: {sr_large_img.shape} at 2.5 m")
```

### Comparing Results for the Larger Region

```{code-cell} ipython3
geoai.plot_sr_comparison(s2_path, sr_large, bands=[1, 2, 3])
plt.show()
```

### Interactive Split Map Comparison

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=sr_large, right_layer="Esri.WorldImagery", left_args={"vmax": 0.3}
)
```

## Limitations and Cautions

## Key Takeaways

## Exercises

### Exercise 1: Sampling Steps and Output Quality

```{code-cell} ipython3

```

### Exercise 2: Uncertainty Across Land Cover Types

```{code-cell} ipython3

```

### Exercise 3: Overlap Parameter and Stitching Quality

```{code-cell} ipython3

```

### Exercise 4: Super-Resolution Across Landscape Types

```{code-cell} ipython3

```

### Exercise 5: Validation Against High-Resolution Imagery

```{code-cell} ipython3

```
