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
# Creating Training Data

## Introduction

## Learning Objectives

## The Training Data Pipeline

## Generating Image Chips from a Single Image

### Download Sample Data

```{code-cell} ipython3
import geoai
```

```{code-cell} ipython3
raster_url = "https://data.source.coop/opengeos/geoai/naip-train.tif"
vector_url = "https://data.source.coop/opengeos/geoai/naip-train-buildings.geojson"
raster_path = geoai.download_file(raster_url)
vector_path = geoai.download_file(vector_url)
```

### Preview Data

```{code-cell} ipython3
geoai.view_image(raster_path, figsize=(18, 10))
```

```{code-cell} ipython3
geoai.view_vector(vector_path, raster_path=raster_path, figsize=(18, 10))
```

```{code-cell} ipython3
geoai.view_vector_interactive(vector_path, tiles=raster_path)
```

### Convert Vector to Raster

```{code-cell} ipython3
output_path = vector_path.replace(".geojson", ".tif")
geoai.vector_to_raster(vector_path, output_path, reference_raster=raster_path)
```

```{code-cell} ipython3
geoai.view_image(output_path, figsize=(18, 10))
```

### Tiling Parameters

### Generate Tiles

```{code-cell} ipython3
tiles = geoai.export_geotiff_tiles(
    in_raster=raster_path,
    out_folder="output",
    in_class_data=vector_path,
    tile_size=512,
    stride=384,
    buffer_radius=0,
    create_overview=True,
    quiet=True,
)
```

### Preview Image Chips

```{code-cell} ipython3
geoai.view_image("output/overview.png", figsize=(18, 10))
```

```{code-cell} ipython3
fig = geoai.display_training_tiles(output_dir="output", num_tiles=4, figsize=(18, 10))
```

## Batch Processing Multiple Images

### Download Batch Sample Data

```{code-cell} ipython3
import os
```

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/naip-rgb-train-tiles.zip"
data_dir = geoai.download_file(url)
```

### Explore Sample Data

```{code-cell} ipython3
print("Images:")
for f in sorted(os.listdir(f"{data_dir}/images")):
    print(f"  - {f}")

print("\nAnnotations (single file):")
for f in sorted(os.listdir(f"{data_dir}/masks1")):
    print(f"  - {f}")

print("\nAnnotations (multiple files):")
for f in sorted(os.listdir(f"{data_dir}/masks2")):
    print(f"  - {f}")
```

### Visualize Image and Annotations

```{code-cell} ipython3
image_path = f"{data_dir}/images/naip_rgb_train_tile1.tif"
mask_path = f"{data_dir}/masks2/naip_rgb_train_tile1.geojson"

fig, axes, info = geoai.display_image_with_vector(image_path, mask_path)
print(f"Number of buildings: {info['num_features']}")
```

### Method 1: Single Vector File Covering All Images

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_file=f"{data_dir}/masks1/naip_train_buildings.geojson",
    output_folder="output/method1_single_mask",
    tile_size=256,
    stride=128,
    class_value_field="class",
    skip_empty_tiles=True,
    quiet=False,
)

print(f"\n{'='*60}")
print("Results:")
print(f"  Images processed: {stats['processed_pairs']}")
print(f"  Total tiles generated: {stats['total_tiles']}")
print(f"  Tiles with features: {stats['tiles_with_features']}")
print(f"  Feature percentage: {stats['tiles_with_features']/stats['total_tiles']*100:.1f}%")
```

### Method 2: Multiple Vector Files Matched by Sorted Order

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_folder=f"{data_dir}/masks2",
    output_folder="output/method2_sorted_order",
    tile_size=256,
    stride=128,
    class_value_field="class",
    skip_empty_tiles=True,
    match_by_name=False,
)

print(f"\n{'='*60}")
print("Results:")
print(f"  Images processed: {stats['processed_pairs']}")
print(f"  Total tiles generated: {stats['total_tiles']}")
print(f"  Tiles with features: {stats['tiles_with_features']}")
```

### Method 3: Multiple Vector Files Matched by Filename

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_folder=f"{data_dir}/masks2",
    output_folder="output/method3_matched_name",
    tile_size=256,
    stride=128,
    class_value_field="class",
    skip_empty_tiles=True,
    match_by_name=True,
)

print(f"\n{'='*60}")
print("Results:")
print(f"  Images processed: {stats['processed_pairs']}")
print(f"  Total tiles generated: {stats['total_tiles']}")
print(f"  Tiles with features: {stats['tiles_with_features']}")
```

### Visualize Generated Tiles

```{code-cell} ipython3
output_dir = "output/method1_single_mask"
fig = geoai.display_training_tiles(output_dir, num_tiles=4, figsize=(18, 10))
```

### Advanced Usage: Custom Parameters

```{code-cell} ipython3
stats = geoai.export_geotiff_tiles_batch(
    images_folder=f"{data_dir}/images",
    masks_file=f"{data_dir}/masks1/naip_train_buildings.geojson",
    output_folder="output/advanced_example",
    tile_size=512,
    stride=256,
    class_value_field="class",
    buffer_radius=0.5,
    skip_empty_tiles=True,
    all_touched=True,
    max_tiles=10,
    quiet=False,
)

print(f"\nGenerated {stats['total_tiles']} tiles with 50% overlap")
print(f"Output structure:")
print(f"  - output/advanced_example/images/  (image tiles)")
print(f"  - output/advanced_example/masks/   (mask tiles)")
```

## Batch Processing with Raster Masks

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/landcover-sample-data.zip"
data_dir2 = geoai.download_file(url)
```

```{code-cell} ipython3
images_dir = f"{data_dir2}/images"
masks_dir = f"{data_dir2}/masks"
tiles_dir = f"{data_dir2}/tiles"
```

```{code-cell} ipython3
result = geoai.export_geotiff_tiles_batch(
    images_folder=images_dir,
    masks_folder=masks_dir,
    output_folder=tiles_dir,
    tile_size=512,
    stride=384,
    quiet=True,
)
```

## Label Quality Considerations

## Dataset Organization

### Train/Validation/Test Splits

### Directory Structure

## Summary

## Key Takeaways

## Exercises

### Exercise 1: Generate Image Chips with Different Overlap Settings

```{code-cell} ipython3

```

### Exercise 2: Batch Process with Different Pairing Methods

```{code-cell} ipython3

```

### Exercise 3: Visualize and Validate Training Data

```{code-cell} ipython3

```

### Exercise 4: Prepare a Complete Training Dataset

```{code-cell} ipython3

```
