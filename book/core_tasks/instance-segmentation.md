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
# Instance Segmentation

## Introduction

## Learning Objectives

## Instance Segmentation vs. Semantic Segmentation

## Mask R-CNN Architecture

### Backbone Encoder

### Region Proposal Network

### Detection Head

### Mask Head

### RoI Align: Preserving Spatial Precision

### The Complete Pipeline

## Downloading the FTW Dataset

```{code-cell} ipython3
from pathlib import Path

import geopandas as gpd
import geoai
```

```{code-cell} ipython3
geoai.download_ftw(countries=["luxembourg"], output_dir="ftw_data")
```

### Exploring the Dataset

```{code-cell} ipython3
country_dir = Path("ftw_data") / "luxembourg"
chips_gdf = gpd.read_parquet(country_dir / "chips_luxembourg.parquet")

print(f"Total chips: {len(chips_gdf)}")
print(f"\nSplit distribution:")
print(chips_gdf["split"].value_counts())
```

```{code-cell} ipython3
geoai.view_vector_interactive(chips_gdf, column="split")
```

```{code-cell} ipython3
geoai.display_ftw_samples("ftw_data", country="luxembourg", num_samples=4)
```

## Preparing Training Data

```{code-cell} ipython3
data = geoai.prepare_ftw("ftw_data", country="luxembourg")
data
```

```{code-cell} ipython3
geoai.display_training_tiles(
    output_dir="field_boundaries",
    num_tiles=4,
    figsize=(12, 6),
    cmap="tab20",
)
```

## Training a Mask R-CNN Model

```{code-cell} ipython3
geoai.train_instance_segmentation_model(
    images_dir=data["images_dir"],
    labels_dir=data["labels_dir"],
    output_dir="field_boundaries/models",
    num_classes=2,
    num_channels=4,
    batch_size=4,
    num_epochs=20,
    learning_rate=0.005,
    val_split=0.2,
    instance_labels=True,
    visualize=True,
    verbose=True,
)
```

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path="field_boundaries/models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

## Running Inference

```{code-cell} ipython3
test_images = sorted(Path(data["test_dir"]).glob("*.tif"))
test_image_path = str(test_images[0])
masks_path = "field_boundary_prediction.tif"
model_path = "field_boundaries/models/best_model.pth"

result = geoai.instance_segmentation(
    input_path=test_image_path,
    output_path=masks_path,
    model_path=model_path,
    num_classes=2,
    num_channels=4,
    window_size=256,
    overlap=128,
    confidence_threshold=0.5,
    batch_size=4,
    vectorize=True,
    class_names=["background", "building"],
)
result
```

### Visualizing Raw Predictions

```{code-cell} ipython3
geoai.view_raster(
    result["instance"],
    nodata=0,
    cmap="tab20",
    basemap=test_image_path,
    backend="ipyleaflet",
)
```

```{code-cell} ipython3
geoai.view_raster(
    result["class_label"],
    nodata=0,
    cmap="binary",
    basemap=test_image_path,
    backend="ipyleaflet",
)
```

```{code-cell} ipython3
geoai.view_raster(
    result["score"], nodata=0, basemap=test_image_path, backend="ipyleaflet"
)
```

```{code-cell} ipython3
geoai.view_vector_interactive(result["vector"], tiles=test_image_path, column="score")
```

## Post-Processing Predictions

```{code-cell} ipython3
cleaned_masks_path = "field_boundary_prediction_cleaned.tif"
geoai.clean_instance_mask(
    result["instance"], cleaned_masks_path, min_area=100, max_hole_area=100
)
```

```{code-cell} ipython3
geoai.view_raster(
    cleaned_masks_path,
    nodata=0,
    cmap="tab20",
    basemap=test_image_path,
    backend="ipyleaflet",
)
```

### Vectorizing Predictions

```{code-cell} ipython3
output_vector_path = "field_boundary_prediction.geojson"
gdf = geoai.raster_to_vector(cleaned_masks_path, output_vector_path)
```

### Comparing Predictions with Imagery

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf,
    right_layer=test_image_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=test_image_path,
)
```

## Extracting Geometric Properties

```{code-cell} ipython3
gdf_props = geoai.add_geometric_properties(gdf, area_unit="ha", length_unit="m")
gdf_props.head()
```

```{code-cell} ipython3
gdf_props.describe()
```

### Visualizing Fields by Property

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_props, column="area_ha", tiles=test_image_path)
```

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_props, column="elongation", tiles=test_image_path)
```

## Batch Processing

```{code-cell} ipython3
geoai.instance_segmentation_batch(
    input_dir=data["test_dir"],
    output_dir="field_boundaries/predictions",
    model_path=model_path,
    num_classes=2,
    num_channels=4,
    window_size=256,
    overlap=128,
    confidence_threshold=0.5,
    batch_size=4,
)
```

## Key Takeaways

## Exercises

### Exercise 1: Confidence Threshold Analysis

```{code-cell} ipython3

```

### Exercise 2: Multi-Country Comparison

```{code-cell} ipython3

```

### Exercise 3: Field Size Classification

```{code-cell} ipython3

```

### Exercise 4: Post-Processing Parameter Sensitivity

```{code-cell} ipython3

```

### Exercise 5: End-to-End Field Boundary Pipeline

```{code-cell} ipython3

```
