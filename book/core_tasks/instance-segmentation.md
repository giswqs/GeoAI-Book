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

## Preparing Instance Segmentation Data

```{code-cell} python
import geoai
import geopandas as gpd

# Download training imagery and building annotations
train_image_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_train.tif"
train_labels_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_train_buildings.geojson"
test_image_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_test.tif"
test_labels_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_test_buildings.geojson"

train_image = geoai.download_file(train_image_url)
train_labels = geoai.download_file(train_labels_url)
test_image = geoai.download_file(test_image_url)
test_labels = geoai.download_file(test_labels_url)

print(f"Training image: {train_image}")
print(f"Test image: {test_image}")
```

### Inspecting Annotations

```{code-cell} python
import rasterio
import matplotlib.pyplot as plt

# Load and inspect the building annotations
gdf = gpd.read_file(train_labels)
print(f"Number of buildings: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(f"Geometry types: {gdf.geometry.type.unique()}")
print(f"Columns: {list(gdf.columns)}")
gdf.head()
```

```{code-cell} python
# Visualize the training image with building annotations
with rasterio.open(train_image) as src:
    img = src.read([1, 2, 3])
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img.transpose(1, 2, 0), extent=extent)
gdf.boundary.plot(ax=ax, color="red", linewidth=0.5)
ax.set_title(f"Training Image with {len(gdf)} Building Annotations")
ax.axis("off")
plt.show()
```

## Training a Mask R-CNN Model

```{code-cell} python
# Initialize a Mask R-CNN model for building extraction (1 class: building)
num_classes = 2  # background + building
model = geoai.get_instance_segmentation_model(num_classes=num_classes)
print(f"Model type: {type(model).__name__}")
```

```{code-cell} python
# Train the model on the NAIP training data
output_dir = "instance_seg_output"

model, history = geoai.train_instance_segmentation_model(
    model=model,
    train_raster=train_image,
    train_vector=train_labels,
    val_raster=test_image,
    val_vector=test_labels,
    output_dir=output_dir,
    num_epochs=10,
    batch_size=4,
    patch_size=256,
    learning_rate=0.005,
    class_column=None,
)
```

```{code-cell} python
# Plot the training loss curve
geoai.plot_training_history(history, output_dir=output_dir)
```

## Evaluating Instance Segmentation

### Running Inference on Test Data

```{code-cell} python
# Run instance segmentation on the test image
predictions = geoai.instance_segmentation(
    model=model,
    raster=test_image,
    confidence_threshold=0.5,
    patch_size=256,
    overlap=64,
)
print(f"Detected {len(predictions)} building instances")
```

```{code-cell} python
# Convert predicted masks to vector polygons
predicted_gdf = geoai.masks_to_vector(
    predictions=predictions,
    raster=test_image,
)
print(f"Predicted polygons: {len(predicted_gdf)}")
predicted_gdf.head()
```

### Visualizing Predictions

```{code-cell} python
gt_gdf = gpd.read_file(test_labels)
with rasterio.open(test_image) as src:
    test_img = src.read([1, 2, 3])
    test_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, gdf_plot, title in zip(axes, [gt_gdf, predicted_gdf], ["Ground Truth", "Predictions"]):
    ax.imshow(test_img.transpose(1, 2, 0), extent=test_extent)
    gdf_plot.boundary.plot(ax=ax, color="red", linewidth=0.5)
    ax.set_title(f"{title} ({len(gdf_plot)} buildings)")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Post-Processing Predictions

### Orthogonalization

```{code-cell} python
# Orthogonalize building polygons to enforce right angles
ortho_gdf = geoai.orthogonalize(predicted_gdf)
print(f"Orthogonalized polygons: {len(ortho_gdf)}")
```

### Regularization

```{code-cell} python
# Regularize polygons to simplify geometry
regular_gdf = geoai.regularize(predicted_gdf)
print(f"Regularized polygons: {len(regular_gdf)}")
```

### Smoothing

```{code-cell} python
# Smooth polygon edges
smooth_gdf = geoai.smooth_vector(predicted_gdf)
print(f"Smoothed polygons: {len(smooth_gdf)}")
```

### Comparing Post-Processing Methods

```{code-cell} python
# Visualize the effects of different post-processing methods
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, gdf_plot, title in zip(
    axes,
    [predicted_gdf, ortho_gdf, regular_gdf, smooth_gdf],
    ["Raw Predictions", "Orthogonalized", "Regularized", "Smoothed"],
):
    ax.imshow(test_img.transpose(1, 2, 0), extent=test_extent)
    gdf_plot.boundary.plot(ax=ax, color="red", linewidth=0.8)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

### Choosing a Post-Processing Strategy

### Extracting Geometric Properties

```{code-cell} python
# Add geometric properties to the orthogonalized polygons
props_gdf = geoai.add_geometric_properties(ortho_gdf)
print(f"Columns: {list(props_gdf.columns)}")
props_gdf.describe()
```

## Case Study: Building Footprint Extraction

```{code-cell} python
# Download Las Vegas building training data for a larger-scale example
lv_buildings_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/las_vegas_buildings_train.geojson"
lv_buildings = geoai.download_file(lv_buildings_url)

lv_gdf = gpd.read_file(lv_buildings)
print(f"Las Vegas training buildings: {len(lv_gdf)}")
print(f"Total building area: {lv_gdf.geometry.area.sum():,.0f} sq meters")
print(f"Average building area: {lv_gdf.geometry.area.mean():,.1f} sq meters")
```

```{code-cell} python
# Run batch inference on the test image with overlap for seamless results
batch_predictions = geoai.instance_segmentation_batch(
    model=model,
    raster=test_image,
    confidence_threshold=0.5,
    patch_size=256,
    overlap=64,
    batch_size=4,
)

# Convert batch predictions to vector format
batch_gdf = geoai.masks_to_vector(
    predictions=batch_predictions,
    raster=test_image,
)
print(f"Batch inference detected {len(batch_gdf)} buildings")
```

```{code-cell} python
# Apply orthogonalization, add geometric properties, and filter small detections
final_gdf = geoai.orthogonalize(batch_gdf)
final_gdf = geoai.add_geometric_properties(final_gdf)
min_area = 20  # square meters
final_gdf = final_gdf[final_gdf["area"] >= min_area].reset_index(drop=True)
print(f"Buildings after filtering: {len(final_gdf)}")
print(f"  Total footprint area: {final_gdf['area'].sum():,.0f} sq meters")
print(f"  Mean building area: {final_gdf['area'].mean():,.1f} sq meters")
print(f"  Median building area: {final_gdf['area'].median():,.1f} sq meters")
```

```{code-cell} python
# Export final results as a GeoPackage for use in GIS software
output_path = "building_footprints.gpkg"
final_gdf.to_file(output_path, driver="GPKG")
print(f"Saved {len(final_gdf)} building footprints to {output_path}")
```

```{code-cell} python
# Create a summary visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(test_img.transpose(1, 2, 0), extent=test_extent)
final_gdf.plot(ax=axes[0], facecolor="none", edgecolor="red", linewidth=0.8)
axes[0].set_title(f"Extracted Building Footprints ({len(final_gdf)} buildings)")
axes[0].axis("off")
axes[1].hist(final_gdf["area"], bins=30, color="steelblue", edgecolor="white")
axes[1].set_xlabel("Building Area (sq meters)")
axes[1].set_ylabel("Count")
axes[1].set_title("Building Area Distribution")
axes[1].axvline(final_gdf["area"].median(), color="red", linestyle="--",
                label=f"Median: {final_gdf['area'].median():,.0f} sq m")
axes[1].legend()
plt.tight_layout()
plt.show()
```

## Key Takeaways

## Exercises

### Exercise 1: Confidence Threshold Analysis

### Exercise 2: Comparing Post-Processing Methods

### Exercise 3: Building Size Classification

### Exercise 4: Training with Different Patch Sizes

### Exercise 5: End-to-End Building Extraction Pipeline
