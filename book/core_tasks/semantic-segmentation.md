---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: geo
  language: python
  name: python3
---
# Semantic Segmentation

## Introduction

## Learning Objectives

## Foundations of Semantic Segmentation

### Segmentation Architectures

### Encoders and Transfer Learning

### Practical Implementation

## Import Libraries

```{code-cell} ipython3
import geoai
```

## Building Detection from Aerial Imagery

### Download Sample Data

```{code-cell} ipython3
train_raster_url = (
    "https://data.source.coop/opengeos/geoai/naip_rgb_train.tif"
)
train_vector_url = "https://data.source.coop/opengeos/geoai/naip_train_buildings.geojson"
test_raster_url = (
    "https://data.source.coop/opengeos/geoai/naip_test.tif"
)
```

```{code-cell} ipython3
train_raster_path = geoai.download_file(train_raster_url)
train_vector_path = geoai.download_file(train_vector_url)
test_raster_path = geoai.download_file(test_raster_url)
```

### Visualize Sample Data

```{code-cell} ipython3
geoai.print_raster_info(train_raster_path, show_preview=False)
```

```{code-cell} ipython3
geoai.view_vector_interactive(train_vector_path, tiles=train_raster_path)
```

```{code-cell} ipython3
geoai.view_raster(test_raster_path)
```

### Create Training Tiles

```{code-cell} ipython3
out_folder = "buildings"
tiles = geoai.export_geotiff_tiles(
    in_raster=train_raster_path,
    out_folder=out_folder,
    in_class_data=train_vector_path,
    tile_size=512,
    stride=256,
    buffer_radius=0,
)
```

### Train the Model

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/labels",
    output_dir=f"{out_folder}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=3,
    num_classes=2,
    batch_size=8,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
)
```

### Evaluate the Model

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/unet_models/training_history.pth",
    figsize=(15, 5),
)
```

### Run Inference

```{code-cell} ipython3
masks_path = "naip_buildings_prediction.tif"
probability_path = "naip_test_probability_map.tif"
model_path = f"{out_folder}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_raster_path,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
    probability_path=probability_path
)
```

### Visualize Raster Masks

```{code-cell} ipython3
geoai.view_raster(
    masks_path,
    nodata=0,
    colormap="binary",
    basemap=test_raster_path,
)
```

```{code-cell} ipython3
geoai.view_raster(
    probability_path, indexes=[2], basemap=test_raster_path
)
```

### Vectorize Predictions

```{code-cell} ipython3
output_vector_path = "naip_buildings_prediction.geojson"
gdf = geoai.orthogonalize(masks_path, output_vector_path, epsilon=2)
```

### Add Geometric Properties

```{code-cell} ipython3
gdf_props = geoai.add_geometric_properties(gdf, area_unit="m2", length_unit="m")
```

### Visualize Results

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_props, column="area_m2", tiles=test_raster_path)
```

```{code-cell} ipython3
gdf_filtered = gdf_props[(gdf_props["area_m2"] > 10)]
geoai.view_vector_interactive(gdf_filtered, column="area_m2", tiles=test_raster_path)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf_filtered,
    right_layer=test_raster_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=test_raster_path,
)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Surface Water Mapping

### Water Mapping with Non-Georeferenced Satellite Imagery

#### Download Sample Data

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/waterbody-dataset.zip"
```

```{code-cell} ipython3
out_folder = geoai.download_file(url)
print(f"Downloaded dataset to {out_folder}")
```

#### Train the Model

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/masks",
    output_dir=f"{out_folder}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=3,
    num_classes=2,
    batch_size=16,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
    target_size=(512, 512),
    verbose=True,
)
```

#### Evaluate the Model

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/unet_models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

#### Run Inference on a Single Image

```{code-cell} ipython3
index = 3
test_image_path = f"{out_folder}/images/water_body_{index}.jpg"
ground_truth_path = f"{out_folder}/masks/water_body_{index}.jpg"
prediction_path = f"{out_folder}/prediction/water_body_{index}.png"
model_path = f"{out_folder}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_image_path,
    output_path=prediction_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=32,
)
```

```{code-cell} ipython3
fig = geoai.plot_prediction_comparison(
    original_image=test_image_path,
    prediction_image=prediction_path,
    ground_truth_image=ground_truth_path,
    titles=["Original", "Prediction", "Ground Truth"],
    figsize=(15, 5),
    save_path=f"{out_folder}/prediction/water_body_{index}_comparison.png",
    show_plot=True,
)
```

#### Run Inference on Multiple Images

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/waterbody-dataset-sample.zip"
```

```{code-cell} ipython3
data_dir = geoai.download_file(url)
print(f"Downloaded dataset to {data_dir}")
```

```{code-cell} ipython3
images_dir = f"{data_dir}/images"
predictions_dir = f"{data_dir}/predictions"
```

```{code-cell} ipython3
geoai.semantic_segmentation_batch(
    input_dir=images_dir,
    output_dir=predictions_dir,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=3,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=4,
    quiet=True,
)
```

```{code-cell} ipython3
geoai.empty_cache()
```

### Water Mapping with Sentinel-2 Multispectral Imagery

#### Download Sample Data

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/dset-s2.zip"
data_dir = geoai.download_file(url, output_path="dset-s2.zip")
```

```{code-cell} ipython3
images_dir = f"{data_dir}/tra_scene"
masks_dir = f"{data_dir}/tra_truth"
tiles_dir = f"{data_dir}/tiles"
```

#### Create Training Tiles

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

#### Train the Model

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{tiles_dir}/images",
    labels_dir=f"{tiles_dir}/masks",
    output_dir=f"{tiles_dir}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=6,
    num_classes=2,
    batch_size=32,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
)
```

#### Evaluate the Model

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{tiles_dir}/unet_models/training_history.pth",
    figsize=(15, 5),
)
```

#### Run Inference on Validation Data

```{code-cell} ipython3
images_dir = f"{data_dir}/val_scene"
masks_dir = f"{data_dir}/val_truth"
predictions_dir = f"{data_dir}/predictions"
model_path = f"{tiles_dir}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation_batch(
    input_dir=images_dir,
    output_dir=predictions_dir,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=6,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=32,
    quiet=True,
)
```

#### Visualize Results

```{code-cell} ipython3
image_id = "S2A_L2A_20190318_N0211_R061"
test_image_path = f"{data_dir}/val_scene/{image_id}_6Bands_S2.tif"
ground_truth_path = f"{data_dir}/val_truth/{image_id}_S2_Truth.tif"
prediction_path = f"{data_dir}/predictions/{image_id}_6Bands_S2_mask.tif"
save_path = f"{data_dir}/{image_id}_6Bands_S2_comparison.png"

fig = geoai.plot_prediction_comparison(
    original_image=test_image_path,
    prediction_image=prediction_path,
    ground_truth_image=ground_truth_path,
    titles=["Original", "Prediction", "Ground Truth"],
    figsize=(15, 5),
    save_path=save_path,
    show_plot=True,
    indexes=[5, 4, 3],
    divider=5000,
)
```

#### Apply the Model to New Sentinel-2 Imagery

```{code-cell} ipython3
s2_path = "s2.tif"
url = "https://data.source.coop/opengeos/geoai/s2-minnesota-2025-08-31-subset.tif"
geoai.download_file(url, output_path=s2_path)
```

```{code-cell} ipython3
geoai.view_raster(
    s2_path, indexes=[4, 3, 2], vmin=0, vmax=5000, layer_name="Sentinel-2"
)
```

```{code-cell} ipython3
s2_mask = "s2_mask.tif"
model_path = f"{tiles_dir}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=s2_path,
    output_path=s2_mask,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=6,
    num_classes=2,
    window_size=512,
    overlap=256,
    batch_size=32,
)
```

```{code-cell} ipython3
geoai.view_raster(
    s2_mask,
    nodata=0,
    colormap="binary",
    layer_name="Water",
    basemap=s2_path,
    basemap_args={"indexes": [4, 3, 2], "vmin": 0, "vmax": 5000},
)
```

#### Vectorize Water Mask

```{code-cell} ipython3
output_vector_path = "s2_water_polygons.geojson"
gdf = geoai.raster_to_vector(
    raster_path=s2_mask,
    output_path=output_vector_path,
    min_area=100,
    simplify_tolerance=None,
)
```

#### Smooth Water Body Polygons

```{code-cell} ipython3
smoothed_path = "s2_water_smoothed.geojson"
gdf = geoai.smooth_vector(
    gdf,
    smooth_iterations=3,
    output_path=smoothed_path,
)
```

#### Add Geometric Properties

```{code-cell} ipython3
gdf_props = geoai.add_geometric_properties(gdf, area_unit="m2", length_unit="m")
gdf_props.head()
```

#### Filter Small Artifacts

```{code-cell} ipython3
gdf_filtered = gdf_props[gdf_props["area_m2"] > 100]
print(f"Water bodies detected: {len(gdf_filtered)}")
print(f"Removed {len(gdf_props) - len(gdf_filtered)} small artifacts")
```

#### Visualize Water Body Polygons

```{code-cell} ipython3
geoai.view_vector_interactive(
    gdf_filtered,
    column="area_m2",
    opacity=0.5,
    tiles=s2_path,
    tiles_args={"indexes": [4, 3, 2], "vmin": 0, "vmax": 5000},
)
```

#### Split Map Comparison

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf_filtered,
    right_layer=s2_path,
    left_args={"style": {"color": "blue", "fillOpacity": 0.3}},
    right_args={"indexes": [4, 3, 2], "vmin": 0, "vmax": 5000},
    basemap=s2_path,
)
```

#### Water Body Area Statistics

```{code-cell} ipython3
print(gdf_filtered["area_m2"].describe())
```

#### Save Results

```{code-cell} ipython3
gdf_filtered.to_file("s2_water_bodies_final.geojson", driver="GeoJSON")
print(f"Saved {len(gdf_filtered)} water body polygons to s2_water_bodies_final.geojson")
```

### Water Mapping with NAIP Aerial Imagery

#### Download Sample Data

```{code-cell} ipython3
train_raster_url = "https://data.source.coop/opengeos/geoai/naip_water_train.tif"
train_masks_url = "https://data.source.coop/opengeos/geoai/naip_water_masks.tif"
test_raster_url = "https://data.source.coop/opengeos/geoai/naip_water_test.tif"
```

```{code-cell} ipython3
train_raster_path = geoai.download_file(train_raster_url)
train_masks_path = geoai.download_file(train_masks_url)
test_raster_path = geoai.download_file(test_raster_url)
```

```{code-cell} ipython3
geoai.print_raster_info(train_raster_path, show_preview=False)
```

#### Visualize Sample Data

```{code-cell} ipython3
geoai.view_raster(train_masks_path, nodata=0, opacity=0.5, basemap=train_raster_path)
```

```{code-cell} ipython3
geoai.view_raster(test_raster_path)
```

#### Create Training Tiles

```{code-cell} ipython3
out_folder = "naip"
```

```{code-cell} ipython3
tiles = geoai.export_geotiff_tiles(
    in_raster=train_raster_path,
    out_folder=out_folder,
    in_class_data=train_masks_path,
    tile_size=512,
    stride=256,
    buffer_radius=0,
)
```

#### Train the Model

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/labels",
    output_dir=f"{out_folder}/models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=4,
    num_classes=2,
    batch_size=8,
    num_epochs=20,
    learning_rate=0.005,
    val_split=0.2,
)
```

#### Evaluate the Model

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

#### Run Inference

```{code-cell} ipython3
masks_path = "naip_water_prediction.tif"
model_path = f"{out_folder}/models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_raster_path,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=4,
    num_classes=2,
    window_size=512,
    overlap=128,
    batch_size=32,
)
```

```{code-cell} ipython3
geoai.view_raster(
    masks_path,
    nodata=0,
    layer_name="Water",
    basemap=test_raster_path,
)
```

#### Vectorize and Analyze Predictions

```{code-cell} ipython3
output_path = "naip_water_prediction.geojson"
gdf = geoai.raster_to_vector(
    masks_path, output_path, min_area=1000, simplify_tolerance=1
)
```

```{code-cell} ipython3
gdf = geoai.add_geometric_properties(gdf)
len(gdf)
```

```{code-cell} ipython3
geoai.view_vector_interactive(gdf, tiles=test_raster_path)
```

```{code-cell} ipython3
gdf["elongation"].hist()
```

```{code-cell} ipython3
gdf_filtered = gdf[gdf["elongation"] < 10]
```

```{code-cell} ipython3
len(gdf_filtered)
```

#### Visualize Results

```{code-cell} ipython3
geoai.view_vector_interactive(gdf_filtered, tiles=test_raster_path)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=gdf_filtered,
    right_layer=test_raster_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=test_raster_path,
)
```

### Sensor-Agnostic Water Segmentation

#### Sentinel-2 Water Segmentation

##### Download Sentinel-2 Data

```{code-cell} ipython3
s2_url = "https://data.source.coop/opengeos/geoai/S2A-L2A-20190318-N0211-R061-6Bands-S2.tif"
s2_path = geoai.download_file(s2_url)
```

##### Visualize Input Data

```{code-cell} ipython3
geoai.view_raster(s2_path, indexes=[4, 3, 2], vmax=3000)
```

##### Run Water Segmentation

```{code-cell} ipython3
s2_mask_path = geoai.segment_water(
    s2_path,
    band_order="sentinel2",
    output_raster="s2_owm_water_mask.tif",
)
```

##### Visualize Raster Mask

```{code-cell} ipython3
geoai.view_raster(
    s2_mask_path,
    nodata=0,
    basemap=s2_path,
    opacity=0.5,
    backend="ipyleaflet",
)
```

##### Vectorize and Smooth Water Bodies

```{code-cell} ipython3
s2_gdf = geoai.segment_water(
    s2_path,
    band_order="sentinel2",
    output_raster="s2_owm_water_mask.tif",
    output_vector="s2_owm_water_bodies.geojson",
    smooth=True,
    smooth_iterations=3,
    min_size=100,
)
```

##### Filter Small Artifacts

```{code-cell} ipython3
s2_filtered = s2_gdf[s2_gdf["area_m2"] > 100]
print(f"Water bodies detected: {len(s2_filtered)}")
print(f"Removed {len(s2_gdf) - len(s2_filtered)} small artifacts")
```

##### Visualize Water Body Polygons

```{code-cell} ipython3
geoai.view_vector_interactive(
    s2_filtered,
    column="area_m2",
    tiles=s2_path,
    tiles_args={"indexes": [4, 3, 2], "vmax": 3000},
)
```

##### Split Map Comparison

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=s2_filtered,
    right_layer=s2_path,
    left_args={"style": {"color": "blue", "fillOpacity": 0.3}},
    right_args={"indexes": [4, 3, 2], "vmax": 3000},
    basemap=s2_path,
)
```

#### NAIP Water Segmentation

##### Download NAIP Data

```{code-cell} ipython3
naip_url = "https://data.source.coop/opengeos/geoai/naip_water_test_subset.tif"
naip_path = geoai.download_file(naip_url)
```

##### Visualize NAIP Imagery

```{code-cell} ipython3
geoai.view_raster(naip_path, indexes=[4, 1, 2])
```

##### Run Water Segmentation

```{code-cell} ipython3
naip_gdf = geoai.segment_water(
    naip_path,
    band_order="naip",
    output_raster="naip_owm_water_mask.tif",
    output_vector="naip_owm_water_bodies.geojson",
    smooth=True,
    smooth_iterations=3,
    min_size=100,
)
```

##### Visualize Raster Mask

```{code-cell} ipython3
geoai.view_raster(
    "naip_owm_water_mask.tif",
    nodata=0,
    basemap=naip_path,
    opacity=0.5,
    backend="ipyleaflet",
)
```

##### Filter Small Artifacts

```{code-cell} ipython3
naip_filtered = naip_gdf[naip_gdf["area_m2"] > 100]
print(f"Water bodies detected: {len(naip_filtered)}")
print(f"Removed {len(naip_gdf) - len(naip_filtered)} small artifacts")
```

##### Visualize Water Body Polygons

```{code-cell} ipython3
geoai.view_vector_interactive(
    naip_filtered,
    column="area_m2",
    tiles=naip_path,
)
```

##### Split Map Comparison

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=naip_filtered,
    right_layer=naip_path,
    left_args={"style": {"color": "blue", "fillOpacity": 0.3}},
    basemap=naip_path,
)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Cloud and Cloud Shadow Detection

### Download Sample Data

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/S2C-MSIL2A-20250920T162001-subset.tif"
s2_path = geoai.download_file(url)
```

### Predict Cloud Mask

```{code-cell} ipython3
pred_path = "cloud_mask.tif"
geoai.predict_cloud_mask_from_raster(
    input_path=s2_path,
    output_path=pred_path,
    red_band=1,
    green_band=2,
    nir_band=4,
    batch_size=4,
    inference_dtype="bf16",
)
```

### Cloud Statistics

```{code-cell} ipython3
import rasterio as rio
import numpy as np

with rio.open(pred_path) as src:
    pred_array = src.read(1)

stats = geoai.calculate_cloud_statistics(pred_array)
for key, value in stats.items():
    if "percent" in key:
        print(f"{key}: {value:.2f}%")
    else:
        print(f"{key}: {value:,}")
```

### Visualize Results

```{code-cell} ipython3
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

with rio.open(s2_path) as src:
    scene_fc = src.read([4, 1, 2]).astype(np.float32)

# Percentile-based contrast stretch per band
for i in range(3):
    band = scene_fc[i]
    p2, p98 = np.percentile(band, (2, 98))
    scene_fc[i] = (band - p2) / (p98 - p2)
scene_fc = np.clip(scene_fc, 0, 1)

cmap = ListedColormap(["green", "white", "gray", "black"])
labels = ["Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow"]
patches = [
    Patch(facecolor=cmap(i), edgecolor="black", label=labels[i]) for i in range(4)
]

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(scene_fc.transpose(1, 2, 0))
ax[0].set_title("False Color (NIR/R/G)")
ax[0].set_axis_off()
ax[1].imshow(pred_array, cmap=cmap, vmin=0, vmax=3)
ax[1].set_title("Cloud Mask")
ax[1].legend(handles=patches, loc="upper left", fontsize=12)
ax[1].set_axis_off()
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
from scipy import ndimage

cloud_binary = (pred_array == 1) | (pred_array == 2)
labeled, _ = ndimage.label(cloud_binary)
if labeled.max() > 0:
    sizes = ndimage.sum(cloud_binary, labeled, range(1, labeled.max() + 1))
    largest_label = np.argmax(sizes) + 1
    cy, cx = ndimage.center_of_mass(cloud_binary, labeled, largest_label)
    cy, cx = int(cy), int(cx)
else:
    cy, cx = pred_array.shape[0] // 2, pred_array.shape[1] // 2

# Crop a 1000x1000 window around the center
h, w = pred_array.shape
half = 500
r0, r1 = max(0, cy - half), min(h, cy + half)
c0, c1 = max(0, cx - half), min(w, cx + half)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(scene_fc[:, r0:r1, c0:c1].transpose(1, 2, 0))
ax[0].contour(pred_array[r0:r1, c0:c1], levels=[0.5, 1.5, 2.5], colors="cyan")
ax[0].set_title("False Color with Cloud Contours")
ax[0].set_axis_off()
ax[1].imshow(pred_array[r0:r1, c0:c1], cmap=cmap, vmin=0, vmax=3)
ax[1].set_title("Cloud Mask (Zoomed)")
ax[1].legend(handles=patches, loc="upper left", fontsize=12)
ax[1].set_axis_off()
plt.tight_layout()
plt.show()
```

### Post-Processing

```{code-cell} ipython3
cleaned_mask_path = "cleaned_mask.tif"
geoai.clean_raster(pred_path, cleaned_mask_path)
```

```{code-cell} ipython3
cloud_vector = "cloud_vector.geojson"
cloud_gdf = geoai.raster_to_vector(cleaned_mask_path, cloud_vector)
```

```{code-cell} ipython3
smoothed_cloud_gdf = geoai.smooth_vector(
    cloud_gdf, smooth_iterations=3, output_path="smoothed_cloud_vector.geojson"
)
```

### Interactive Visualization

```{code-cell} ipython3
geoai.view_vector_interactive(
    smoothed_cloud_gdf, tiles=s2_path, tiles_args={"indexes": [4, 1, 2], "vmax": 4000}
)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=smoothed_cloud_gdf,
    right_layer=s2_path,
    left_args={"style": {"color": "cyan", "fillOpacity": 0.2}},
    right_args={"indexes": [4, 1, 2], "vmax": 4000},
    basemap=s2_path,
)
```

### Geometric Properties

```{code-cell} ipython3
props_gdf = geoai.add_geometric_properties(cloud_gdf)
props_gdf.describe()
```

### Cloud-Free Mask

```{code-cell} ipython3
cloud_free = geoai.create_cloud_free_mask(pred_array)
usable_pct = cloud_free.sum() / cloud_free.size * 100
print(f"Usable (cloud-free) pixels: {usable_pct:.2f}%")

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(scene_fc.transpose(1, 2, 0))
ax[0].set_title("False Color (NIR/R/G)")
ax[0].set_axis_off()
ax[1].imshow(cloud_free, cmap="RdYlGn", vmin=0, vmax=1)
ax[1].set_title("Cloud-Free Mask (green = usable)")
ax[1].set_axis_off()
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Land Cover Classification

### Download Sample Data

```{code-cell} ipython3
train_raster_url = "https://data.source.coop/opengeos/geoai/m_3807511_ne_18_060_20181104.tif"
train_landcover_url = "https://data.source.coop/opengeos/geoai/m_3807511_ne_18_060_20181104_landcover.tif"
test_raster_url = "https://data.source.coop/opengeos/geoai/m_3807511_se_18_060_20181104.tif"
```

```{code-cell} ipython3
train_raster_path = geoai.download_file(train_raster_url)
train_landcover_path = geoai.download_file(train_landcover_url)
test_raster_path = geoai.download_file(test_raster_url)
```

### Visualize Sample Data

```{code-cell} ipython3
legend_args = {"builtin_legend": "Chesapeake", "title": "Land Cover Type"}
geoai.view_raster(train_landcover_path, basemap=train_raster_path, legend_args=legend_args)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=train_landcover_path,
    right_layer=train_raster_path,
)
```

```{code-cell} ipython3
geoai.view_raster(test_raster_path)
```

### Create Training Tiles

```{code-cell} ipython3
out_folder = "landcover"
```

```{code-cell} ipython3
tiles = geoai.export_geotiff_tiles(
    in_raster=train_raster_path,
    out_folder=out_folder,
    in_class_data=train_landcover_path,
    tile_size=512,
    stride=256,
    buffer_radius=0,
)
```

### Train the Model

```{code-cell} ipython3
geoai.train_segmentation_model(
    images_dir=f"{out_folder}/images",
    labels_dir=f"{out_folder}/labels",
    output_dir=f"{out_folder}/unet_models",
    architecture="unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    num_channels=4,
    num_classes=13,
    batch_size=8,
    num_epochs=20,
    learning_rate=0.001,
    val_split=0.2,
    verbose=True,
    plot_curves=True,
)
```

### Evaluate the Model

```{code-cell} ipython3
geoai.plot_performance_metrics(
    history_path=f"{out_folder}/unet_models/training_history.pth",
    figsize=(15, 5),
    verbose=True,
)
```

### Run Inference

```{code-cell} ipython3
masks_path = "landcover_prediction.tif"
model_path = f"{out_folder}/unet_models/best_model.pth"
```

```{code-cell} ipython3
geoai.semantic_segmentation(
    input_path=test_raster_path,
    output_path=masks_path,
    model_path=model_path,
    architecture="unet",
    encoder_name="resnet34",
    num_channels=4,
    num_classes=13,
    window_size=512,
    overlap=128,
    batch_size=4,
)
```

### Visualize Results

```{code-cell} ipython3
geoai.write_colormap(masks_path, train_landcover_path, output=masks_path)
```

```{code-cell} ipython3
geoai.view_raster(masks_path, basemap=test_raster_path, legend_args=legend_args)
```

## Publish and Reuse Models

### Push to Hugging Face Hub

```{code-cell} ipython3
from huggingface_hub import notebook_login

notebook_login()
```

```{code-cell} ipython3
repo_url = geoai.push_timm_model_to_hub(
    model_path=model_path,
    repo_id="your-username/chesapeake-landcover-unet-resnet34",
    encoder_name="resnet34",
    architecture="unet",
    num_channels=4,
    num_classes=13,
)
print(repo_url)
```

### Run Inference from Hub

```{code-cell} ipython3
hub_output = "landcover_hub_prediction.tif"
geoai.timm_segmentation_from_hub(
    input_path=test_raster_path,
    output_path=hub_output,
    repo_id="your-username/chesapeake-landcover-unet-resnet34",
    window_size=512,
    overlap=128,
    batch_size=4,
)
```

```{code-cell} ipython3
geoai.write_colormap(hub_output, train_landcover_path, output=hub_output)
geoai.view_raster(hub_output, basemap=test_raster_path, legend_args=legend_args)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Key Takeaways

## Exercises

### Exercise 1: Training with a Different Architecture

```{code-cell} ipython3

```

### Exercise 2: Evaluating the Effect of Additional Spectral Bands

```{code-cell} ipython3

```

### Exercise 3: Tile Size and Overlap Sensitivity

```{code-cell} ipython3

```

### Exercise 4: Multi-Class Post-Processing

```{code-cell} ipython3

```
