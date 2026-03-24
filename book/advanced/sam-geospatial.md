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
# Segment Anything for Geospatial

## Introduction

## Learning Objectives

## How SAM 3 Works

## Setting Up the Environment

```{code-cell} ipython3
# %pip install geoai-py "segment-geospatial[samgeo3]"
```

```{code-cell} ipython3
import os
import geoai
import leafmap
from samgeo import SamGeo3, SamGeo3Video, download_file, show_image
from samgeo.common import raster_to_vector, regularize
```

```{code-cell} ipython3
# from huggingface_hub import login
# login()
```

## Image Segmentation

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/uc-berkeley.tif"
image_path = download_file(url)
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite image")
m
```

```{code-cell} ipython3
sam3 = SamGeo3(backend="meta", device=None, checkpoint_path=None, load_from_HF=True)
sam3.set_image(image_path)
```

### Text-Prompted Segmentation

```{code-cell} ipython3
sam3.generate_masks(prompt="building")
```

```{code-cell} ipython3
sam3.show_anns()
```

```{code-cell} ipython3
sam3.show_masks()
```

```{code-cell} ipython3
sam3.save_masks(output="building_masks.tif", unique=True)
```

```{code-cell} ipython3
sam3.save_masks(
    output="building_masks_with_scores.tif",
    save_scores="building_scores.tif",
    unique=True,
)
```

```{code-cell} ipython3
sam3.show_masks(cmap="coolwarm")
```

```{code-cell} ipython3
m.add_raster("building_masks.tif", layer_name="Building masks", visible=False)
m.add_raster(
    "building_scores.tif",
    layer_name="Building scores",
    cmap="coolwarm",
    opacity=0.8,
    nodata=0,
    vmin=0.5,
    vmax=1.0,
)
m.add_colormap(cmap="coolwarm", vmin=0.5, vmax=1.0, label="Confidence score")
m
```

### Box-Prompted Segmentation

```{code-cell} ipython3
# Define boxes in [xmin, ymin, xmax, ymax] format
boxes = [[-122.2597, 37.8709, -122.2587, 37.8717]]
box_labels = [True]  # True=include, False=exclude

sam3.generate_masks_by_boxes(boxes, box_labels, box_crs="EPSG:4326")
```

```{code-cell} ipython3
sam3.show_boxes(boxes, box_labels, box_crs="EPSG:4326")
```

```{code-cell} ipython3
sam3.show_anns()
```

```{code-cell} ipython3
building_mask_path = "building_masks.tif"
sam3.save_masks(output=building_mask_path, unique=True)
```

```{code-cell} ipython3
geoai.view_raster(
    building_mask_path, nodata=0, opacity=0.7, basemap=image_path
)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Point Prompts for Instance Segmentation

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/truck-example.jpg"
image_path = download_file(url)
```

```{code-cell} ipython3
show_image(image_path, axis="on")
```

```{code-cell} ipython3
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
sam.set_image(image_path)
```

### Single Point Prompt

```{code-cell} ipython3
sam.generate_masks_by_points([[750, 370]])
```

```{code-cell} ipython3
sam.show_points([[750, 370]], [1])
```

```{code-cell} ipython3
sam.show_anns()
```

```{code-cell} ipython3
sam.save_masks("truck_mask.png", unique=True)
```

### Multiple Points

```{code-cell} ipython3
sam.generate_masks_by_points([[500, 375], [1125, 625]], point_labels=[1, 1])
```

```{code-cell} ipython3
sam.show_points([[500, 375], [1125, 625]], [1, 1])
```

```{code-cell} ipython3
sam.show_anns()
```

### Background Points

```{code-cell} ipython3
sam.generate_masks_by_points([[750, 370], [1125, 625]], point_labels=[1, 0])
```

```{code-cell} ipython3
sam.show_points([[750, 370], [1125, 625]], [1, 0])
```

```{code-cell} ipython3
sam.show_anns()
```

### Multiple Box Prompts

```{code-cell} ipython3
boxes = [
    [75, 275, 1725, 850],  # Whole truck
    [425, 600, 700, 875],  # Rear wheel
    [1375, 550, 1650, 800],  # Front wheel on the passenger side
    [1240, 675, 1400, 750],  # Front wheel on the driver's side
]
sam.generate_masks_by_boxes_inst(boxes)
```

```{code-cell} ipython3
sam.show_boxes(boxes)
```

```{code-cell} ipython3
sam.show_anns()
```

```{code-cell} ipython3
sam.save_masks("truck_boxes_mask.png", unique=True)
```

```{code-cell} ipython3
geoai.empty_cache()
```

### Batch Point Prompts for Geospatial Data

```{code-cell} ipython3
image_url = "https://data.source.coop/opengeos/geoai/wa-building-image.tif"
geojson_url = "https://data.source.coop/opengeos/geoai/wa-building-centroids.geojson"
image_path = download_file(image_url)
geojson_path = download_file(geojson_url)
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite image")
m
```

```{code-cell} ipython3
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
sam.set_image(image_path)
```

```{code-cell} ipython3
point_coords_batch = [
    [-117.599896, 47.655345],
    [-117.59992, 47.655167],
    [-117.599928, 47.654974],
    [-117.599518, 47.655337],
]

sam.generate_masks_by_points_patch(
    point_coords_batch=point_coords_batch,
    point_crs="EPSG:4326",
    output="masks.tif",
    dtype="uint8",
)
```

```{code-cell} ipython3
sam.show_points(point_coords_batch, point_crs="EPSG:4326")
```

```{code-cell} ipython3
m.add_raster("masks.tif", cmap="viridis", nodata=0, opacity=0.7, layer_name="Mask")
m
```

```{code-cell} ipython3
sam.generate_masks_by_points_patch(
    point_coords_batch=geojson_path,
    point_crs="EPSG:4326",
    output="building_masks.tif",
    dtype="uint16",
)
```

```{code-cell} ipython3
m.add_raster(
    "building_masks.tif", cmap="jet", nodata=0, opacity=0.7, layer_name="Building masks"
)
m.add_circle_markers_from_xy(
    geojson_path, radius=3, color="red", fill_color="yellow", fill_opacity=0.8
)
m
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Box Prompts for Building Extraction

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite image")
m
```

```{code-cell} ipython3
sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
sam.set_image(image_path)
```

```{code-cell} ipython3
if m.user_rois is not None:
    boxes = m.user_rois
else:
    boxes = [
        [-117.5995, 47.6518, -117.5988, 47.652],
        [-117.5987, 47.6518, -117.5979, 47.652],
    ]
```

```{code-cell} ipython3
sam.generate_masks_by_boxes_inst(boxes=boxes, box_crs="EPSG:4326")
```

```{code-cell} ipython3
sam.save_masks(output="mask.tif", dtype="uint8")
```

```{code-cell} ipython3
m.add_raster("mask.tif", cmap="viridis", nodata=0, opacity=0.5, layer_name="Mask")
m
```

### Using a Vector File as Box Prompts

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/wa-building-bboxes.geojson"
geojson_path = download_file(url)
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Image")
style = {
    "color": "#ffff00",
    "weight": 2,
    "fillColor": "#7c4185",
    "fillOpacity": 0,
}
m.add_vector(geojson_path, style=style, zoom_to_layer=True, layer_name="Bboxes")
m
```

```{code-cell} ipython3
output_masks = "building_masks.tif"
sam.generate_masks_by_boxes_inst(
    boxes=geojson_path,
    box_crs="EPSG:4326",
    output=output_masks,
    dtype="uint16",
    multimask_output=False,
)
```

```{code-cell} ipython3
m.add_raster(
    output_masks, cmap="jet", nodata=0, opacity=0.5, layer_name="Building masks"
)
m
```

### Converting to Vector and Regularizing

```{code-cell} ipython3
output_vector = "building_vector.geojson"
raster_to_vector(output_masks, output_vector)
```

```{code-cell} ipython3
output_regularized = "building_regularized.geojson"
regularize(output_vector, output_regularized)
```

```{code-cell} ipython3
m.add_vector(
    output_regularized, style=style, layer_name="Building regularized", info_mode=None
)
m
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Batch Segmentation

```{code-cell} ipython3
image_paths = []
for i in range(1, 5):
    url = f"https://data.source.coop/opengeos/geoai/uc-berkeley-{i}.tif"
    image_path = download_file(url)
    image_paths.append(image_path)
```

```{code-cell} ipython3
m = leafmap.Map()
for i, image_path in enumerate(image_paths):
    m.add_raster(image_path, layer_name=f"image_{i + 1}")
m
```

```{code-cell} ipython3
sam3 = SamGeo3(backend="meta", device=None, checkpoint_path=None, load_from_HF=True)
sam3.set_image_batch(image_paths)
```

```{code-cell} ipython3
sam3.generate_masks_batch("building", min_size=100)
```

```{code-cell} ipython3
for i, result in enumerate(sam3.batch_results):
    print(f"Image {i + 1}: Found {len(result['masks'])} objects")
```

```{code-cell} ipython3
sam3.show_anns_batch(ncols=2, show_bbox=True, show_score=True, figsize=(12, 8))
```

```{code-cell} ipython3
sam3.show_anns_batch(output_dir="output/annotations/", prefix="ann", dpi=300)
```

```{code-cell} ipython3
saved_files = sam3.save_masks_batch(
    output_dir="output/", prefix="building_mask", unique=True
)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Tiled Segmentation for Large Images

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/naip_water_train.tif"
image_path = download_file(url)
```

```{code-cell} ipython3
geoai.print_raster_info(image_path)
```

```{code-cell} ipython3
sam = SamGeo3(backend="meta")

output_path = "segmentation_mask.tif"

sam.generate_masks_tiled(
    source=image_path,
    prompt="water",
    output=output_path,
    tile_size=1024,
    overlap=128,
    min_size=100,
    unique=False,
    dtype="int32",
    verbose=True,
)
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Original Image")
m.add_raster(
    output_path, nodata=0, opacity=0.8, cmap="Blues", layer_name="Segmentation Mask"
)
m
```

```{code-cell} ipython3
vector_path = "segmentation_mask.gpkg"
geoai.raster_to_vector(output_path, vector_path)
```

```{code-cell} ipython3
smooth_vector_path = "segmentation_mask_smooth.gpkg"
gdf = geoai.smooth_vector(vector_path, smooth_vector_path)
```

```{code-cell} ipython3
style = {
    "color": "#ff0000",
    "weight": 2,
    "fillOpacity": 0,
}
m.add_gdf(gdf, layer_name="Smoothed Vector", info_mode=None, style=style)
m
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Interactive Segmentation

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/uc-berkeley.tif"
image_path = download_file(url)
```

```{code-cell} ipython3
sam3 = SamGeo3(
    backend="transformers", device=None, checkpoint_path=None, load_from_HF=True
)
sam3.set_image(image_path)
```

```{code-cell} ipython3
sam3.generate_masks(prompt="building")
sam3.save_masks("masks.tif")
```

```{code-cell} ipython3
sam3.show_map(height="700px", min_size=10)
```

```{code-cell} ipython3
geoai.empty_cache()
```

## Video Segmentation

### Text-Prompted Video Segmentation

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/cars.mp4"
video_path = download_file(url)
```

```{code-cell} ipython3
sam = SamGeo3Video()
sam.set_video(video_path)
sam.show_video(video_path)
```

```{code-cell} ipython3
sam.generate_masks("car")
```

```{code-cell} ipython3
sam.show_frame(0, axis="on")
```

```{code-cell} ipython3
sam.show_frames(frame_stride=20, ncols=3)
```

```{code-cell} ipython3
sam.remove_object(2)
sam.propagate()
sam.show_frame(0)
```

```{code-cell} ipython3
os.makedirs("output", exist_ok=True)
sam.save_masks("output/masks")
```

```{code-cell} ipython3
sam.save_video("output/segmented.mp4", fps=25)
```

```{code-cell} ipython3
sam.close()
```

### Point-Prompted Video Segmentation

```{code-cell} ipython3
sam = SamGeo3Video()
sam.set_video(video_path)
sam.init_tracker()
sam.show_frame(0, axis="on")
```

```{code-cell} ipython3
sam.add_point_prompts([[300, 200]], [1], obj_id=1, frame_idx=0)
sam.add_point_prompts([[420, 200]], [1], obj_id=2, frame_idx=0)
sam.propagate()
```

```{code-cell} ipython3
sam.show_frame(0, axis="on")
```

```{code-cell} ipython3
sam.show_frames(frame_stride=20, ncols=3)
```

```{code-cell} ipython3
# Positive point on windshield, negative point on car body
sam.add_point_prompts(
    points=[[335, 195], [335, 220]],
    labels=[1, 0],
    obj_id=1,
    frame_idx=0,
)
sam.propagate()
sam.show_frames(frame_stride=20, ncols=3)
```

```{code-cell} ipython3
sam.save_masks("output/masks")
sam.save_video("output/segmented.mp4", fps=25)
sam.close()
```

### Object Tracking

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/basketball.mp4"
video_path = download_file(url)
```

```{code-cell} ipython3
sam = SamGeo3Video()
sam.set_video(video_path)
sam.show_video(video_path)
```

```{code-cell} ipython3
sam.generate_masks("player")
```

```{code-cell} ipython3
player_names = {}
for i in range(15):
    player_names[i] = f"Player {i}"
sam.show_frame(0, axis="on", show_ids=player_names)
```

```{code-cell} ipython3
sam.remove_object(obj_id=[5, 8, 12, 13])
sam.propagate()
sam.show_frame(0, show_ids=player_names)
```

```{code-cell} ipython3
os.makedirs("output", exist_ok=True)
sam.save_masks("output/masks")
sam.save_video("output/players_segmented.mp4", fps=60, show_ids=player_names)
sam.show_video("output/players_segmented.mp4")
```

```{code-cell} ipython3
sam.close()
sam.shutdown()
```

## Key Takeaways

## Exercises

### Exercise 1: Comparing Prompt Types

```{code-cell} ipython3

```

### Exercise 2: Batch Segmentation Across Tiles

```{code-cell} ipython3

```

### Exercise 3: Tiled Segmentation on a Large Image

```{code-cell} ipython3

```

### Exercise 4: Point-Prompted Instance Segmentation with Geographic Coordinates

```{code-cell} ipython3

```

### Exercise 5: End-to-End Video Segmentation Pipeline

```{code-cell} ipython3

```
