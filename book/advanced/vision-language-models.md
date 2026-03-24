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
# Vision-Language Models

## Introduction

## Learning Objectives

## How Vision-Language Models Work

## Setting Up the Environment

```{code-cell} ipython3
# %pip install geoai-py transformers==4.57.6
```

```{code-cell} ipython3
import geoai
import leafmap
from geoai import MoondreamGeo
```

## Sample Data

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/parking-lot.tif"
image_path = geoai.download_file(url)
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite Image")
m
```

## Initializing the Moondream Processor

```{code-cell} ipython3
processor = MoondreamGeo(
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
```

## Image Captioning

```{code-cell} ipython3
result = processor.caption(image_path, length="short")
print(result["caption"])
```

```{code-cell} ipython3
result = processor.caption(image_path, length="normal")
print(result["caption"])
```

```{code-cell} ipython3
result = processor.caption(image_path, length="long")
print(result["caption"])
```

## Visual Question Answering

```{code-cell} ipython3
result = processor.query("How many buildings are in the image?", image_path)
print(result["answer"])
```

```{code-cell} ipython3
result = processor.query("What are the building roof colors?", image_path)
print(result["answer"])
```

```{code-cell} ipython3
result = processor.query("What types of vehicles are visible in the parking areas?", image_path)
print(result["answer"])
```

## Object Detection and Point Localization

### Detect Buildings

```{code-cell} ipython3
result = processor.detect(image_path, "building", output_path="buildings.geojson")
print(f"Detected {len(result['objects'])} buildings")
```

```{code-cell} ipython3
result["gdf"]
```

```{code-cell} ipython3
style = {"color": "red", "weight": 2}
m.add_gdf(result["gdf"], layer_name="Buildings", style=style)
m
```

### Locate Building Centroids

```{code-cell} ipython3
result = processor.point(
    image_path, "building", output_path="building_centroids.geojson"
)
print(f"Found {len(result['points'])} building centroids")
```

```{code-cell} ipython3
m.add_gdf(result["gdf"], layer_name="Building Centroids")
m
```

### Detect Trees

```{code-cell} ipython3
result = processor.detect(image_path, "tree", output_path="trees.geojson")
print(f"Detected {len(result['objects'])} trees")
```

```{code-cell} ipython3
m.add_gdf(result["gdf"], layer_name="Trees", style={"color": "green", "weight": 2})
```

### Locate Tree Centroids

```{code-cell} ipython3
result = processor.point(image_path, "tree", output_path="tree_centroids.geojson")
print(f"Found {len(result['points'])} tree centroids")
```

```{code-cell} ipython3
m.add_gdf(result["gdf"], layer_name="Tree Centroids")
m
```

## Interactive GUI

```{code-cell} ipython3
moondream = MoondreamGeo(
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
moondream.load_image(image_path)
m_gui = moondream.show_gui()
m_gui
```

```{code-cell} ipython3
gdf = m_gui.last_result_as_gdf
gdf
```

## Sliding Window Analysis for Large Rasters

### Object Detection with Sliding Window

```{code-cell} ipython3
result = processor.detect_sliding_window(
    image_path,
    "car",
    window_size=512,
    overlap=64,
    iou_threshold=0.5,
    output_path="cars_sliding_window.geojson",
)
print(f"Detected {len(result['objects'])} cars")
```

```{code-cell} ipython3
result["gdf"].head()
```

```{code-cell} ipython3
m2 = leafmap.Map()
m2.add_raster(image_path, layer_name="Satellite Image")
m2.add_gdf(
    result["gdf"],
    layer_name="Detected Cars",
    style={"color": "red", "fillOpacity": 0.3},
)
m2
```

### Point Detection with Sliding Window

```{code-cell} ipython3
trees = processor.point_sliding_window(
    image_path,
    "tree",
    window_size=512,
    overlap=64,
    output_path="trees_sliding_window.geojson",
)
print(f"Found {len(trees['points'])} tree locations")
```

```{code-cell} ipython3
m3 = leafmap.Map()
m3.add_raster(image_path, layer_name="Satellite Image")
m3.add_gdf(trees["gdf"], layer_name="Trees", style={"color": "green", "radius": 3})
m3
```

### Visual Question Answering with Sliding Window

```{code-cell} ipython3
result = processor.query_sliding_window(
    "What types of vehicles are visible?",
    image_path,
    window_size=512,
    overlap=64,
    combine_strategy="concatenate",
)
print(result["answer"])
```

```{code-cell} ipython3
result = processor.query_sliding_window(
    "Describe the land use and features in this area.",
    image_path,
    window_size=512,
    overlap=64,
    combine_strategy="summarize",
)
print(result["answer"])
```

```{code-cell} ipython3
for tile in result["tile_answers"][:2]:  # Show first 2 tiles
    print(f"Tile {tile['tile_id']}: {tile['answer']}\n")
```

### Image Captioning with Sliding Window

```{code-cell} ipython3
result = processor.caption_sliding_window(
    image_path,
    window_size=512,
    overlap=64,
    length="normal",
    combine_strategy="concatenate",
)
print(result["caption"])
```

```{code-cell} ipython3
result = processor.caption_sliding_window(
    image_path,
    window_size=512,
    overlap=64,
    length="long",
    combine_strategy="summarize",
)
print(result["caption"])
```

```{code-cell} ipython3
geoai.empty_cache()
```

### Convenience Functions

```{code-cell} ipython3
from geoai import moondream_detect_sliding_window

result = moondream_detect_sliding_window(
    image_path,
    "car",
    window_size=512,
    overlap=64,
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
print(f"Detected {len(result['objects'])} cars")
```

```{code-cell} ipython3
geoai.view_vector_interactive(result["gdf"], tiles=image_path)
```

### Comparing Regular vs. Sliding Window Detection

```{code-cell} ipython3
processor = MoondreamGeo(
    model_name="vikhyatk/moondream2",
    revision="2025-06-21",
)
```

```{code-cell} ipython3
regular_result = processor.detect(image_path, "car")
print(f"Regular detection: {len(regular_result['objects'])} cars")

sliding_result = processor.detect_sliding_window(
    image_path, "car", window_size=512, overlap=64
)
print(f"Sliding window detection: {len(sliding_result['objects'])} cars")
```

```{code-cell} ipython3
geoai.empty_cache()
```

### Performance Tips

## CLIP-Based Segmentation

```{code-cell} ipython3
clip_image_url = "https://data.source.coop/opengeos/geoai/uc-berkeley.tif"
clip_image_path = geoai.download_file(clip_image_url)
```

```{code-cell} ipython3
geoai.view_raster(clip_image_path)
```

```{code-cell} ipython3
segmenter = geoai.CLIPSegmentation(tile_size=512, overlap=32)
```

```{code-cell} ipython3
mask_output_path = "tree_masks.tif"
text_prompt = "trees"
```

```{code-cell} ipython3
segmenter.segment_image(
    clip_image_path,
    output_path=mask_output_path,
    text_prompt=text_prompt,
    threshold=0.5,
    smoothing_sigma=1.0,
)
```

```{code-cell} ipython3
geoai.view_raster(
    mask_output_path,
    nodata=0,
    opacity=0.7,
    colormap="greens",
    layer_name="Trees",
    basemap=clip_image_path,
)
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=mask_output_path,
    right_layer=clip_image_path,
    left_label="Trees",
    right_label="Satellite Image",
    left_args={"nodata": 0, "opacity": 0.8, "colormap": "greens"},
    basemap=clip_image_path,
)
```

## Practical Applications in Earth Observation

## Limitations and Considerations

## Key Takeaways

## Exercises

### Exercise 1: Caption Length Comparison

```{code-cell} ipython3

```

### Exercise 2: Geospatial Visual Question Answering

```{code-cell} ipython3

```

### Exercise 3: Object Detection and Counting

```{code-cell} ipython3

```

### Exercise 4: Multi-Class CLIP Segmentation

```{code-cell} ipython3

```
