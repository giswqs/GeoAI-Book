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
# Vision-Language Models

## Introduction

## Learning Objectives

## How Vision-Language Models Work

## Setting Up the Environment

```{code-cell} python
# %pip install geoai leafmap
```

```{code-cell} python
import geoai
import leafmap
```

## Image Captioning

```{code-cell} python
building_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/caption-building.webp"
building_image = geoai.download_file(building_url)
```

```{code-cell} python
caption = geoai.moondream_caption(building_image)
print(caption)
```

```{code-cell} python
traffic_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/caption-traffic-sign.webp"
traffic_image = geoai.download_file(traffic_url)
caption = geoai.moondream_caption(traffic_image)
print(caption)
```

```{code-cell} python
water_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/caption-water.webp"
water_image = geoai.download_file(water_url)
caption = geoai.moondream_caption(water_image)
print(caption)
```

## Visual Question Answering

```{code-cell} python
answer = geoai.moondream_query(building_image, "How many stories does this building have?")
print(answer)
```

```{code-cell} python
answer = geoai.moondream_query(water_image, "What type of water body is shown in this image?")
print(answer)
```

```{code-cell} python
answer = geoai.moondream_query(traffic_image, "What text appears on the sign?")
print(answer)
```

```{code-cell} python
response = geoai.moondream(building_image, "Describe the architectural style and surrounding environment.")
print(response)
```

## Text-Guided Object Detection

```{code-cell} python
detections = geoai.moondream_detect(building_image, "window")
print(detections)
```

```{code-cell} python
points = geoai.moondream_point(building_image, "window")
print(points)
```

```{code-cell} python
detections = geoai.moondream_detect(traffic_image, "sign")
print(detections)
```

## Sliding Window Analysis for Large Rasters

```{code-cell} python
aerial_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/aerial.tif"
aerial_path = geoai.download_file(aerial_url)
```

```{code-cell} python
m = leafmap.Map()
m.add_raster(aerial_path, layer_name="Aerial Image")
m
```

### Sliding Window Captioning

```{code-cell} python
captions_gdf = geoai.moondream_caption_sliding_window(aerial_path)
captions_gdf.head()
```

```{code-cell} python
m2 = leafmap.Map()
m2.add_raster(aerial_path, layer_name="Aerial Image")
m2.add_gdf(captions_gdf, layer_name="Captions")
m2
```

### Sliding Window Query

```{code-cell} python
query_gdf = geoai.moondream_query_sliding_window(
    aerial_path,
    "What land cover types are visible in this area?"
)
query_gdf.head()
```

### Sliding Window Detection

```{code-cell} python
detect_gdf = geoai.moondream_detect_sliding_window(aerial_path, "building")
detect_gdf.head()
```

```{code-cell} python
m3 = leafmap.Map()
m3.add_raster(aerial_path, layer_name="Aerial Image")
m3.add_gdf(detect_gdf, layer_name="Building Detections", style={"color": "red", "weight": 2})
m3
```

### Sliding Window Point Localization

```{code-cell} python
points_gdf = geoai.moondream_point_sliding_window(aerial_path, "tree")
points_gdf.head()
```

```{code-cell} python
m4 = leafmap.Map()
m4.add_raster(aerial_path, layer_name="Aerial Image")
m4.add_gdf(points_gdf, layer_name="Tree Points", style={"color": "green"})
m4
```

## CLIP-Based Segmentation

```{code-cell} python
berkeley_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif"
berkeley_path = geoai.download_file(berkeley_url)
```

```{code-cell} python
m5 = leafmap.Map()
m5.add_raster(berkeley_path, layer_name="UC Berkeley")
m5
```

```{code-cell} python
result = geoai.CLIPSegmentation(
    berkeley_path,
    text_prompts=["building", "tree", "road"],
    output_dir="clip_seg_output",
)
```

```{code-cell} python
m6 = leafmap.Map()
m6.add_raster(berkeley_path, layer_name="UC Berkeley")
m6.add_raster("clip_seg_output/building.tif", layer_name="Buildings", colormap="Reds", opacity=0.6)
m6.add_raster("clip_seg_output/tree.tif", layer_name="Trees", colormap="Greens", opacity=0.6)
m6
```

## Practical Applications in Earth Observation

## Limitations and Considerations

## Key Takeaways

## Exercises

### Exercise 1: Comparative Captioning

```{code-cell} python
# Your code here
```

### Exercise 2: Geospatial Visual Question Answering

```{code-cell} python
# Your code here
```

### Exercise 3: Object Detection and Counting

```{code-cell} python
# Your code here
```

### Exercise 4: Multi-Class CLIP Segmentation

```{code-cell} python
# Your code here
```

### Exercise 5: VLM-Assisted Damage Assessment

```{code-cell} python
# Your code here
```
