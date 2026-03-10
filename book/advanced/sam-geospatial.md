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
# Segment Anything for Geospatial

## Introduction

## Learning Objectives

## How SAM Works

## SAM Model Variants

## Setting Up the Environment

```{code-cell} python
# %pip install segment-geospatial leafmap geoai
```

```{code-cell} python
import leafmap
import geoai
from samgeo import SamGeo
from samgeo.text_sam import LangSAM
```

## Automatic Mask Generation

```{code-cell} python
image_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/sam_demo_image.tif"
image_path = geoai.download_file(image_url)
```

```{code-cell} python
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite Image")
m
```

```{code-cell} python
sam = SamGeo(
    model_type="vit_h",
    automatic=True,
    sam_kwargs=None,
)
```

```{code-cell} python
mask_output = "sam_masks.tif"
sam.generate(
    image_path,
    mask_output,
    batch=True,
    foreground=True,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
)
```

```{code-cell} python
m2 = leafmap.Map()
m2.add_raster(image_path, layer_name="Image")
m2.add_raster(mask_output, layer_name="SAM Masks", nodata=0, opacity=0.7)
m2
```

```{code-cell} python
vector_output = "sam_masks.gpkg"
sam.tiff_to_gpkg(mask_output, vector_output, simplify_tolerance=None)
```

```{code-cell} python
style = {
    "color": "#3388ff",
    "weight": 2,
    "fillColor": "#7c4185",
    "fillOpacity": 0.5,
}
m2.add_vector(vector_output, layer_name="Segments", style=style)
m2
```

## Point and Box Prompts

```{code-cell} python
aerial_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/aerial.tif"
aerial_path = geoai.download_file(aerial_url)
```

```{code-cell} python
m3 = leafmap.Map()
m3.add_raster(aerial_path, layer_name="Aerial Image")
m3
```

```{code-cell} python
sam2 = SamGeo(
    model_type="vit_h",
    automatic=False,
    sam_kwargs=None,
)
sam2.set_image(aerial_path)
```

### Point Prompts

```{code-cell} python
point_coords = [[-122.1464, 37.6431], [-122.1449, 37.6415], [-122.1451, 37.6395]]
sam2.predict(
    point_coords,
    point_labels=1,
    point_crs="EPSG:4326",
    output="point_mask.tif",
)
```

```{code-cell} python
m4 = leafmap.Map()
m4.add_raster(aerial_path, layer_name="Aerial")
m4.add_raster("point_mask.tif", layer_name="Point Mask", nodata=0, cmap="Greens", opacity=0.7)
m4
```

### Box Prompts

```{code-cell} python
box_coords = [-122.1480, 37.6390, -122.1440, 37.6420]
sam2.predict(
    point_coords=None,
    point_labels=None,
    box=box_coords,
    point_crs="EPSG:4326",
    output="box_mask.tif",
)
```

```{code-cell} python
m5 = leafmap.Map()
m5.add_raster(aerial_path, layer_name="Aerial")
m5.add_raster("box_mask.tif", layer_name="Box Mask", nodata=0, cmap="Blues", opacity=0.7)
m5
```

## Text-Prompted Segmentation

```{code-cell} python
berkeley_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif"
berkeley_path = geoai.download_file(berkeley_url)
```

```{code-cell} python
m6 = leafmap.Map()
m6.add_raster(berkeley_path, layer_name="UC Berkeley")
m6
```

```{code-cell} python
lang_sam = LangSAM()
```

```{code-cell} python
lang_sam.predict(
    berkeley_path,
    text_prompt="building",
    box_threshold=0.24,
    text_threshold=0.24,
    output="building_masks.tif",
)
```

```{code-cell} python
lang_sam.show_anns(
    cmap="Greens",
    box_color="red",
    title="Buildings Detected by LangSAM",
    blend=True,
)
```

```{code-cell} python
lang_sam.raster_to_vector("building_masks.tif", "buildings.gpkg")
```

```{code-cell} python
m7 = leafmap.Map()
m7.add_raster(berkeley_path, layer_name="Image")
m7.add_vector("buildings.gpkg", layer_name="Buildings", style={"color": "red", "fillOpacity": 0.3})
m7
```

## Geospatial SAM Workflows

## Practical Tips and Best Practices

## Case Study: Feature Extraction from Aerial Imagery

```{code-cell} python
aerial_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/aerial.tif"
features_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/aerial_features.json"
aerial_path = geoai.download_file(aerial_url)
features_path = geoai.download_file(features_url)
```

```{code-cell} python
m8 = leafmap.Map()
m8.add_raster(aerial_path, layer_name="Aerial Imagery")
m8.add_vector(features_path, layer_name="Reference Features", style={"color": "yellow", "weight": 2})
m8
```

```{code-cell} python
sam_auto = SamGeo(
    model_type="vit_h",
    automatic=True,
    sam_kwargs={
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
    },
)
```

```{code-cell} python
case_mask = "case_study_masks.tif"
sam_auto.generate(
    aerial_path,
    case_mask,
    batch=True,
    foreground=True,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
)
```

```{code-cell} python
case_vector = "case_study_segments.gpkg"
sam_auto.tiff_to_gpkg(case_mask, case_vector, simplify_tolerance=None)
```

```{code-cell} python
m9 = leafmap.Map()
m9.add_raster(aerial_path, layer_name="Aerial")
m9.add_vector(
    case_vector,
    layer_name="SAM Segments",
    style={"color": "#3388ff", "weight": 1, "fillOpacity": 0.3},
)
m9.add_vector(
    features_path,
    layer_name="Reference",
    style={"color": "red", "weight": 2, "fillOpacity": 0},
)
m9
```

## Key Takeaways

## Exercises

### Exercise 1: Exploring Automatic Mask Generation Parameters

### Exercise 2: Interactive Segmentation with Point Prompts

### Exercise 3: Text-Prompted Feature Extraction

### Exercise 4: Comparing SAM Variants

### Exercise 5: End-to-End Feature Extraction Pipeline
