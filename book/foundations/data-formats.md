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
# Geospatial Data Essentials

## Introduction

## Learning Objectives

## Raster Data

### What is Raster Data?

### Multispectral and Hyperspectral Imagery

### Spatial Resolution

### Common Raster Formats

### Reading Raster Data with Python

```{code-cell} ipython3
import geoai
import rasterio

raster_url = "https://data.source.coop/opengeos/geoai/naip-train.tif"
raster_path = geoai.download_file(raster_url)

with rasterio.open(raster_path) as src:
    print(f"Shape: {src.height} x {src.width}")
    print(f"Bands: {src.count}")
    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res}")
    band1 = src.read(1)  # Read the first band as a NumPy array
    print(f"Band 1 dtype: {band1.dtype}")
```

```{code-cell} ipython3
geoai.print_raster_info(raster_path, figsize=(18, 10))
```

```{code-cell} ipython3
clip_raster_path = "naip_clip.tif"
geoai.clip_raster_by_bbox(
    raster_path,
    clip_raster_path,
    bbox=(0, 0, 500, 500),
    bands=[1, 2, 3],
    bbox_type="pixel",
)
geoai.view_image(clip_raster_path)
```

## Vector Data

### What is Vector Data?

### Common Vector Formats

### Reading Vector Data with Python

```{code-cell} ipython3
import geopandas as gpd

vector_url = "https://data.source.coop/opengeos/geoai/naip-train-buildings.geojson"
vector_path = geoai.download_file(vector_url)
gdf = gpd.read_file(vector_path)
print(f"Features: {len(gdf)}")
print(f"CRS: {gdf.crs}")
gdf.head()
```

```{code-cell} ipython3
geoai.print_vector_info(vector_path, figsize=(18, 10))
```

```{code-cell} ipython3
geoai.view_vector(vector_path, basemap=True)
```

```{code-cell} ipython3
geoai.view_vector(
    vector_path,
    raster_path=raster_path,
    outline_only=True,
    edge_color="red",
    figsize=(18, 10),
)
```

## Coordinate Reference Systems

### Geographic vs. Projected CRS

### Common CRS

### Reprojection

```{code-cell} ipython3
gdf_utm = gdf.to_crs(crs="EPSG:26911")  # Reproject to UTM Zone 11N
print(f"Original CRS: {gdf.crs}")
print(f"New CRS: {gdf_utm.crs}")
print(f"Sample coordinates before: {gdf.geometry.iloc[0].centroid}")
print(f"Sample coordinates after:  {gdf_utm.geometry.iloc[0].centroid}")
```

```{code-cell} ipython3
import leafmap

leafmap.reproject(
    image=raster_path,
    output="naip_reprojected.tif",
    dst_crs="EPSG:3857",
    resampling="bilinear",
)
```

## Annotation Formats for Deep Learning

### Why Annotation Formats Matter

### COCO Format

### YOLO Format

### Pascal VOC Format

### Raster Masks

### Choosing the Right Format

## Image Tiling and Chips

### Why Tiling is Necessary

### Tile Size Considerations

### Maintaining Geospatial Context

```{code-cell} ipython3
leafmap.split_raster(
    raster_path,
    out_dir="naip_tiles",
    tile_size=512,
    overlap=128,
)
```

## Key Takeaways

## Exercises

### Exercise 1: Inspect a Raster File

```{code-cell} ipython3
```

### Exercise 2: Read and Explore Vector Data

```{code-cell} ipython3
```

### Exercise 3: Reproject a Dataset

```{code-cell} ipython3
```

### Exercise 4: Tile a Raster Image
