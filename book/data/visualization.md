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
# Interactive Mapping & Visualization

## Introduction

## Learning Objectives

## Creating Interactive Maps with Leafmap

### Your First Map

```{code-cell} ipython3
import leafmap

m = leafmap.Map(center=[36.1617, -115.1524], zoom=11, height="600px")
m
```

### Adding Basemaps

```{code-cell} ipython3
m = leafmap.Map()
m.add_basemap("Esri.WorldImagery")
m
```

```{code-cell} ipython3
basemaps = list(leafmap.basemaps.keys())
print(f"Total basemaps: {len(basemaps)}")
print("First 10:", basemaps[:10])
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_basemap("Esri.WorldImagery")
m.add_basemap("OpenTopoMap")
m
```

## Working with Raster Data on Maps

### Adding GeoTIFF Layers

```{code-cell} ipython3
import geoai
```

```{code-cell} ipython3
naip_url = "https://data.source.coop/opengeos/geoai/las-vegas-train-naip.tif"
hag_url = "https://data.source.coop/opengeos/geoai/las-vegas-train-hag.tif"
buildings_url = (
    "https://data.source.coop/opengeos/geoai/las-vegas-buildings-train.geojson"
)
buildings_mask_url = (
    "https://data.source.coop/opengeos/geoai/las-vegas-buildings-mask.tif"
)
```

```{code-cell} ipython3
naip_path = geoai.download_file(naip_url)
hag_path = geoai.download_file(hag_url)
buildings_path = geoai.download_file(buildings_url)
mask_path = geoai.download_file(buildings_mask_url)
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path, layer_name="NAIP Image")
m
```

```{code-cell} ipython3
m.add_raster(
    hag_path, vmin=0, vmax=10, colormap="terrain", layer_name="Height Above Ground"
)
```

### Cloud-Optimized GeoTIFF (COG)

```{code-cell} ipython3
m = leafmap.Map()
m.add_cog_layer(naip_url, name="Las Vegas NAIP")
m
```

### Band Combinations

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path, indexes=[4, 1, 2], layer_name="False Color")
m
```

## Visualizing Data from Planetary Computer

### Browsing Available Collections

```{code-cell} ipython3
collections = geoai.pc_collection_list()
print(f"Total collections: {len(collections)}")
collections.head(10)
```

### Searching for STAC Items

```{code-cell} ipython3
naip_items = geoai.pc_stac_search(
    collection="naip",
    bbox=[-76.6657, 39.2648, -76.6478, 39.2724],
    time_range="2013-01-01/2014-12-31",
)
naip_items
```

```{code-cell} ipython3
geoai.pc_item_asset_list(naip_items[0])
```

### Visualizing NAIP Imagery

```{code-cell} ipython3
geoai.view_pc_item(item=naip_items[0])
```

### Visualizing Land Cover Data

```{code-cell} ipython3
lc_items = geoai.pc_stac_search(
    collection="chesapeake-lc-13",
    bbox=[-76.6657, 39.2648, -76.6478, 39.2724],
    time_range="2013-01-01/2014-12-31",
    max_items=10,
)
lc_items
```

```{code-cell} ipython3
geoai.view_pc_item(item=lc_items[0], colormap_name="tab10", basemap="SATELLITE")
```

### Visualizing Landsat Imagery

```{code-cell} ipython3
landsat_items = geoai.pc_stac_search(
    collection="landsat-c2-l2",
    bbox=[-76.6657, 39.2648, -76.6478, 39.2724],
    time_range="2024-10-27/2024-12-31",
    query={"eo:cloud_cover": {"lt": 1}},
    max_items=10,
)
landsat_items
```

```{code-cell} ipython3
geoai.pc_item_asset_list(landsat_items[0])
```

```{code-cell} ipython3
geoai.view_pc_item(item=landsat_items[0], assets=["red", "green", "blue"])
```

```{code-cell} ipython3
geoai.view_pc_item(item=landsat_items[0], assets=["nir08", "red", "green"])
```

```{code-cell} ipython3
geoai.view_pc_item(
    item=landsat_items[0],
    expression="(nir08-red)/(nir08+red)",
    rescale="-1,1",
    colormap_name="greens",
    name="NDVI",
)
```

### Downloading Data from Planetary Computer

```{code-cell} ipython3
geoai.pc_stac_download(naip_items, output_dir="data", assets=["image", "thumbnail"])
```

```{code-cell} ipython3
ds = geoai.read_pc_item_asset(lc_items[0], asset="data")
ds
```

## Working with Vector Data on Maps

### Adding GeoJSON and GeoDataFrame

```{code-cell} ipython3
import geopandas as gpd

gdf = gpd.read_file(buildings_path)
print(f"Features: {len(gdf)}")
gdf.head()
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path)
m.add_gdf(gdf, layer_name="Building Footprints", zoom_to_layer=True)
m
```

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path)
m.add_geojson(buildings_url, layer_name="Buildings", zoom_to_layer=True)
m
```

### Styling Vector Layers

```{code-cell} ipython3
style = {
    "color": "red",
    "weight": 2,
    "fillColor": "yellow",
    "fillOpacity": 0.3,
}
m = leafmap.Map()
m.add_raster(naip_path)
m.add_gdf(gdf, layer_name="Styled Buildings", style=style, zoom_to_layer=True)
m
```

### Adding Markers and Points

```{code-cell} ipython3
m = leafmap.Map(center=[36.1617, -115.1524], zoom=11)
m.add_marker(location=[36.1617, -115.1524])
m
```

## Split-Panel Maps for Comparison

### Side-by-Side Comparison

```{code-cell} ipython3
m = leafmap.Map()
m.split_map(
    left_layer=naip_path,
    right_layer=hag_path,
    left_args={"indexes": [4, 1, 2]},
    right_args={"vmin": 0, "vmax": 10, "cmap": "terrain"},
    left_label="NAIP",
    right_label="HAG",
)
m
```

## Visualizing Model Results

### Overlaying Predictions

```{code-cell} ipython3
m = leafmap.Map()
m.add_raster(naip_path, layer_name="NAIP Imagery")
m.add_raster(mask_path, opacity=0.8, nodata=0, layer_name="Building Mask")
m
```

### Comparing Labels with Source Imagery

```{code-cell} ipython3
import leafmap

m = leafmap.Map()
m.split_map(
    left_layer=buildings_path,
    right_layer=naip_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
)
m
```

```{code-cell} ipython3
geoai.create_split_map(
    left_layer=buildings_path,
    right_layer=naip_path,
    left_args={"style": {"color": "red", "fillOpacity": 0.2}},
    basemap=naip_path,
)
```

## Best Practices for GeoAI Visualization

## Key Takeaways

## Exercises

### Exercise 1: Create a Custom Basemap Map

```{code-cell} ipython3
```

### Exercise 2: Visualize a Raster with Multiple Colormaps

```{code-cell} ipython3
```

### Exercise 3: Style and Explore Vector Data

```{code-cell} ipython3
```

### Exercise 4: Build a Split-Panel Comparison

```{code-cell} ipython3
```

### Exercise 5: Overlay and Compare Building Layers

```{code-cell} ipython3
```
