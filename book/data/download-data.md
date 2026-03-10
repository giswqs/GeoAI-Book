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
# Downloading Remote Sensing Data

## Introduction

## Learning Objectives

## Satellite Imagery Sources

### NAIP (National Agriculture Imagery Program)

### Sentinel-2

### Landsat

### Commercial Imagery

### Vantor Open Data Program

## Searching with STAC

### What is STAC?

### The STAC Specification

### STAC API Endpoints

### Python Libraries for STAC

```{code-cell} python
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
print(f"Catalog title: {catalog.title}")
print(f"Catalog description: {catalog.description}")

collections = list(catalog.get_collections())
print(f"\nNumber of collections: {len(collections)}")
print("\nFirst 10 collections:")
for collection in collections[:10]:
    print(f"  {collection.id}: {collection.title}")
```

### Exploring a STAC Collection

```{code-cell} python
collection = catalog.get_collection("sentinel-2-l2a")
print(f"Title: {collection.title}")
print(f"Description: {collection.description[:200]}...")
print(f"License: {collection.license}")
print(f"Temporal extent: {collection.extent.temporal.intervals}")
print(f"Spatial extent: {collection.extent.spatial.bboxes}")
```

### Searching for Items

```{code-cell} python
bbox = [-83.95, 35.94, -83.91, 35.98]  # Small area near Knoxville, TN

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=3,
)

items = search.item_collection()
print(f"Found {len(items)} items\n")

for item in items:
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover:.1f}%" if cloud_cover is not None else "N/A"
    )
    print(f"ID: {item.id}")
    print(f"  Date: {item.datetime}")
    print(f"  Cloud cover: {cloud_cover_text}")
    print(f"  Bounding box: {item.bbox}")
    print()
```

### Inspecting Items and Assets

```{code-cell} python
if items:
    item = items[0]
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover}%" if cloud_cover is not None else "N/A"
    )
    print(f"Item ID: {item.id}")
    print(f"Date: {item.datetime}")
    print(f"Cloud cover: {cloud_cover_text}")
    print(f"Platform: {item.properties.get('platform', 'N/A')}")
    print(f"\nAvailable assets ({len(item.assets)}):")
    for key, asset in item.assets.items():
        roles = ", ".join(asset.roles) if asset.roles else "N/A"
        print(f"  {key}: {asset.title or 'No title'} [{roles}]")
```

## Downloading NAIP Imagery

```{code-cell} python
import geoai

bbox = [-83.94, 35.96, -83.92, 35.98]  # Small area near Knoxville, TN
output_dir = "naip_data"

filepaths = geoai.download_naip(
    bbox=bbox,
    output_dir=output_dir,
    year=2021,
    max_items=1,
)
print(f"Downloaded {len(filepaths)} file(s):")
for fp in filepaths:
    print(f"  {fp}")
```

```{code-cell} python
import rasterio

if filepaths:
    with rasterio.open(filepaths[0]) as src:
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res[0]:.2f} m")
        print(f"Bounds: {src.bounds}")
        print(f"Data type: {src.dtypes[0]}")
```

```{code-cell} python
geoai.print_raster_info(filepaths[0])
```

## Downloading Sentinel-2 Data

### Searching for Sentinel-2 Items

```{code-cell} python
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
bbox = [-83.94, 35.96, -83.92, 35.98]

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=1,
)

items = search.item_collection()
if items:
    item = items[0]
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover}%" if cloud_cover is not None else "N/A"
    )
    print(f"Selected item: {item.id}")
    print(f"Date: {item.datetime}")
    print(f"Cloud cover: {cloud_cover_text}")
    item_url = item.self_href
    print(f"Item URL: {item_url}")
```

### Downloading and Merging Bands

```{code-cell} python
if items:
    result = geoai.download_pc_stac_item(
        item_url=item_url,
        bands=["B02", "B03", "B04", "B08"],  # Blue, Green, Red, NIR
        output_dir="sentinel2_data",
        merge_bands=True,
        merged_filename="knoxville_s2_rgbn.tif",
        overwrite=False,
    )

    for band_name, path in result.items():
        print(f"  {band_name}: {path}")
```

## Downloading Landsat Data

```{code-cell} python
from pystac_client import Client

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
bbox = [-83.94, 35.96, -83.92, 35.98]

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
    max_items=1,
)

items = search.item_collection()
if items:
    item = items[0]
    cloud_cover = item.properties.get("eo:cloud_cover")
    cloud_cover_text = (
        f"{cloud_cover}%" if cloud_cover is not None else "N/A"
    )
    print(f"Selected item: {item.id}")
    print(f"Date: {item.datetime}")
    print(f"Cloud cover: {cloud_cover_text}")
    print(f"\nAvailable assets:")
    for key in list(item.assets.keys())[:10]:
        print(f"  {key}")
```

```{code-cell} python
if items:
    landsat_url = items[0].self_href
    result = geoai.download_pc_stac_item(
        item_url=landsat_url,
        bands=["blue", "green", "red", "nir08"],
        output_dir="landsat_data",
        merge_bands=True,
        merged_filename="knoxville_landsat_rgbn.tif",
        overwrite=False,
    )

    for band_name, path in result.items():
        print(f"  {band_name}: {path}")
```

```{code-cell} python
if items:
    downloaded = geoai.pc_stac_download(
        items=items[0],
        output_dir="landsat_data_raw",
        assets=["blue", "green", "red", "nir08"],
        max_workers=4,
    )
    for item_id, assets in downloaded.items():
        print(f"Item: {item_id}")
        for asset_key, fpath in assets.items():
            print(f"  {asset_key}: {fpath}")
```

## Downloading Vantor Open Data

```{code-cell} python
from pystac_client import Client

vantor_catalog_url = (
    "https://vantor-opendata.s3.amazonaws.com/events/catalog.json"
)
vantor_catalog = Client.open(vantor_catalog_url)

collections = list(vantor_catalog.get_collections())
print(f"Number of event collections: {len(collections)}")
for collection in collections:
    print(f"  {collection.id}: {collection.title}")
```

```{code-cell} python
if collections:
    event = collections[0]
    print(f"Event: {event.title}")
    print(f"Description: {event.description}")
    print(f"License: {event.license}")
    print(f"Temporal extent: {event.extent.temporal.intervals}")
    print(f"Spatial extent: {event.extent.spatial.bboxes}")
```

## Accessing Vector Data

### Overture Maps Buildings

```{code-cell} python
import geoai

bbox = (-83.94, 35.96, -83.92, 35.98)  # Knoxville area
output_path = "buildings.geojson"

geoai.download_overture_buildings(
    bbox=bbox,
    output=output_path,
    overture_type="building",
)
print(f"Buildings saved to {output_path}")
```

```{code-cell} python
gdf = geoai.get_overture_data(
    overture_type="building",
    bbox=(-83.94, 35.96, -83.92, 35.98),
    output="buildings.parquet",
)
print(f"Downloaded {len(gdf)} buildings")
gdf.head()
```

### OpenStreetMap Data

```{code-cell} python
import leafmap.osm as osm

bbox = (-83.94, 35.96, -83.92, 35.98)  # Knoxville area
buildings = osm.quackosm_gdf_from_bbox(bbox, tags={"building": True})
print(f"Downloaded {len(buildings)} buildings")
buildings.head()
```

```{code-cell} python
roads = osm.quackosm_gdf_from_place("Knoxville, Tennessee", tags={"highway": True})
print(f"Downloaded {len(roads)} road segments")
roads.head()
```

```{code-cell} python
from shapely.geometry import Polygon

polygon = Polygon([
    (-83.94, 35.96),
    (-83.92, 35.96),
    (-83.92, 35.98),
    (-83.94, 35.98),
])
natural = osm.quackosm_gdf_from_geometry(polygon, tags={"natural": True})
print(f"Downloaded {len(natural)} natural features")
natural.head()
```

## Organizing Your Data

## Key Takeaways

## Exercises

### Exercise 1: Search for NAIP Imagery

```{code-cell} python
```

### Exercise 2: Download and Inspect Sentinel-2 Data

```{code-cell} python
```

### Exercise 3: Compare Landsat and Sentinel-2

```{code-cell} python
```

### Exercise 4: Download Building Footprints

```{code-cell} python
```

### Exercise 5: Build a Data Acquisition Script

```{code-cell} python
```
