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
# Satellite Embeddings

## Introduction

## Learning Objectives

## What Are Satellite Embeddings?

## Setting Up the Environment

```{code-cell} ipython3
# %pip install geoai-py
```

```{code-cell} ipython3
import geoai
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
```

## Browsing Available Embedding Datasets

```{code-cell} ipython3
df = geoai.list_embedding_datasets(verbose=False)
df
```

```{code-cell} ipython3
geoai.list_embedding_datasets(kind="patch", verbose=False)
```

```{code-cell} ipython3
geoai.list_embedding_datasets(kind="pixel", verbose=False)
```

```{code-cell} ipython3
info = geoai.get_embedding_info("clay")
for key, value in info.items():
    print(f"{key}: {value}")
```

## Exploring Patch-Based Embeddings

### Downloading Clay Embeddings

```{code-cell} ipython3
repo_id = "made-with-clay/classify-embeddings-sf-baseball-marinas"

api = HfApi()
embedding_files = [
    f.path
    for f in api.list_repo_tree(repo_id, repo_type="dataset")
    if f.path.endswith(".gpq")
]
print(f"Found {len(embedding_files)} embedding tiles")
```

```{code-cell} ipython3
all_gdfs = []
for f in embedding_files:
    path = hf_hub_download(repo_id, f, repo_type="dataset")
    gdf = gpd.read_parquet(path)
    all_gdfs.append(gdf)

embeddings_gdf = pd.concat(all_gdfs, ignore_index=True)
embeddings_gdf = gpd.GeoDataFrame(
    embeddings_gdf, geometry="geometry", crs=all_gdfs[0].crs
)
print(f"Combined: {len(embeddings_gdf)} patches")
print(f"Bounds: {embeddings_gdf.total_bounds}")
print(f"Embedding dimension: {len(embeddings_gdf.iloc[0]['embeddings'])}")
```

```{code-cell} ipython3
labels_file = hf_hub_download(repo_id, "baseball.geojson", repo_type="dataset")
labels_gdf = gpd.read_file(labels_file)
print(f"Labeled locations: {len(labels_gdf)}")
print(f"Class distribution:")
print(labels_gdf["class"].value_counts())
```

### Extracting Embedding Vectors

```{code-cell} ipython3
embeddings = np.stack(embeddings_gdf["embeddings"].values)
centroids = embeddings_gdf.geometry.centroid
coords_x = centroids.x.values
coords_y = centroids.y.values

print(f"Embeddings shape: {embeddings.shape}")
print(f"X range: [{coords_x.min():.4f}, {coords_x.max():.4f}]")
print(f"Y range: [{coords_y.min():.4f}, {coords_y.max():.4f}]")
```

## Visualizing Embeddings

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for i, ax in enumerate(axes):
    idx = i * (len(embeddings) // 3)
    ax.plot(embeddings[idx], linewidth=0.5)
    ax.set_title(f"Patch {idx}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
fig = geoai.visualize_embeddings(
    embeddings,
    method="pca",
    figsize=(10, 8),
    s=3,
    alpha=0.4,
    title="PCA of Clay Embeddings (SF Bay Area)",
)
plt.show()
```

## Clustering Embeddings

```{code-cell} ipython3
result = geoai.cluster_embeddings(embeddings, n_clusters=8, method="kmeans")
cluster_labels = result["labels"]
print(f"Number of clusters: {result['n_clusters']}")
print(f"Cluster sizes: {np.bincount(cluster_labels)}")
```

```{code-cell} ipython3
fig = geoai.visualize_embeddings(
    embeddings,
    labels=cluster_labels,
    method="pca",
    figsize=(10, 8),
    s=5,
    alpha=0.5,
    title="K-Means Clusters of Clay Embeddings",
)
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))
n_clusters = len(set(cluster_labels))
scatter = ax.scatter(
    coords_x,
    coords_y,
    c=cluster_labels,
    cmap=plt.colormaps["tab10"].resampled(n_clusters),
    vmin=-0.5,
    vmax=n_clusters - 0.5,
    s=3,
    alpha=0.6,
)
plt.colorbar(scatter, ax=ax, label="Cluster", ticks=range(n_clusters))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution of Embedding Clusters")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

## Similarity Search

```{code-cell} ipython3
query_idx = 0
query = embeddings[query_idx]
print(f"Query location (lat, lon): ({coords_y[query_idx]:.4f}, {coords_x[query_idx]:.4f})")

results = geoai.embedding_similarity(
    query=query, embeddings=embeddings, metric="cosine", top_k=10
)

print("\nTop 10 most similar locations:")
for rank, (idx, score) in enumerate(
    zip(results["indices"], results["scores"]), start=1
):
    print(
        f"  {rank}. Index {idx}: similarity={score:.4f}, "
        f"location=({coords_y[idx]:.4f}, {coords_x[idx]:.4f})"
    )
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 8))

# Background: all embeddings in gray
ax.scatter(coords_x, coords_y, c="lightgray", s=1, alpha=0.3)

# Highlight nearest neighbors
nn_indices = results["indices"]
ax.scatter(
    coords_x[nn_indices],
    coords_y[nn_indices],
    c="blue",
    s=50,
    marker="o",
    label="Nearest Neighbors",
    edgecolors="black",
    linewidths=0.5,
)

# Highlight the query point
ax.scatter(
    coords_x[query_idx],
    coords_y[query_idx],
    c="red",
    s=100,
    marker="*",
    label="Query",
    edgecolors="black",
    linewidths=0.5,
    zorder=5,
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Similarity Search: Query and Nearest Neighbors")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

## Training Classifiers on Embeddings

```{code-cell} ipython3
# Ensure both GeoDataFrames use the same CRS
if labels_gdf.crs != embeddings_gdf.crs:
    labels_gdf = labels_gdf.to_crs(embeddings_gdf.crs)

# Spatial join: find which embedding patch each labeled point falls within
joined = gpd.sjoin(labels_gdf, embeddings_gdf, how="inner", predicate="within")
print(f"Matched {len(joined)} labeled points to embedding patches")
print(f"Class distribution: {joined['class'].value_counts().to_dict()}")
```

```{code-cell} ipython3
labeled_embeddings = np.stack(
    [embeddings_gdf.iloc[idx]["embeddings"] for idx in joined["index_right"]]
)
class_labels = joined["class"].values

print(f"Labeled embeddings shape: {labeled_embeddings.shape}")
print(f"Labels shape: {class_labels.shape}")
```

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    labeled_embeddings,
    class_labels,
    test_size=0.3,
    random_state=42,
    stratify=class_labels,
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")
```

```{code-cell} ipython3
label_names = ["Baseball Field", "Marina"]

result = geoai.train_embedding_classifier(
    train_embeddings=X_train,
    train_labels=y_train,
    val_embeddings=X_val,
    val_labels=y_val,
    method="knn",
    n_neighbors=5,
    label_names=label_names,
)

print(f"\nTrain accuracy: {result['train_accuracy']:.2%}")
print(f"Val accuracy:   {result['val_accuracy']:.2%}")
```

```{code-cell} ipython3
methods = ["knn", "random_forest", "logistic_regression"]
results_summary = []

for method in methods:
    res = geoai.train_embedding_classifier(
        train_embeddings=X_train,
        train_labels=y_train,
        val_embeddings=X_val,
        val_labels=y_val,
        method=method,
        label_names=label_names,
        verbose=False,
    )
    results_summary.append(
        {
            "Method": method,
            "Train Acc": f"{res['train_accuracy']:.2%}",
            "Val Acc": f"{res['val_accuracy']:.2%}",
        }
    )

pd.DataFrame(results_summary)
```

```{code-cell} ipython3
fig = geoai.visualize_embeddings(
    labeled_embeddings,
    labels=class_labels,
    label_names=label_names,
    method="pca",
    figsize=(8, 8),
    s=30,
    alpha=0.8,
    title="PCA of Labeled Embeddings (Baseball vs Marina)",
)
plt.show()
```

## Comparing Embeddings for Change Detection

```{code-cell} ipython3
n = len(embeddings)
half = n // 2
emb_a = embeddings[:half]
emb_b = embeddings[half : half + half]

similarity = geoai.compare_embeddings(emb_a, emb_b, metric="cosine")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(similarity, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(
    similarity.mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {similarity.mean():.3f}",
)
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Count")
ax.set_title("Embedding Similarity Distribution")
ax.legend()
plt.tight_layout()
plt.show()
```

## Loading Datasets with TorchGeo

```{code-cell} ipython3
single_file = hf_hub_download(repo_id, embedding_files[0], repo_type="dataset")
ds = geoai.load_embedding_dataset("clay", root=single_file)

print(f"Dataset length: {len(ds)}")
print(f"Dataset type: {type(ds).__name__}")
```

```{code-cell} ipython3
try:
    sample = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Embedding shape: {sample['embedding'].shape}")

    fig = ds.plot(sample)
    plt.show()
except KeyError as e:
    print(
        f"Note: This parquet file is missing the '{e.args[0]}' column "
        f"expected by TorchGeo's ClayEmbeddings class."
    )
    print("For such files, use geopandas directly (as shown above).")
    print("The TorchGeo class works best with official Clay data products.")
```

## Working with TESSERA Temporal Embeddings

```{code-cell} ipython3
# %pip install geoai-py geotessera
```

### Checking Data Availability

```{code-cell} ipython3
years = geoai.tessera_available_years()
print(f"Available years: {years}")
```

```{code-cell} ipython3
bbox = (0.05, 52.15, 0.25, 52.25)
count = geoai.tessera_tile_count(bbox=bbox, year=2024)
print(f"{count} tiles available for the specified region")
```

```{code-cell} ipython3
geoai.tessera_coverage(year=2024, output_path="tessera_coverage_2024.png")
```

```{code-cell} ipython3
geoai.tessera_coverage(
    year=2024, region_bbox=(-10, 35, 40, 60), output_path="tessera_coverage_europe.png"
)
```

### Downloading Embeddings

```{code-cell} ipython3
bbox = (0.05, 52.15, 0.25, 52.25)

cambridge_files = geoai.tessera_download(
    bbox=bbox, year=2024, output_dir="./tessera_cambridge", output_format="tiff"
)
print(f"Downloaded {len(cambridge_files)} files")
for f in cambridge_files:
    print(f"  {f}")
```

```{code-cell} ipython3
files = geoai.tessera_download(
    lon=0.15, lat=52.05, year=2024, output_dir="./tessera_single_tile"
)
print(f"Downloaded {len(files)} file(s)")
```

```{code-cell} ipython3
files = geoai.tessera_download(
    bbox=bbox, year=2024, bands=[0, 1, 2], output_dir="./tessera_rgb_only"
)
print(f"Downloaded {len(files)} files with 3 bands each")
```

### Visualizing TESSERA Embeddings

```{code-cell} ipython3
geoai.tessera_visualize_rgb(
    str(cambridge_files[0]), bands=(0, 1, 2), title="Cambridge - TESSERA Bands 0, 1, 2"
)
```

```{code-cell} ipython3
geoai.tessera_visualize_rgb(
    str(cambridge_files[0]),
    bands=(30, 60, 90),
    title="Cambridge - TESSERA Bands 30, 60, 90",
)
```

### Fetching Embeddings to Memory

```{code-cell} ipython3
tiles = geoai.tessera_fetch_embeddings(bbox=(0.05, 52.15, 0.25, 52.25), year=2024)

for tile in tiles:
    print(f"Tile ({tile['lon']:.2f}, {tile['lat']:.2f}):")
    print(f"  Shape: {tile['embedding'].shape}")
    print(f"  CRS: {tile['crs']}")
    print(f"  Mean: {tile['embedding'].mean():.4f}")
    print(f"  Std: {tile['embedding'].std():.4f}")
```

### Sampling Embeddings at Point Locations

```{code-cell} ipython3
from shapely.geometry import Point

points = gpd.GeoDataFrame(
    {"name": ["Point A", "Point B", "Point C"]},
    geometry=[Point(0.12, 52.20), Point(0.15, 52.18), Point(0.20, 52.22)],
    crs="EPSG:4326",
)

result = geoai.tessera_sample_points(points, year=2024)

print(f"Result shape: {result.shape}")
print(
    f"\nOriginal columns: {[c for c in result.columns if not c.startswith('tessera_')]}"
)
print(f"Embedding columns: tessera_0 through tessera_127")
print(f"\nFirst few embedding values for Point A:")
print(result.iloc[0][[f"tessera_{i}" for i in range(5)]])
```

### Alternative Download Formats

```{code-cell} ipython3
import json

files = geoai.tessera_download(
    bbox=bbox, year=2024, output_dir="./tessera_numpy", output_format="npy"
)

with open("./tessera_numpy/metadata.json") as f:
    meta = json.load(f)

print(f"Year: {meta['year']}")
print(f"Number of tiles: {len(meta['tiles'])}")
for tile_info in meta["tiles"]:
    arr = np.load(f"./tessera_numpy/{tile_info['file']}")
    print(f"  {tile_info['file']}: shape={arr.shape}, dtype={arr.dtype}")
```

```{code-cell} ipython3
from shapely.geometry import box

region = gpd.GeoDataFrame(geometry=[box(0.05, 52.15, 0.25, 52.25)], crs="EPSG:4326")
region.to_file("cambridge_region.geojson", driver="GeoJSON")

files = geoai.tessera_download(
    region_file="cambridge_region.geojson",
    year=2024,
    output_dir="./tessera_from_region",
)
print(f"Downloaded {len(files)} files")
```

## Exploring AlphaEarth Satellite Embeddings

```{code-cell} ipython3
# %pip install -U geemap
```

```{code-cell} ipython3
import ee
import geoai
```

```{code-cell} ipython3
ee.Authenticate()
ee.Initialize(project="your-ee-project")
```

### Interactive Map with AlphaEarth GUI

```{code-cell} ipython3
m = geoai.Map(projection="globe", sidebar_visible=True)
m.add_basemap("USGS.Imagery")
m.add_alphaearth_gui()
m
```

### Visualizing Embeddings as RGB Composites

```{code-cell} ipython3
m = geoai.Map(projection="globe", sidebar_visible=True)
m.add_basemap("USGS.Imagery")

lon = -121.8036
lat = 39.0372
m.set_center(lon, lat, zoom=12)
m
```

```{code-cell} ipython3
point = ee.Geometry.Point(lon, lat)
dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

image1 = dataset.filterDate("2017-01-01", "2018-01-01").filterBounds(point).first()
image2 = dataset.filterDate("2024-01-01", "2025-01-01").filterBounds(point).first()
```

```{code-cell} ipython3
vis_params = {"min": -0.3, "max": 0.3, "bands": ["A01", "A16", "A09"]}
m.add_ee_layer(image1, vis_params, name="2017 embeddings")
m.add_ee_layer(image2, vis_params, name="2024 embeddings")
```

### Change Detection via Embedding Similarity

```{code-cell} ipython3
dot_prod = image1.multiply(image2).reduce(ee.Reducer.sum())
```

```{code-cell} ipython3
vis_params = {"min": 0, "max": 1, "palette": ["#ffffff", "#000000"]}
m.add_ee_layer(dot_prod, vis_params, name="Similarity")
m.add_colorbar(cmap="gray", label="Similarity")
m
```

## Key Takeaways

## Exercises

### Exercise 1: Exploring the Embedding Registry

```{code-cell} ipython3

```

### Exercise 2: Clustering with Different K Values

```{code-cell} ipython3

```

### Exercise 3: Similarity-Based Retrieval

```{code-cell} ipython3

```

### Exercise 4: TESSERA Multi-Year Comparison

```{code-cell} ipython3

```

### Exercise 5: AlphaEarth Change Detection for a Different Region

```{code-cell} ipython3

```
