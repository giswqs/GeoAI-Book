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
# Satellite Embeddings

## Introduction

## Learning Objectives

## What Are Satellite Embeddings?

```{code-cell} python
# %pip install geoai leafmap
```

```{code-cell} python
import geoai
```

```{code-cell} python
geoai.list_embedding_datasets(as_dataframe=True)
```

```{code-cell} python
geoai.list_embedding_datasets(kind="patch", as_dataframe=True)
```

```{code-cell} python
geoai.list_embedding_datasets(kind="pixel", as_dataframe=True)
```

```{code-cell} python
geoai.get_embedding_info("clay")
```

## Setting Up the Environment

```{code-cell} python
import geoai
import leafmap
```

## Exploring Patch-Based Embeddings

```{code-cell} python
clay_dataset = geoai.load_embedding_dataset("clay")
clay_dataset
```

```{code-cell} python
clay_result = geoai.extract_patch_embeddings(clay_dataset, max_samples=500)
clay_embeddings = clay_result["embeddings"]
clay_metadata = clay_result["metadata"]
print(f"Embeddings shape: {clay_embeddings.shape}")
```

## Exploring Pixel-Based Embeddings

```{code-cell} python
google_dataset = geoai.load_embedding_dataset("google_satellite")
google_dataset
```

```{code-cell} python
google_result = geoai.extract_pixel_embeddings(google_dataset, num_samples=50, size=256)
google_embeddings = google_result["embeddings"]
print(f"Embeddings shape: {google_embeddings.shape}")
```

```{code-cell} python
geoai.plot_embedding_vector(google_embeddings[0])
```

```{code-cell} python
geoai.plot_embedding_raster(google_result["images"][0])
```

## Visualizing Embeddings

```{code-cell} python
geoai.visualize_embeddings(clay_embeddings, method="pca")
```

```{code-cell} python
geoai.visualize_embeddings(clay_embeddings, method="tsne")
```

```{code-cell} python
geoai.visualize_embeddings(clay_embeddings, method="umap")
```

## Similarity Search with Embeddings

```{code-cell} python
query = clay_embeddings[0]
sim_result = geoai.embedding_similarity(query, clay_embeddings, metric="cosine", top_k=10)
print("Top 10 most similar tiles:")
print("Indices:", sim_result["indices"])
print("Scores:", sim_result["scores"])
```

## Clustering Embeddings

```{code-cell} python
cluster_result = geoai.cluster_embeddings(clay_embeddings, n_clusters=5, method="kmeans")
labels = cluster_result["labels"]
print(f"Cluster labels: {labels[:20]}")
```

```{code-cell} python
geoai.visualize_embeddings(clay_embeddings, method="umap", labels=labels)
```

## Training Classifiers on Embeddings

```{code-cell} python
import numpy as np

n = len(clay_embeddings)
train_idx = np.arange(0, int(n * 0.7))
val_idx = np.arange(int(n * 0.7), n)

train_emb = clay_embeddings[train_idx]
val_emb = clay_embeddings[val_idx]
train_labels = labels[train_idx]
val_labels = labels[val_idx]

classifier_result = geoai.train_embedding_classifier(
    train_emb, train_labels, val_emb, val_labels, method="knn", n_neighbors=5
)
print(f"Validation accuracy: {classifier_result['val_accuracy']:.3f}")
```

## Comparing Embeddings for Change Detection

```{code-cell} python
emb_t1 = clay_embeddings[0]
emb_t2 = clay_embeddings[1]
change_score = geoai.compare_embeddings(emb_t1, emb_t2, metric="cosine")
print(f"Change score: {change_score:.4f}")
```

## Exporting Embeddings as GeoTIFF

```{code-cell} python
geoai.embedding_to_geotiff(
    google_embeddings[0],
    bounds=google_result.get("bounds", (-122.5, 37.5, -122.0, 38.0)),
    output_path="embeddings_export.tif",
)
```

## Working with TESSERA Embeddings

```{code-cell} python
tessera_data = geoai.tessera_download(
    lon=-122.4194,
    lat=37.7749,
    year=2024,
    output_dir="./tessera_output",
    output_format="tiff",
)
```

```{code-cell} python
geoai.tessera_visualize_rgb(tessera_data)
```

```{code-cell} python
bbox = (-122.5, 37.7, -122.3, 37.8)
tessera_emb = geoai.tessera_fetch_embeddings(bbox, year=2024)
print(f"TESSERA embeddings shape: {tessera_emb.shape}")
```

## Using Prithvi for Earth Observation

```{code-cell} python
models = geoai.get_available_prithvi_models()
print(models)
```

```{code-cell} python
model = geoai.load_prithvi_model(models[0])
```

```{code-cell} python
raster_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/hls_sample.tif"
raster_path = geoai.download_file(raster_url)
result = geoai.prithvi_inference(model, raster_path)
print(type(result))
```

```{code-cell} python
processor = geoai.PrithviProcessor(
    model_name=models[0],
    bands=["B02", "B03", "B04", "B05", "B06", "B07"],
)
```

## DINOv2 Feature Extraction

```{code-cell} python
image_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/dino_sample.tif"
image_path = geoai.download_file(image_url)
```

```{code-cell} python
dino_processor = geoai.DINOv3GeoProcessor(model_name="dinov2_vits14")
features = geoai.dinov3(dino_processor, image_path)
print(f"Feature shape: {features.shape}")
```

```{code-cell} python
similarity_map = geoai.create_similarity_map(features, reference_point=(100, 100))
```

```{code-cell} python
m = leafmap.Map()
m.add_raster(image_path, layer_name="Satellite Image")
m.add_raster(similarity_map, layer_name="Similarity", cmap="RdYlGn", opacity=0.7)
m
```

## Key Takeaways

## Exercises

### Exercise 1: Exploring the Embedding Registry

```{code-cell} python
# Your code here
```

### Exercise 2: Clustering and Visualization

```{code-cell} python
# Your code here
```

### Exercise 3: Similarity-Based Retrieval

```{code-cell} python
# Your code here
```

### Exercise 4: TESSERA Temporal Analysis

```{code-cell} python
# Your code here
```

### Exercise 5: Embedding-Based Change Detection

```{code-cell} python
# Your code here
```
