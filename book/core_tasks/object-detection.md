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
# Object Detection

## Introduction

## Learning Objectives

## Understanding Object Detection

### Classification vs. Detection

### Key Concepts

## Detection Architectures

### Two-Stage Detectors

### Single-Stage Detectors

### Transformer-Based Detectors

### Zero-Shot Detection

### Choosing an Architecture

## Preparing Detection Datasets

### Annotation Formats Recap

### Creating Detection Annotations

## Training an Object Detector

### Using geoai for Object Detection Training

```{code-cell} python
from geoai.train import train_MaskRCNN_model

train_MaskRCNN_model(
    images_dir="data/images",
    labels_dir="data/labels",
    output_dir="models/car_detector",
    input_format="directory",  # or "coco" or "yolo"
    num_epochs=20,
    batch_size=4,
    learning_rate=0.005,
    val_split=0.2,
    pretrained=True,
    seed=42,
)
```

### Configuration

### Training and Monitoring

## Evaluating Detection Results

### Mean Average Precision (mAP)

### Precision-Recall Curves

### IoU Thresholds

## Running Inference

### Detecting Objects in New Images

```{code-cell} python
from geoai import CarDetector

detector = CarDetector()
results = detector.process_raster(
    raster_path="cars_15cm.tif",
    output_path="car_detections.geojson",
    batch_size=4,
)
print(f"Detected {len(results)} vehicles")
```

```{code-cell} python
from geoai import ObjectDetector

detector = ObjectDetector(
    model_path="models/car_detector/best_model.pth",
    num_classes=2,
)
results = detector.process_raster(
    raster_path="new_image.tif",
    output_path="detections.geojson",
)
```

### Filtering by Confidence Threshold

```{code-cell} python
# High threshold: fewer detections, higher precision
results_strict = detector.process_raster(
    raster_path="cars_15cm.tif",
    confidence_threshold=0.8,
)

# Low threshold: more detections, higher recall
results_lenient = detector.process_raster(
    raster_path="cars_15cm.tif",
    confidence_threshold=0.3,
)
```

### Visualizing Detection Results on Maps

```{code-cell} python
import leafmap

m = leafmap.Map()
m.add_raster("cars_15cm.tif", layer_name="Aerial Image")
style = {
    "color": "red",
    "weight": 2,
    "fillOpacity": 0.0,
}
m.add_gdf(results, layer_name="Car Detections", style=style)
m
```

## Case Study: Vehicle Detection

```{code-cell} python
import geopandas as gpd

url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/"
gdf = gpd.read_file(url + "car_detection.geojson")
print(f"Ground truth annotations: {len(gdf)} vehicles")
print(gdf.head())
```

```{code-cell} python
from geoai import CarDetector

detector = CarDetector()
detections = detector.process_raster(
    raster_path="cars_15cm.tif",
    output_path="car_results.geojson",
    confidence_threshold=0.5,
    overlap=0.25,
)
print(f"Detected {len(detections)} vehicles")
```

```{code-cell} python
import leafmap

m = leafmap.Map()
m.add_raster("cars_15cm.tif", layer_name="Aerial Image")
m.add_gdf(gdf, layer_name="Ground Truth", style={"color": "green", "weight": 2, "fillOpacity": 0})
m.add_gdf(detections, layer_name="Detections", style={"color": "red", "weight": 2, "fillOpacity": 0})
m
```

```{code-cell} python
print(f"Mean confidence: {detections['confidence'].mean():.3f}")
print(f"Median confidence: {detections['confidence'].median():.3f}")
print(f"Detections above 0.8 confidence: {(detections['confidence'] > 0.8).sum()}")
```

## Key Takeaways

## Exercises

### Exercise 1: Comparing Detection and Segmentation

### Exercise 2: Vehicle Detection with Confidence Analysis

### Exercise 3: Multi-Object Detection Comparison

### Exercise 4: Architecture Selection

### Exercise 5: Detection Post-Processing Pipeline
