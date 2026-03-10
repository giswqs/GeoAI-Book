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
# Pixel-Level Regression

## Introduction

## Learning Objectives

## Understanding Pixel Regression

### Classification vs. Regression

### Applications

## Regression Architectures

### Adapting Segmentation Models for Regression

### Loss Functions for Regression

### Output Activation and Scaling

## Preparing Regression Data

### Input Features

### Continuous Labels

### Using PixelRegressionDataset

```{code-cell} python
import geoai
```

```{code-cell} python
train_image_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/las_vegas_train_naip.tif"
train_target_url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/las_vegas_train_hag.tif"
train_image = geoai.download_file(train_image_url)
train_target = geoai.download_file(train_target_url)
```

```{code-cell} python
import rasterio

with rasterio.open(train_image) as src:
    print(f"Input shape: {src.height} x {src.width}, {src.count} bands")
    print(f"Input CRS: {src.crs}")
    print(f"Input resolution: {src.res}")

with rasterio.open(train_target) as src:
    print(f"Target shape: {src.height} x {src.width}, {src.count} band(s)")
    target_data = src.read(1)
    print(f"Target value range: [{target_data.min():.2f}, {target_data.max():.2f}]")
```

```{code-cell} python
image_tiles, target_tiles = geoai.create_regression_tiles(
    input_raster=train_image,
    target_raster=train_target,
    output_dir="regression_tiles",
    tile_size=256,
    stride=256,
    min_valid_ratio=0.8,
    target_min=0.0,
    target_max=100.0,
)
print(f"Created {len(image_tiles)} paired tiles")
```

```{code-cell} python
from sklearn.model_selection import train_test_split

train_imgs, val_imgs, train_tgts, val_tgts = train_test_split(
    image_tiles, target_tiles, test_size=0.2, random_state=42
)
print(f"Training tiles: {len(train_imgs)}")
print(f"Validation tiles: {len(val_imgs)}")
```

## Training a Regression Model

### Using geoai.PixelRegressionModel

### Configuration and Training

```{code-cell} python
model = geoai.train_pixel_regressor(
    train_image_paths=train_imgs,
    train_target_paths=train_tgts,
    val_image_paths=val_imgs,
    val_target_paths=val_tgts,
    encoder_name="resnet50",
    architecture="unet",
    in_channels=4,
    encoder_weights="imagenet",
    loss_type="mse",
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=50,
    patience=10,
    output_dir="regression_output",
    num_workers=0,
    verbose=False,
)
```

### Monitoring Regression Loss

```{code-cell} python
fig, metrics_df = geoai.plot_training_history(
    log_dir="regression_output",
    metrics=["loss", "r2"],
)
```

## Evaluating Regression Results

### Metrics

### Residual Analysis

```{code-cell} python
pred_raster = geoai.predict_raster(
    model=model,
    input_raster=train_image,
    output_raster="regression_output/prediction.tif",
    tile_size=256,
    overlap=64,
    input_bands=[1, 2, 3, 4],
    batch_size=4,
    clip_range=(0, 100),
)
```

```{code-cell} python
fig, metrics = geoai.plot_regression_comparison(
    true_raster=train_target,
    pred_raster="regression_output/prediction.tif",
    title="Height Above Ground - Regression Results",
    cmap="viridis",
    vmin=0,
    vmax=30,
    valid_range=(0, 100),
)
```

### Spatial Error Patterns

```{code-cell} python
fig, scatter_metrics = geoai.plot_scatter(
    true_raster=train_target,
    pred_raster="regression_output/prediction.tif",
    sample_size=10000,
    title="Predicted vs Actual Height",
    valid_range=(0, 100),
    fit_line=True,
)
```

## Case Study: Building Height Estimation

### Data Preparation

### Training and Results

```{code-cell} python
fig = geoai.visualize_prediction(
    input_raster=train_image,
    pred_raster="regression_output/prediction.tif",
    rgb_bands=[1, 2, 3],
    cmap="viridis",
    vmin=0,
    vmax=30,
)
```

## Key Takeaways

## Exercises

### Exercise 1: Comparing Loss Functions

### Exercise 2: Encoder Architecture Comparison

### Exercise 3: The Effect of Tile Size

### Exercise 4: Residual Analysis and Error Mapping

### Exercise 5: Transfer to a New Area
