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

## Case Study: NDVI Prediction from Landsat Imagery

### Environment Setup

```{code-cell} ipython3
import geoai
from sklearn.model_selection import train_test_split
```

### Download Data

```{code-cell} ipython3
train_raster = geoai.download_file(
    "https://data.source.coop/opengeos/geoai/tn_landsat_2022.tif"
)
train_target = geoai.download_file(
    "https://data.source.coop/opengeos/geoai/tn_ndvi_2022.tif"
)
test_raster = geoai.download_file(
    "https://data.source.coop/opengeos/geoai/tn_landsat_2023.tif"
)
```

### Inspect the Data

```{code-cell} ipython3
import rasterio

with rasterio.open(train_raster) as src:
    in_channels = src.count
    print(f"Input shape: {src.height} x {src.width}, {src.count} bands")
    print(f"Input CRS: {src.crs}")
    print(f"Input resolution: {src.res}")

with rasterio.open(train_target) as src:
    print(f"Target shape: {src.height} x {src.width}, {src.count} band(s)")
    target_data = src.read(1)
    print(f"Target value range: [{target_data.min():.2f}, {target_data.max():.2f}]")
```

### Create Training Tiles

```{code-cell} ipython3
image_paths, target_paths = geoai.create_regression_tiles(
    input_raster=train_raster,
    target_raster=train_target,
    output_dir="ndvi_tiles",
    tile_size=256,
    stride=128,
    target_band=1,
    min_valid_ratio=0.9,
    target_min=-1.0,
    target_max=1.0,
)
print(f"Created {len(image_paths)} tiles")
```

### Split Data

```{code-cell} ipython3
train_imgs, val_imgs, train_tgts, val_tgts = train_test_split(
    image_paths, target_paths, test_size=0.2, random_state=42
)
print(f"Training: {len(train_imgs)}, Validation: {len(val_imgs)}")
```

### Train the Model

```{code-cell} ipython3
model = geoai.train_pixel_regressor(
    train_image_paths=train_imgs,
    train_target_paths=train_tgts,
    val_image_paths=val_imgs,
    val_target_paths=val_tgts,
    encoder_name="resnet34",
    architecture="unet",
    in_channels=in_channels,
    output_dir="ndvi_model",
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-3,
    num_workers=0,
    loss_type="mse",
    patience=20,
    devices=1,
    verbose=False,
)
```

### Monitor Training History

```{code-cell} ipython3
fig, history_df = geoai.plot_training_history(
    log_dir="ndvi_model",
    metrics=["loss", "r2"],
)
```

### Run Inference on Training Area

```{code-cell} ipython3
geoai.predict_raster(
    model=model,
    input_raster=train_raster,
    output_raster="ndvi_model/predicted_ndvi_2022.tif",
    tile_size=256,
    overlap=64,
    batch_size=8,
    clip_range=(-1.0, 1.0),
)
```

### Evaluate Results

```{code-cell} ipython3
fig, metrics = geoai.plot_regression_comparison(
    true_raster=train_target,
    pred_raster="ndvi_model/predicted_ndvi_2022.tif",
    title="NDVI Prediction Results",
    cmap="RdYlGn",
    vmin=-0.2,
    vmax=0.8,
    valid_range=(-1.0, 1.0),
)
```

```{code-cell} ipython3
fig, metrics = geoai.plot_scatter(
    true_raster=train_target,
    pred_raster="ndvi_model/predicted_ndvi_2022.tif",
    sample_size=50000,
    valid_range=(-1.0, 1.0),
    fit_line=True,
)
```

### Predict on New Data (2023)

```{code-cell} ipython3
geoai.predict_raster(
    model=model,
    input_raster=test_raster,
    output_raster="ndvi_model/predicted_ndvi_2023.tif",
    tile_size=256,
    overlap=64,
    batch_size=8,
    clip_range=(-1.0, 1.0),
)
```

```{code-cell} ipython3
geoai.visualize_prediction(
    input_raster=test_raster,
    pred_raster="ndvi_model/predicted_ndvi_2023.tif",
    cmap="RdYlGn",
    vmin=-0.2,
    vmax=0.8,
)
```

## Evaluation Metrics

## Key Takeaways

## Exercises

### Exercise 1: Comparing Loss Functions

```{code-cell} ipython3

```

### Exercise 2: Encoder Architecture Comparison

```{code-cell} ipython3

```

### Exercise 3: The Effect of Tile Size and Stride

```{code-cell} ipython3

```

### Exercise 4: Residual Analysis and Error Mapping

```{code-cell} ipython3

```
