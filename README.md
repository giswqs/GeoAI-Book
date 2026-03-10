# GeoAI with Python: A Hands-On Guide to Geospatial AI

[![Jupyter Book](https://img.shields.io/badge/Jupyter%20Book-v2-orange?logo=jupyter)](https://jupyterbook.org)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

A comprehensive guide to applying artificial intelligence to geospatial data using Python. This book covers everything from downloading satellite imagery to building QGIS plugins, with hands-on code examples using the [geoai](https://github.com/opengeos/geoai) package and the broader Python geospatial ecosystem.

## Book Structure

### Part I: Foundations (Chapters 1-3)
Set up your development environment, understand geospatial data formats, and learn the building blocks for GeoAI workflows.

### Part II: Data Acquisition and Preparation (Chapters 4-6)
Download remote sensing data from the Planetary Computer and HuggingFace, create interactive visualizations, and prepare training datasets with proper tiling and augmentation.

### Part III: Core AI Tasks (Chapters 7-12)
Master the fundamental computer vision tasks for geospatial analysis: image segmentation, object detection, instance segmentation, land cover classification, change detection, and pixel-level regression.

### Part IV: Foundation Models and Satellite Embeddings (Chapters 13-16)
Explore cutting-edge approaches including the Segment Anything Model (SAM), vision-language models, satellite embeddings from foundation models, and AI agents for geospatial analysis.

### Part V: QGIS Plugins (Chapters 17-20)
Bring GeoAI capabilities into QGIS through the GeoAI plugin, covering installation, Segment Anything, training and inference, and vision-language models in a familiar desktop GIS environment.

## Key Technologies

- **[geoai](https://github.com/opengeos/geoai)** - High-level Python package for geospatial AI
- **[segment-geospatial](https://github.com/opengeos/segment-geospatial)** - SAM for geospatial data
- **[leafmap](https://github.com/opengeos/leafmap)** - Interactive mapping and visualization
- **[torchgeo](https://github.com/microsoft/torchgeo)** - PyTorch datasets and models for geospatial data
- **[rasterio](https://github.com/rasterio/rasterio)** / **[geopandas](https://github.com/geopandas/geopandas)** - Core geospatial I/O

## Getting Started

### Prerequisites

- Python 3.12+
- NVIDIA GPU recommended (but not required)
- Conda or Miniconda

### Installation

```bash
conda create -n geoai python=3.12
conda activate geoai
pip install geoai-py segment-geospatial leafmap
```

### Building the Book

Build the HTML version:

```bash
myst build --html
```

Build the PDF via Typst:

```bash
myst build --typst
python myst_to_typst.py
cd _build/typst/ && typst compile main.typ
```

## Sample Data

All code examples use freely available data from the [giswqs/geospatial](https://huggingface.co/datasets/giswqs/geospatial) HuggingFace dataset, including NAIP aerial imagery, Landsat satellite imagery, building footprints, and land cover labels.

## Author

**Qiusheng Wu**
Department of Geography & Sustainability, University of Tennessee, Knoxville
[GitHub](https://github.com/giswqs) | [Website](https://wetlands.io)

## License

This book is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Credits

Built with [Jupyter Book](https://jupyterbook.org/) and [MyST Markdown](https://mystmd.org/).
