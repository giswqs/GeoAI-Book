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
# Setting Up the GeoAI QGIS Plugin

## Introduction

## Learning Objectives

## Plugin Architecture Overview

## Installing the QGIS Plugin

### Prerequisites

### Option A: QGIS Plugin Manager (Recommended)

### Option B: Helper Script

```bash
git clone https://github.com/opengeos/geoai.git
cd geoai/qgis_plugin
python install.py
```

```bash
python install.py --remove
```

### Option C: Manual Installation

### Enabling the Plugin

## Setting Up Dependencies

### Option A: Built-in Dependency Installer (Recommended)

```bash
# Linux/macOS
export GEOAI_CACHE_DIR=/path/to/writable/directory

# Windows (Command Prompt)
set GEOAI_CACHE_DIR=D:\geoai_cache

# Windows (PowerShell)
$env:GEOAI_CACHE_DIR = "D:\geoai_cache"
```

### Option B: Pixi Environment (Advanced)

#### Installing Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

```bash
pixi --version
```

#### Creating the Python Environment

```bash
pixi init geo
cd geo
```

#### Installing the Environment

```bash
pixi install
```

#### Verifying PyTorch and CUDA

```bash
pixi run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'))"
```

#### Installing DeepForest

```bash
pixi run pip install deepforest
```

```bash
pixi run pip install -U numpy transformers
```

#### Launching QGIS from Pixi

```bash
pixi run qgis
```

## Setting Up SAM 3 Access

### Requesting Access on Hugging Face

### Authenticating and Downloading the Model

```bash
pixi run hf auth login
```

```bash
pixi run hf download facebook/sam3
```

## GPU Memory Management

### When to Clear GPU Memory

### Monitoring GPU Usage

```bash
nvidia-smi
```

```bash
watch -n 1 nvidia-smi
```

## Checking for Plugin Updates

## Key Takeaways

## Exercises

### Exercise 1: Install and Verify the Plugin

### Exercise 2: Built-in Dependency Installer and Environment Verification

### Exercise 3: GPU Memory Management and Plugin Maintenance
