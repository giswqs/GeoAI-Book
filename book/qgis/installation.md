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

## Installing Pixi

### Linux and macOS

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Windows

### Verifying the Installation

```bash
pixi --version
```

## Creating the Python Environment

### GPU with CUDA 12.x

### GPU with CUDA 13.x

### CPU Only

### Installing the Environment

```bash
pixi install
```

### Verifying PyTorch and CUDA

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Installing the QGIS Plugin

### Prerequisites

### Option A: QGIS Plugin Manager (Recommended)

### Option B: Helper Script

```bash
git clone https://github.com/opengeos/geoai.git
cd geoai
python install.py
```

### Option C: Manual Installation

### Enabling the Plugin

## Setting Up SAM 3 Access

### Requesting Access on HuggingFace

### Authenticating with the HuggingFace CLI

```bash
pixi run huggingface-cli login
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

```bash
pixi update
```

## Key Takeaways

## Exercises
