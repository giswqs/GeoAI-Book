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
# Setting Up Your Environment

## Introduction

## Learning Objectives

## Hardware Requirements

### Minimum Requirements

### Recommended Requirements

### Cloud Alternatives

## Installing NVIDIA Drivers

### macOS

### Windows

```bash
nvidia-smi
```

### Linux

```bash
sudo apt update
sudo ubuntu-drivers install
```

```bash
sudo apt install nvidia-driver-580
```

```bash
sudo reboot
```

```bash
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
    https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install akmod-nvidia
```

```bash
sudo pacman -S nvidia nvidia-utils
```

### Verifying the Driver Installation

```bash
nvidia-smi
```

## Installing Python with Miniconda

### Windows Installation

   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
   start /wait "" .\miniconda.exe /S
   del .\miniconda.exe
   ```

   ```bash
   conda --version
   python --version
   ```

### macOS Installation

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

```bash
source ~/miniconda3/bin/activate
conda init --all
```

```bash
conda --version
python --version
```

### Linux Installation

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

```bash
source ~/miniconda3/bin/activate
conda init --all
```

```bash
conda --version
python --version
```

## Creating a Conda Environment

```bash
conda create -n geoai python=3.13 -y
```

```bash
conda activate geoai
```

```bash
conda deactivate
```

## Installing GeoAI and PyTorch with GPU Support

```bash
conda activate geoai
```

### Checking Your GPU

```bash
nvidia-smi
```

### Installing GeoAI and PyTorch with CUDA

```bash
conda install -c conda-forge geoai segment-geospatial "pytorch=*=cuda*"
```

```bash
conda install -c conda-forge geoai segment-geospatial "pytorch=2.7.0=cuda128"
```

### Installing GeoAI and PyTorch for CPU Only

```bash
conda install -c conda-forge geoai segment-geospatial
```

### Verifying GPU Access

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"
```

## Installing uv as an Alternative Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
pip install uv
```

```bash
uv venv --python 3.13
```

```bash
source .venv/bin/activate
```

```bash
uv pip install geoai-py segment-geospatial jupyterlab
```

## Setting Up Jupyter

```bash
jupyter lab
```

```{code-cell} python
import torch
import geoai

print(torch.cuda.is_available())
print(geoai.__version__)
```

## Installing Visual Studio Code

### Download and Installation

### Recommended Extensions

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension charliermarsh.ruff
```

## Verifying Your Setup

```{code-cell} python
import sys
import importlib

print(f"Python version: {sys.version}\n")

# Check core packages
packages = {
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "geoai": "GeoAI",
    "leafmap": "Leafmap",
    "geopandas": "GeoPandas",
    "rasterio": "Rasterio",
    "samgeo": "Segment-Geospatial",
    "torchgeo": "TorchGeo",
}

for module_name, display_name in packages.items():
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  {display_name}: {version}")
    except ImportError:
        print(f"  {display_name}: NOT FOUND")

# Check GPU availability
import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Running in CPU-only mode.")
```

```{code-cell} python
import leafmap

m = leafmap.Map(center=[40, -100], zoom=4)
m
```

## Exercises

### Exercise 1: Install and Verify Your Environment

### Exercise 2: Check GPU Availability

### Exercise 3: Explore Conda Environment Management

### Exercise 4: Create an Interactive Map
