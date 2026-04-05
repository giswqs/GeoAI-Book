# GeoAI with Python

## Introduction

Welcome to the official repository for _**GeoAI with Python: A Practical Guide to Open-Source Geospatial AI**_. This repository contains all the code examples featured in the book, designed to help you learn and apply GeoAI using open-source tools.

## Get the Book

### Print Edition

- 🇺🇸 Englsih Full-Color Print Edition (430 pages) is available on Amazon ([link](https://www.amazon.com/dp/B0GTVFY3PQ))

### PDF and EPUB Editions

🇺🇸 [English](https://leanpub.com/geoai) | 🇲🇽 [Spanish](https://leanpub.com/geoai-es) | 🇨🇳 [Chinese](https://leanpub.com/geoai-zh) | 🇫🇷 [French](https://leanpub.com/geoai-fr) | 🇩🇪 [German](https://leanpub.com/geoai-de) | 🇯🇵 [Japanese](https://leanpub.com/geoai-ja) | 🇰🇷 [Korean](https://leanpub.com/geoai-ko) | 🇮🇩 [Indonesian](https://leanpub.com/geoai-id) | 🇵🇹 [Portuguese](https://leanpub.com/geoai-pt) | 🇷🇺 [Russian](https://leanpub.com/geoai-ru) | 🇮🇹 [Italian](https://leanpub.com/geoai-it)

## Cite the Book

If you use this book in your research or teaching, please consider citing it as follows:

> Wu, Q. (2026). *GeoAI with Python: A Practical Guide to Open-Source Geospatial AI*. Independently published. PDF edition ISBN 979-8993859729; Print edition ISBN 979-8253507414. Available at [book.opengeoai.org](https://book.opengeoai.org).

![](https://books.gishub.org/geoai/front-cover.webp)

## About This Book

Learn to apply deep learning and AI to satellite imagery, aerial photos, and geospatial data using Python. This practical, hands-on guide walks you from downloading remote sensing data to training and evaluating deep learning models, all using open-source tools.

**What you’ll learn**

- Set up a complete GeoAI environment with Python, PyTorch, and GPU acceleration.
- Download satellite imagery from Microsoft Planetary Computer and open data portals.
- Create interactive maps and prepare training datasets from large satellite images.
- Train and evaluate models for seven core geospatial AI tasks: image recognition, object detection, semantic segmentation, instance segmentation, image translation, change detection, and pixel-level regression.
- Apply foundation models, including the Segment Anything Model (SAM), vision-language models, and satellite embeddings, to real-world Earth observation problems.
- Run AI workflows in QGIS without writing code using plugins for tree segmentation, water detection, and more.

**Structure and format**

- 23 chapters of executable code examples organized in five parts: Foundations, Data Acquisition and Preparation, Core AI Tasks, Foundation Models, and QGIS Plugins.
- All examples use real satellite imagery with PyTorch, torchgeo, segment-geospatial, leafmap, and geoai.
- All code and datasets are freely available on GitHub and Source Cooperative for full reproducibility.

**Who it’s for**

GIS professionals, remote sensing scientists, data scientists, and students who want to apply AI to geospatial data using Python and open-source tools.



## Table of Contents

### Preface

- [Preface](book/preface.md)

### Part I: Foundations

1. [Introduction to GeoAI](book/foundations/introduction.md)
2. [Setting Up Your Environment](book/foundations/setup.md)
3. [Geospatial Data Essentials](book/foundations/data-formats.md)

### Part II: Data Acquisition and Preparation

4. [Downloading Remote Sensing Data](book/data/download-data.md)
5. [Interactive Mapping and Visualization](book/data/visualization.md)
6. [Preparing Training Data](book/data/training-data.md)

### Part III: Core AI Tasks

7. [Image Recognition](book/core_tasks/image-recognition.md)
8. [Object Detection](book/core_tasks/object-detection.md)
9. [Semantic Segmentation](book/core_tasks/semantic-segmentation.md)
10. [Instance Segmentation](book/core_tasks/instance-segmentation.md)
11. [Image Translation](book/core_tasks/image-translation.md)
12. [Change Detection](book/core_tasks/change-detection.md)
13. [Pixel-Level Regression](book/core_tasks/pixel-regression.md)

### Part IV: Foundation Models and Satellite Embeddings

14. [SAM for Geospatial Applications](book/advanced/sam-geospatial.md)
15. [Vision-Language Models](book/advanced/vision-language-models.md)
16. [Satellite Embeddings](book/advanced/foundation-models.md)

### Part V: QGIS Plugins

17. [Setting Up the GeoAI QGIS Plugin](book/qgis/installation.md)
18. [Tree Segmentation in QGIS](book/qgis/tree-segmentation.md)
19. [Water Segmentation in QGIS](book/qgis/water-segmentation.md)
20. [Vision-Language Models in QGIS](book/qgis/vlm-plugin.md)
21. [Segment Anything in QGIS](book/qgis/samgeo-plugin.md)
22. [Semantic Segmentation in QGIS](book/qgis/semantic-segmentation.md)
23. [Instance Segmentation in QGIS](book/qgis/instance-segmentation.md)


## About the Author

Dr. Qiusheng Wu is an Associate Professor and the Director of Graduate Studies in the Department of Geography & Sustainability at the University of Tennessee, Knoxville. He also serves as an Amazon Scholar. Dr. Wu's research focuses on advancing open-source geospatial analytics through cloud computing and GeoAI, with an emphasis on leveraging big geospatial data and artificial intelligence to study environmental change, particularly surface water and wetland inundation dynamics.

He is the creator and maintainer of several widely used open-source Python packages, including [geemap](https://geemap.org) for interactive Google Earth Engine visualization, [leafmap](https://leafmap.org) for versatile geospatial mapping, [segment-geospatial](https://samgeo.gishub.org) for applying the Segment Anything Model to geospatial data, and [geoai](https://opengeoai.org) for high-level GeoAI workflows. His open-source projects, available through the [Open Geospatial Solutions](https://github.com/opengeos) organization on GitHub, have been widely adopted by researchers, educators, and practitioners worldwide.

Dr. Wu's work bridges remote sensing, Earth observation, and artificial intelligence to make large-scale geospatial data more accessible, reproducible, and intelligent. He is passionate about open science and believes that the best tools for understanding our planet should be freely available to everyone.

## Licensing and Copyright

This book embraces the principles of open science and open education. To support transparency, learning, and reuse, the **code examples** in this book are released under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This means you are free to copy, modify, and distribute the code, even for commercial purposes, as long as appropriate credit is given.

Please attribute code usage by citing the book or linking to the GitHub repository:

Wu, Q. (2026). *GeoAI with Python: A Practical Guide to Open-Source Geospatial AI*. Independently published. PDF edition ISBN 979-8993859729; Print edition ISBN 979-8253507414. Available at [book.opengeoai.org](https://book.opengeoai.org).

While the code is freely available, the **text, figures, and images** in this book are **copyrighted** © 2026 by the author and may not be reproduced, redistributed, or modified without explicit permission. This includes all written content, custom diagrams, and embedded visualizations unless otherwise noted.

If you wish to reuse or adapt any non-code material from the book (for example, for teaching, presentations, or publications), please contact the author to request permission.

This dual licensing approach helps balance open access to learning materials with the protection of original creative work. Thank you for respecting these terms and supporting the open-source geospatial community.
