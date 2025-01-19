# Land Use Detector AI

This project implements an artificial intelligence system for land use classification in satellite images, focusing on any given area from a shapefile. The goal is to provide a detailed analysis of land use and cover.

## Features

- U-Net Model Training for Semantic Segmentation:
  - Multi-class classification: Urban, Agriculture, Pasture, Forest, Water, Bare Soil, and Unknown.
  - Supports data augmentation and training performance visualization.
- Land Use Prediction:
  - Processing of GeoTIFF and PNG images.
  - Mosaic tiling for large areas.
  - Generation of predictive masks and recomposition of processed images.
- Statistical Analysis:
  - Chart generation (pie, bar, boxplot) for class distribution.
  - Metrics such as IoU, F1-Score, and Confusion Matrix.
- Export and Visualization:
  - Export predictive images in GIS-compatible formats.
  - Intuitive visualization of original images and masks.
