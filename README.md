# Multimodal detection CNN and LiDAR

# Table of contents
- [Getting Started](#getting-started)
- [Dataset Handling and calibration](#Dataset-Handling-and-calibration)
- [Late fusion approach](#late-fusion-approach)
- [Middle fusion approach](#middle-fusion-approach)
- [Sources](#sources)
- [Future and improvments](#future-and-improvments)


# Getting started
(to do at the end of the project)
To use this repo you need:

...

# Dataset Handling and calibration

## Coordinate system

## Projection pipeline
Conversion des coordonnées Homogènes.

Matrice de rotation/translation (Trvelo_to_cam​).

Matrice de rectification (R0​).

Matrice intrinsèque caméra (P).

## Data Filtering (only keep whats in front of us)
 Pourquoi et comment on retire les points derrière la caméra (x<0).

# Late fusion approach

## 1. CNN

## 2. LiDAR

## 3. Fusion


# Middle fusion approach



# Sources
- [KITTI Coordinate Transformations](https://medium.com/data-science/kitti-coordinate-transformations-125094cd42fb)
- [Vision meets Robotics: The KITTI Dataset](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
- [Camera-Lidar Projection: Navigating between 2D and 3D](https://medium.com/swlh/camera-lidar-projection-navigating-between-2d-and-3d-911c78167a94)

## Data Sources

# Future and improvments