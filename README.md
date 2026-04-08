# Multimodal detection CNN and LiDAR

The goal is to implement a multimodal approach for detection combining LiDAR and computer vision.

## What is LiDAR?
LiDAR is for ***Light Detection And Ranging*** it works like a sonar with light. It generates a 3D Scatter plot where each point has an intensity/reflectance.

- pros:
    - works in dark conditions
    - can measure distances
- cons:
    - doesnt react well with water and fog
    - if objects are far away they can be avoided by the light rays and not detected

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
Velodyn (named velo sometimes) refers to the LiDAR. I will use frames from camera 2 as it's the colored version.

![Sensor Setup](images/sensor_setup.png)
## Coordinate system
From the paper we know:
![cord system](images/cord_system.png)
## Projection pipeline

The LiDAR sensor and Camera 2 don't have the same position. The Kittit Dataset contain the data to project the LiDAR results on the camera referential.
We  can summarize:
$P_{2d} = P_{rect} \times R_{rect} \times Tr_{velo-to-cam}\times P_{3d}$
where:
- $P_{3d}$ is the LiDAR measurement in it's referential
- $Tr_{velo-to-cam}$ Is the translation to move $P_{3d}$ to the camera referential
- $R_{rect}$ To align the two referentiels on the same axle
- $P_{rect}$ To Transform the 3D to 2D using the physicals of the camera (focal, lens)
- $P_{2d}$ Is the projected results of the LiDAR on the 2d images

We have :
- calib_velo_to_cam: With the ***Tr_velo_to_cam***
- calib_cam_to_cam: With ***Rrect*** and ***Prect***

***[cordonnées homogenes/ Homogeneous coordinates](https://fr.wikipedia.org/wiki/Coordonn%C3%A9es_homog%C3%A8nes)***

In the calibration file, there are times where matrix has a 1 line added. its for doing the translation. The explanation are in the link.

## Data Filtering 
(only keep whats in front of us)/ Pourquoi et comment on retire les points derrière la caméra (x<0).

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