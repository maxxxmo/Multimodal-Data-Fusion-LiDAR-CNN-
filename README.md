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
The dataset is composed of a tracking sub dataset, a velodyne dataset and videos to test the results.
<!-- I did select 3 videos:
![alt text](images/dataset.png) -->
Velodyn (named velo sometimes) refers to the LiDAR. I will use frames from camera 2 as it's the colored version.

![Sensor Setup](images/sensor_setup.png)
## Coordinate system
From the paper we know:
![cord system](images/cord_system.png)

and labels are:

61 3 Car 1 0 -2.101562 858.090004 200.535261 1241.000000 374.000000 1.377451 1.491847 3.318948 3.256830 1.648443 5.626493 -1.600523

for

Frame_ID Track_ID Type Truncated Occluded Alpha bbox_left bbox_top bbox_right bbox_bottom Height Width Length Location_X Location_Y Location_Z Rotation_y
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

Then we filter the data to keep only what's in front of the LiDAR sensor.

# Late fusion approach
First we create a YOLO type Dataset using the tracking part of the dataset
## 1. CNN
### First Training
![Confusion](images/confusion_matrix_normalized.png)
![results](images/results.png)

TO DO improve the model...

## 2. LiDAR

## 3. Fusion


# Middle fusion approach



# Sources
- [KITTI Coordinate Transformations](https://medium.com/data-science/kitti-coordinate-transformations-125094cd42fb)
- [Vision meets Robotics: The KITTI Dataset](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
- [Camera-Lidar Projection: Navigating between 2D and 3D](https://medium.com/swlh/camera-lidar-projection-navigating-between-2d-and-3d-911c78167a94)

- @article{Zhou2018,
   author  = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
   title   = {{Open3D}: {A} Modern Library for {3D} Data Processing},
   journal = {arXiv:1801.09847},
   year    = {2018},
}


## Data Sources

# Future and improvments