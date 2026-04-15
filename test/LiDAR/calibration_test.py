import numpy as np
import matplotlib.pyplot as plt
import cv2 
from src.late_fusion.utils.calibration import KittiCalibration 

# config 
calib_dir = 'data/data_tracking_calib/training/calib/000000.txt'
img_path = 'data/data_object_label_2/training/label_2/000000.txt'
velo_path = 'data/kitti/2011_09_26/velodyne_points/data/000000.bin'


# Image loading
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# LiDAR loading
scan = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
points_3d = scan[:, :3]

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.scatter(points_3d[:, 0], points_3d[:, 1], s=1, c='red')
plt.title('LiDAR points in image coordinates')

# Calibration loading
calib = KittiCalibration(calib_dir)

# Only Calibration for training our detector (Warning the order is IMPORTANT)
## Filtering

## Transormation
points_rect = calib.transform_velo_to_rect(points_3d)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.scatter(points_rect[:, 0], points_rect[:, 1], s=1, c='red')
plt.title('LiDAR points in image coordinates')

# Projection for the points detected by our detection system for the fusion
pixels = calib.project_rect_to_image(points_rect) #Projection of 3D points to 2D image plane the result is in px

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.scatter(pixels[:, 0], pixels[:, 1], s=1, c='red')
plt.title('LiDAR points projected on image')