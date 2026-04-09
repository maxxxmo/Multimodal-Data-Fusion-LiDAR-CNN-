import numpy as np
'''This script contains functions for reading and handling KITTI calibration data.'''

def read_calib_file(filepath):
    """Read in a calibration file and parse it into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                data[key] = np.array([float(x) for x in value.split()])
    return data


class KittiCalibration:
    """class to handle calibration data for KITTI dataset"""
    def __init__(self, calib_dir):
        # Opening files and reading calibration data
        cam_to_cam = read_calib_file(f"{calib_dir}/calib_cam_to_cam.txt")
        velo_to_cam = read_calib_file(f"{calib_dir}/calib_velo_to_cam.txt")

        # matrix are written in a flat format, we need to reshape them
        self.Tr_velo_to_cam = velo_to_cam['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_velo_to_cam = np.vstack([self.Tr_velo_to_cam, [0, 0, 0, 1]]) # np.vstack stacks arrays in vertical sequences
        self.R_rect = np.eye(4) #np.eye(4) creates a 4x4 identity matrix
        self.R_rect[:3, :3] = cam_to_cam['R_rect_00'].reshape(3, 3)
        self.P_rect2 = cam_to_cam['P_rect_02'].reshape(3, 4)
        self.T_rect = self.R_rect @ self.Tr_velo_to_cam

    def transform_velo_to_rect(self, points_3d):
        """Transform 3D points from LiDAR to Camera 3D """
        # Convert to homogeneous coordinates
        n = points_3d.shape[0]# [n, 3] --> correspond to [x,y,z]
        points_homo = np.hstack([points_3d, np.ones((n, 1))]) # [n, 4], (x,y,z,1),
        points_rect = points_homo @ self.T_rect.T # result in (N, 4), (X',Y',Z')
        return points_rect[:, :3]
            

    def project_rect_to_image(self, points_3d_rect):
        """Project 3D points from camera (3D) to Pixels (2D)"""
        # Convert to homogeneous coordinates
        n = points_3d_rect.shape[0] # [n, 3D] --> correspond to [x,y,z]
        points_homo = np.hstack([points_3d_rect, np.ones((n, 1))]) # [n, 4], (x,y,z,1),
        # Projection
        points_2d = points_homo @ self.P_rect2.T
        # Normalization, from 3D world to 2D world
        # U horizontal, V vertical
        # u = x/z , v = y/z
        u = points_2d[:, 0] / points_2d[:, 2]
        v = points_2d[:, 1] / points_2d[:, 2]
        return np.vstack([u, v]).T
    

    def filter_points_on_image(self, points_3d, pixels, img_shape):
        """Keep only points that are visible in the image. To keep alignement between sensors
        Reference: camera02"""
        h, w = img_shape
        # points in front of the LiDAR and vehicle
        front_mask = points_3d[:, 0] > 0
        # Point in (u,v) range
        u, v = pixels[:, 0], pixels[:, 1]
        image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        # Using the 2 mask to filter
        final_mask = front_mask & image_mask
        return final_mask
    
    def filter_ground_simple(self, points_3d, ground_threshold=-1.5):
        """Point only from the ground. The LiDAR is 1.73m above the ground, we filter to decrease number of points for detection.
        reference: lidar"""
        not_ground_mask = points_3d[:, 2] > ground_threshold
        return not_ground_mask

    def filter_front_velo(self, points_3d, min_dist=0.0):
        """Points only in front of the LiDAR and vehicle.we filter to decrease number of points for detection.
        reference: lidar"""
        return points_3d[:, 0] > min_dist
