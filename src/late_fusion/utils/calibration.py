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
        self.M = self.P_rect2 @ self.R_rect @ self.Tr_velo_to_cam
        
        
    def project_velo_to_image(self, points_3d):
        """Project 3D points from LiDAR to 2D image plane. The result is LiDAR points in pixel coordinates."""
        # Convert to homogeneous coordinates
        n = points_3d.shape[0] # [n, 3] --> correspond to [x,y,z]
        points_homo = np.hstack([points_3d, np.ones((n, 1))]) # [n, 4], (x,y,z,1),

        # Projection
        points_2d = points_homo @ self.M.T  # Result (N, 3), (X',Y',Z')

        # Normalization, from 3D world to 2D world
        # U horizontal, V vertical
        # u = x/z , v = y/z
        u = points_2d[:, 0] / points_2d[:, 2]
        v = points_2d[:, 1] / points_2d[:, 2]
        return np.vstack([u, v]).T
    
    def filter_points_on_image(self, points_3d, pixels, img_shape):
        """Keep only points that are visible in the image."""
        h, w = img_shape
        # points in front of the LiDAR and vehicle
        front_mask = points_3d[:, 0] > 0
        # Point in (u,v) range
        u, v = pixels[:, 0], pixels[:, 1]
        image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        # Using the 2 mask to filter
        final_mask = front_mask & image_mask
        return final_mask