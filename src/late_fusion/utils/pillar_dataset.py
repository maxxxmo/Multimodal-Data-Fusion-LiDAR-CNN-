import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.late_fusion.utils.calibration import KittiCalibration
import yaml

class KittiPillarDataset(Dataset):
    def __init__(self, data_dir, config_path, split='train'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.root_dir = Path(data_dir).resolve()
        self.base_path = self.root_dir / split
        
        self.lidar_path = self.base_path / "velodyne"
        self.label_path = self.base_path / "labels"
        self.calib_path = self.base_path / "calib"
        
        print(f"Base Path: {self.base_path}")
    
        if self.lidar_path.exists():
            self.file_list = sorted([f.stem for f in self.lidar_path.glob("*.bin")])
            print(f"📊 files found  : {len(self.file_list)}")
        else:
            self.file_list = []
            print(f"❌ Error :  LiDAR folder not found : {self.lidar_path}")
            
    def __len__(self):
        return len(self.file_list)

    def load_label(self, file_id, calib):
        """Load and transform labels from camera to LiDAR coordinates"""
        annotations = []
        label_file = self.label_path / f"{file_id}.txt"
        if not label_file.exists(): return np.array([])

        with open(label_file, 'r') as f:
            for line in f.readlines():
                p = line.split()
                if p[0] == 'DontCare': continue
                h, w, l = float(p[8]), float(p[9]), float(p[10])
                loc_cam = np.array([[float(p[11]), float(p[12]), float(p[13])]])
                ry = float(p[14]) # camera rotation around Y-axis (vertical axis)
                loc_lidar = calib.project_rect_to_velo(loc_cam).flatten() # Cam to LiDAR
                loc_lidar[2] += h / 2 # BBOX correction, in LiDAR we use the center of the box
                yaw_lidar = -ry - np.pi / 2 # angle of the object corrected for LiDAR coordinates
                annotations.append([loc_lidar[0], loc_lidar[1], loc_lidar[2], l, w, h, yaw_lidar])
        
        return np.array(annotations, dtype=np.float32)

    def transform_to_pillars(self, points):
        """Simplified version, Create a pseudo-image from LiDAR points"""
        pc_range = self.dataset_config['pc_range']
        grid_size = self.dataset_config['grid_size']

        # FFiltering points within the defined range
        mask = ((points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
                (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]))
        points = points[mask]

        # Calcul of the resolution of each grid cell
        res_x = (pc_range[3] - pc_range[0]) / grid_size[0]
        res_y = (pc_range[4] - pc_range[1]) / grid_size[1]
        # Mapping points to grid cells
        u = ((points[:, 0] - pc_range[0]) / res_x).astype(np.int32)
        v = ((points[:, 1] - pc_range[1]) / res_y).astype(np.int32)

        # Creating a pseudo image with 2 channels: max height and density
        pseudo_image = np.zeros((2, grid_size[0], grid_size[1]), dtype=np.float32)
        for i in range(len(points)):
            # Canal 0: max height (z) in the cell
            if points[i, 2] > pseudo_image[0, u[i], v[i]]:
                pseudo_image[0, u[i], v[i]] = points[i, 2]
            # Canal 1: Density of the pillar
            pseudo_image[1, u[i], v[i]] += 0.1 
        return pseudo_image

    def __getitem__(self, idx):
        """Return a single sample from the dataset"""
        """
        - Read a binar file
        - calibrate labels
        - transform points to pseudo-image
        - transform to tensors"""
        
        file_id = self.file_list[idx]
        calib = KittiCalibration(self.calib_path / f"{file_id}.txt")
        points = np.fromfile(self.lidar_path / f"{file_id}.bin", dtype=np.float32).reshape(-1, 4)
        
        gt_boxes = self.load_label(file_id, calib)
        pseudo_image = self.transform_to_pillars(points)

        return {
            "input": torch.from_numpy(pseudo_image),
            "target": torch.from_numpy(gt_boxes),
            "id": file_id
        }
