import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import h5py



class KittiPillarDataset(Dataset):
    """This class is responsible for loading and preprocessing the KITTI dataset for the PointPillar model.
    It reads LiDAR point clouds, transforms them into a pseudo-image format suitable for BEV (Bird's Eye View)
    detection, and loads corresponding labels"""
    
    def __init__(self, data_dir="./data/kitti_lidar", config_path= "./src/late_fusion/LiDAR/config.yaml", split='train', anchor_gen=None, target_assigner=None):
        self.anchor_gen = anchor_gen
        self.target_assigner = target_assigner
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_config = self.config['dataset']
        self.root_dir = Path(data_dir).resolve()
        self.base_path = self.root_dir / split    
        

        # Paths
        self.label_path = self.base_path / "labels"
        h5_path = self.base_path / "h_data" / "dataset_unified.h5"
        self.h5_path = h5_path
        self.h5_file = None
        self.lidar_path = self.base_path / "velodyne"
        
        if self.lidar_path.exists():
            # On cherche les fichiers .bin dans le dossier velodyne
            self.file_list = sorted([f.stem for f in self.lidar_path.glob("*.bin")])
            print(f"✅ {len(self.file_list)} fichiers trouvés.")
        else:
            self.file_list = []
            print(f"❌ ERREUR : Chemin introuvable -> {self.lidar_path}")
            
    def __getstate__(self):
        """Usefull for parallelization, so we can copy the h5 dataset on different threads of the CPU. 
        (h55py cant share) an open folder this reset file state to none before copy so it doesnt crash"""
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state
    
    def __len__(self):
        """tell pytorch how many samples are available"""
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
                ry = float(p[14]) # camera rotation
                loc_lidar = calib.project_rect_to_velo(loc_cam).flatten() # Cam to LiDAR
                loc_lidar[2] += h / 2 # BBOX correction, in LiDAR we use the center of the box
                yaw_lidar = -ry - np.pi / 2 # angle of the object corrected for LiDAR coordinates
                annotations.append([loc_lidar[0], loc_lidar[1], loc_lidar[2], w, l, h, yaw_lidar])
        
        return np.array(annotations, dtype=np.float32)
    

    def transform_to_pillars(self, points):
        """
        Transofrm 3d points to  BEV PointPillar (C, H, W)

        - H = axe Y 
        - W = axe X 
        
        """

        pc_range = self.dataset_config['pc_range']
        grid_size = self.dataset_config['grid_size']  # [H, W]

        H, W = grid_size

        # Filter point inside pc_range

        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4])
        )
        points = points[mask]

        # Resolution calculus

        res_x = (pc_range[3] - pc_range[0]) / W
        res_y = (pc_range[4] - pc_range[1]) / H

        # 3. GRID INDEXING (u=W, v=H)

        u = np.clip(((points[:, 0] - pc_range[0]) / res_x).astype(np.int32), 0, W - 1)
        v = np.clip(((points[:, 1] - pc_range[1]) / res_y).astype(np.int32), 0, H - 1)

        # 4. BEV MAP (C, H, W)

        pseudo_image = np.zeros((3, H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        # Filling the bev map

        for i in range(len(points)):
            uu = u[i]
            vv = v[i]

            z = points[i, 2]
            intensity = points[i, 3]

            # Keeping higher point of the column
            if z > pseudo_image[0, vv, uu]:
                pseudo_image[0, vv, uu] = z
            # keeping density and intensity
            pseudo_image[1, vv, uu] += 1.0
            pseudo_image[2, vv, uu] += intensity
            count_map[vv, uu] += 1

        # normalization

        mask_non_empty = count_map > 0

        # height --> point on the ground is now 0
        pseudo_image[0, mask_non_empty] = ((pseudo_image[0, mask_non_empty] + 1.6) / 0.5)

        # Density --> compress values
        pseudo_image[1] = np.log1p(pseudo_image[1])

        # Intensity --> decrease values
        pseudo_image[2, mask_non_empty] /= count_map[mask_non_empty]
        pseudo_image[2, mask_non_empty] = np.tanh(pseudo_image[2, mask_non_empty] / 10.0)
        
        return np.nan_to_num(pseudo_image)


    def __getitem__(self, idx):
        """The dataloader call this function to load Data, needs to be fast"""
        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r') 
            
        file_id = str(self.file_list[idx])
        group = self.h5_file[file_id]
        
        return {
            'id': file_id,
            'inputs': torch.from_numpy(group['pseudo_image'][:]).float(),
            'targets': {
                'cls': torch.from_numpy(group['targets_cls'][:]).float(),
                'reg': torch.from_numpy(group['targets_reg'][:]).float(),
                'bev_label': torch.from_numpy(group['bev_label'][:]).float()
            },
            'pos_mask': torch.from_numpy(group['pos_mask'][:]).bool(),
        }