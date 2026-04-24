import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import h5py

class KittiPillarDataset(Dataset):
    def __init__(self, data_dir, config_path, split='train', anchor_gen=None, target_assigner=None):
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
        # On ne veut pas sérialiser (copier) le pointeur du fichier vers les workers
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state
    
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
        """
        Version finalisée : Alignement (3, Y, X) avec normalisation robuste
        pour une convergence stable des gradients.
        """
        pc_range = self.dataset_config['pc_range'] # [x_min, y_min, z_min, x_max, y_max, z_max]
        grid_size = self.dataset_config['grid_size'] 

        # 1. Filtrage spatial
        mask = ((points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
                (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]))
        points = points[mask]

        res_x = (pc_range[3] - pc_range[0]) / grid_size[0]
        res_y = (pc_range[4] - pc_range[1]) / grid_size[1]

        u = np.clip(((points[:, 0] - pc_range[0]) / res_x).astype(np.int32), 0, grid_size[0] - 1)
        v = np.clip(((points[:, 1] - pc_range[1]) / res_y).astype(np.int32), 0, grid_size[1] - 1)

        # 2. Création des maps
        pseudo_image = np.zeros((3, grid_size[1], grid_size[0]), dtype=np.float32)
        count_map = np.zeros((grid_size[1], grid_size[0]), dtype=np.float32)

        # 3. Remplissage
        for i in range(len(points)):
            curr_u, curr_v = u[i], v[i]
            z, intensity = points[i, 2], points[i, 3]

            if z > pseudo_image[0, curr_v, curr_u]:
                pseudo_image[0, curr_v, curr_u] = z
            
            pseudo_image[1, curr_v, curr_u] += 1.0 # Nombre de points par pilier
            pseudo_image[2, curr_v, curr_u] += intensity
            count_map[curr_v, curr_u] += 1

        # 4. Normalisation robuste
        mask_non_empty = count_map > 0
        
        # Canal 0 (Z) : Centré et réduit par la plage [z_min, z_max]
        z_min, z_max = pc_range[2], pc_range[5]
        pseudo_image[0, mask_non_empty] = (pseudo_image[0, mask_non_empty] - z_min) / (z_max - z_min)
        
        # Canal 1 (Nombre de points) : log-échelle pour éviter les pics extrêmes
        pseudo_image[1] = np.log1p(pseudo_image[1])
        
        # Canal 2 (Intensité) : Moyenne normalisée
        pseudo_image[2, mask_non_empty] /= count_map[mask_non_empty]
        pseudo_image[2, mask_non_empty] = np.clip(pseudo_image[2, mask_non_empty] / 255.0, 0, 1)

        # Sécurité finale : aucune valeur infinie ou NaN
        return np.nan_to_num(pseudo_image)
    
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r') # Pointant sur le fichier unifié
            
        file_id = str(self.file_list[idx])
        group = self.h5_file[file_id]
        
        # Tout est lu au même endroit
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