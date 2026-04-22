import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.late_fusion.utils.calibration import KittiCalibration
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
        h5_path = self.base_path / "h_data" / "dataset_cache.h5"
        self.chunk_size = 100
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
        """Version avec Intensité : 3 canaux (Z max, Densité, Intensité moyenne)"""
        pc_range = self.dataset_config['pc_range']
        grid_size = self.dataset_config['grid_size']

        # Filtrage des points (X et Y)
        mask = ((points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
                (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]))
        points = points[mask]

        res_x = (pc_range[3] - pc_range[0]) / grid_size[0]
        res_y = (pc_range[4] - pc_range[1]) / grid_size[1]
        
        u = ((points[:, 0] - pc_range[0]) / res_x).astype(np.int32)
        v = ((points[:, 1] - pc_range[1]) / res_y).astype(np.int32)
        
        # Limiter u et v pour éviter les index out of bounds
        u = np.clip(u, 0, grid_size[0] - 1)
        v = np.clip(v, 0, grid_size[1] - 1)

        # Création de la pseudo-image : (3, H, W)
        # Canal 0: Z max, Canal 1: Densité, Canal 2: Intensité
        pseudo_image = np.zeros((3, grid_size[0], grid_size[1]), dtype=np.float32)
        
        # Pour l'intensité moyenne, on va avoir besoin de compter les points par pilier
        count_map = torch.zeros((grid_size[0], grid_size[1])) 

        for i in range(len(points)):
            curr_u, curr_v = u[i], v[i]
            z, intensity = points[i, 2], points[i, 3]

            # Canal 0 : Hauteur Max
            if z > pseudo_image[0, curr_u, curr_v]:
                pseudo_image[0, curr_u, curr_v] = z
            
            # Canal 1 : Densité (cumulée)
            pseudo_image[1, curr_u, curr_v] += 0.1
            
            # Canal 2 : Intensité (cumulée pour faire la moyenne après)
            pseudo_image[2, curr_u, curr_v] += intensity
            count_map[curr_u, curr_v] += 1

        # Finalisation des canaux
        # Moyenne de l'intensité sur les piliers non vides
        mask_non_empty = count_map > 0
        pseudo_image[2, mask_non_empty] /= count_map[mask_non_empty].numpy()
        
        # Normalisation de la densité (déjà dans ton code)
        pseudo_image[1] = np.log1p(pseudo_image[1])
        
        return pseudo_image


    def __getitem__(self, idx):
            # 1. Gestion de l'ouverture du fichier H5 (sécurisé)
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')
                
            file_id = self.file_list[idx]
            group = self.h5_file[str(file_id)]
            
            # 2. Chargement des données brutes
            # Assure-toi que ces clés correspondent à ton fichier H5
            input_data = torch.from_numpy(group['pseudo_image'][:]).float()
            target_data = torch.from_numpy(group['labels'][:])
            
            # 3. Assignation à la volée
            # On passe les ancres et les labels récupérés
            cls_t, reg_t, mask = self.target_assigner.assign(self.anchor_gen.anchors, target_data)
            
            # 4. Retourne la structure attendue
            return {
                'id': file_id,
                'inputs': input_data,
                'targets': {
                    'cls': cls_t.float(), 
                    'reg': reg_t.float()
                },
                'pos_mask': mask.to(torch.uint8)
            }
    
