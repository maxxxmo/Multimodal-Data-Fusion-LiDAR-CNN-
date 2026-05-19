import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as T
from src.late_fusion.utils.calibration import KittiCalibration


class KittiMultiModalDataset(Dataset):
    """
    Dataset multimodal chargeant les pseudo-images LiDAR (via H5) 
    et les images/labels au format YOLO pour le CNN.
    """
    def __init__(self, 
                 data_dir="./data/kitti_lidar", 
                 yolo_dir="./data/kitti_yolo",
                 config_path="./src/late_fusion/LiDAR/config.yaml", 
                 split='train', 
                 anchor_gen=None, 
                 target_assigner=None,
                 img_size=(640, 640)): 
        
        self.anchor_gen = anchor_gen
        self.target_assigner = target_assigner
        self.img_size = img_size
        
        #  LiDAR Configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_config = self.config['dataset']
        
        self.root_dir = Path(data_dir).resolve()
        self.base_path = self.root_dir / split    
        self.lidar_path = self.base_path / "velodyne"
        
        # Camera configuration
        self.yolo_base_path = Path(yolo_dir).resolve() / split
        self.yolo_img_path = self.yolo_base_path / "images"
        self.yolo_label_path = self.yolo_base_path / "labels"
        
        # H5 file for pseudo image
        self.h5_path = self.base_path / "h_data" / "dataset_unified.h5"
        self.h5_file = None
        
        # calibration
        
        self.calib_path = self.base_path / "calib"
        
        self.img_transforms = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(), # Convertit en [C, H, W] et scale entre 0.0 et 1.0
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        if self.lidar_path.exists():
            self.file_list = sorted([f.stem for f in self.lidar_path.glob("*.bin")])
            print(f"✅ {len(self.file_list)} samples multimodaux trouvés.")
        else:
            self.file_list = []
            print(f"❌ ERREUR : Chemin LiDAR introuvable -> {self.lidar_path}")

    def __getstate__(self):
        # for parallelisation 
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state



    def __len__(self):
        return len(self.file_list)

    def load_yolo_label(self, file_id):
        """load  labels  [class, x, y, w, h] normalized"""
        label_file = self.yolo_label_path / f"{file_id}.txt"
        if not label_file.exists():
            return np.array([], dtype=np.float32)
        
        try:
            labels = np.loadtxt(label_file, dtype=np.float32)
            if labels.ndim == 1: 
                labels = np.expand_dims(labels, axis=0)
            return labels
        except Exception:
            return np.array([], dtype=np.float32)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r') 
            
        file_id = str(self.file_list[idx])
        group = self.h5_file[file_id]
        
        # LiDAR pseudo image
        lidar_inputs = torch.from_numpy(group['pseudo_image'][:]).float()
        
        # image
        img_file = self.yolo_img_path / f"{file_id}.jpg"
        if img_file.exists():
            img = Image.open(img_file).convert('RGB')
            img_tensor = self.img_transforms(img)
        else:
            img_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            
        # targets
        yolo_labels = self.load_yolo_label(file_id)


        calib_file = self.calib_path / f"{file_id}.txt"
        if calib_file.exists():
            calib_obj = KittiCalibration(calib_file)
        else:
            calib_obj = None
        
        return {
            'id': file_id,
            'lidar_inputs': lidar_inputs,
            'camera_inputs': img_tensor,
            'calib': calib_obj,
            'lidar_targets': {
                'cls': torch.from_numpy(group['targets_cls'][:]).float(),
                'reg': torch.from_numpy(group['targets_reg'][:]).float(),
                'bev_label': torch.from_numpy(group['bev_label'][:]).float()
            },
            'camera_targets': torch.from_numpy(yolo_labels).float(),
            'pos_mask': torch.from_numpy(group['pos_mask'][:]).bool(),
        }