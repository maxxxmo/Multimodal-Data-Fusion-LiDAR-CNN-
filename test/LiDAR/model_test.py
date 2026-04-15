import torch
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np

from src.late_fusion.utils.pillar_dataset import KittiPillarDataset 
from src.late_fusion.LiDAR.pillarbackbone2 import PillarBackbone
from src.late_fusion.LiDAR.inference import get_detected_boxes
from src.late_fusion.utils.display_lidar import visualize_point_cloud, visualize_point_cloud_with_boxes

CONFIG_PATH ="./src/late_fusion/LiDAR/config.yaml"
DATA_DIR="./data/kitti_lidar"
MODEL_PATH= "./mlruns/1/models/m-7ba4c8f3691f405196b3b69b08196a24/artifacts/data/model.pth" 
confidence_threshold = 0.1
torch.serialization.add_safe_globals([PillarBackbone])


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test sur : {device}")

    # 1. Charger le Dataset de Validation
    val_dataset = KittiPillarDataset(config_path=CONFIG_PATH, data_dir=DATA_DIR, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) 
    print(f"Dataset de validation chargé ({len(val_dataset)} fichiers).")

    # 2. Charger le Modèle entraîné
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    model = PillarBackbone(in_channels=config['dataset']['num_channels'])
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint.state_dict())
        
    print("model loaded successfully")
    model.to(device)
    model.eval()
    
    # testing on 3 examples
    for i, batch in enumerate(val_loader):
        if i >= 3: break
        print(f"\n Example {i+1} :")
        pseudo_image = batch['input'].to(device).float()
        file_id = batch['id'][0]
        
        # Prediction
        with torch.no_grad():
            preds = model(pseudo_image) 

        # bbox extraction
        detected_boxes = get_detected_boxes(preds, threshold=confidence_threshold) 
        print(f"Voitures détectées (score > {confidence_threshold}) : {len(detected_boxes)}")

        # Display
        bin_path = os.path.join(DATA_DIR, 'val', 'velodyne', f"{file_id}.bin")
        if os.path.exists(bin_path):
            point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            
            visualize_point_cloud(point_cloud)
            if len(detected_boxes) > 0:
                visualize_point_cloud_with_boxes(point_cloud, detected_boxes)
            else:
                print("no box to display")
        else:
            print(f"file .bin not found {file_id}, cant be displayed.")
    
if __name__ == "__main__":
    run_test()