import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from pathlib import Path

from src.late_fusion.utils.pillar_dataset import KittiPillarDataset 
from src.late_fusion.LiDAR.pillarbackbone3 import PillarBackbone
from src.late_fusion.LiDAR.anchors import TargetAssigner, AnchorGenerator
from src.late_fusion.LiDAR.inference import get_detected_boxes
from src.late_fusion.utils.display_lidar import (
    visualize_point_cloud, 
    visualize_point_cloud_with_boxes,
    get_projected_bbox_coords, 
    visualize_boxes_on_image
)

# --- CONFIGURATION ---
CONFIG_PATH = "./src/late_fusion/LiDAR/config.yaml"
DATA_DIR = "./data/kitti_lidar"
MODEL_PATH = Path("./mlruns/429621317766757696/models/m-74fbda6fe5674fc5ab02ed325c54a953/artifacts/data/model.pth")
IMG_ROOT = Path("./data/kitti_yolo/val/images")
CONFIDENCE_THRESHOLD = 0.9
torch.serialization.add_safe_globals([PillarBackbone])

# Initialisation anchors
anchor_sizes = [[3.9, 1.6, 1.56]]
anchor_rotations = [0]
total_anchors = len(anchor_sizes) * len(anchor_rotations)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Test lancé sur : {device}")

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config_dataset = config['dataset']
    H, W = config_dataset['grid_size']
    pc_range = config_dataset['pc_range'] 
    
    anchor_gen = AnchorGenerator(
        feature_map_size=(H, W), 
        anchor_sizes=anchor_sizes,
        anchor_rotations=anchor_rotations, 
        pc_range=pc_range
    )
    target_assigner = TargetAssigner(iou_thresholds=(0.6, 0.8))
    
    val_dataset = KittiPillarDataset(config_path=CONFIG_PATH, data_dir=DATA_DIR, split='val', anchor_gen=anchor_gen, target_assigner=target_assigner)
    val_loader = DataLoader(val_dataset, batch_size=3, num_workers=0, shuffle=False) 
    print(f"✅ Dataset chargé ({len(val_dataset)} fichiers).")

    model = PillarBackbone(in_channels=config['dataset']['num_channels'], num_anchors=total_anchors)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict())
        print("✅ Modèle chargé avec succès.")
    else:
        print(f"❌ ERREUR : Modèle introuvable à {MODEL_PATH}")
        return

    model.to(device).eval()
    
    
    batch = next(iter(val_loader))

    
    pseudo_image = batch['inputs'].to(device).float()
    
    with torch.no_grad():
        cls_logits, reg_preds = model(pseudo_image)
        batch_size = cls_logits.shape[0]

        for b in range(batch_size):
            file_id = batch['id'][b]
            print(f"\n🔎 --- Traitement image {b+1}/{batch_size} (ID: {file_id}) ---")
            
            # Inférence image par image
            detected_boxes, scores = get_detected_boxes(
                cls_logits[b:b+1], 
                reg_preds[b:b+1], 
                val_dataset.anchor_gen, 
                score_thresh=CONFIDENCE_THRESHOLD
            )
            
            # Conversion propre
            detected_boxes_list = [
                {'box': detected_boxes[j].cpu().numpy(), 'score': scores[j].item()}
                for j in range(len(detected_boxes))
            ]
            
            # Visualisation
            lidar_val_dir = Path(DATA_DIR) / "val"
            bin_path = lidar_val_dir / "velodyne" / f"{file_id}.bin"
            
            if bin_path.exists():
                pc = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
                visualize_point_cloud_with_boxes(pc, detected_boxes_list) if detected_boxes_list else visualize_point_cloud(pc)
            
            # Visualisation 2D (optionnel, nécessite la gestion de calib/image ici)
            print(f"✅ {len(detected_boxes_list)} boîtes détectées.")

if __name__ == "__main__":
    run_test()