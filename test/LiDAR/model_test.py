import torch
import yaml
import numpy as np
from pathlib import Path
import h5py
from src.late_fusion.utils.pillar_dataset import KittiPillarDataset 
from src.late_fusion.LiDAR.model.pillarbackbone3 import PillarBackbone
from src.late_fusion.LiDAR.anchors import TargetAssigner, AnchorGenerator
from src.late_fusion.LiDAR.inference import get_detected_boxes
from src.late_fusion.utils.display_lidar import get_color_map
from src.late_fusion.utils.display_lidar import visualize_lidar_and_bboxes

# --- CONFIGURATION ---
CONFIG_PATH = "./src/late_fusion/LiDAR/config.yaml"
DATA_DIR = "./data/kitti_lidar"
MODEL_PATH = Path("./mlruns/429621317766757696/models/m-3476de56f923442e86650d04c6287326/artifacts/data/model.pth")
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 1 


split_eval = 'val'
id = "0014_000089"
def run_inference(pattern, split=split_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inférence sur : {device}")
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Initialisation des composants
    anchor_gen = AnchorGenerator(
        feature_map_size=tuple(config['dataset']['grid_size']), 
        anchor_sizes=config['dataset']['anchor_sizes'], 
        anchor_rotations=config['dataset']['anchor_rotations'], 
        pc_range=config['dataset']['pc_range']
    )
    total_anchors = len(config['dataset']['anchor_sizes'])*len(config['dataset']['anchor_rotations'])
    dataset = KittiPillarDataset(
        config_path=CONFIG_PATH, 
        data_dir=DATA_DIR, 
        split=split, 
        anchor_gen=anchor_gen, 
        target_assigner=TargetAssigner(iou_thresholds=(0.5, 0.7))
    )

    # 2. Résolution de l'indice basé sur le pattern
    indices = [i for i, fid in enumerate(dataset.file_list) if pattern in fid]
    if not indices:
        print(f"❌ Aucun fichier trouvé pour le pattern : {pattern}")
        return
    print(f"✅ {len(indices)} échantillon(s) identifié(s).")

    # 3. Chargement du modèle
    model = PillarBackbone(in_channels=config['dataset']['num_channels'], num_anchors=total_anchors)
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict())
        model.to(device).eval()
        print("✅ Modèle chargé.")
    else:
        print(f"❌ Modèle introuvable à {MODEL_PATH}")
        return

    # 4. Inférence
    for i in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch_samples = [dataset[idx] for idx in batch_indices]
        
        # Stack des inputs
        batch_inputs = torch.stack([s['inputs'] for s in batch_samples]).to(device).float()
        
        with torch.no_grad():
            cls_logits, reg_preds = model(batch_inputs)
            
            for j, original_idx in enumerate(batch_indices):
                file_id = dataset.file_list[original_idx]
                
                detected_boxes, scores = get_detected_boxes(
                    cls_logits[j:j+1], 
                    reg_preds[j:j+1], 
                    dataset.anchor_gen, 
                    score_thresh=CONFIDENCE_THRESHOLD
                )
                boxes_list = [
                    {'box': detected_boxes[k].cpu().numpy(), 'score': scores[k].item()} 
                    for k in range(len(detected_boxes))
                ]
                

                bin_path=Path(DATA_DIR,split,'velodyne',f"{file_id}.bin")
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

                formatted_annotations = [item['box'] for item in boxes_list]
                point_colors=get_color_map(points[:, 2])
                visualize_lidar_and_bboxes(points=points[:, :3], annotations=formatted_annotations,point_colors=point_colors)
                
                
if __name__ == "__main__":

    run_inference(id)