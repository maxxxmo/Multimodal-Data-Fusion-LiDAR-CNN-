import torch
import yaml
import numpy as np
from pathlib import Path
import h5py
from src.late_fusion.utils.pillar_dataset import KittiPillarDataset 
from src.late_fusion.LiDAR.pillarbackbone3 import PillarBackbone
from src.late_fusion.LiDAR.anchors import TargetAssigner, AnchorGenerator
from src.late_fusion.LiDAR.inference import visualize_boxes_on_pseudo_image, get_detected_boxes


from test.LiDAR.test_label_calibration import visualize_lidar_and_bboxes, get_color_map

# --- CONFIGURATION ---
CONFIG_PATH = "./src/late_fusion/LiDAR/config.yaml"
DATA_DIR = "./data/kitti_lidar"
MODEL_PATH = Path("./mlruns/429621317766757696/models/m-35d124cbb3064eb09864b4d77b3304a2/artifacts/data/model.pth")
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 1 
num_types = len([[3.9, 1.6, 1.56]])       
num_rots = len([0])
total_anchors = num_types * num_rots 
split_eval = 'train'
id = "0001_000010"
def run_inference(pattern, split=split_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inférence sur : {device}")
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Initialisation des composants
    anchor_gen = AnchorGenerator(
        feature_map_size=tuple(config['dataset']['grid_size']), 
        anchor_sizes=[[3.9, 1.6, 1.56]], 
        anchor_rotations=[0], 
        pc_range=config['dataset']['pc_range']
    )
    
    dataset = KittiPillarDataset(
        config_path=CONFIG_PATH, 
        data_dir=DATA_DIR, 
        split=split, 
        anchor_gen=anchor_gen, 
        target_assigner=TargetAssigner(iou_thresholds=(0.3, 0.5))
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