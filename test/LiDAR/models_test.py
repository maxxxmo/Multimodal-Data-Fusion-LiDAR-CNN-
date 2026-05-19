import torch
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO  


from src.late_fusion.utils.kittimultimodaldataset import KittiMultiModalDataset
from src.late_fusion.LiDAR.model.pillarbackbone3 import PillarBackbone
from src.late_fusion.LiDAR.anchors import TargetAssigner, AnchorGenerator
from src.late_fusion.LiDAR.inference import get_detected_boxes
from src.late_fusion.utils.display_lidar import get_color_map
from src.late_fusion.utils.display_lidar import visualize_lidar_and_bboxes
from src.late_fusion.utils.nms import apply_nms

# Configuration des chemins
CONFIG_PATH = "./src/late_fusion/LiDAR/config.yaml"
DATA_DIR = "./data/kitti_lidar"
YOLO_DIR = "./data/kitti_yolo"

# Tes checkpoints de modèles
MODEL_LIDAR_PATH = Path("./mlruns/429621317766757696/models/m-3476de56f923442e86650d04c6287326/artifacts/data/model.pth")
MODEL_CNN_PATH = Path("./runs/detect/runs/train/yolov8n.pt_20260417-0935/weights/best.pt")

CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 1 
split_eval = 'val'
id = "0014_000089"

def run_multimodal_inference(pattern, split=split_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Configuration des Anchors LiDAR
    anchor_gen = AnchorGenerator(
        feature_map_size=tuple(config['dataset']['grid_size']), 
        anchor_sizes=config['dataset']['anchor_sizes'], 
        anchor_rotations=config['dataset']['anchor_rotations'], 
        pc_range=config['dataset']['pc_range']
    )
    total_anchors = len(config['dataset']['anchor_sizes']) * len(config['dataset']['anchor_rotations'])
    
    # 2. Dataset Multimodal
    dataset = KittiMultiModalDataset(
        config_path=CONFIG_PATH, 
        data_dir=DATA_DIR, 
        yolo_dir=YOLO_DIR,
        split=split, 
        anchor_gen=anchor_gen, 
        target_assigner=TargetAssigner(iou_thresholds=(0.5, 0.7))
    )

    indices = [i for i, fid in enumerate(dataset.file_list) if pattern in fid]
    if not indices:
        print(f"❌ Aucun fichier trouvé pour l'identifiant : {pattern}")
        return

    # 3. Chargement du modèle LiDAR
    model_lidar = PillarBackbone(in_channels=config['dataset']['num_channels'], num_anchors=total_anchors)
    if MODEL_LIDAR_PATH.exists():
        checkpoint_lidar = torch.load(MODEL_LIDAR_PATH, map_location=device, weights_only=False)
        model_lidar.load_state_dict(checkpoint_lidar if isinstance(checkpoint_lidar, dict) else checkpoint_lidar.state_dict())
        model_lidar.to(device).eval()
        print("✅ Modèle PointPillar LiDAR chargé.")
    else:
        print(f"❌ Modèle LiDAR introuvable à {MODEL_LIDAR_PATH}")
        return

    # 4. Chargement du modèle YOLOv8
    if MODEL_CNN_PATH.exists():
        # La classe YOLO d'ultralytics gère automatiquement le device et le chargement des poids
        model_cnn = YOLO(str(MODEL_CNN_PATH))
        print("✅ Modèle CNN YOLOv8 chargé.")
    else:
        print(f"❌ Modèle CNN introuvable à {MODEL_CNN_PATH}")
        return

    # 5. Boucle d'inférence
    for i in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]
        batch_samples = [dataset[idx] for idx in batch_indices]
        
        # Envoi de la pseudo-image LiDAR sur le device
        batch_inputs_lidar = torch.stack([s['lidar_inputs'] for s in batch_samples]).to(device).float()
        
        with torch.no_grad():
            # Inférence LiDAR
            cls_logits, reg_preds = model_lidar(batch_inputs_lidar)
            
            for j, original_idx in enumerate(batch_indices):
                file_id = dataset.file_list[original_idx]
                print(f"\n======== SÉQUENCE DE VISUALISATION : {file_id} ========")
                
                # ---------------------------------------------------------
                # PARTIE 1 : Traitement et Visualisation LiDAR 3D
                # ---------------------------------------------------------
                detected_boxes, scores = get_detected_boxes(
                    cls_logits[j:j+1], 
                    reg_preds[j:j+1], 
                    dataset.anchor_gen, 
                    score_thresh=CONFIDENCE_THRESHOLD
                )
                
                detected_boxes_nms, scores_nms = apply_nms(detected_boxes, scores, iou_threshold=0.1)
                
                bin_path = Path(DATA_DIR, split, 'velodyne', f"{file_id}.bin")
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                point_colors = get_color_map(points[:, 2])
                
                print("📺 [1/2] Affichage du nuage de points LiDAR (3D)...")
                visualize_lidar_and_bboxes(
                    points=points[:, :3], 
                    annotations=[b.cpu().numpy() for b in detected_boxes_nms],
                    point_colors=point_colors
                )
                
                # ---------------------------------------------------------
                # PARTIE 2 : Inférence et Visualisation Caméra 2D (YOLOv8)
                # ---------------------------------------------------------
                # On récupère le chemin de l'image d'origine pour YOLOv8 (extension .txt -> .jpg/.png)
                # Note : Ajuste l'extension si tes images YOLO sont en .png
                raw_img_path = Path(YOLO_DIR, split, "images", f"{file_id}.png")
                
                if raw_img_path.exists():
                    print("📺 [2/2] Affichage des détections Caméra (2D)...")
                    
                    # Exécution de l'inférence YOLOv8 directement sur le fichier
                    results = model_cnn(str(raw_img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
                    
                    # .plot() génère l'image avec les boîtes dessinées au format BGR Numpy Array
                    annotated_frame = results[0].plot()
                    # Conversion BGR vers RGB pour un affichage correct dans Matplotlib
                    annotated_frame_rgb = annotated_frame[:, :, ::-1]
                    
                    # Affichage graphique de l'image annotée
                    plt.figure(figsize=(12, 6))
                    plt.imshow(annotated_frame_rgb)
                    plt.title(f"Détections Caméra YOLOv8 - {file_id}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"❌ Image d'origine introuvable pour la caméra : {raw_img_path}")

if __name__ == "__main__":
    run_multimodal_inference(id)