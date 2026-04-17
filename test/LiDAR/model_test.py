import torch
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np

from src.late_fusion.utils.pillar_dataset import KittiPillarDataset 
from src.late_fusion.LiDAR.pillarbackbone2 import PillarBackbone
from src.late_fusion.LiDAR.inference import get_detected_boxes
from src.late_fusion.utils.display_lidar import visualize_point_cloud, visualize_point_cloud_with_boxes
from src.late_fusion.utils.display_lidar import get_projected_bbox_coords, visualize_boxes_on_image

CONFIG_PATH ="./src/late_fusion/LiDAR/config.yaml"
DATA_DIR="./data/kitti_lidar"
MODEL_PATH= "./mlruns/429621317766757696/models/m-9b4d0a4b7fe144f7b34c834b0548a1c4/artifacts/data/model.pth" 
confidence_threshold = 0.1
torch.serialization.add_safe_globals([PillarBackbone])


# def run_test():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Test sur : {device}")

#     # 1. Charger le Dataset de Validation
#     val_dataset = KittiPillarDataset(config_path=CONFIG_PATH, data_dir=DATA_DIR, split='val')
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) 
#     print(f"Dataset de validation chargé ({len(val_dataset)} fichiers).")

#     # 2. Charger le Modèle entraîné
#     with open(CONFIG_PATH, 'r') as f:
#         config = yaml.safe_load(f)
    
#     model = PillarBackbone(in_channels=config['dataset']['num_channels'])
    
#     if os.path.exists(MODEL_PATH):
#         checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
#         if isinstance(checkpoint, dict):
#             model.load_state_dict(checkpoint)
#         else:
#             model.load_state_dict(checkpoint.state_dict())
        
#     print("model loaded successfully")
#     model.to(device)
#     model.eval()
    
#     # testing on 3 examples
#     for i, batch in enumerate(val_loader):
#         if i >= 3: break
#         print(f"\n Example {i+1} :")
#         pseudo_image = batch['input'].to(device).float()
#         file_id = batch['id'][0]
        
#         # Prediction
#         with torch.no_grad():
#             preds = model(pseudo_image) 

#         # bbox extraction
#         detected_boxes = get_detected_boxes(preds, threshold=confidence_threshold) 
#         print(f"Voitures détectées (score > {confidence_threshold}) : {len(detected_boxes)}")

#         # Display
#         bin_path = os.path.join(DATA_DIR, 'val', 'velodyne', f"{file_id}.bin")
#         if os.path.exists(bin_path):
#             point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            
#             visualize_point_cloud(point_cloud)
#             if len(detected_boxes) > 0:
#                 visualize_point_cloud_with_boxes(point_cloud, detected_boxes)
#             else:
#                 print("no box to display")
#         else:
#             print(f"file .bin not found {file_id}, cant be displayed.")
            
#         projected = get_projected_bbox_coords(calib_path, preds[0], score_threshold=confidence_threshold)
#         visualize_boxes_on_image(image_file, projected)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def run_test():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Test sur : {device}")

#     # 1. Charger le Dataset de Validation
#     val_dataset = KittiPillarDataset(config_path=CONFIG_PATH, data_dir=DATA_DIR, split='val')
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) 
#     print(f"Dataset de validation chargé ({len(val_dataset)} fichiers).")

#     # 2. Charger le Modèle
#     with open(CONFIG_PATH, 'r') as f:
#         config = yaml.safe_load(f)
    
#     model = PillarBackbone(in_channels=config['dataset']['num_channels'])
    
#     if os.path.exists(MODEL_PATH):
#         # Utilisation de weights_only=True par sécurité si possible
#         checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
#         if isinstance(checkpoint, dict):
#             model.load_state_dict(checkpoint)
#         else:
#             model.load_state_dict(checkpoint.state_dict())
#         print("Modèle chargé avec succès.")
#     else:
#         print(f"ERREUR : Modèle non trouvé à {MODEL_PATH}")
#         return

#     model.to(device)
#     model.eval()
    
#     # 3. Boucle de test
#     for i, batch in enumerate(val_loader):
#         if i >= 3: break
        
#         file_id = batch['id'][0]
#         print(f"\n--- Exemple {i+1} (ID: {file_id}) ---")
        
#         pseudo_image = batch['input'].to(device).float()
        
#         # Chemins dynamiques basés sur le file_id
#         # On suppose que DATA_DIR pointe vers la racine contenant 'val' ou 'training'
#         split_dir = os.path.join(DATA_DIR, 'val') 
#         bin_path = os.path.join(split_dir, 'velodyne', f"{file_id}.bin")
#         calib_path = os.path.join(split_dir, 'calib', f"{file_id}.txt")
#         image_file = os.path.join(split_dir, 'image_2', f"{file_id}.png")

#         # Inférence
#         with torch.no_grad():
#             preds = model(pseudo_image) 

#         # Extraction pour Open3D (liste de dictionnaires avec 'box' et 'score')
#         detected_boxes = get_detected_boxes(preds, threshold=confidence_threshold) 
#         print(f"Objets détectés : {len(detected_boxes)}")

#         # --- VISUALISATION 3D (LIDAR) ---
#         if os.path.exists(bin_path):
#             point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#             if len(detected_boxes) > 0:
#                 visualize_point_cloud_with_boxes(point_cloud, detected_boxes)
#             else:
#                 visualize_point_cloud(point_cloud)
#         else:
#             print(f"Fichier .bin manquant : {bin_path}")

#         # --- VISUALISATION 2D (IMAGE) ---
#         if os.path.exists(calib_path) and os.path.exists(image_file):
#             # Attention : preds[0] car on a un batch de 1
#             # get_projected_bbox_coords utilise ta classe KittiCalibration
#             projected = get_projected_bbox_coords(calib_path, preds[0], score_threshold=confidence_threshold)
            
#             if len(projected) > 0:
#                 visualize_boxes_on_image(image_file, projected)
#             else:
#                 print("Aucune boîte projetée sur l'image (score trop bas ou hors champ).")
#         else:
#             print(f"path Calib pour {calib_path} et path image {image_file} de l'exemple {file_id}.")
             
            
            
# if __name__ == "__main__":
#     run_test()






































import torch
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
from pathlib import Path

# Tes imports locaux
from src.late_fusion.utils.pillar_dataset import KittiPillarDataset 
from src.late_fusion.LiDAR.pillarbackbone2 import PillarBackbone
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
# On utilise Path pour éviter les soucis de slash

MODEL_PATH = Path("./mlruns/429621317766757696/models/m-9b4d0a4b7fe144f7b34c834b0548a1c4/artifacts/data/model.pth" )
IMG_ROOT = Path("./data/kitti_yolo/val/images")
CONFIDENCE_THRESHOLD = 0.1

torch.serialization.add_safe_globals([PillarBackbone])

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Test lancé sur : {device}")

    # 1. Charger le Dataset de Validation
    val_dataset = KittiPillarDataset(config_path=CONFIG_PATH, data_dir=DATA_DIR, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) 
    print(f"✅ Dataset chargé ({len(val_dataset)} fichiers).")

    # 2. Charger le Modèle
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    model = PillarBackbone(in_channels=config['dataset']['num_channels'])
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint.state_dict())
        print("✅ Modèle chargé avec succès.")
    else:
        print(f"❌ ERREUR : Modèle introuvable à {MODEL_PATH}")
        return

    model.to(device)
    model.eval()
    
    # 3. Boucle de test sur 3 exemples
    for i, batch in enumerate(val_loader):
        if i >= 3: break
        
        file_id = batch['id'][0]
        print(f"\n🔎 --- Exemple {i+1} (ID: {file_id}) ---")
        
        # Reconstruction des chemins spécifiques
        lidar_val_dir = Path(DATA_DIR) / "val"
        bin_path = lidar_val_dir / "velodyne" / f"{file_id}.bin"
        calib_path = lidar_val_dir / "calib" / f"{file_id}.txt"
        image_file = IMG_ROOT / f"{file_id}.png"

        # Inférence
        pseudo_image = batch['input'].to(device).float()
        with torch.no_grad():
            preds = model(pseudo_image) 
            scores = torch.sigmoid(preds[0, 0, :, :]) # On applique sigmoid sur le canal de classif
            print(f"Max score dans la heatmap : {scores.max().item():.4f}")
            print(f"Moyenne des scores : {scores.mean().item():.4f}")
        # 4. Extraction des boîtes pour Open3D
        detected_boxes = get_detected_boxes(preds, threshold=CONFIDENCE_THRESHOLD) 
        print(f"📦 Objets détectés : {len(detected_boxes)}")

        # --- VISUALISATION 3D (LIDAR) ---
        if bin_path.exists():
            point_cloud = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
            if len(detected_boxes) > 0:
                # visualize_point_cloud_with_boxes attend une liste de dicts {'box': [], 'score': float}
                visualize_point_cloud_with_boxes(point_cloud, detected_boxes)
            else:
                visualize_point_cloud(point_cloud)
        else:
            print(f"⚠️ Fichier .bin manquant : {bin_path}")

        # --- VISUALISATION 2D (IMAGE) ---
        if calib_path.exists() and image_file.exists():
            # preds[0] car on retire la dimension de batch (1, 8, H, W) -> (8, H, W)
            projected = get_projected_bbox_coords(
                str(calib_path), 
                preds[0], 
                score_threshold=CONFIDENCE_THRESHOLD
            )
            
            if len(projected) > 0:
                print(f"🎨 Projection de {len(projected)} boîtes sur l'image...")
                visualize_boxes_on_image(str(image_file), projected)
            else:
                print("ℹ️ Aucune boîte n'a survécu au seuil ou n'est dans le champ caméra.")
        else:
            if not calib_path.exists(): print(f"⚠️ Calib manquant : {calib_path}")
            if not image_file.exists(): print(f"⚠️ Image manquante : {image_file}")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"💥 Erreur critique : {e}")