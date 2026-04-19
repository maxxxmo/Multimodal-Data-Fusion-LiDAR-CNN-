import os
import yaml
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from src.late_fusion.utils.pillar_dataset import KittiPillarDataset
from src.late_fusion.LiDAR.pillarbackbone2 import PillarBackbone
from src.late_fusion.LiDAR.detectionloss2 import DetectionLoss

model_name ="best_Lidar_model.pth"




def draw_gaussian(heatmap, center, radius):
    """Dessine une gaussienne 2D sur la heatmap pour guider l'apprentissage."""
    diameter = 2 * radius + 1
    # Création d'une gaussienne 1D
    gaussian_1d = np.exp(-((np.arange(diameter) - radius) ** 2) / (2 * (radius / 3) ** 2))
    # Création du kernel 2D
    x, y = np.meshgrid(gaussian_1d, gaussian_1d)
    kernel = torch.from_numpy(x * y).float()
    
    height, width = heatmap.shape
    y_c, x_c = center
    
    # Calcul des zones de collage
    top, bottom = max(0, y_c - radius), min(height, y_c + radius + 1)
    left, right = max(0, x_c - radius), min(width, x_c + radius + 1)
    
    # Collage avec maintien du maximum (pour gérer les chevauchements)
    masked_gaussian = kernel[max(0, radius - y_c) : min(diameter, radius + height - y_c),
                             max(0, radius - x_c) : min(diameter, radius + width - x_c)]
    
    heatmap[top:bottom, left:right] = torch.max(heatmap[top:bottom, left:right], masked_gaussian)
    return heatmap




# def create_heatmaps(targets, grid_size=(432, 496), pc_range=[0, -39.68, 69.12, 39.68], radius=5):
#     """
#     targets: tenseur (N, 7) [x, y, z, w, l, h, theta]
#     """
#     H, W = grid_size
#     x_min, y_min, x_max, y_max = pc_range
    
#     voxel_size_x = (x_max - x_min) / W
#     voxel_size_y = (y_max - y_min) / H
    
#     cls_heatmap = torch.zeros((1, H, W))
#     reg_grid = torch.zeros((7, H, W))
    
#     for obj in targets:
#         gx_float = (obj[0] - x_min) / voxel_size_x
#         gy_float = (obj[1] - y_min) / voxel_size_y
        
#         gx, gy = int(gx_float), int(gy_float)
        
#         if 0 <= gx < W and 0 <= gy < H:
#             draw_gaussian(cls_heatmap[0], (gy, gx), radius)
            
#             # --- 1. Offsets (x, y) ---
#             pixel_center_x = (gx + 0.5) * voxel_size_x + x_min
#             pixel_center_y = (gy + 0.5) * voxel_size_y + y_min
#             reg_grid[0, gy, gx] = (obj[0] - pixel_center_x) / voxel_size_x # dx
#             reg_grid[1, gy, gx] = (obj[1] - pixel_center_y) / voxel_size_y # dy
            
#             # --- 2. Z : Normalisation simple par la range ---
#             reg_grid[2, gy, gx] = obj[2] 
            
#             # --- 3. Dimensions (w, l, h) : Log-transform ---
#             # Le log aide à gérer les ordres de grandeur (ex: piéton vs camion)
#             # On ajoute un epsilon pour éviter log(0)
#             reg_grid[3:6, gy, gx] = torch.log(obj[3:6] + 1e-6) 
            
#             # --- 4. Theta (Angle) : Sin/Cos Encoding ---
#             # Ne jamais régresser l'angle brut (0-2pi), car 0 et 2pi sont proches
#             # mais le modèle les voit comme opposés.
#             reg_grid[6, gy, gx] = obj[6] # Garde theta ici, mais regarde l'astuce ci-dessous
            
#     return cls_heatmap, reg_grid



# def create_heatmaps(targets, grid_size=(432, 496), pc_range=[0, -39.68, 69.12, 39.68], radius=5):
#     """
#     targets: tenseur (N, 7) [x, y, z, w, l, h, theta]
#     """
#     H, W = grid_size
#     x_min, y_min, x_max, y_max = pc_range
    
#     voxel_size_x = (x_max - x_min) / W
#     voxel_size_y = (y_max - y_min) / H
    
#     cls_heatmap = torch.zeros((1, H, W))
#     reg_grid = torch.zeros((7, H, W))
    
#     for obj in targets:
#         gx_float = (obj[0] - x_min) / voxel_size_x
#         gy_float = (obj[1] - y_min) / voxel_size_y
        
#         gx, gy = int(gx_float), int(gy_float)
        
#         if 0 <= gx < W and 0 <= gy < H:
#             draw_gaussian(cls_heatmap[0], (gy, gx), radius)
            
#             # --- 1. Offsets (x, y) ---
#             pixel_center_x = (gx + 0.5) * voxel_size_x + x_min
#             pixel_center_y = (gy + 0.5) * voxel_size_y + y_min
#             reg_grid[0, gy, gx] = (obj[0] - pixel_center_x) / voxel_size_x # dx
#             reg_grid[1, gy, gx] = (obj[1] - pixel_center_y) / voxel_size_y # dy
            
#             # --- 2. Z : Normalisation simple par la range ---
#             reg_grid[2, gy, gx] = obj[2] 
            
#             # --- 3. Dimensions (w, l, h) : Log-transform ---
#             # Le log aide à gérer les ordres de grandeur (ex: piéton vs camion)
#             # On ajoute un epsilon pour éviter log(0)
#             reg_grid[3:6, gy, gx] = torch.log(obj[3:6] + 1e-6) 
            
#             # --- 4. Theta (Angle) : Sin/Cos Encoding ---
#             # Ne jamais régresser l'angle brut (0-2pi), car 0 et 2pi sont proches
#             # mais le modèle les voit comme opposés.
#             reg_grid[6, gy, gx] = obj[6] # Garde theta ici, mais regarde l'astuce ci-dessous
            
#     return cls_heatmap, reg_grid


def create_heatmaps(targets, grid_size=(432, 496), pc_range=[0, -39.68, 69.12, 39.68], radius=5):
    H, W = grid_size
    x_min, y_min, x_max, y_max = pc_range
    voxel_size_x = (x_max - x_min) / W
    voxel_size_y = (y_max - y_min) / H
    
    cls_heatmap = torch.zeros((1, H, W))
    reg_grid = torch.zeros((7, H, W))
    
    for obj in targets:
        gx = int((obj[0] - x_min) / voxel_size_x)
        gy = int((obj[1] - y_min) / voxel_size_y)
        
        if 0 <= gx < W and 0 <= gy < H:
            draw_gaussian(cls_heatmap[0], (gy, gx), radius)
            
            # Offsets normalisés pour rester dans [-1, 1]
            reg_grid[0, gy, gx] = (obj[0] - ((gx + 0.5) * voxel_size_x + x_min)) / voxel_size_x
            reg_grid[1, gy, gx] = (obj[1] - ((gy + 0.5) * voxel_size_y + y_min)) / voxel_size_y
            
            # Z: Normalisation par la plage totale (ici 4m, de -3 à 1)
            reg_grid[2, gy, gx] = (obj[2] + 3.0) / 4.0 
            
            # Dimensions: Log transformé mais divisé par une valeur typique (ex: 2.0)
            reg_grid[3:6, gy, gx] = torch.log(obj[3:6] + 1e-6) / 2.0
            
            # Theta: inchangé, c'est déjà un angle
            reg_grid[6, gy, gx] = obj[6] 
            
    return cls_heatmap, reg_grid


def kitti_collate_fn(batch):
    """How to stack samples, images are already the same sizes so easy to stack but labels are lists of varying lengths"""

    inputs = torch.stack([item['input'] for item in batch])
    
    batch_cls = []
    batch_reg = []
    
    for item in batch:
        # On appelle la fonction pour chaque échantillon du batch
        # Assure-toi que item['target'] est bien le tenseur (N, 7)
        c_map, r_map = create_heatmaps(item['target'])
        batch_cls.append(c_map)
        batch_reg.append(r_map)
    
    # On stacke tout pour créer les tenseurs (Batch, C, H, W)
    cls_maps = torch.stack(batch_cls)
    reg_maps = torch.stack(batch_reg)
    
    return {
        "inputs": inputs, 
        "targets": {"cls": cls_maps, "reg": reg_maps}, 
        "ids": [item['id'] for item in batch]
    }

def validate(model, loader, criterion, device):
    model.eval() 
    val_loss = 0
    total_obs = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device).float()
            targets = {
            "cls": batch["targets"]["cls"].to(device).float(),
            "reg": batch["targets"]["reg"].to(device).float()
        }

            cls_logits, reg_preds = model(inputs)
            loss, n_obs, losses = criterion((cls_logits, reg_preds), targets)


            
            total_obs += n_obs
            batch_size = inputs.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size

    return val_loss / total_samples



def train_one_epoch(model, loader, optimizer, criterion, device, scaler=torch.amp.GradScaler('cuda')):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        inputs = batch["inputs"].to(device).float()
        
        # On définit les cibles
        reg_target = batch["targets"]["reg"].to(device).float()
        cls_target = batch["targets"]["cls"].to(device).float()
        targets = {"cls": cls_target, "reg": reg_target}
        
        optimizer.zero_grad()
        
        # Passage dans le modèle
        with torch.amp.autocast('cuda'):
            cls_logits, reg_preds = model(inputs)
            
            # Correction du debug pour matcher les dimensions
            if i % 20 == 0:
                # print(f"predictions: cls_logits {cls_logits}, reg_preds {reg_preds}")
                print(f"Stats cibles: Min={reg_target.min():.2f}, Max={reg_target.max():.2f}")
                # 1. On crée le masque spatial (batch, H, W)
                # On prend le premier canal de cls_target qui est (B, 1, H, W) -> on squeeze
                mask = (cls_target.squeeze(1) > 0.1) 
                
                # 2. On applique le masque sur reg_preds
                # reg_preds est (B, 7, H, W)
                # mask est (B, H, W)
                # On veut les valeurs où mask est True. 
                # .permute(1, 0, 2, 3) permet d'isoler les 7 canaux avant le masquage
                significant_preds = reg_preds.permute(1, 0, 2, 3)[:, mask]
                significant_targets = reg_target.permute(1, 0, 2, 3)[:, mask]
                
                if significant_preds.shape[1] > 0:
                    # On ne prend que les canaux 0 et 1 (dx, dy) pour le print
                    pred_mean = significant_preds[0:2].mean().item()
                    target_mean = significant_targets[0:2].mean().item()
                    
                    print(f"Step {i} | Pred Mean (dx,dy): {pred_mean:.4f} | Target Mean (dx,dy): {target_mean:.4f}")



            loss, n_obs, losses = criterion((cls_logits, reg_preds), targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)


def run_train(config_path="./src/late_fusion/LiDAR/config.yaml"):
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    mlflow_path = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_path.replace(os.sep, '/')}")
    
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration non trouvée : {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"{config['model_variant']}_{timestamp}"


    dataset = KittiPillarDataset(config_path= "./src/late_fusion/LiDAR/config.yaml", data_dir="./data/kitti_lidar", split='train')
    loader = DataLoader(dataset, batch_size=config['train_params']['batch'], collate_fn=kitti_collate_fn, shuffle=True, num_workers=8,pin_memory=True)

    val_dataset = KittiPillarDataset(config_path=config_path, data_dir="./data/kitti_lidar", split='val')
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch'], collate_fn=kitti_collate_fn, num_workers=8,pin_memory=True)
    
    model = PillarBackbone(in_channels=config['dataset']['num_channels'])
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = DetectionLoss() 
    
    best_val_loss = float('inf')
    #  MLflow Tracking
    mlflow.set_experiment(config['experiment_name'])
    save_path = Path("checkpoints")
    save_path.mkdir(exist_ok=True)
    
    clean_dataset_config = {k: (str(v) if isinstance(v, list) else v) for k, v in config['dataset'].items()}
    clean_train_params = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in config['train_params'].items()}

    with mlflow.start_run(run_name=run_id):
        print(f"debug params and dataset type for mlflow logging: config['dataset']={clean_dataset_config}, type={clean_train_params}")
        mlflow.log_params(clean_dataset_config)
        mlflow.log_params(clean_train_params)

        print(f"starting training on : {device}")
        
        for epoch in range(config['train_params']['epochs']):
            avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
            
            mlflow.log_metric("train_loss", float(avg_loss), step=epoch)
            print(f"Epoch {epoch+1}/{config['train_params']['epochs']} - Loss: {avg_loss:.4f}")
            
            avg_val_loss = validate(model, val_loader, criterion, device)
            mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
            print(f"Epoch {epoch+1}/{config['train_params']['epochs']} - Val Loss: {avg_val_loss:.4f}")
            print(f"debug metrics types: train_loss={type(avg_loss)}, val_loss={type(avg_val_loss)}")
            
            scheduler.step(avg_val_loss)
            
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        mlflow.pytorch.log_model(model, "pillar_backbone_best_model")
        print(f"training done, saved in: {run_id}")
        
    torch.save(model.state_dict(), save_path / model_name)
if __name__ == "__main__":
    try:
            run_train()
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")