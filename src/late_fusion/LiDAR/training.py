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
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from src.late_fusion.LiDAR.anchors import AnchorGenerator, TargetAssigner, encode_targets
from src.late_fusion.utils.pillar_dataset import KittiPillarDataset
from src.late_fusion.LiDAR.pillarbackbone3 import PillarBackbone
from src.late_fusion.LiDAR.anchorloss import AnchorDetectionLoss as DetectionLoss

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

def validate(model, loader, criterion, device, precision_metric, recall_metric):
    model.eval()
    val_loss = 0
    total_samples = 0
    
    # Reset des métriques pour cette validation
    precision_metric.reset()
    recall_metric.reset()
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device).float()
            cls_target = batch["targets"]["cls"].to(device).float()
            cls_target[cls_target == -1] = 0

            reg_target = batch["targets"]["reg"].to(device).float()
            pos_mask = batch["pos_mask"].to(device).bool()
            
            cls_logits, reg_preds = model(inputs)
            loss = criterion(cls_logits, reg_preds, cls_target, reg_target, pos_mask)
            
            # Calcul des métriques
            probs = torch.sigmoid(cls_logits)
            precision_metric(probs, cls_target)
            recall_metric(probs, cls_target)
            
            batch_size = inputs.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            
    # Calcul des scores finaux
    final_prec = precision_metric.compute()
    final_recall = recall_metric.compute()
    
    return val_loss / total_samples, final_prec, final_recall

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0.0
    
    # On itère avec un tqdm pour suivre la progression en temps réel
    for i, batch in enumerate(loader):
        inputs = batch["inputs"].to(device).float()
        cls_target = batch["targets"]["cls"].to(device).float()
        reg_target = batch["targets"]["reg"].to(device).float()
        pos_mask = batch["pos_mask"].to(device).bool() # Le mask doit être un booléen
        
        optimizer.zero_grad()
        
        # 1. Utilisation du Mixed Precision (autocast)
        with torch.amp.autocast('cuda'):
            cls_logits, reg_preds = model(inputs)
            loss = criterion(cls_logits, reg_preds, cls_target, reg_target, pos_mask)
        
        # 2. Scaler pour éviter les underflows de gradient (Float16)
        scaler.scale(loss).backward()
        
        # 3. Gradient Clipping : Crucial pour les modèles 3D pour éviter l'explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Cumul de la perte pour les logs
        epoch_loss += loss.item()
        
        # Logging léger pour le débug
        if i % 50 == 0:
            print(f"Batch {i} | Loss: {loss.item():.4f} | Pos Anchors: {pos_mask.sum().item()}")

    return epoch_loss / len(loader)

def run_train(config_path="./src/late_fusion/LiDAR/config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    mlflow_path = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_path.replace(os.sep, '/')}")
    
    precision_metric = BinaryPrecision(threshold=0.8).to(device)
    recall_metric = BinaryRecall(threshold=0.8).to(device)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration non trouvée : {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"{config['model_variant']}_{timestamp}"

    
    config_dataset = config['dataset']

    # Extraction des dimensions
    # Note: config['dataset']['grid_size'] est [432, 496] -> (H, W)
    H, W = config_dataset['grid_size']
    pc_range = config_dataset['pc_range'] # [0, -39.68, -3, 69.12, 39.68, 1]


    anchor_gen = AnchorGenerator(
        feature_map_size=(H, W), 
        # anchor_sizes=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73]],  # TO DO , put in the YAML config
        # anchor_rotations=[0, np.pi/2],  # TO DO , put in the YAML config
        anchor_sizes=[[3.9, 1.6, 1.56]] , # TO DO , put in the YAML config
        anchor_rotations=[0], # TO DO , put in the YAML config
        pc_range=pc_range
    )
    target_assigner = TargetAssigner(iou_thresholds=(0.6, 0.8))
    

    dataset = KittiPillarDataset(config_path= "./src/late_fusion/LiDAR/config.yaml", data_dir="./data/kitti_lidar", split='train',anchor_gen=anchor_gen,target_assigner=target_assigner)
    loader = DataLoader(dataset, batch_size=config['train_params']['batch'], shuffle=True, num_workers=0,pin_memory=False)

    val_dataset = KittiPillarDataset(config_path=config_path, data_dir="./data/kitti_lidar", split='val',anchor_gen=anchor_gen,target_assigner=target_assigner)
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch'], num_workers=0,pin_memory=False)
    
    num_types = len([[3.9, 1.6, 1.56]])       
    num_rots = len([0])
    total_anchors = num_types * num_rots 
    
    model = PillarBackbone(in_channels=config['dataset']['num_channels'], num_anchors=total_anchors)
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = DetectionLoss(num_anchors=total_anchors) 
    
    best_val_loss = float('inf')
    #  MLflow Tracking
    mlflow.set_experiment(config['experiment_name'])
    save_path = Path("checkpoints")
    save_path.mkdir(exist_ok=True)
    
    clean_dataset_config = {k: (str(v) if isinstance(v, list) else v) for k, v in config['dataset'].items()}
    clean_train_params = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in config['train_params'].items()}

    # 1. Instancie le scaler avant la boucle
    scaler = torch.amp.GradScaler('cuda')

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(clean_dataset_config)
        mlflow.log_params(clean_train_params)
        
        for epoch in range(config['train_params']['epochs']):
            # 2. Passe le scaler ici
            avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, scaler)
            mlflow.log_metric("train_loss", float(avg_loss), step=epoch)
            
            # avg_val_loss = validate(model, val_loader, criterion, device)
            avg_val_loss, val_prec, val_recall = validate(model, val_loader, criterion, device, precision_metric, recall_metric)
            mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
            mlflow.log_metric("val_precision", float(val_prec), step=epoch)
            mlflow.log_metric("val_recall", float(val_recall), step=epoch)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Prec: {val_prec:.4f} | Rec: {val_recall:.4f}")            
            scheduler.step(avg_val_loss)
            
            # 3. Sauvegarde uniquement si c'est le meilleur modèle
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Sauvegarde locale
                torch.save(model.state_dict(), save_path / model_name)
                # Enregistrement MLflow
                mlflow.pytorch.log_model(model, "pillar_backbone_best_model")
                print(">>> Meilleur modèle sauvegardé !")
                
                
        print(f"training done, saved in: {run_id}")
        
    torch.save(model.state_dict(), save_path / model_name)
    
    
            
if __name__ == "__main__":
    try:
            run_train()
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")