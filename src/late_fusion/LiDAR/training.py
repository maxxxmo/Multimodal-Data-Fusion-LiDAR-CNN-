import os
import yaml
import datetime
import torch
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from torchmetrics.classification import BinaryPrecision, BinaryRecall

from src.late_fusion.LiDAR.anchors import AnchorGenerator, TargetAssigner
from src.late_fusion.utils.pillar_dataset import KittiPillarDataset
from late_fusion.LiDAR.model.pillarbackbone3 import PillarBackbone
from late_fusion.LiDAR.model.anchorloss import AnchorDetectionLoss as DetectionLoss
from src.late_fusion.LiDAR.evaluation import validate

model_name ="best_Lidar_model.pth"





def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0.0
    
    for i, batch in enumerate(loader):
        inputs = batch["inputs"].to(device).float()
        cls_target = batch["targets"]["cls"].to(device).float()
        reg_target = batch["targets"]["reg"].to(device).float()
        pos_mask = batch["pos_mask"].to(device).bool() # Le mask doit être un booléen
        
        optimizer.zero_grad()
        
        # 1. Utilisation du Mixed Precision (autocast)
        with torch.amp.autocast('cuda'):
            cls_logits, reg_preds = model(inputs)
            loss = criterion(cls_logits, reg_preds, cls_target, reg_target,pos_mask)
        
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
    
    precision_metric = BinaryPrecision(threshold=0.3).to(device)
    recall_metric = BinaryRecall(threshold=0.3).to(device)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration non trouvée : {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_id = f"{config['model_variant']}_{timestamp}"

    
    config_dataset = config['dataset']

    # Extraction des dimensions

    H, W = config_dataset['grid_size']
    pc_range = config_dataset['pc_range'] # [0, -39.68, -3, 69.12, 39.68, 1]
    anchor_sizes = config_dataset['dataset']['anchor_sizes']
    anchor_rotations = config_dataset['dataset']['anchor_rotations']
    anchor_gen = AnchorGenerator(
        feature_map_size=(H, W), 
        anchor_sizes=anchor_sizes,
        anchor_rotations= anchor_rotations,
        pc_range=pc_range
    )
    target_assigner = TargetAssigner(iou_thresholds=(0.3, 0.5))
    

    dataset = KittiPillarDataset(config_path= "./src/late_fusion/LiDAR/config.yaml", data_dir="./data/kitti_lidar", split='train',anchor_gen=anchor_gen,target_assigner=target_assigner)
    loader = DataLoader(dataset, batch_size=config['train_params']['batch'], shuffle=True, num_workers=2,prefetch_factor=1,persistent_workers=True,pin_memory=False)

    val_dataset = KittiPillarDataset(config_path=config_path, data_dir="./data/kitti_lidar", split='val',anchor_gen=anchor_gen,target_assigner=target_assigner)
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch'], num_workers=2,prefetch_factor=1,persistent_workers=True,pin_memory=False)
    
    total_anchors = len(anchor_sizes) * (anchor_rotations) 
    
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
    scaler = torch.amp.GradScaler('cuda')

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(clean_dataset_config)
        mlflow.log_params(clean_train_params)
        
        for epoch in range(config['train_params']['epochs']):
            
            avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, scaler)
            mlflow.log_metric("train_loss", float(avg_loss), step=epoch)
            
            avg_val_loss, val_prec, val_recall = validate(model, val_loader, criterion, device, precision_metric, recall_metric)
            mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
            mlflow.log_metric("val_precision", float(val_prec), step=epoch)
            mlflow.log_metric("val_recall", float(val_recall), step=epoch)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Prec: {val_prec:.4f} | Rec: {val_recall:.4f}")  
                      
            scheduler.step(avg_val_loss)
            
            # save if best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path / model_name)
                mlflow.pytorch.log_model(model, "pillar_backbone_best_model")
                print(">>> Best model saved !")
                
                
        print(f"training done, saved in: {run_id}")
        
    torch.save(model.state_dict(), save_path / model_name)
    
    
            
if __name__ == "__main__":
    try:
            run_train()
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")