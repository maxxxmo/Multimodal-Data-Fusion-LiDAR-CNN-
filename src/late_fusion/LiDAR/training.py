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
from src.late_fusion.LiDAR.model.pillarbackbone3 import PillarBackbone
from src.late_fusion.LiDAR.model.anchorloss import AnchorDetectionLoss as DetectionLoss
from src.late_fusion.LiDAR.evaluation import validate

model_name ="best_Lidar_model.pth"



threshold = 0.5
threshold_val = 0.5

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_reg_loss = 0.0
    for i, batch in enumerate(loader):
        # print(batch["targets"]["reg"].shape) # Doit être [B, H, W, 1, 2, 8]
        inputs = batch["inputs"].to(device).float()
        cls_target = batch["targets"]["cls"].to(device).float()
        reg_target = batch["targets"]["reg"].to(device).float()
        pos_mask = batch["pos_mask"].to(device).bool() # Le mask doit être un booléen
        
        optimizer.zero_grad()
        
        # 1. cls_target: [B, H, W, 1, 2] -> [B, 1*2, H, W] -> [B, 2, H, W]
        # On flatten la dimension 3 et 4 (les 1 et 2)
        # B, H, W, N_a1, N_a2 = cls_target.shape
        B, H, W, N_a1, N_a2, dim_reg = reg_target.shape
        # print("reg_target shape BEFORE view:", reg_target.shape)
        # print("numel:", reg_target.numel())
        # print("expected:", B * N_a1 * N_a2 * dim_reg * H * W)
        # print("view target:", B, N_a1 * N_a2 * dim_reg, H, W)
        # print("view numel:", B * (N_a1 * N_a2 * dim_reg) * H * W)
        
        cls_target = cls_target.permute(0, 3, 4, 1, 2).contiguous().view(B, N_a1 * N_a2, H, W)
        pos_mask = pos_mask.permute(0, 3, 4, 1, 2).contiguous().view(B, N_a1 * N_a2, H, W)
        
        # 2. reg_target: [B, H, W, 1, 2, 7] -> [B, 1*2*7, H, W] -> [B, 14, H, W]
        # reg_target = reg_target.permute(0, 3, 4, 5, 1, 2).contiguous().view(B, N_a1 * N_a2 * 8, H, W)
        reg_target = reg_target.permute(0, 3, 4, 5, 1, 2).contiguous()
        reg_target = reg_target.view(B, N_a1 * N_a2 * dim_reg, H, W)
        
        
        
        
        # 1. Récupère les dimensions réelles de la cible
    
        
        # 1. Utilisation du Mixed Precision (autocast)
        with torch.amp.autocast('cuda'):
            cls_logits, reg_preds = model(inputs)
            # print("cls_logits", cls_logits.shape)
            # print("cls_target", cls_target.shape)

            # print("reg_preds", reg_preds.shape)
            # print("reg_target", reg_target.shape)

            # print("pos_mask", pos_mask.shape)
            # print("one anchor",cls_target[0, :, 10, 10])
            
            loss= criterion(cls_logits, reg_preds, cls_target, reg_target,pos_mask)
            current_cls_loss, current_reg_loss = criterion.get_losses()
        # print(f"cls {cls},reg {reg}")
        # 2. Scaler pour éviter les underflows de gradient (Float16)
        scaler.scale(loss).backward()
        
        # 3. Gradient Clipping : Crucial pour les modèles 3D pour éviter l'explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Cumul de la perte pour les logs
        epoch_loss += loss.item()
        epoch_cls_loss += current_cls_loss
        epoch_reg_loss += current_reg_loss
        # Logging léger pour le débug
        if i % 50 == 0:
            total_elements = pos_mask.numel()
            pos_count = pos_mask.sum().item()
            pos_ratio = pos_count / total_elements
            
            # On vérifie aussi la moyenne des logits pour voir si le modèle est "confiant"
            with torch.no_grad():
                mean_logit = cls_logits.mean().item()
            
            print(f"Batch {i} | Loss: {loss.item():.4f} | Pos: {pos_count} ({pos_ratio:.6%}) | Mean Logit: {mean_logit:.4f}")

        # Logging léger pour le débug
        if i % 50 == 0:
            print(f"Batch {i} | Loss: {loss.item():.4f} | Pos Anchors: {pos_mask.sum().item()}")
            print(f"(Cls: {current_cls_loss:.4f}, Reg: {current_reg_loss:.6f})")
        n = len(loader)
    return epoch_loss / n, epoch_cls_loss/n , epoch_reg_loss/n

def run_train(config_path="./src/late_fusion/LiDAR/config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    mlflow_path = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_path.replace(os.sep, '/')}")
    
    precision_metric = BinaryPrecision(threshold=threshold).to(device)
    recall_metric = BinaryRecall(threshold=threshold).to(device)      

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
    anchor_sizes = config_dataset['anchor_sizes']
    anchor_rotations = config_dataset['anchor_rotations']
    anchor_gen = AnchorGenerator(
        feature_map_size=(H, W), 
        anchor_sizes=anchor_sizes,
        anchor_rotations= anchor_rotations,
        pc_range=pc_range
    )
    print(f"anchor_gen.anchors.shape:{anchor_gen.anchors.shape}")
    print(f"anchor_gen.anchors.numel:{anchor_gen.anchors.numel()}")
    
    target_assigner = TargetAssigner(iou_thresholds=(0.3, 0.4))
    

    dataset = KittiPillarDataset( split='train',anchor_gen=anchor_gen,target_assigner=target_assigner)
    loader = DataLoader(dataset, batch_size=config['train_params']['batch'], shuffle=True, num_workers=0,pin_memory=False)

    val_dataset = KittiPillarDataset(split='val',anchor_gen=anchor_gen,target_assigner=target_assigner)
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch'], num_workers=0,pin_memory=False)
    
    total_anchors = len(anchor_sizes) * len(anchor_rotations) 
    
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
            

            print('train')
            avg_loss, avg_cls_loss, avg_reg_loss = train_one_epoch(model, loader, optimizer, criterion, device, scaler)
            mlflow.log_metric("train_loss", float(avg_loss), step=epoch)
            print('val')
            avg_val_loss, val_prec, val_recall = validate(model, val_loader, criterion, device, precision_metric, recall_metric,thresold=threshold_val)
            mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
            mlflow.log_metric("val_precision", float(val_prec), step=epoch)
            mlflow.log_metric("val_recall", float(val_recall), step=epoch)
            mlflow.log_metric("avg_cls_loss", float(avg_cls_loss), step=epoch)
            mlflow.log_metric("avg_reg_loss", float(avg_reg_loss), step=epoch)
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