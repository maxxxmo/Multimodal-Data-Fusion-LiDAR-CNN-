import os
import yaml
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader
from pathlib import Path


from src.late_fusion.utils.pillar_dataset import KittiPillarDataset
from src.late_fusion.LiDAR.pillarbackbone2 import PillarBackbone
from src.late_fusion.LiDAR.detectionloss2 import DetectionLoss

model_name ="best_Lidar_model.pth"

def kitti_collate_fn(batch):
    """How to stack samples, images are already the same sizes so easy to stack but labels are lists of varying lengths
    Dataloaders in PyTorch require a collate function to specify how to combine individual samples into a batch.
    Thats wath this function is for"""

    inputs = torch.stack([item['input'] for item in batch]) # images [Batch_size, Channels (2), Height, Width]
    targets = [item['target'] for item in batch] # Liste of tensors with varying shapes [num_objects, 7] (x,y,z,l,w,h,yaw)
    ids = [item['id'] for item in batch] # List of strings (file ids)
    return {"inputs": inputs, "targets": targets, "ids": ids}

def validate(model, loader, criterion, device):
    model.eval() 
    val_loss = 0
    total_obs = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device).float()
            targets = [t.to(device).float() for t in batch["targets"]]

            preds = model(inputs)
            loss, n_obs = criterion(preds, targets)
            
            val_loss += loss.item()
            total_obs += n_obs
            

    return val_loss / len(loader)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        inputs = batch["inputs"].to(device).float()
        targets = [t.to(device).float() for t in batch["targets"]]

        optimizer.zero_grad()
        preds = model(inputs)

        # loss est le total (cls + reg), n_obs est le nombre d'objets réels
        loss, n_obs = criterion(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if i % 10 == 0:
            # On affiche aussi le nombre d'objets pour vérifier que le loader fonctionne
            print(f"Batch {i}/{len(loader)} | Loss: {loss.item():.4f} | Objs: {int(n_obs)}")
        
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


    dataset = KittiPillarDataset(config_path= "./src/late_fusion/LiDAR/config.yaml", data_dir="./data/kitti_lidar")
    loader = DataLoader(dataset, batch_size=config['train_params']['batch'], collate_fn=kitti_collate_fn, shuffle=True, num_workers=8)

    val_dataset = KittiPillarDataset(config_path=config_path, data_dir="./data/kitti_lidar", split='val')
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch'], collate_fn=kitti_collate_fn, shuffle=False, num_workers=4)
    
    model = PillarBackbone(in_channels=config['dataset']['num_channels'])
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
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