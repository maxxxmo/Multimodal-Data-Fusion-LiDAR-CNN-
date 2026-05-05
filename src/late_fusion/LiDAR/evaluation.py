import torch
from torchvision.ops import nms

def validate(model, loader, criterion, device, precision_metric, recall_metric,thresold=0.35):

    model.eval()
    val_loss = 0
    total_samples = 0
    
    # Reset des métriques pour chaque validation
    precision_metric.reset()
    recall_metric.reset()
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device).float()
            cls_target = batch["targets"]["cls"].to(device).float()
            reg_target = batch["targets"]["reg"].to(device).float()
            pos_mask = batch["pos_mask"].to(device).bool()
            
            # --- 1. Remise en forme des tenseurs ---
            # Correspond exactement à ce que tu fais dans le train_one_epoch
            B, H, W, N_a1, N_a2 = cls_target.shape
            cls_target = cls_target.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, H, W)
            pos_mask = pos_mask.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, H, W)
            reg_target = reg_target.permute(0, 3, 4, 5, 1, 2).contiguous().view(B, -1, H, W)
            
            # --- 2. Inférence ---
            cls_logits, reg_preds = model(inputs)
            loss = criterion(cls_logits, reg_preds, cls_target, reg_target, pos_mask)
            
            # --- 3. Calcul des métriques ---
            # On applique la sigmoid pour obtenir des probabilités [0, 1]
            probs = torch.sigmoid(cls_logits)

            preds = (probs > thresold)

            tp = ((preds == 1) & (cls_target == 1)).sum()
            fp = ((preds == 1) & (cls_target == 0)).sum()
            fn = ((preds == 0) & (cls_target == 1)).sum()

            print(f"tp{tp}, fp{fp}, fn {fn}")
            
  

            precision_metric(probs, cls_target.long())
            recall_metric(probs, cls_target.long())
            
            # --- 4. Accumulation de la loss ---
            batch_size = inputs.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # --- 5. Logging debug ---
            # if total_samples % (len(loader) * batch_size // 2 + 1) == 0:
            print(f"DEBUG: Confiance Max={probs.max():.4f}, Moy={probs.mean():.4f}")
    
    # Compute final metrics
    return val_loss / total_samples, precision_metric.compute(), recall_metric.compute()