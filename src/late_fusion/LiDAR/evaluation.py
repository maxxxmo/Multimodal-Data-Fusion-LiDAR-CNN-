import torch

def validate(model, loader, criterion, device, precision_metric, recall_metric,thresold=0.35):

    model.eval()
    val_loss = 0
    total_samples = 0
    precision_metric.reset()
    recall_metric.reset()
    
    with torch.no_grad():
        for batch in loader:
            # inputs & outputs
            inputs = batch["inputs"].to(device).float()
            cls_target = batch["targets"]["cls"].to(device).float()
            reg_target = batch["targets"]["reg"].to(device).float()
            pos_mask = batch["pos_mask"].to(device).bool()
            # Shaping
            B, H, W, N_a1, N_a2 = cls_target.shape
            cls_target = cls_target.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, H, W)
            pos_mask = pos_mask.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, H, W)
            reg_target = reg_target.permute(0, 3, 4, 5, 1, 2).contiguous().view(B, -1, H, W)
            
            # Inference
            cls_logits, reg_preds = model(inputs)
            loss = criterion(cls_logits, reg_preds, cls_target, reg_target, pos_mask)
            probs = torch.sigmoid(cls_logits)

            # metrics
            precision_metric(probs, cls_target.long())
            recall_metric(probs, cls_target.long())
            
            # loss
            batch_size = inputs.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # print(f"DEBUG: Confiance Max={probs.max():.4f}, Moy={probs.mean():.4f}")
    return val_loss / total_samples, precision_metric.compute(), recall_metric.compute()