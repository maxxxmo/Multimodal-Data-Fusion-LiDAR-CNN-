import torch

def validate(model, loader, criterion, device, precision_metric, recall_metric):
    model.eval()
    val_loss = 0
    total_samples = 0
    precision_metric.reset()
    recall_metric.reset()
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device).float()
            
            
            
            cls_target = batch["targets"]["cls"].to(device).float()
            cls_target[cls_target == -1] = 1
            reg_target = batch["targets"]["reg"].to(device).float()
            pos_mask = batch["pos_mask"].to(device).bool()
            
            cls_logits, reg_preds = model(inputs)
            loss = criterion(cls_logits, reg_preds, cls_target, reg_target,pos_mask)
            
            # Calcul des métriques
            probs = torch.sigmoid(cls_logits)
            print(f"Proba Max: {probs.max().item():.4f} | Proba Moyenne: {probs.mean().item():.6f}")
            precision_metric(probs, cls_target)
            recall_metric(probs, cls_target)
            
            batch_size = inputs.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            
    # Calcul des scores finaux
    final_prec = precision_metric.compute()
    final_recall = recall_metric.compute()
    
    return val_loss / total_samples, final_prec, final_recall