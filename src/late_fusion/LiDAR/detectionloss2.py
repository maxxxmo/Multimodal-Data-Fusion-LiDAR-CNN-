import torch
import torch.nn as nn


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    modulating_factor = (1 - p_t)**gamma
    
    loss = alpha_t * modulating_factor * bce
    
    # AU LIEU DE .mean(), on normalise uniquement par le nombre de positifs
    num_pos = (targets > 0.1).float().sum() + 1e-6
    return loss.sum() / num_pos

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # On garde SmoothL1, c'est très bien.
        
    def forward(self, preds, targets):
        cls_logits, reg_preds = preds
        cls_target = targets['cls']  # (B, 1, H, W)
        reg_target = targets['reg']  # (B, 7, H, W)

        # 1. Classification
        cls_loss = focal_loss(cls_logits, cls_target)

        # 2. Régression Masquée
        # Utilise un seuil binaire pour le masque, pas la gaussienne floue
        mask = (cls_target > 0.1).float() 
        
        reg_loss_map = nn.functional.smooth_l1_loss(reg_preds, reg_target, reduction='none')
        
        # On normalise par le nombre d'objets (nombre de pixels > seuil)
        num_pos = mask.sum() + 1e-6
        reg_loss = (reg_loss_map * mask).sum() / num_pos
        
        return cls_loss + reg_loss, num_pos.item()