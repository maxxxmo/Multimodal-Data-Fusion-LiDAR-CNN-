import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.8, gamma=2.0):
    mask = (targets >= 0).float()
    targets_clamped = targets.clone()
    targets_clamped[targets < 0] = 0
    
    pos_mask = (targets == 1).float()
    n_pos = pos_mask.sum()
    
    bce = F.binary_cross_entropy_with_logits(logits, targets_clamped, reduction='none')
    
    probs = torch.sigmoid(logits)
    p_t = probs * targets_clamped + (1 - probs) * (1 - targets_clamped)
    
    alpha_factor = targets_clamped * alpha + (1 - targets_clamped) * (1 - alpha)
    focal_weight = alpha_factor * (1 - p_t)**gamma
    
    loss = focal_weight * bce
    
    # MODIFICATION CRUCIALE :
    # On isole la perte des positifs et la perte des négatifs
    loss_pos = (loss * pos_mask).sum() / (n_pos + 1.0)
    loss_neg = (loss * (1 - pos_mask) * mask).sum() / (mask.sum() - n_pos + 1.0)
    
    # Maintenant, tu peux pondérer le déséquilibre manuellement
    return loss_pos + 0.1 * loss_neg # On réduit le poids du fond de 90% !



class AnchorDetectionLoss(nn.Module):
    def __init__(self, num_anchors, cls_weight=2.0, reg_weight=1.0): 
        super().__init__()
        self.num_anchors = num_anchors  # Sauvegarde-le
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, preds_cls, preds_reg, targets_cls, targets_reg, pos_mask):
        # 1. Classification
        cls_loss = focal_loss(preds_cls, targets_cls)
        
        # 2. Régression
        n_pos = pos_mask.sum()
        if n_pos > 0:
            # On utilise self.num_anchors ici
            reg_mask = pos_mask.unsqueeze(2).repeat(1, 1, 7, 1, 1).view(preds_reg.shape)
            preds_reg_pos = preds_reg[reg_mask].view(-1, 7)
            targets_reg_pos = targets_reg[reg_mask].view(-1, 7)
            preds_reg_pos = torch.clamp(preds_reg_pos, min=-2.0, max=2.0)
            reg_loss = self.smooth_l1(preds_reg_pos, targets_reg_pos).mean()
        else:
            reg_loss = torch.tensor(0.0, device=preds_reg.device)
            
        return (self.cls_weight * cls_loss) + (self.reg_weight * reg_loss)



