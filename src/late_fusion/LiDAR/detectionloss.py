import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # SmoothL1 est beaucoup plus stable que MSE pour la 3D
        self.reg_loss = nn.SmoothL1Loss(reduction='none') 

    def forward(self, preds, targets_list):
        device = preds.device
        B, C, H, W = preds.shape
        
        # 1. On prépare la Target Map et le Masque
        target_map = torch.zeros_like(preds).to(device)
        mask = torch.zeros((B, 1, H, W)).to(device)

        for b in range(B):
            targets = targets_list[b]
            for t in targets:
                # Projection (Assure-toi que ces échelles correspondent à ton grid)
                grid_x = int(torch.clamp(t[0] * (W / 80.0), 0, W - 1)) 
                grid_y = int(torch.clamp(t[1] * (H / 80.0), 0, H - 1))
                
                target_map[b, :, grid_y, grid_x] = t
                mask[b, 0, grid_y, grid_x] = 1.0

        # 2. Calcul de la Loss de régression
        # On calcule la loss partout, puis on masque
        raw_loss = self.reg_loss(preds, target_map)
        
        num_objects = mask.sum()
        if num_objects > 0:
            # On ne garde que les pixels où il y a un objet
            # On fait la somme de tous les canaux (7) pour ces pixels
            masked_loss = (raw_loss * mask).sum() 
            # Normalisation par le nombre d'objets pour un gradient stable
            return masked_loss / (num_objects + 1e-6), num_objects.item()
        else:
            # Si pas d'objets, on réduit juste les prédictions vers 0
            return raw_loss.mean() * 0.1, 0.0