import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Pour la géométrie (x, y, z, l, w, h, yaw)
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
        # Pour la présence d'objet (Objectness)
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, preds, targets_list):
        """
        preds: (B, 8, H, W) -> Canal 0: Cls, Canaux 1-7: Reg
        targets_list: Liste de tenseurs [N, 7]
        """
        device = preds.device
        B, _, H, W = preds.shape
        
        # 1. Séparation des prédictions
        cls_preds = preds[:, 0, :, :]       # (B, H, W)
        reg_preds = preds[:, 1:, :, :]      # (B, 7, H, W)

        # 2. Création de la Target Map et du Masque (Comme avant)
        target_map = torch.zeros_like(reg_preds).to(device)
        mask = torch.zeros((B, H, W)).to(device)

        for b in range(B):
            for t in targets_list[b]:
                # On projette x,y sur la grille (adapter l'échelle 80m si besoin)
                grid_x = int(torch.clamp(t[0] * (W / 80.0), 0, W - 1))
                grid_y = int(torch.clamp(t[1] * (H / 80.0), 0, H - 1))
                
                target_map[b, :, grid_y, grid_x] = t
                mask[b, grid_y, grid_x] = 1.0

        # --- BRANCH 1: CLASSIFICATION ---
        # On apprend au modèle à prédire 1 sur le mask, 0 ailleurs
        # C'est cette loss qui va supprimer tes "cubes rouges" partout
        cls_loss = self.cls_loss_fn(cls_preds, mask)

        # --- BRANCH 2: REGRESSION ---
        # On ne calcule l'erreur de position/taille QUE là où il y a un objet
        num_objects = mask.sum()
        if num_objects > 0:
            # On étend le masque pour couvrir les 7 canaux de régression
            reg_mask = mask.unsqueeze(1).expand_as(reg_preds)
            
            # Calcul de la SmoothL1 uniquement sur les objets
            reg_loss_raw = self.reg_loss_fn(reg_preds, target_map)
            reg_loss = (reg_loss_raw * reg_mask).sum() / (num_objects + 1e-6)
        else:
            reg_loss = torch.tensor(0.0).to(device)

        # --- FUSION DES LOSSES ---
        # On peut pondérer : on donne souvent plus de poids à la régression (ex: 2.0)
        total_loss = cls_loss + 2.0 * reg_loss
        
        return total_loss, num_objects.item()