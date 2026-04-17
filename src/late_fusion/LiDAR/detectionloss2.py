# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class DetectionLoss(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         # Pour la géométrie (x, y, z, l, w, h, yaw)
# #         self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
# #         # Pour la présence d'objet (Objectness)
# #         self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

# #     def forward(self, preds, targets_list):
# #         """
# #         preds: (B, 8, H, W) -> Canal 0: Cls, Canaux 1-7: Reg
# #         targets_list: Liste de tenseurs [N, 7]
# #         """
# #         device = preds.device
# #         B, _, H, W = preds.shape
        
# #         # 1. Séparation des prédictions
# #         cls_preds = preds[:, 0, :, :]       # (B, H, W)
# #         reg_preds = preds[:, 1:, :, :]      # (B, 7, H, W)

# #         # 2. Création de la Target Map et du Masque (Comme avant)
# #         target_map = torch.zeros_like(reg_preds).to(device)
# #         mask = torch.zeros((B, H, W)).to(device)

# #         for b in range(B):
# #             for t in targets_list[b]:
# #                 # On projette x,y sur la grille (adapter l'échelle 80m si besoin)
# #                 grid_x = int(torch.clamp(t[0] * (W / 80.0), 0, W - 1))
# #                 grid_y = int(torch.clamp(t[1] * (H / 80.0), 0, H - 1))
                
# #                 target_map[b, :, grid_y, grid_x] = t
# #                 mask[b, grid_y, grid_x] = 1.0

# #         # --- BRANCH 1: CLASSIFICATION ---
# #         # On apprend au modèle à prédire 1 sur le mask, 0 ailleurs
# #         # C'est cette loss qui va supprimer tes "cubes rouges" partout
# #         cls_loss = self.cls_loss_fn(cls_preds, mask)

# #         # --- BRANCH 2: REGRESSION ---
# #         # On ne calcule l'erreur de position/taille QUE là où il y a un objet
# #         num_objects = mask.sum()
# #         if num_objects > 0:
# #             # On étend le masque pour couvrir les 7 canaux de régression
# #             reg_mask = mask.unsqueeze(1).expand_as(reg_preds)
            
# #             # Calcul de la SmoothL1 uniquement sur les objets
# #             reg_loss_raw = self.reg_loss_fn(reg_preds, target_map)
# #             reg_loss = (reg_loss_raw * reg_mask).sum() / (num_objects + 1e-6)
# #         else:
# #             reg_loss = torch.tensor(0.0).to(device)

# #         # --- FUSION DES LOSSES ---
# #         # On peut pondérer : on donne souvent plus de poids à la régression (ex: 2.0)
# #         total_loss = cls_loss + 2.0 * reg_loss
        
# #         return total_loss, num_objects.item()

# import torch
# import torch.nn as nn

# class DetectionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
#         # On utilise une pondération pour aider le modèle à voir les objets
#         self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, preds, targets_list):
#         device = preds.device
#         B, _, H, W = preds.shape
        
#         cls_preds = preds[:, 0, :, :] 
#         reg_preds = preds[:, 1:, :, :]

#         target_map = torch.zeros_like(reg_preds).to(device)
#         mask = torch.zeros((B, H, W)).to(device)

#         # CONFIGURATION - Doit matcher ton pc_range [0, -40, -3, 70, 40, 1]
#         x_min, y_min, x_max, y_max = 0.0, -40.0, 70.0, 40.0 

#         for b in range(B):
#             for t in targets_list[b]:
#                 # 1. Mapping CORRECT (Mètres vers Pixels)
#                 norm_x = (t[0] - x_min) / (x_max - x_min)
#                 norm_y = (t[1] - y_min) / (y_max - y_min)
                
#                 gx = int(torch.clamp(norm_x * W, 0, W - 1))
#                 gy = int(torch.clamp(norm_y * H, 0, H - 1))
#                 pixel_center_x = (gx + 0.5) / W
#                 pixel_center_y = (gy + 0.5) / H
                
#                 t_encoded = t.clone()
#                 t_encoded[0] = (norm_x - pixel_center_x) * W # Offset X
#                 t_encoded[1] = (norm_y - pixel_center_y) * H # Offset Y
#                 target_map[b, :, gy, gx] = t_encoded
                
#                 mask[b, gy, gx] = 1.0

#         # --- CLASSIFICATION AVEC PONDÉRATION (Positives vs Negatives) ---
#         raw_cls_loss = self.cls_loss_fn(cls_preds, mask)
#         # On donne 10x plus d'importance aux pixels avec une voiture
#         pos_weight = 10.0
#         weight_mask = torch.where(mask > 0, pos_weight, 1.0)
#         cls_loss = (raw_cls_loss * weight_mask).mean()

#         # --- REGRESSION ---
#         num_objects = mask.sum()
#         if num_objects > 0:
#             reg_mask = mask.unsqueeze(1).expand_as(reg_preds)
#             reg_loss_raw = self.reg_loss_fn(reg_preds, target_map)
#             reg_loss = (reg_loss_raw * reg_mask).sum() / (num_objects + 1e-6)
#         else:
#             reg_loss = torch.tensor(0.0).to(device)

#         # Équilibre Cls / Reg
#         total_loss = cls_loss + 1.0 * reg_loss
        
#         return total_loss, num_objects.item()

import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self, pos_weight=30.0, reg_coef=2.0):
        super().__init__()
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
        # BCEWithLogitsLoss est idéale ici, elle inclut le Sigmoid interne
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.pos_weight = pos_weight
        self.reg_coef = reg_coef

    def forward(self, preds, targets_list):
        """
        preds: (B, 8, H, W) -> [score, x, y, z, l, w, h, yaw]
        targets_list: Liste de tenseurs [N, 7] (x, y, z, l, w, h, yaw)
        """
        device = preds.device
        B, C, H, W = preds.shape
        
        # 1. Séparation des prédictions (déjà transformées par le Backbone)
        cls_preds = preds[:, 0, :, :] # (B, H, W)
        reg_preds = preds[:, 1:, :, :] # (B, 7, H, W)

        # Initialisation des cibles
        target_reg = torch.zeros_like(reg_preds).to(device)
        mask = torch.zeros((B, H, W)).to(device)

        # CONFIGURATION - Doit matcher strictement ton Dataset
        x_min, y_min, x_max, y_max = 0.0, -40.0, 70.0, 40.0 

        for b in range(B):
            if len(targets_list[b]) == 0:
                continue
                
            for t in targets_list[b]:
                # Mapping normalisé (0 à 1)
                norm_x = (t[0] - x_min) / (x_max - x_min)
                norm_y = (t[1] - y_min) / (y_max - y_min)
                
                # Vérification que l'objet est bien dans la zone
                if 0 <= norm_x < 1 and 0 <= norm_y < 1:
                    # Indexation : gx = colonnes (W), gy = lignes (H)
                    gx = int(norm_x * W)
                    gy = int(norm_y * H)
                    
                    # Correction des bords
                    gx = min(max(gx, 0), W - 1)
                    gy = min(max(gy, 0), H - 1)

                    # ENCODAGE DES TARGETS
                    # On convertit t en t_encoded pour les offsets de centre
                    pixel_center_x = (gx + 0.5) / W
                    pixel_center_y = (gy + 0.5) / H
                    
                    t_encoded = t.clone()
                    t_encoded[0] = (norm_x - pixel_center_x) * W # Offset X local
                    t_encoded[1] = (norm_y - pixel_center_y) * H # Offset Y local
                    
                    # Le reste (z, l, w, h, yaw) reste en valeurs physiques
                    target_reg[b, :, gy, gx] = t_encoded
                    mask[b, gy, gx] = 1.0

        # --- LOSS DE CLASSIFICATION (Heatmap) ---
        raw_cls_loss = self.cls_loss_fn(cls_preds, mask)
        # Pondération pour compenser le déséquilibre massif (beaucoup plus de vide que d'objets)
        weight_mask = torch.where(mask > 0, self.pos_weight, 1.0)
        cls_loss = (raw_cls_loss * weight_mask).mean()

        # --- LOSS DE RÉGRESSION ---
        num_objects = mask.sum()
        if num_objects > 0:
            # On ne calcule la régression QUE sur les pixels contenant un objet
            # On crée un masque 4D (B, 7, H, W)
            reg_mask = mask.unsqueeze(1).expand_as(reg_preds)
            
            # Calcul de la perte sur tous les paramètres (x,y,z,l,w,h,yaw)
            reg_loss_all = self.reg_loss_fn(reg_preds, target_reg)
            
            # On applique le masque et on normalise par le nombre d'objets
            reg_loss = (reg_loss_all * reg_mask).sum() / (num_objects * 7 + 1e-6)
        else:
            reg_loss = torch.zeros(1, device=device, requires_grad=True)

        # Équilibre total
        total_loss = cls_loss + (self.reg_coef * reg_loss)
        
        return total_loss, num_objects.item()