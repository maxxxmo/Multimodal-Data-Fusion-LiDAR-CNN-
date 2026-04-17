# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PillarBackbone(nn.Module):
#     """
#     Architecture type PointPillars / CenterPoint.
#     Entrée : Pseudo-image (B, Channels, H, W)
#     Sortie : Heatmap (Classification) + Boîtes (Régression)
#     """
#     def __init__(self, in_channels=2):
#         super(PillarBackbone, self).__init__()

#         # --- PARTIE 1 : EXTRACTEUR DE CARACTÉRISTIQUES (BACKBONE) ---
#         # On traite la pseudo-image issue des piliers
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )

#         # Une couche de downsampling pour capter plus de contexte spatial
#         self.block2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # Réduit H,W par 2
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )

#         # Up-sampling (Deconv) pour revenir à la taille initiale (108x124)
#         self.up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

#         # --- PARTIE 2 : LES TÊTES DE DÉTECTION (DETECTION HEADS) ---
#         # 1. Tête de CLASSIFICATION : "Est-ce qu'il y a un objet ?" (1 canal)
#         # On utilise 1 seul canal pour le score d'objectivité (0 à 1 après Sigmoid)
#         self.cls_head = nn.Conv2d(64, 1, kernel_size=1)

#         # 2. Tête de RÉGRESSION : "Où et comment est l'objet ?" (7 canaux)
#         # Canaux : x, y, z, l, w, h, yaw
#         self.reg_head = nn.Conv2d(64, 7, kernel_size=1)
#     def forward(self, x):
#         x1 = self.block1(x)
#         x2 = self.block2(x1)
#         out = self.up(x2)
        
#         if out.shape != x1.shape:
#             out = F.interpolate(out, size=(x1.shape[2], x1.shape[3]))
        
#         out = out + x1

#         # 1. On récupère les sorties brutes (logits)
#         cls_score = self.cls_head(out) # (B, 1, H, W)
#         reg_raw = self.reg_head(out)   # (B, 7, H, W)

#         # 2. On applique les transformations SANS modification in-place
#         # On sépare les canaux pour les traiter
#         pos_z = reg_raw[:, 0:3, :, :] # x, y, z (on les laisse tels quels)
        
#         # l, w, h : On applique exp pour garantir des valeurs positives
#         dims = torch.exp(torch.clamp(reg_raw[:, 3:6, :, :], max=5.0))
        
#         # yaw : On applique tanh pour rester entre -pi et pi
#         yaw = torch.tanh(reg_raw[:, 6:7, :, :]) * 3.14159

#         # 3. On concatène tout dans un NOUVEAU tenseur
#         # Ordre : [score, x, y, z, l, w, h, yaw]
#         full_preds = torch.cat([cls_score, pos_z, dims, yaw], dim=1)

#         return full_preds

import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarBackbone(nn.Module):
    """
    Architecture améliorée (Itération 2) pour PointPillars.
    Optimisée pour réduire l'overfitting sur les séquences de validation.
    """
    def __init__(self, in_channels=2):
        super(PillarBackbone, self).__init__()

        # --- BLOC 1 : Extraction haute résolution ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1) # Régularisation pour aider la val loss
        )

        # --- BLOC 2 : Downsampling (Contexte spatial) ---
        # On augmente légèrement les canaux (128) pour mieux généraliser
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        # --- UP-SAMPLING & FUSION ---
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Couche de fusion pour combiner intelligemment x1 et x2
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # --- TÊTES DE DÉTECTION (Feature Sharing) [1, 2] ---
        # 1. Classification : Logits pour BCEWithLogitsLoss [1, 3]
        self.cls_head = nn.Conv2d(64, 1, kernel_size=1)

        # 2. Régression : 7 paramètres pour SmoothL1Loss [1, 3]
        self.reg_head = nn.Conv2d(64, 7, kernel_size=1)

    def forward(self, x):
        # Extraction des caractéristiques via le backbone [4, 5]
        x1 = self.block1(x)
        x2 = self.block2(x1)
        
        # Upsampling pour revenir à la résolution des piliers
        up_feat = self.up(x2)
        
        # Redimensionnement robuste si les dimensions diffèrent légèrement
        if up_feat.shape[2:] != x1.shape[2:]:
            up_feat = F.interpolate(up_feat, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        # Concaténation au lieu d'une addition pour plus de flexibilité
        combined = torch.cat([up_feat, x1], dim=1)
        out = self.fusion(combined)

        # Sorties des têtes [2, 6]
        cls_score = self.cls_head(out)
        reg_raw = self.reg_head(out)
        
        # Post-processing des prédictions de régression
        pos_z = reg_raw[:, 0:3, :, :] # x, y, z
        # Clamp pour éviter les explosions numériques dans l'exponentielle
        dims = torch.exp(torch.clamp(reg_raw[:, 3:6, :, :], min=-5.0, max=5.0))
        yaw = torch.tanh(reg_raw[:, 6:7, :, :]) * 3.14159

        # Résultat final : [score, x, y, z, l, w, h, yaw]
        return torch.cat([cls_score, pos_z, dims, yaw], dim=1)