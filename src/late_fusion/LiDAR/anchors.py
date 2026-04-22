import torch
import numpy as np

class AnchorGenerator:
    def __init__(self, feature_map_size, anchor_sizes, anchor_rotations, pc_range):
        """
        feature_map_size: (H, W)
        anchor_sizes: liste de [w, l, h] par type d'ancre
        anchor_rotations: liste des angles (ex: [0, np.pi/2])
        pc_range: [x_min, y_min, x_max, y_max]
        """
        self.H, self.W = feature_map_size
        self.anchor_sizes = torch.tensor(anchor_sizes) # (N_anchors, 3)
        self.rotations = torch.tensor(anchor_rotations)
        self.pc_range = pc_range
        self.anchors = self._generate_anchors()
        # print(f"DEBUG ANCHOR: Initialisation avec pc_range={self.pc_range} et feature_map={feature_map_size}")
        
    def _generate_anchors(self):
        # 1. Extraction des bornes
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        # 2. Calcul des centres avec décalage de demi-voxel
        res_x = (x_max - x_min) / self.W
        res_y = (y_max - y_min) / self.H
        
        x_centers = torch.linspace(x_min + res_x / 2, x_max - res_x / 2, self.W)
        y_centers = torch.linspace(y_min + res_y / 2, y_max - res_y / 2, self.H)
        
        # 3. Meshgrid : xv (H, W), yv (H, W)
        xv, yv = torch.meshgrid(x_centers, y_centers, indexing='xy')
        
        # 4. Préparation des dimensions pour le broadcasting
        # On ajoute des dimensions pour atteindre (H, W, 1, 1, 1)
        x_grid = xv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y_grid = yv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z_grid = torch.full_like(x_grid, -1.0) # Hauteur Z fixe (ex: -1m)

        # 5. Préparation des dimensions pour sizes et rotations
        # sizes: (1, 1, N_types, 1, 3) -> W, L, H
        # rots: (1, 1, 1, N_rots, 1)
        sizes = self.anchor_sizes.view(1, 1, -1, 1, 3)
        rots = self.rotations.view(1, 1, 1, -1, 1)

        # 6. Expansion broadcastée
        # Toutes les dimensions deviennent (H, W, N_types, N_rots, dim)
        x_exp = x_grid.expand(self.H, self.W, len(self.anchor_sizes), len(self.rotations), 1)
        y_exp = y_grid.expand(self.H, self.W, len(self.anchor_sizes), len(self.rotations), 1)
        z_exp = z_grid.expand(self.H, self.W, len(self.anchor_sizes), len(self.rotations), 1)
        
        # On extrait W, L, H depuis sizes
        w_exp = sizes[..., 0].unsqueeze(-1).expand(self.H, self.W, -1, len(self.rotations), 1)
        l_exp = sizes[..., 1].unsqueeze(-1).expand(self.H, self.W, -1, len(self.rotations), 1)
        h_exp = sizes[..., 2].unsqueeze(-1).expand(self.H, self.W, -1, len(self.rotations), 1)
        rot_exp = rots.expand(self.H, self.W, len(self.anchor_sizes), -1, 1)

        # 7. Concaténation finale
        # all_anchors: (H, W, N_types, N_rots, 7)
        all_anchors = torch.cat([x_exp, y_exp, z_exp, w_exp, l_exp, h_exp, rot_exp], dim=-1)
        
        print(f"DEBUG: Shape finale des ancres : {all_anchors.shape}")
        return all_anchors
    # def _generate_anchors(self):
    #     # 1. Extraction explicite pour éviter l'erreur de mapping
    #     # pc_range est attendu sous la forme [x_min, y_min, z_min, x_max, y_max, z_max]
    #     # x_min, y_min, _, x_max, y_max, _ = self.pc_range
        
    #     x_min, y_min = self.pc_range[0], self.pc_range[1]
    #     x_max, y_max = self.pc_range[3], self.pc_range[4]
        
    #     print(f"DEBUG: Génération ancres X [{x_min:.2f}, {x_max:.2f}] | Y [{y_min:.2f}, {y_max:.2f}]")
    #     res_x = (x_max - x_min) / self.W
    #     res_y = (y_max - y_min) / self.H
        
    #     # 3. Appliquer le décalage du demi-voxel
    #     # On commence au premier centre (x_min + demi_pas) et on finit au dernier
    #     x_start = x_min + res_x / 2
    #     x_end = x_max - res_x / 2
    #     y_start = y_min + res_y / 2
    #     y_end = y_max - res_y / 2
        
        
    #     # 2. Création des centres avec le bon nombre de points (W pour X, H pour Y)
    #     x_centers = torch.linspace(x_start, x_end, self.W)
    #     y_centers = torch.linspace(y_start, y_end, self.H)
        
    #     # 3. Meshgrid avec indexing='xy' (plus intuitif pour les coordonnées spatiales)
    #     # xv: (H, W), yv: (H, W)
    #     # xv, yv = torch.meshgrid(x_centers, y_centers, indexing='xy')
                
                
    #     xv, yv = torch.meshgrid(x_centers, y_centers, indexing='xy')

    #     # 4. Création de la grille (H, W, 1, 1, 3)
    #     # On empile les coordonnées et un Z=0 (ou hauteur fixe si besoin)
    #     print(f"DEBUG: xv unique values: {xv.unique().shape} | yv unique values: {yv.unique().shape}")
    #     grid = torch.stack([xv, yv, torch.zeros_like(xv)], dim=-1)
    #     grid = grid.unsqueeze(2).unsqueeze(3) 
        
    #     # 5. Gestion des dimensions et rotations (Broadcasting)
    #     N_types = len(self.anchor_sizes)
    #     N_rots = len(self.rotations)
        
    #     # S'assurer que sizes et rots sont des tenseurs pour le broadcast
    #     sizes = self.anchor_sizes.view(1, 1, N_types, 1, 3)
    #     rots = self.rotations.view(1, 1, 1, N_rots, 1)
        
    #     # 6. Expansion pour remplir la grille (H, W, N_types, N_rots, 7)
    #     # Grid est (H, W, 1, 1, 3) -> broadcasté en (H, W, N_types, N_rots, 3)
    #     grid_expanded = grid.expand(self.H, self.W, N_types, N_rots, 3)
    #     sizes_expanded = sizes.expand(self.H, self.W, N_types, N_rots, 3)
    #     rots_expanded = rots.expand(self.H, self.W, N_types, N_rots, 1)
        
    #     # 7. Concaténation finale
    #     # Format: [x, y, z, w, l, h, theta]
    #     all_anchors = torch.cat([grid_expanded, sizes_expanded, rots_expanded], dim=-1)
        
    #     print(f"DEBUG: Shape finale des ancres : {all_anchors.shape}")
    #     return all_anchors
        

class TargetAssigner:
    def __init__(self, iou_thresholds=(0.45, 0.6)):
        self.neg_thresh = iou_thresholds[0]
        self.pos_thresh = iou_thresholds[1]
        
    def calculate_iou_3d(self, anchors, gt_boxes):
        """
        Calcul de l'IoU BEV entre les ancres et les GT.
        anchors: (N, 7) [x, y, z, w, l, h, theta]
        gt_boxes: (M, 7) [x, y, z, w, l, h, theta]
        """
        # 1. Extraire les coordonnées BEV (x, y, w, l)
        # On ignore Z, H et Theta pour le moment (calcul "axis-aligned")
        a_x1 = anchors[:, 0] - anchors[:, 3] / 2
        a_y1 = anchors[:, 1] - anchors[:, 4] / 2
        a_x2 = anchors[:, 0] + anchors[:, 3] / 2
        a_y2 = anchors[:, 1] + anchors[:, 4] / 2
        
        g_x1 = gt_boxes[:, 0] - gt_boxes[:, 3] / 2
        g_y1 = gt_boxes[:, 1] - gt_boxes[:, 4] / 2
        g_x2 = gt_boxes[:, 0] + gt_boxes[:, 3] / 2
        g_y2 = gt_boxes[:, 1] + gt_boxes[:, 4] / 2
        
        # 2. Calculer les dimensions de l'intersection
        # On utilise broadcasting pour comparer chaque ancre (N) avec chaque GT (M)
        # Shape finale (N, M)
        inter_x1 = torch.max(a_x1.unsqueeze(1), g_x1.unsqueeze(0))
        inter_y1 = torch.max(a_y1.unsqueeze(1), g_y1.unsqueeze(0))
        inter_x2 = torch.min(a_x2.unsqueeze(1), g_x2.unsqueeze(0))
        inter_y2 = torch.min(a_y2.unsqueeze(1), g_y2.unsqueeze(0))
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # 3. Calculer les aires et l'Union
        area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
        area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
        
        # Union = AreaA + AreaG - Intersection
        union_area = area_a.unsqueeze(1) + area_g.unsqueeze(0) - inter_area
        
        # 4. IoU final
        return inter_area / (union_area + 1e-7)
   
    def assign(self, anchors, gt_boxes):
        """
        anchors: [H, W, N_types, N_rots, 7]
        gt_boxes: [M, 7]
        """
        if gt_boxes.numel() == 0:
            H, W, N_types, N_rots, _ = anchors.shape
            num_anchors = N_types * N_rots
            # Retourner des labels à -1 (background/ignore) et des targets vides
            labels = torch.full((num_anchors, H, W), -1, device=anchors.device)
            reg_targets = torch.zeros((num_anchors * 7, H, W), device=anchors.device)
            pos_mask = torch.zeros((num_anchors, H, W), dtype=torch.bool, device=anchors.device)
            return labels, reg_targets, pos_mask
            
        H, W, N_types, N_rots, _ = anchors.shape
        num_anchors = N_types * N_rots
        
        # 1. Aplatir les ancres pour le calcul IoU
        # On passe de (H, W, N_types, N_rots, 7) à (N_total, 7)
        anchors_flat = anchors.view(-1, 7)
        
        # 2. Calcul IoU (N_total, M)
        iou_matrix = self.calculate_iou_3d(anchors_flat, gt_boxes)

        # Après : iou_matrix = self.calculate_iou_3d(anchors_flat, gt_boxes)
        # print(f"DEBUG: Max IoU dans la matrice : {iou_matrix.max().item():.4f}")
        # print(f"DEBUG: Anchors Shape: {anchors_flat.shape} | GT Shape: {gt_boxes.shape}")

        # Si le Max IoU est très bas, vérifions la première ancre et le premier GT
        # if iou_matrix.max() < 0.01:
        #     print(f"ANCRE EXEMPLE : {anchors_flat[0]}") # [x, y, z, w, l, h, theta]
        #     print(f"GT EXEMPLE     : {gt_boxes[0]}")
                
        
        max_iou, argmax_iou = iou_matrix.max(dim=1)
        
        # 3. Initialisation des buffers pour les cibles (format 2D)
        # On veut (N_anchors_flat, 7) pour le reg et (N_anchors_flat,) pour le cls
        labels = torch.zeros(anchors_flat.shape[0], device=anchors.device)
        reg_targets = torch.zeros_like(anchors_flat)
        pos_mask = (max_iou >= self.pos_thresh)
        
        # 4. Encodage des cibles positives
        if pos_mask.any():
            pos_anchors = anchors_flat[pos_mask]
            matched_gt = gt_boxes[argmax_iou[pos_mask]]
            reg_targets[pos_mask] = encode_targets(pos_anchors, matched_gt)
            labels[pos_mask] = 1
        
        # 5. Gestion des négatifs et ignores
        labels[(max_iou < self.pos_thresh) & (max_iou >= self.neg_thresh)] = -1 # Ignore
        
        # 6. RESHAPE FINAL vers (Channels, H, W)
        # Format attendu par le réseau: (N_anchors * 7, H, W) pour reg, (N_anchors, H, W) pour cls
        
        # Reshape CLS: (H, W, num_anchors) -> (num_anchors, H, W)
        labels_2d = labels.view(H, W, num_anchors).permute(2, 0, 1)
        
        # Reshape REG: (H, W, num_anchors, 7) -> (H, W, num_anchors * 7) -> (num_anchors * 7, H, W)
        reg_targets_2d = reg_targets.view(H, W, num_anchors * 7).permute(2, 0, 1)
        
        # Pos mask pour la loss: (H, W, num_anchors) -> (num_anchors, H, W)
        pos_mask_2d = pos_mask.view(H, W, num_anchors).permute(2, 0, 1)
        
        return labels_2d, reg_targets_2d, pos_mask_2d

    
def encode_targets(anchors, gt_boxes):
    """
    anchors: (N, 7)
    gt_boxes: (N, 7) - (déjà matchés avec les ancres correspondantes)
    """
    # Calcul de la diagonale de l'ancre
    diagonal = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    
    # Encodage (x, y, z)
    deltas_x = (gt_boxes[:, 0] - anchors[:, 0]) / diagonal
    deltas_y = (gt_boxes[:, 1] - anchors[:, 1]) / diagonal
    deltas_z = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    
    # Encodage (w, l, h)
    deltas_w = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    deltas_l = torch.log(gt_boxes[:, 4] / anchors[:, 4])
    deltas_h = torch.log(gt_boxes[:, 5] / anchors[:, 5])
    
    # Encodage angle
    deltas_theta = gt_boxes[:, 6] - anchors[:, 6]
    
    return torch.stack([deltas_x, deltas_y, deltas_z, 
                        deltas_w, deltas_l, deltas_h, 
                        deltas_theta], dim=1)