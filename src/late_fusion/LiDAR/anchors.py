import torch
import numpy as np



class AnchorGenerator:
    """
    Docstring for AnchorGenerator
    
    :var anchors: Description
    :var gt_boxes: groiund truth boxes
    :var gt: ground truth
    
    
    
    """
    def __init__(self, feature_map_size, anchor_sizes, anchor_rotations, pc_range):
        self.H, self.W = feature_map_size
        # self.W, self.H = feature_map_size
        self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        self.rotations = torch.tensor(anchor_rotations, dtype=torch.float32)
        self.pc_range = pc_range
        self.anchors = self._generate_anchors()
        
    def _generate_anchors(self):
        """
        Create a grid of anchors following config.yaml parameters
        """
        # BBOX centers
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        res_x = (x_max - x_min) / self.W
        res_y = (y_max - y_min) / self.H
        
        x_centers = torch.linspace(x_min + res_x / 2, x_max - res_x / 2, self.W)
        y_centers = torch.linspace(y_min + res_y / 2, y_max - res_y / 2, self.H)
        
        # grid (H,W)
        yv, xv = torch.meshgrid(y_centers, x_centers, indexing='ij')
        
        N_types = self.anchor_sizes.shape[0]
        N_rots = self.rotations.shape[0]
        
        # Anchors tensor (H, W, N_types, N_rots, 7)
        anchors = torch.zeros((self.H, self.W, N_types, N_rots, 7), dtype=torch.float32)
        
        # 4. BROADCASTING FOR EFFICIENCY
        #  (H, W) to (H, W, 1, 1) the broadcasting to (H, W, N_types, N_rots) 
        xv_exp = xv.unsqueeze(-1).unsqueeze(-1).expand(self.H, self.W, N_types, N_rots)
        yv_exp = yv.unsqueeze(-1).unsqueeze(-1).expand(self.H, self.W, N_types, N_rots)
        
        anchors[..., 0] = xv_exp  # X
        anchors[..., 1] = yv_exp  # Y
        anchors[..., 2] = -0.7   # Z fixed -1.6 m + ((Car height =1.5)/2) 
        
        # adding sizes
        sizes = self.anchor_sizes.view(1, 1, N_types, 1, 3).expand(self.H, self.W, N_types, N_rots, 3)
        anchors[..., 3:6] = sizes
        
        # adding rotations
        rots = self.rotations.view(1, 1, 1, N_rots).expand(self.H, self.W, N_types, N_rots)
        anchors[..., 6] = rots
        
        return anchors


class TargetAssigner:

    def __init__(
        self,
        iou_thresholds=(0.2, 0.45),
        pc_range=None
    ):

        self.neg_thresh, self.pos_thresh = iou_thresholds
        self.pc_range = pc_range
    
    
    def assign(self, anchors, gt_boxes):
        """
        anchors : [H, W, N_types, N_rots, 7]
        gt_boxes: [M, 7]
        """

        H, W, N_t, N_r, _ = anchors.shape
        device = anchors.device

        # =====================================================
        # INIT
        # =====================================================

        # -1 = ignore
        #  0 = negative
        #  1 = positive

        # labels = torch.full(
        #     (H, W, N_t, N_r),
        #     -1,
        #     dtype=torch.float32,
        #     device=device
        # )
        labels = torch.zeros(
            (H, W, N_t, N_r),
            dtype=torch.float32,
            device=device
        )

        pos_mask = torch.zeros(
            (H, W, N_t, N_r),
            dtype=torch.bool,
            device=device
        )

        reg_targets = torch.zeros(
            (H, W, N_t, N_r, 8),
            dtype=torch.float32,
            device=device
        )

        # =====================================================
        # GRID RESOLUTION
        # =====================================================

        res_x = (
            self.pc_range[3] - self.pc_range[0]
        ) / W

        res_y = (
            self.pc_range[4] - self.pc_range[1]
        ) / H

        # =====================================================
        # LOOP GT
        # =====================================================

        for gt in gt_boxes:

            # -------------------------------------------------
            # GT -> GRID
            # -------------------------------------------------

            u = int(torch.clamp(
                ((gt[0] - self.pc_range[0]) / res_x),
                0,
                W - 1
            ))

            v = int(torch.clamp(
                ((gt[1] - self.pc_range[1]) / res_y),
                0,
                H - 1
            ))

            # -------------------------------------------------
            # LOCAL WINDOW
            # -------------------------------------------------

            v_start = max(0, v - 2)
            v_end = min(H, v + 3)

            u_start = max(0, u - 2)
            u_end = min(W, u + 3)

            local_anchors = anchors[
                v_start:v_end,
                u_start:u_end,
                :,
                :,
                :
            ]

            local_shape = local_anchors.shape

            anchors_flat = local_anchors.reshape(-1, 7)

            # -------------------------------------------------
            # IOU
            # -------------------------------------------------

            ious = self.calculate_iou_bev(
                anchors_flat,
                gt.unsqueeze(0)
            ).squeeze(-1)

            # =====================================================
            # NEGATIVES
            # =====================================================

            negative_indices = (
                ious < self.neg_thresh
            ).nonzero(as_tuple=False)

            for idx_tensor in negative_indices:

                flat_idx = idx_tensor.item()

                idx_v = flat_idx // (
                    local_shape[1]
                    * local_shape[2]
                    * local_shape[3]
                )

                rem = flat_idx % (
                    local_shape[1]
                    * local_shape[2]
                    * local_shape[3]
                )

                idx_u = rem // (
                    local_shape[2]
                    * local_shape[3]
                )

                rem = rem % (
                    local_shape[2]
                    * local_shape[3]
                )

                idx_t = rem // local_shape[3]

                idx_r = rem % local_shape[3]

                v_glob = v_start + idx_v
                u_glob = u_start + idx_u

                # seulement si pas déjà positif
                if labels[v_glob, u_glob, idx_t, idx_r] != 1:
                    labels[v_glob, u_glob, idx_t, idx_r] = 0

            # =====================================================
            # POSITIVES
            # =====================================================

            positive_indices = (
                ious > self.pos_thresh
            ).nonzero(as_tuple=False)

            # -------------------------------------------------
            # SAFETY POSITIVE
            # -------------------------------------------------

            if len(positive_indices) == 0:

                best_idx = torch.argmax(ious)

                if ious[best_idx] > 0.25:
                    positive_indices = torch.tensor(
                        [[best_idx]],
                        device=device
                    )

            # =====================================================
            # ASSIGN POSITIVES
            # =====================================================

            for idx_tensor in positive_indices:

                flat_idx = idx_tensor.item()

                idx_v = flat_idx // (
                    local_shape[1]
                    * local_shape[2]
                    * local_shape[3]
                )

                rem = flat_idx % (
                    local_shape[1]
                    * local_shape[2]
                    * local_shape[3]
                )

                idx_u = rem // (
                    local_shape[2]
                    * local_shape[3]
                )

                rem = rem % (
                    local_shape[2]
                    * local_shape[3]
                )

                idx_t = rem // local_shape[3]

                idx_r = rem % local_shape[3]

                v_glob = v_start + idx_v
                u_glob = u_start + idx_u

                # positive
                labels[
                    v_glob,
                    u_glob,
                    idx_t,
                    idx_r
                ] = 1

                pos_mask[
                    v_glob,
                    u_glob,
                    idx_t,
                    idx_r
                ] = True

                # regression target

                anchor = anchors[
                    v_glob,
                    u_glob,
                    idx_t,
                    idx_r
                ].unsqueeze(0)

                encoded = encode_targets(
                    anchor,
                    gt.unsqueeze(0)
                ).squeeze(0)

                reg_targets[
                    v_glob,
                    u_glob,
                    idx_t,
                    idx_r
                ] = encoded
        return labels, reg_targets, pos_mask
    

    def calculate_iou_bev(self, anchors, gt):
        """
        Calcule l'IoU BEV (2D) entre N ancres et M GT en prenant en compte l'angle.
        Format attendu pour anchors et gt : [..., 7] -> (x, y, z, w, l, h, theta)
        """
        device = anchors.device
        N, M = anchors.shape[0], gt.shape[0]
        
        # 1. Extraction des paramètres BEV pertinents
        # On utilise l'index 3 pour Width (w) et 4 pour Length (l)
        a_x, a_y, a_w, a_l, a_theta = anchors[:, 0], anchors[:, 1], anchors[:, 3], anchors[:, 4], anchors[:, 6]
        g_x, g_y, g_w, g_l, g_theta = gt[:, 0], gt[:, 1], gt[:, 3], gt[:, 4], gt[:, 6]

        iou_matrix = torch.zeros((N, M), device=device)

        for j in range(M):
            # 2. Différence d'angle relative
            # On utilise cos/sin absolus pour projeter l'ancre dans le repère du GT
            delta_theta = a_theta - g_theta[j]
            cos_rel = torch.abs(torch.cos(delta_theta))
            sin_rel = torch.abs(torch.sin(delta_theta))
            
            # 3. Calcul des dimensions effectives (projetées)
            # Imagine l'ombre de l'ancre tournée sur les axes de la voiture GT
            a_l_eff = a_l * cos_rel + a_w * sin_rel
            a_w_eff = a_l * sin_rel + a_w * cos_rel
            
            # 4. Intersection "Axis-Aligned" dans le repère local du GT
            dx = torch.abs(a_x - g_x[j])
            dy = torch.abs(a_y - g_y[j])
            
            # chevauchement sur X et Y
            inter_w = torch.clamp((a_l_eff + g_l[j]) / 2 - dx, min=0)
            inter_h = torch.clamp((a_w_eff + g_w[j]) / 2 - dy, min=0)
            
            inter_area = inter_w * inter_h
            
            # 5. Union et IoU
            area_a = a_w * a_l
            area_g = g_w[j] * g_l[j]
            union_area = area_a + area_g - inter_area
            
            iou_matrix[:, j] = inter_area / (union_area + 1e-7)

        return iou_matrix
    # def calculate_iou_bev(self, anchors, gt):
    #     """
    #     IoU BEV axis-aligned
    #     """

    #     a_w = anchors[:, 4]
    #     a_l = anchors[:, 5]

    #     g_w = gt[:, 4]
    #     g_l = gt[:, 5]

    #     # -----------------------------------------
    #     # anchor corners
    #     # -----------------------------------------

    #     a_x1 = anchors[:, 0] - a_l / 2
    #     a_x2 = anchors[:, 0] + a_l / 2

    #     a_y1 = anchors[:, 1] - a_w / 2
    #     a_y2 = anchors[:, 1] + a_w / 2

    #     # -----------------------------------------
    #     # gt corners
    #     # -----------------------------------------

    #     g_x1 = gt[:, 0] - g_l / 2
    #     g_x2 = gt[:, 0] + g_l / 2

    #     g_y1 = gt[:, 1] - g_w / 2
    #     g_y2 = gt[:, 1] + g_w / 2

    #     # -----------------------------------------
    #     # intersection
    #     # -----------------------------------------

    #     inter_w = torch.clamp(
    #         torch.min(a_x2, g_x2)
    #         - torch.max(a_x1, g_x1),
    #         min=0
    #     )

    #     inter_h = torch.clamp(
    #         torch.min(a_y2, g_y2)
    #         - torch.max(a_y1, g_y1),
    #         min=0
    #     )

    #     inter = inter_w * inter_h

    #     # -----------------------------------------
    #     # union
    #     # -----------------------------------------

    #     area_a = a_w * a_l
    #     area_g = g_w * g_l

    #     union = area_a + area_g - inter

    #     return inter / (union + 1e-7)



# def encode_targets(anchors, gt_boxes):
#     """
#     anchors: (N, 7)
#     gt_boxes: (N, 7) - (déjà matchés avec les ancres correspondantes)
#     """
#     # Calcul de la diagonale de l'ancre
#     diagonal = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    
#     # Encodage (x, y, z)
#     deltas_x = (gt_boxes[:, 0] - anchors[:, 0]) / diagonal
#     deltas_y = (gt_boxes[:, 1] - anchors[:, 1]) / diagonal
#     deltas_z = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    
#     # Encodage (w, l, h)
#     deltas_w = torch.log(gt_boxes[:, 3] / anchors[:, 3])
#     deltas_l = torch.log(gt_boxes[:, 4] / anchors[:, 4])
#     deltas_h = torch.log(gt_boxes[:, 5] / anchors[:, 5])
    
#     # Encodage angle
#     deltas_theta = gt_boxes[:, 6] - anchors[:, 6]
    
#     return torch.stack([deltas_x, deltas_y, deltas_z, 
#                         deltas_w, deltas_l, deltas_h, 
#                         deltas_theta], dim=1)

def encode_targets(anchors, gt_boxes):
    diagonal = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    
    deltas_x = (gt_boxes[:, 0] - anchors[:, 0]) / diagonal
    deltas_y = (gt_boxes[:, 1] - anchors[:, 1]) / diagonal
    deltas_z = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    
    deltas_w = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    deltas_l = torch.log(gt_boxes[:, 4] / anchors[:, 4])
    deltas_h = torch.log(gt_boxes[:, 5] / anchors[:, 5])
    
    # --- LA MAGIE EST ICI ---
    # Au lieu d'un delta linéaire, on prédit le sinus et le cosinus de la différence
    angle_diff = gt_boxes[:, 6] - anchors[:, 6]
    deltas_sin = torch.sin(angle_diff)
    deltas_cos = torch.cos(angle_diff)
    
    return torch.stack([
        deltas_x, deltas_y, deltas_z, 
        deltas_w, deltas_l, deltas_h, 
        deltas_sin, deltas_cos # 8 paramètres
    ], dim=1)