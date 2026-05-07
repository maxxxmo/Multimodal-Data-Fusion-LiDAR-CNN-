import torch
import numpy as np



class AnchorGenerator:
    """
    """
    def __init__(self, feature_map_size, anchor_sizes, anchor_rotations, pc_range):
        self.H, self.W = feature_map_size
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
        #  (H, W) to (H, W, 1, 1) then broadcasting to (H, W, N_types, N_rots) 
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
    """
    Assign GT to Anchor
    """
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
        Determinate for each Anchor on the grid if it's 
        positive --> can predict
        negative --> predict nothing (background)
        ignored --> not used for loss calculus
        
        It doesnt compare all anchors and GT. For each GT it looks in a 5*5 Anchor Grid Radius
        If positive we use encode targets to stock deltas
        """

        H, W, N_t, N_r, _ = anchors.shape
        device = anchors.device

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

        # GRID RESOLUTION
        res_x = (self.pc_range[3] - self.pc_range[0]) / W
        res_y = (self.pc_range[4] - self.pc_range[1]) / H

        
        for gt in gt_boxes:
            # Projection of GT in GRID
            u = int(torch.clamp(((gt[0] - self.pc_range[0]) / res_x),0,W - 1))
            v = int(torch.clamp(((gt[1] - self.pc_range[1]) / res_y),0,H - 1))

            # LOCAL WINDOW

            v_start = max(0, v - 2)
            v_end = min(H, v + 3)
            u_start = max(0, u - 2)
            u_end = min(W, u + 3)

            local_anchors = anchors[v_start:v_end,u_start:u_end,:,:,:] # Anchors within the Window
            local_shape = local_anchors.shape
            anchors_flat = local_anchors.reshape(-1, 7)

            # IOU
            ious = self.calculate_iou_bev(anchors_flat,gt.unsqueeze(0)).squeeze(-1)

            # NEGATIVES
            negative_indices = (ious < self.neg_thresh).nonzero(as_tuple=False)

            for idx_tensor in negative_indices:
                # Local anchor are flattened so we need to find back the indexes of the negative indices
                # Format of local window before flattening : [H_local, W_local, N_types, N_rots] after flattening [N_total_anchors, 7]
                # Now we have flat_idx = 73 and we want to find (v=2, u=2, t=0, r=1) 
                flat_idx = idx_tensor.item()
                # We have one long dimension we divide to get v, each line is  W_local, N_types, N_rots
                idx_v = flat_idx // (local_shape[1]* local_shape[2]* local_shape[3])
                # what remains of the indexe for the other parameters
                rem = flat_idx % (local_shape[1]* local_shape[2]* local_shape[3])
                # now u
                idx_u = rem // (local_shape[2]* local_shape[3])
                rem = rem % (local_shape[2]* local_shape[3])
                # now t and r
                idx_t = rem // local_shape[3]
                idx_r = rem % local_shape[3]
                
                # we start from the window coordinates
                v_glob = v_start + idx_v
                u_glob = u_start + idx_u

                # To not overwrite positives ones
                if labels[v_glob, u_glob, idx_t, idx_r] != 1:
                    labels[v_glob, u_glob, idx_t, idx_r] = 0

            # POSITIVES

            positive_indices = (ious > self.pos_thresh).nonzero(as_tuple=False)

            # SAFETY POSITIVE

            if len(positive_indices) == 0:

                best_idx = torch.argmax(ious)

                if ious[best_idx] > 0.25:
                    positive_indices = torch.tensor(
                        [[best_idx]],
                        device=device
                    )

            # ASSIGN POSITIVES

            for idx_tensor in positive_indices:
                flat_idx = idx_tensor.item()
                idx_v = flat_idx // (local_shape[1]* local_shape[2]* local_shape[3])
                rem = flat_idx % (local_shape[1]* local_shape[2]* local_shape[3])
                idx_u = rem // (local_shape[2]* local_shape[3])
                rem = rem % (local_shape[2]* local_shape[3])
                idx_t = rem // local_shape[3]
                idx_r = rem % local_shape[3]
                v_glob = v_start + idx_v
                u_glob = u_start + idx_u

                # positive
                labels[v_glob,u_glob,idx_t,idx_r] = 1

                pos_mask[v_glob,u_glob,idx_t,idx_r] = True

                # regression target

                anchor = anchors[v_glob,u_glob,idx_t,idx_r].unsqueeze(0)

                encoded = encode_targets(anchor,gt.unsqueeze(0)).squeeze(0)

                reg_targets[v_glob,u_glob,idx_t,idx_r] = encoded
                
        return labels, reg_targets, pos_mask
    

    def calculate_iou_bev(self, anchors, gt):
        """
        Calculate IoU in 2D between N anchors and M GroundTruths 
        Anchors and GroundTruth are: (x, y, z, w, l, h, theta)
        """
        device = anchors.device
        N, M = anchors.shape[0], gt.shape[0]
        
        # Parameters
        a_x, a_y, a_w, a_l, a_theta = anchors[:, 0], anchors[:, 1], anchors[:, 3], anchors[:, 4], anchors[:, 6]
        g_x, g_y, g_w, g_l, g_theta = gt[:, 0], gt[:, 1], gt[:, 3], gt[:, 4], gt[:, 6]

        iou_matrix = torch.zeros((N, M), device=device)

        for j in range(M):
            # Projecting angle using sin and cos
            delta_theta = a_theta - g_theta[j]
            cos_rel = torch.abs(torch.cos(delta_theta))
            sin_rel = torch.abs(torch.sin(delta_theta))
            
            # Projected dimensions
            a_l_eff = a_l * cos_rel + a_w * sin_rel
            a_w_eff = a_l * sin_rel + a_w * cos_rel
            
            # Intersection calculus
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
    
def encode_targets(anchors, gt_boxes):
    """Use anchors and gt boxes to create the 8 targets--> (dx, dy, dz, dw, dl, dh, dsin(theta), dcos(theta))"""
    diagonal = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    """Anchors are fix and targets are rotating rectangles. When rotating, the relatives offset of 
    x and y change depending on the object orientation. But the Diagonal (or
    in math the norm=sqrt(w²+l²)) of the anchor is invariant"""
    deltas_x = (gt_boxes[:, 0] - anchors[:, 0]) / diagonal
    deltas_y = (gt_boxes[:, 1] - anchors[:, 1]) / diagonal
    deltas_z = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    
    # Normalisation 0 to 1  
    deltas_w = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    deltas_l = torch.log(gt_boxes[:, 4] / anchors[:, 4])
    deltas_h = torch.log(gt_boxes[:, 5] / anchors[:, 5])
    

    # Using sin and cosin
    angle_diff = gt_boxes[:, 6] - anchors[:, 6]
    deltas_sin = torch.sin(angle_diff)
    deltas_cos = torch.cos(angle_diff)
    
    return torch.stack([
        deltas_x, deltas_y, deltas_z, 
        deltas_w, deltas_l, deltas_h, 
        deltas_sin, deltas_cos # 8 paramters
    ], dim=1)