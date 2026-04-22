import torch
def decode_boxes(anchors, deltas):
    """
    anchors: (N, 7) [x, y, z, w, l, h, theta]
    deltas: (N, 7) [dx, dy, dz, dw, dl, dh, dtheta]
    """
    # 1. Calcul de la diagonale (nécessaire pour l'inversion de x, y)
    diagonal = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    
    # 2. Inversion x, y
    res_x = anchors[:, 0] + deltas[:, 0] * diagonal
    res_y = anchors[:, 1] + deltas[:, 1] * diagonal
    res_z = anchors[:, 2] + deltas[:, 2] * anchors[:, 5]
    
    # 3. Inversion dimensions (w, l, h)
    res_w = torch.exp(deltas[:, 3]) * anchors[:, 3]
    res_l = torch.exp(deltas[:, 4]) * anchors[:, 4]
    res_h = torch.exp(deltas[:, 5]) * anchors[:, 5]
    
    # 4. Inversion angle
    res_theta = deltas[:, 6] + anchors[:, 6]
    
    return torch.stack([res_x, res_y, res_z, res_w, res_l, res_h, res_theta], dim=1)



def get_detected_boxes(cls_logits, reg_preds, anchor_generator, score_thresh=0.5):
    print(f"DEBUG: cls_logits shape: {cls_logits.shape}")
    print(f"DEBUG: reg_preds shape: {reg_preds.shape}")
    print(f"DEBUG: anchor_gen.anchors shape: {anchor_generator.anchors.shape}")
    # 1. Préparation des ancres (reforme en (N, 7))
    anchors = anchor_generator.anchors.view(-1, 7).to(cls_logits.device)
    
    # 2. Flatten des prédictions (N, C)
    # cls_logits: (1, num_anchors, H, W) -> (N, 1)
    # reg_preds: (1, num_anchors * 7, H, W) -> (N, 7)
    num_anchors, H, W = cls_logits.shape[1], cls_logits.shape[2], cls_logits.shape[3]
    
    cls_probs = torch.sigmoid(cls_logits).permute(0, 2, 3, 1).reshape(-1, 1)
    deltas = reg_preds.permute(0, 2, 3, 1).reshape(-1, 7)
    
    # 3. Filtrage par score (rapide avant décodage lourd)
    keep_mask = (cls_probs.squeeze() > score_thresh)
    if not keep_mask.any():
        return torch.zeros((0, 7)), torch.zeros(0)
        
    final_anchors = anchors[keep_mask]
    final_deltas = deltas[keep_mask]
    final_scores = cls_probs[keep_mask]
    
    # 4. Décodage
    pred_boxes = decode_boxes(final_anchors, final_deltas)
    
    return pred_boxes, final_scores