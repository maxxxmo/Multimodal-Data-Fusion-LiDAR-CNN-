import torch
import matplotlib.pyplot as plt

import numpy as np


def decode_boxes(anchors, deltas):
    """
    Inputs
    anchors: (N, 7)
    deltas: (N, 8) [dx, dy, dz, dw, dl, dh, sin, cos]
    Outputs:
    Create final coordinates of the detected bbox
    """
    diagonal = torch.sqrt(anchors[:, 3]**2 + anchors[:, 4]**2)
    
    # Position
    res_x = anchors[:, 0] + deltas[:, 0] * diagonal
    res_y = anchors[:, 1] + deltas[:, 1] * diagonal
    res_z = anchors[:, 2] + deltas[:, 2] * anchors[:, 5]
    
    # Dimensions
    res_w = torch.exp(deltas[:, 3]) * anchors[:, 3]
    res_l = torch.exp(deltas[:, 4]) * anchors[:, 4]
    res_h = torch.exp(deltas[:, 5]) * anchors[:, 5]
    

    angle_diff = torch.atan2(deltas[:, 6], deltas[:, 7])
    res_theta = anchors[:, 6] + angle_diff
    
    return torch.stack([res_x, res_y, res_z, res_w, res_l, res_h, res_theta],dim=1)

def get_detected_boxes(cls_logits, reg_preds, anchor_generator, score_thresh=0.5):
    """
    Takes predictions > threshold and return the bbox and their score

    """
    anchors = anchor_generator.anchors.view(-1, 7).to(cls_logits.device)
    num_anchors, H, W = cls_logits.shape[1], cls_logits.shape[2], cls_logits.shape[3]
    cls_probs = torch.sigmoid(cls_logits).permute(0, 2, 3, 1).reshape(-1, 1)
    deltas = reg_preds.permute(0, 2, 3, 1).reshape(-1, 8)
    
    # Score filtering
    keep_mask = (cls_probs.squeeze() > score_thresh)
    if not keep_mask.any():
        return torch.zeros((0, 7)), torch.zeros(0)
        
    final_anchors = anchors[keep_mask]
    final_deltas = deltas[keep_mask]
    final_scores = cls_probs[keep_mask]
    
    # Decoding
    pred_boxes = decode_boxes(final_anchors, final_deltas)
    
    return pred_boxes, final_scores

