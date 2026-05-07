import torch
from torchvision.ops import nms

def apply_nms(boxes, scores, iou_threshold=0.1):
    """
    Args:
        boxes (Tensor): [N, 7] (x, y, z, w, l, h, yaw)
        scores (Tensor): [N]
        iou_threshold (float): if bbox<threshold the bbox is deleted
    """
    
    if len(boxes) == 0:
        return torch.empty((0, 7)), torch.empty((0,))

    scores = scores.reshape(-1)
    
    bevs = torch.zeros((boxes.shape[0], 4), device=boxes.device)
    bevs[:, 0] = boxes[:, 0] - boxes[:, 4] / 2 # x_min
    bevs[:, 1] = boxes[:, 1] - boxes[:, 3] / 2 # y_min
    bevs[:, 2] = boxes[:, 0] + boxes[:, 4] / 2 # x_max
    bevs[:, 3] = boxes[:, 1] + boxes[:, 3] / 2 # y_max

    keep_indices = nms(bevs, scores, iou_threshold)

    return boxes[keep_indices], scores[keep_indices]