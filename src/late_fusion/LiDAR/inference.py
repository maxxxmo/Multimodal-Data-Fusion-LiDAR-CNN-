import torch

def get_detected_boxes(preds, threshold=0.1):
    """Extract detected boxes from model predictions based on a confidence threshold. """
    if preds.dim() == 4:
        preds = preds.squeeze(0)
    
    score_map = torch.sigmoid(preds[0, :, :]) 
    keep_indices = (score_map > threshold).nonzero()
    
    detected_boxes = []
    for idx in keep_indices:
        yy, xx = idx
        box_params = preds[1:, yy, xx].cpu().numpy()
        conf = score_map[yy, xx].item()
        detected_boxes.append({
            'box': box_params,
            'score': conf
        })
        
    return detected_boxes