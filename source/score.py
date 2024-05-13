import torch

def iou_score(pred_mask, true_mask):
    intersection = torch.logical_and(true_mask, pred_mask).sum().float()
    union = torch.logical_or(true_mask, pred_mask).sum().float()
    iou = intersection / union if union > 0 else 0.0
    return iou
    
