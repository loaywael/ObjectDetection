import torch
import  numpy as np



def intersection_over_union(target_boxes, predicted_boxes):
    """
    Evaluates Predicted box location relative to Ground Truth box
    based on Intersection over Union metric to score high matching.

    Params
    ------
    target_boxes : (torch.tensor)
        midpoint format boxes {cx, cy, w, h} of shape --> (N, 4) 
    predicted_boxes : (torch.tensor)
        {cx, cy, w, h} of shape --> (N, 4) 
    
    Return
    ------
    iou_scores : (torch.tensor)
        iou scores of shape --> (N, 1)
    """
    tb_cx, tb_cy, tb_w, tb_h  = target_boxes.split(1, dim=1)
    pb_cx, pb_cy, pb_w, pb_h = predicted_boxes.split(1, dim=1)
    # midpoint anchor box format
    tb_x1, tb_y1 = tb_cx - tb_w/2, tb_cy - tb_h/2
    tb_x2, tb_y2 = tb_cx + tb_w/2, tb_cy + tb_h/2
    pb_x1, pb_y1 = pb_cx - pb_w, pb_cy - pb_h
    pb_x2, pb_y2 = pb_cx + pb_w/2, pb_cy + pb_h/2
    # intersection corners estimation
    inter_x1 = torch.max(tb_x1, bp_x1)
    inter_y1 = torch.max(tb_y1, bp_y1)
    inter_x2 = torch.min(tb_x2, bp_x2)
    inter_y2 = torch.min(tb_y2, bp_y2)
    # iou evaluation
    inter_area = (inter_x2-inter_x1).clamp(0) * (inter_y2-inter_y1).clamp(0)
    union_area = torch.abs((tb_w * tb_h) + (pb_w * pb_h)) - inter_area
    iou_scores = inter_area / union_area
    
    return iou_scores + 1e-6
    
    