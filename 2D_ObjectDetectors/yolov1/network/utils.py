import torch
import  numpy as np


def eval_iou(target_boxes, predicted_boxes):
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
    gboxes = change_boxes_format(target_boxes)
    pboxes = change_boxes_format(predicted_boxes)
    g_x1, g_y1, g_x2, g_y2  = gboxes.split(1, dim=-1)
    p_x1, p_y1, p_x2, p_y2 = pboxes.split(1, dim=-1)
    # intersection corners estimation
    inter_x1 = torch.max(g_x1, p_x1)
    inter_y1 = torch.max(g_y1, p_y1)
    inter_x2 = torch.min(g_x2, p_x2)
    inter_y2 = torch.min(g_y2, p_y2)
    # iou evaluation
    inter_area = (inter_x2-inter_x1).clamp(0) * (inter_y2-inter_y1).clamp(0)
    g_area = torch.abs((g_x2 - g_x1) * (g_y2 - g_y1))
    p_area = torch.abs((p_x2 - p_x1) * (p_y2 - p_y1))
    union_area = g_area + p_area - inter_area
    iou_scores = inter_area / union_area
    return iou_scores + 1e-6


def change_boxes_format(boxes, in_format="midpoint", out_format="corners"):
    if in_format == "midpoint":
        if out_format == "corners":
            cx, cy, w, h = boxes.split(1, dim=-1)
            x1, y1 = cx-w/2, cy-h/2
            x2, y2 = cx+w/2, cy+h/2
            return torch.hstack([x1, y1, x2, y2])


def non_max_suppression(bboxes, iou_threshold, prob_threshold):
    assert type(bboxes) == list
    bboxes = [bbox for bbox in bboxes if bbox[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    filtered_bboxes = []
    while bboxes:
        chosen_bbox = bboxes.pop(0)
        bboxes = []
        for bbox in bboxes:
            if bbox[0] != chosen_bbox[0] or eval_iou(
                torch.tensor(chosen_bbox[2:]), 
                torch.tensor(bbox[2:])
            ) < iou_threshold:
                bboxes.append(bbox)
        filtered_bboxes.append(chosen_bbox)
    return filtered_bboxes