import torch
import  numpy as np



def eval_iou(target_boxes, predicted_boxes):
    """
    Evaluates Predicted box location relative to Ground Truth box
    based on Intersection over Union metric to score high matching.

    Params
    ------
    target_boxes : (torch.tensor)
        midpoint format boxes {cx, cy, w, h} of shape --> (N, S, S, 4) 
    predicted_boxes : (torch.tensor)
        {cx, cy, w, h} of shape --> (N, S, S, 4) 
    
    Return
    ------
    iou_scores : (torch.tensor)
        iou scores of shape --> (N, S, S, 1)
    """
    tb_cx, tb_cy, tb_w, tb_h  = target_boxes.split(1, dim=-1)
    pb_cx, pb_cy, pb_w, pb_h = predicted_boxes.split(1, dim=-1)
    print(tb_cx.shape, tb_cy.shape, tb_w.shape, tb_h.shape)
    print(pb_cx.shape, pb_cy.shape, pb_w.shape, pb_h.shape)
    # midpoint anchor box format
    tb_x1, tb_y1 = tb_cx - tb_w/2, tb_cy - tb_h/2
    tb_x2, tb_y2 = tb_cx + tb_w/2, tb_cy + tb_h/2
    pb_x1, pb_y1 = pb_cx - pb_w/2, pb_cy - pb_h/2
    pb_x2, pb_y2 = pb_cx + pb_w/2, pb_cy + pb_h/2
    # intersection corners estimation
    inter_x1 = torch.max(tb_x1, pb_x1)
    inter_y1 = torch.max(tb_y1, pb_y1)
    inter_x2 = torch.min(tb_x2, pb_x2)
    inter_y2 = torch.min(tb_y2, pb_y2)
    # iou evaluation
    inter_area = (inter_x2-inter_x1).clamp(0) * (inter_y2-inter_y1).clamp(0)
    union_area = torch.abs((tb_w * tb_h) + (pb_w * pb_h)) - inter_area
    iou_scores = inter_area / union_area
    
    return iou_scores + 1e-6
    
    
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

