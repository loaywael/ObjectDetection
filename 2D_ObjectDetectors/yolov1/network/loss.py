import torch
import torch.nn as nn
from network.utils import eval_iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, C):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.noobj = 0.5
        self.coord = 5.0

    def forward(self, predection_matrix, target_matrix):
        """
        Estimate the loss for boxes regression, confidence of predictions, 
        and classes category predictions.

        Params
        ------
        predection_matrix : (torch.tensor)
            output of the FCL layers of shape --> (N, S*S*(C+5*B))

        target_matrix : (torch.tensor)
            true target matrix of shape --> (N, S, S, C+5*B)

        Returns
        -------
        loss : (torch.tensor)
            the batch mean squared error of all grid cell boxes per image 
        """
        # reshaping predicted matrix from (N, S*S*(C+5*B)) --> (N, S, S, C+5*B)
        N = predection_matrix.size(0)
        S, B, C = self.S, self.B, self.C
        predection_matrix = predection_matrix.reshape((-1, S, S, C+B*5))
        objness_mask = target_matrix[..., C] > 0
        objness_mask = objness_mask.unsqueeze(-1).expand_as(target_matrix)
        noobj_mask = ~ objness_mask
        # ---------------------------------------
        # reshaping predicted matrix from (N, S, S, C+5*B) --> (N*S*S, C+5*B)
        obj_pred_mat = predection_matrix[objness_mask].reshape(-1, C+5*B)
        pred_boxes = obj_pred_mat[:, C:].contiguous().reshape(-1, 5)    # (N*S*S*B, 5)
        pred_classes = obj_pred_mat[:, :C]  # (N*S*S, 20)
        # print(pred_boxes.shape, pred_classes.shape)
        # reshaping target matrix from (N, S, S, C+5*B) --> (N*S*S, C+5*B)
        obj_target_mat = target_matrix[objness_mask].reshape(-1, C+5*B)
        target_boxes = obj_target_mat[:, C:].contiguous().reshape(-1, 5)    # (N*S*S*B, 5)
        target_classes = obj_target_mat[:, :C]  # (N*S*S, 20)
        # print(target_boxes.shape, target_classes.shape)
        # ---------------------------------------
        noobj_pred_mat = predection_matrix[noobj_mask].reshape(-1, C+5*B)
        noobj_target_mat = target_matrix[noobj_mask].reshape(-1, C+5*B)
        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred_mat.size()).fill_(0)

        for b in range(B):
            noobj_conf_mask[:, C+5*b] = 1   # pred_mat[noobj_ids, box_conf_id]--> 1 
        noobj_pred_conf = noobj_pred_mat[noobj_conf_mask]
        noobj_target_conf = noobj_target_mat[noobj_conf_mask]
        # print(noobj_pred_conf.shape, noobj_target_conf.shape)
        # scoring anchor boxes
        obj_resp_mask = torch.cuda.BoolTensor(target_boxes.size()).fill_(0)
        noobj_resp_mask = torch.cuda.BoolTensor(target_boxes.size()).fill_(1)
        target_iou_score = torch.zeros(target_boxes.size()).cuda()
        for i in range(0, target_boxes.size(0), B):
            pred_cell_boxes = pred_boxes[i:i+B]     # (B, 5)
            target_cell_box = target_boxes[i].reshape(-1, 5)
            pred_cell_xyxy = torch.FloatTensor(pred_cell_boxes.size())  # (B, 5) midpoint --> corner format
            target_cell_xyxy = torch.FloatTensor(pred_cell_boxes.size())  # (B, 5) midpoint --> corner format
            pred_cell_xyxy[:, 1:3] = pred_cell_boxes[:, 1:3]/float(S) - 0.5*pred_cell_boxes[:,3:5]
            pred_cell_xyxy[:, 3:5] = pred_cell_boxes[:, 1:3]/float(S) + 0.5*pred_cell_boxes[:,3:5]
            target_cell_xyxy[:, 1:3] = target_cell_box[:, 1:3]/float(S) - 0.5*target_cell_box[:,3:5]
            target_cell_xyxy[:, 3:5] = target_cell_box[:, 1:3]/float(S) + 0.5*target_cell_box[:,3:5]
            iou = eval_iou(pred_cell_xyxy[:, 1:], target_cell_xyxy[:, 1:])
            max_iou, max_id = iou.max(dim=0)
            max_id = max_id.cuda()
            obj_resp_mask[i+max_id] = 1
            noobj_resp_mask[i+max_id] = 0
            target_iou_score[i+max_id, torch.LongTensor([0]).cuda()] = (max_iou).data.cuda()
        target_iou_score = target_iou_score.cuda()
        pred_box = pred_boxes[obj_resp_mask].reshape(-1, 5)
        target_box = target_boxes[obj_resp_mask].reshape(-1, 5)
        target_iou = target_iou_score[obj_resp_mask].reshape(-1, 5)
        xy_loss = self.mse(pred_box[:, 1:3], target_box[:, 1:3])
        wh_loss = self.mse(torch.sqrt(pred_box[:, 3:5]), torch.sqrt(target_box[:, 3:5]))
        obj_loss = self.mse(pred_box[:, 0], target_box[:, 0])
        noobj_loss = self.mse(noobj_pred_conf, noobj_target_conf)
        class_loss = self.mse(pred_classes, target_classes)
        # print("iou score >>> ", pred_box, target_box, target_iou)
        loss = (self.coord * (xy_loss+wh_loss)) + obj_loss + (self.noobj * noobj_loss) + class_loss
        
        return loss / float(N)
