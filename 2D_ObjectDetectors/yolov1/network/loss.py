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
        self.p_noobj = 0.5
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
        predection_matrix = predection_matrix.reshape((-1, self.S, self.S, self.C + self.B*5))
        # # scoring anchor boxes
        iou_b1 = eval_iou(predection_matrix[..., 21:25], target_matrix[..., 21:25])
        iou_b2 = eval_iou(predection_matrix[..., 26:30], target_matrix[..., 21:25])
        iou_scores = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # print("ious shape >>> ", iou_scores.shape)

        # filtering predicted anchor boxes
        iou_max_scores, bestboxes_args = torch.max(iou_scores, dim=0)
        # print("maxargs >>>\t ", bestboxes_args.shape)
        objness = target_matrix[..., self.C].unsqueeze(3)
        # print("\n"*3, "objness >>> ", objness, "\n"*3)

        # indexing box confidence, location
        gndtruth_box = objness * target_matrix[..., 21:25] 
        gndtruth_box[..., 2:4] = torch.sqrt(gndtruth_box[..., 2:4])
        # p_anchor_box shape: (N, S, S, 4), where 4: x, y, w, h
        p_anchor_box = objness * (bestboxes_args * predection_matrix[..., 26:30]
            + (1 - bestboxes_args) * predection_matrix[..., 21:25])
        # print("p_anchor_box >>> ", p_anchor_box.shape)
        print("target_matrix >>> ", target_matrix.shape)
        print("predection_matrix >>> ", predection_matrix.shape)

        # prevent grads of being zero or box dims being negative value
        p_anchor_box_w_h = p_anchor_box[..., 2:4]
        p_anchor_box_w_h = torch.sign(p_anchor_box_w_h)*torch.sqrt(torch.abs(p_anchor_box_w_h+1e-6))
        p_anchor_box[..., 2:4] = p_anchor_box_w_h

        # -------------- Box Coordinates Loss --------------
        #           flattent(N, S, S, 4) --> (N*S*S, 4)
        box_loc_loss = self.mse(
            torch.flatten(gndtruth_box, end_dim=-2), 
            torch.flatten(p_anchor_box, end_dim=-2)
        )
        # print("gndtruth_box >>> ", gndtruth_box.shape)
        # print("p_anchor_box >>> ", p_anchor_box.shape)

        # --------------    Object Loss    --------------
        #           flattent(N, S, S, 1) --> (N*S*S, 1)
        p_box_score = objness * (bestboxes_args * predection_matrix[..., 25:26]
            + (1 - bestboxes_args) * predection_matrix[..., 20:21])
        objness_loss = self.mse(
            torch.flatten(objness * target_matrix[..., 20:21], end_dim=-2),
            torch.flatten(objness * p_box_score, end_dim=-2)
        )
        # print("true_objness_vals >>> ", target_matrix[..., 20:21].shape)
        # print("pred_objness_vals >>> ", p_box_score.shape)

        # --------------    No Object Loss    --------------
        #           flattent(N, S, S, 1) --> (N*S*S, 1)
        no_objness_loss = self.mse(
            torch.flatten((1-objness) * target_matrix[..., 20:21], end_dim=-2),
            torch.flatten((1-objness) * predection_matrix[..., 20:21], end_dim=-2)
        )
        no_objness_loss += self.mse(
            torch.flatten((1-objness) * target_matrix[..., 20:21], end_dim=-2),
            torch.flatten((1-objness) * predection_matrix[..., 25:26], end_dim=-2)
        )
        # print("true_no_objness_vals >>> ", ((1-objness)*target_matrix[..., 20:21]).shape)
        # print("pred_no_objness_vals >>> ", ((1-objness)*predection_matrix[..., 20:21]).shape)


        # --------------    Class Loss    --------------
        #           flattent(N, S, S, 20) --> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(objness * target_matrix[..., :self.C], end_dim=-2),
            torch.flatten(objness * predection_matrix[..., :self.C], end_dim=-2)
        )
        # print("true_class >>> ", target_matrix[..., :self.C].shape)
        # print("pred_class >>> ", predection_matrix[..., :self.C].shape)

        loss = (self.coord * box_loc_loss) + objness_loss
        loss +=  (self.p_noobj * no_objness_loss) + class_loss

        return loss

    # def forward(self, predection_matrix, target_matrix):
    #     """
    #     Estimate the loss for boxes regression, confidence of predictions, 
    #     and classes category predictions.

    #     Params
    #     ------
    #     predection_matrix : (torch.tensor)
    #         output of the FCL layers of shape --> (N, S* S*B*(C+5))

    #     target_matrix : (torch.tensor)
    #         true target matrix of shape --> (N, S, S, B, C+5)

    #     Returns
    #     -------
    #     loss : (torch.tensor)
    #         the batch mean squared error of all grid cell boxes per image 
    #     """
    #     # reshaping predicted matrix from (N, S* S*B*(C+5)) --> (N, S, S, B, C+5)
    #     grid_shape = (-1, self.S, self.S, self.B, (self.C+5))
    #     predection_matrix = predection_matrix.reshape(grid_shape)
    #     # -------------------------------------------------
    #     true_classes = target_matrix[..., :self.C]
    #     true_objness = target_matrix[..., self.C:self.C+1]
    #     true_boxes = target_matrix[..., self.C+1:self.C+5]
    #     # -------------------------------------------------
    #     pred_classes = predection_matrix[..., :self.C]
    #     pred_objness = predection_matrix[..., self.C:self.C+1]
    #     pred_boxes = predection_matrix[..., self.C+1:self.C+5]
    #     # -------------------------------------------------
    #     # scoring anchor boxes per cell for all cells for all N-examples
    #     iou_scores = eval_iou(pred_boxes.reshape(-1, 4), true_boxes.reshape(-1, 4))
    #     iou_scores = iou_scores.reshape(-1, self.S, self.S, self.B, 1)
    #     # -----------------------------------------------------------
    #     best_boxes_arg = torch.argmax(iou_scores, dim=-2).unsqueeze(-1)
    #     # matching dims between indices and the target tensor over the last dim
    #     # best_boxes_arg = torch.repeat_interleave(best_boxes_arg, 4, dim=-1)
    #     # indexing the second last dim using the best box arg of the max iou score 
    #     # bestboxes = torch.gather(pred_boxes, dim=-2, index=best_boxes_arg).squeeze()
    #     print("debugger >>> ", bestboxes.shape)
    #     # indexing box confidence, location
    #     # anchor_box shape: (N, S, S, 4), where 4: x, y, w, h
    #     true_boxes = true_boxes[..., 0, :]
    #     true_boxes[..., 2:4] = torch.sqrt(true_boxes[..., 2:4])
    #     print("debugger >>> ", true_boxes.shape)
    #     # prevent grads of being zero or box dims being negative value
    #     # anchor_box shape: (N, S, S, 4), where 4: x, y, w, h
    #     bestboxes_w_h = bestboxes[..., 2:4]
    #     bestboxes_w_h = torch.sign(bestboxes_w_h)*torch.sqrt(torch.abs(bestboxes_w_h+1e-6))
    #     bestboxes[..., 2:4] = bestboxes_w_h
    #     print("debugger >>> ", bestboxes.shape)
    #     # -------------- Box Coordinates Loss --------------
    #     #           flattent(N, S, S, 4) --> (N*S*S, 4)
    #     box_loc_loss = self.mse(
    #         torch.flatten(true_boxes, end_dim=-2), 
    #         torch.flatten(bestboxes, end_dim=-2)
    #     )
    #     # --------------    Object Loss    --------------
    #     #           flattent(N, S, S, 1) --> (N*S*S, 1)
    #     p_box_score = objness * (bestboxes_args * predection_matrix[..., 25:26]
    #         + (1 - bestboxes_args) * predection_matrix[..., 20:21])
    #     objness_loss = self.mse(
    #         torch.flatten(objness * target_matrix[..., 20:21], end_dim=-2),
    #         torch.flatten(objness * p_box_score, end_dim=-2)
    #     )
    #     # print("true_objness_vals >>> ", target_matrix[..., 20:21].shape)
    #     # print("pred_objness_vals >>> ", p_box_score.shape)

    #     # --------------    No Object Loss    --------------
    #     #           flattent(N, S, S, 1) --> (N*S*S, 1)
    #     no_objness_loss = self.mse(
    #         torch.flatten((1-objness) * target_matrix[..., 20:21], end_dim=-2),
    #         torch.flatten((1-objness) * predection_matrix[..., 20:21], end_dim=-2)
    #     )
    #     no_objness_loss += self.mse(
    #         torch.flatten((1-objness) * target_matrix[..., 20:21], end_dim=-2),
    #         torch.flatten((1-objness) * predection_matrix[..., 25:26], end_dim=-2)
    #     )
    #     # --------------    Class Loss    --------------
    #     #           flattent(N, S, S, 20) --> (N*S*S, 20)
    #     class_loss = self.mse(
    #         torch.flatten(objness * target_matrix[..., :self.C], end_dim=-2),
    #         torch.flatten(objness * predection_matrix[..., :self.C], end_dim=-2)
    #     )
    #     loss = (self.coord * box_loc_loss) + objness_loss
    #     loss +=  (self.p_noobj * no_objness_loss) + class_loss

    #     return loss