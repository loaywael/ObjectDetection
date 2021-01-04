import torch
import numpy as np
from unittest import TestCase
from torchvision import ops
from network.utils import change_boxes_format
from network.utils import eval_iou, non_max_suppression



class TestModel(TestCase):
    def setUp(self):
        self.y_boxes = torch.tensor(([[2, 4, 2+8, 4+6], [35, 22, 35+16, 22+9]]))
        self.yhat_boxes = torch.tensor([[2, 4, 2+7, 4+6], [30, 15, 30+23, 15+15]])
        # ---------------------------------------------------------------
        self.y1_boxes = torch.tensor(([[2+8/2, 4+6/2, 8, 6], [35+16/2, 22+9/2, 16, 9]]))
        self.yhat1_boxes = torch.tensor([[2+7/2, 4+6/2, 7, 6], [30+23/2, 15+15/2, 23, 15]])

    def test_change_box_format(self):
        y1_boxes = change_boxes_format(self.y1_boxes)
        np.testing.assert_array_almost_equal(y1_boxes.numpy(), self.y_boxes.numpy())

    def test_iou_scores_value(self):
        y1_boxes = change_boxes_format(self.y1_boxes)
        yhat1_boxes = change_boxes_format(self.yhat1_boxes)
        ref_iou_scores, _ = ops.box_iou(y1_boxes, yhat1_boxes).max(-1)
        iou_scores = eval_iou(self.y1_boxes, self.yhat1_boxes).squeeze()
        np.testing.assert_array_almost_equal(iou_scores.numpy(), ref_iou_scores.numpy())
        # print("iou_scores: ", ref_iou_scores)
        # print("-- "*15)
        # print("iou_scores: ", iou_scores)

    def test_nms(self):
        bboxes = [
            [12, 84, 140, 212], 
            [24, 84, 152, 212], 
            [36, 84, 164, 212], 
            [12, 96, 140, 224], 
            [24, 96, 152, 224], 
            [24, 108, 152, 236], 
        ]
        bboxes1 = torch.tensor(bboxes).type(torch.FloatTensor)
        scores = [0.5, 0.7, 0.3, 0.2, 0.97, 0.95]
        scores1 = torch.tensor(scores)

        # boxes = non_max_suppression(bboxes, 0.5, 0.5)
        boxes1_ids = ops.nms(bboxes1, scores1, 0.5)
        
        print(boxes)
        print(bboxes1[boxes1_ids])



    