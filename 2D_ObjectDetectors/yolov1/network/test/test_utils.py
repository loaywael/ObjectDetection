from unittest import TestCase
import numpy as np
import torch
from network.utils import eval_iou


class TestModel(TestCase):
    def setUp(self):
        self.y_boxes = torch.randn(2, 7, 7, 4)
        self.y_boxes[0:2, 0, 0, :] = torch.tensor(([[2, 4, 8, 6], [35, 22, 16, 9]]))
        self.yhat_boxes = torch.randn(2, 7, 7, 4)
        self.yhat_boxes[0:2, 0, 0, :] = torch.tensor([[2, 4, 7, 6], [30, 15, 23, 15]])
        self.iou_scores = eval_iou(self.y_boxes, self.yhat_boxes)

    def test_iou_scores_shape(self):
        iou_shape = list(self.iou_scores.shape)
        print("iou output shape: ", iou_shape)
        # self.assertEqual(iou_shape, [2, 1])

    def test_iou_scores_value(self):
        print("iou_scores: ", self.iou_scores[:, 0, 0, :])
    