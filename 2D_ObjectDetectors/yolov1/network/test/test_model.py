from unittest import TestCase
import numpy as np
import torch
from model import Yolov1


class TestModel(TestCase):
    def setUp(self):
        self.model = Yolov1(grid_size=7, num_boxes=2, num_classes=20)

    def test_yolov1(self):
        x = torch.randn((2, 3, 448, 448))
        prediction_shape = list(self.model(x).shape)
        # self.assertEqual(prediction_shape, [2, 1470])
        print(prediction_shape)
    