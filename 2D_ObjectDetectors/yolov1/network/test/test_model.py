from unittest import TestCase
import numpy as np
import torch
from network.model import Yolov1, YoloLoss


class TestModel(TestCase):
    def setUp(self):
        self.x = torch.abs(torch.randint(0, 255, (2, 3, 448, 448)))/255.
        self.y = torch.abs(torch.randint(0, 1000, (2, 7, 7, 30)))/1000.

        self.model = Yolov1(grid_size=7, num_boxes=2, num_classes=20)
        self.predictions = self.model(self.x)
        self.model_loss = YoloLoss(7, 2, 20)

    def test_yolov1(self):
        prediction_shape = list(self.predictions.shape)
        self.assertEqual(prediction_shape, [2, 1470])
        print(prediction_shape)

    def test_yolo_loss(self):
        loss = self.model_loss(self.predictions, self.y)
        print(loss)
    